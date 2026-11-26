"""
Fall Detection System - Dataset Handler
========================================
Handles loading, preprocessing, and creating PyTorch datasets
from fall detection video data.
"""

import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from tqdm import tqdm

try:
    from fyp1_config import SystemConfig, CONFIG
except ImportError:
    from config import SystemConfig, CONFIG
from pose_estimator import PoseEstimator
from feature_extractor import FeatureExtractor, TemporalFeatureProcessor


# =============================================================================
# Label Mapping
# =============================================================================

LABEL_MAP = {
    "normal": 0,    # Walking, standing, sitting
    "warning": 1,   # Unstable posture, stumbling
    "fall": 2,      # Fall detected
}

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


# =============================================================================
# Video Preprocessing Utilities
# =============================================================================

def extract_frames_from_video(
    video_path: str,
    target_fps: int = 15,
    max_frames: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Extract frames from a video file at a target FPS.

    Args:
        video_path: Path to video file
        target_fps: Desired output frame rate
        max_frames: Maximum number of frames to extract

    Returns:
        List of BGR frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0

    frame_interval = max(1, int(src_fps / target_fps))
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frames.append(frame)

        if max_frames and len(frames) >= max_frames:
            break

        frame_idx += 1

    cap.release()
    return frames


def extract_keypoints_from_video(
    video_path: str,
    pose_estimator: PoseEstimator,
    target_fps: int = 15,
) -> np.ndarray:
    """
    Extract pose keypoints from all frames of a video.

    Returns:
        keypoints_array: (num_frames, num_keypoints, 3)
    """
    frames = extract_frames_from_video(video_path, target_fps)
    keypoints_list = []

    for frame in frames:
        kps, _ = pose_estimator.process_frame(frame)
        if kps is not None:
            keypoints_list.append(kps)
        else:
            # Zero-pad missing frames
            keypoints_list.append(
                np.zeros((len(CONFIG.pose.CRITICAL_KEYPOINTS), 3), dtype=np.float32)
            )

    return np.array(keypoints_list)


def preprocess_dataset_folder(
    data_dir: str,
    output_dir: str,
    pose_estimator: PoseEstimator,
    feature_config=None,
):
    """
    Preprocess an entire dataset folder structure.
    Expected structure:
        data_dir/
            normal/
                video1.mp4
                video2.avi
            warning/
                video3.mp4
            fall/
                video4.mp4

    Saves processed features and labels as .npz files.
    """
    feature_config = feature_config or CONFIG.features
    extractor = FeatureExtractor(feature_config, CONFIG.pose)
    processor = TemporalFeatureProcessor(feature_config)

    os.makedirs(output_dir, exist_ok=True)

    all_features = []
    all_labels = []
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

    for label_name, label_id in LABEL_MAP.items():
        label_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_dir):
            print(f"Warning: Directory not found: {label_dir}")
            continue

        video_files = [
            f
            for f in os.listdir(label_dir)
            if Path(f).suffix.lower() in video_extensions
        ]

        print(f"\nProcessing {label_name} ({len(video_files)} videos)...")

        for vf in tqdm(video_files, desc=label_name):
            video_path = os.path.join(label_dir, vf)
            try:
                extractor.reset()
                keypoints = extract_keypoints_from_video(
                    video_path, pose_estimator
                )

                # Extract features for each frame
                frame_features = []
                for kps in keypoints:
                    raw = kps.flatten()
                    eng = extractor.extract_frame_features(kps)
                    combined = np.concatenate([raw, eng])
                    frame_features.append(combined)

                frame_features = np.array(frame_features)
                frame_labels = np.full(len(frame_features), label_id, dtype=np.int64)

                # Create temporal sequences
                seqs, seq_labels = processor.create_sequences(
                    frame_features, frame_labels
                )

                if len(seqs) > 0:
                    all_features.append(seqs)
                    all_labels.append(seq_labels)

            except Exception as e:
                print(f"  Error processing {vf}: {e}")
                continue

    if not all_features:
        raise RuntimeError("No features extracted. Check your data directory.")

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # Normalize
    X = processor.fit_normalize(X)
    norm_mean, norm_std = processor.get_normalization_params()

    # Save
    np.savez(
        os.path.join(output_dir, "dataset.npz"),
        X=X,
        y=y,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    print(f"\nDataset saved to {output_dir}/dataset.npz")
    print(f"  Total sequences: {len(X)}")
    for label_name, label_id in LABEL_MAP.items():
        count = (y == label_id).sum()
        print(f"  {label_name}: {count} ({count / len(y) * 100:.1f}%)")

    return X, y, norm_mean, norm_std


# =============================================================================
# PyTorch Dataset
# =============================================================================

from augmentations import PoseSequenceAugmenter


class FallDetectionDataset(Dataset):
    """
    PyTorch Dataset for fall detection temporal sequences.
    Applies biomechanical and temporal augmentations when augment=True:
      - Temporal jitter, warping, and rolling shift
      - Horizontal flipping with symmetrical joint swapping
      - Coordinate noise and spatial distance scaling
      - Joint occlusion / dropout
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        augmenter: Optional[PoseSequenceAugmenter] = None,
        processor: Optional[TemporalFeatureProcessor] = None,
    ):
        """
        Args:
            features: (N, sequence_length, num_features) array
            labels: (N,) array of integer labels
            augment: Whether to apply data augmentation
            augmenter: Optional custom PoseSequenceAugmenter
            processor: TemporalFeatureProcessor for normalization
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment
        self.augmenter = augmenter or (PoseSequenceAugmenter() if augment else None)
        self.processor = processor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]

        if self.augment and self.augmenter is not None:
            # Apply NumPy-based sequence augmentation
            x_np = x.numpy()
            y_int = int(y.item())
            x_aug, y_aug = self.augmenter.augment(x_np, y_int)
            return torch.from_numpy(x_aug).float(), torch.tensor(y_aug, dtype=torch.long)

        return x, y



# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_dataset(
    data_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load preprocessed dataset from .npz file."""
    data = np.load(data_path)
    return data["X"], data["y"], data["norm_mean"], data["norm_std"]


def create_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    augment_train: bool = True,
    class_weights: Optional[List[float]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders with stratified splitting
    and optional class-balanced sampling.

    Returns:
        train_loader, val_loader, test_loader
    """
    from sklearn.model_selection import train_test_split

    # Stratified split: first separate test, then split remaining into train/val
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_split, stratify=y, random_state=42
    )

    relative_val = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=relative_val, stratify=y_trainval, random_state=42
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Datasets
    train_dataset = FallDetectionDataset(X_train, y_train, augment=augment_train)
    val_dataset = FallDetectionDataset(X_val, y_val, augment=False)
    test_dataset = FallDetectionDataset(X_test, y_test, augment=False)

    # Weighted sampling for class imbalance
    train_sampler = None
    shuffle_train = True

    if class_weights:
        sample_weights = [class_weights[label] for label in y_train]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle_train = False  # Sampler handles ordering

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# Synthetic Data Generator (for testing without real data)
# =============================================================================

# ---------------------------------------------------------------------------
# Critical keypoint layout for synthetic generation.
# Must mirror PoseConfig.CRITICAL_KEYPOINTS:
#   [0:nose, 1:l_shoulder, 2:r_shoulder, 3:l_elbow, 4:r_elbow,
#    5:l_wrist, 6:r_wrist, 7:l_hip, 8:r_hip, 9:l_knee, 10:r_knee,
#    11:l_ankle, 12:r_ankle]  → 13 points × 3 = 39 raw features
# ---------------------------------------------------------------------------

def _build_upright_kps(t: int, noise: float) -> np.ndarray:
    """13-keypoint normalized upright skeleton, matching CRITICAL_KEYPOINTS."""
    sway = float(np.sin(t * 0.3) * 0.02)
    cx = 0.5 + sway
    # [x, y, conf] per keypoint
    kps = np.array([
        [cx,        0.15, 0.95],  # 0  nose
        [cx - 0.07, 0.30, 0.95],  # 1  l_shoulder
        [cx + 0.07, 0.30, 0.95],  # 2  r_shoulder
        [cx - 0.10, 0.43, 0.90],  # 3  l_elbow
        [cx + 0.10, 0.43, 0.90],  # 4  r_elbow
        [cx - 0.10, 0.56, 0.85],  # 5  l_wrist
        [cx + 0.10, 0.56, 0.85],  # 6  r_wrist
        [cx - 0.05, 0.58, 0.95],  # 7  l_hip
        [cx + 0.05, 0.58, 0.95],  # 8  r_hip
        [cx - 0.06, 0.72, 0.90],  # 9  l_knee
        [cx + 0.06, 0.72, 0.90],  # 10 r_knee
        [cx - 0.06, 0.88, 0.90],  # 11 l_ankle
        [cx + 0.06, 0.88, 0.90],  # 12 r_ankle
    ], dtype=np.float32)
    kps += np.random.normal(0, noise, kps.shape).astype(np.float32)
    kps = np.clip(kps, 0.0, 1.0)
    return kps  # shape (13, 3)


def _build_tilting_kps(t: int, progress: float, noise: float) -> np.ndarray:
    """Tilting/unstable skeleton – upper body shifts sideways."""
    kps = _build_upright_kps(t, noise)
    tilt = float(progress) * 0.15
    upper = [0, 1, 2, 3, 4, 5, 6]  # upper-body indices in 13-point array
    for i in upper:
        kps[i, 0] = np.clip(kps[i, 0] + tilt, 0.0, 1.0)
        kps[i, 1] = np.clip(kps[i, 1] + progress * 0.05, 0.0, 1.0)
    return kps


def _build_falling_kps(t: int, progress: float, noise: float) -> np.ndarray:
    """All keypoints dropping rapidly toward bottom of frame."""
    kps = _build_upright_kps(t, noise)
    for i in range(13):
        kps[i, 0] = np.clip(kps[i, 0] + progress * 0.1 * np.random.uniform(-1.0, 1.0), 0.0, 1.0)
        kps[i, 1] = np.clip(kps[i, 1] + progress * 0.35, 0.0, 1.0)
    return kps


def _build_lying_kps(t: int, noise: float) -> np.ndarray:
    """Horizontal skeleton near bottom of frame."""
    base_y = 0.85 + float(np.random.uniform(-0.03, 0.03))
    kps = np.zeros((13, 3), dtype=np.float32)
    for i in range(13):
        kps[i, 0] = np.clip(0.25 + (i / 12.0) * 0.50 + np.random.normal(0, noise), 0.0, 1.0)
        kps[i, 1] = np.clip(base_y + np.random.normal(0, 0.02), 0.0, 1.0)
        kps[i, 2] = 0.70 + float(np.random.uniform(0, 0.25))
    return kps


def generate_synthetic_data(
    num_samples_per_class: int = 200,
    sequence_length: int = 30,
    num_features: int = 51,
    noise_level: float = 0.05,
    num_normal: Optional[int] = None,
    num_warning: Optional[int] = None,
    num_fall: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic fall detection data for testing and class-imbalance benchmarks.
    Creates realistic patterns for normal, warning, and fall activities.

    Feature layout (51 values per frame):
      0–38  Raw keypoints  (13 × 3 = 39, matching CRITICAL_KEYPOINTS)
      39–50 Engineered features computed by FeatureExtractor.extract_frame_features()

    Returns:
        X: (total_samples, sequence_length, num_features)
        y: (total_samples,) labels
        norm_mean: (num_features,) mean for normalization
        norm_std: (num_features,) std for normalization
    """
    from feature_extractor import FeatureExtractor, FeatureConfig, PoseConfig, TemporalFeatureProcessor

    np.random.seed(42)
    all_X: List[np.ndarray] = []
    all_y: List[int] = []

    fe = FeatureExtractor(FeatureConfig(), PoseConfig())

    # Determine sample counts per class
    counts = {
        0: num_normal if num_normal is not None else num_samples_per_class,
        1: num_warning if num_warning is not None else num_samples_per_class,
        2: num_fall if num_fall is not None else num_samples_per_class,
    }

    for label in range(3):
        n_samples = counts[label]
        for _ in range(n_samples):
            # Reset extractor state so head-velocity tracks correctly per sequence.
            fe.reset()
            seq = np.zeros((sequence_length, num_features), dtype=np.float32)

            for t in range(sequence_length):
                if label == 0:  # Normal – standing / walking
                    kps = _build_upright_kps(t, noise_level)

                elif label == 1:  # Warning – tilting / stumbling
                    progress = t / sequence_length
                    kps = _build_tilting_kps(t, progress, noise_level)

                else:  # Fall
                    fall_start = int(np.random.randint(8, 18))
                    if t < fall_start:
                        kps = _build_upright_kps(t, noise_level)
                    elif t < fall_start + 5:
                        fall_progress = (t - fall_start) / 5.0
                        kps = _build_falling_kps(t, fall_progress, noise_level)
                    else:
                        kps = _build_lying_kps(t, noise_level)

                raw = kps.flatten()                          # (39,)
                eng = fe.extract_frame_features(kps)         # (12,) via FeatureExtractor
                seq[t] = np.concatenate([raw, eng])          # (51,)

            all_X.append(seq)
            all_y.append(label)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    # Shuffle
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]

    # Normalize
    processor = TemporalFeatureProcessor(FeatureConfig(sequence_length=sequence_length))
    X = processor.fit_normalize(X)
    norm_mean, norm_std = processor.get_normalization_params()

    return X, y, norm_mean, norm_std



# Legacy helper stubs kept so existing imports don't break.
# The public API is now _build_*_kps; these flat-array versions are unused.
def _generate_upright_keypoints(t, noise):
    return _build_upright_kps(t, noise).flatten()


def _generate_tilting_keypoints(t, progress, noise):
    return _build_tilting_kps(t, progress, noise).flatten()


def _generate_falling_keypoints(t, progress, noise):
    return _build_falling_kps(t, progress, noise).flatten()


def _generate_lying_keypoints(t, noise):
    return _build_lying_kps(t, noise).flatten()
