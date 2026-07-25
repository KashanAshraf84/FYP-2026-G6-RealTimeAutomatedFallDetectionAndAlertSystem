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

class FallDetectionDataset(Dataset):
    """PyTorch Dataset for fall detection sequences."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        processor: Optional[TemporalFeatureProcessor] = None,
    ):
        """
        Args:
            features: (N, sequence_length, num_features) array
            labels: (N,) array of integer labels
            augment: Whether to apply data augmentation
            processor: TemporalFeatureProcessor for augmentation
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment
        self.processor = processor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]

        if self.augment:
            # Random noise augmentation during training
            if torch.rand(1).item() > 0.5:
                noise = torch.randn_like(x) * 0.01
                x = x + noise

            # Random temporal jitter
            if torch.rand(1).item() > 0.7:
                shift = torch.randint(-2, 3, (1,)).item()
                if shift != 0:
                    x = torch.roll(x, shifts=shift, dims=0)

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

def generate_synthetic_data(
    num_samples_per_class: int = 200,
    sequence_length: int = 30,
    num_features: int = 51,
    noise_level: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic fall detection data for testing the pipeline.
    Creates realistic patterns for normal, warning, and fall activities.

    Returns:
        X: (total_samples, sequence_length, num_features)
        y: (total_samples,) labels
    """
    np.random.seed(42)
    all_X = []
    all_y = []

    for label in range(3):
        for _ in range(num_samples_per_class):
            seq = np.zeros((sequence_length, num_features), dtype=np.float32)

            if label == 0:  # Normal
                # Stable keypoints with small variations (upright posture)
                base_y = 0.3 + np.random.uniform(-0.05, 0.05)
                for t in range(sequence_length):
                    # Head at top, feet at bottom, slight sway
                    seq[t, :39] = _generate_upright_keypoints(t, noise_level)
                    # Engineered features: low angle, stable height
                    seq[t, 39] = np.random.uniform(0, 0.15)     # Small body angle
                    seq[t, 40] = base_y + np.random.normal(0, 0.01)  # Stable head Y
                    seq[t, 41] = np.random.normal(0, 0.005)      # Small velocity
                    seq[t, 42:44] = [0.5, 0.5]                   # Stable torso center
                    seq[t, 44] = np.random.uniform(0, 0.02)      # Low COM velocity
                    seq[t, 45] = np.random.uniform(0.3, 0.6)     # Normal aspect ratio
                    seq[t, 46] = np.random.uniform(0.6, 0.9)     # Normal body height
                    seq[t, 47] = np.random.uniform(0, 0.1)       # Low hip-shoulder angle
                    seq[t, 48] = np.random.uniform(0.6, 0.9)     # Normal knee angle
                    seq[t, 49] = 0.0                              # No stillness
                    seq[t, 50] = 0.0                              # No transition

            elif label == 1:  # Warning
                # Unstable posture with increasing tilt
                for t in range(sequence_length):
                    progress = t / sequence_length
                    seq[t, :39] = _generate_tilting_keypoints(t, progress, noise_level)
                    seq[t, 39] = np.random.uniform(0.15, 0.4) + progress * 0.1
                    seq[t, 40] = 0.3 + progress * 0.1 + np.random.normal(0, 0.02)
                    seq[t, 41] = np.random.uniform(0.01, 0.05)
                    seq[t, 42:44] = [0.5 + progress * 0.05, 0.5 + progress * 0.05]
                    seq[t, 44] = np.random.uniform(0.02, 0.08)
                    seq[t, 45] = np.random.uniform(0.5, 0.8)
                    seq[t, 46] = np.random.uniform(0.4, 0.7)
                    seq[t, 47] = np.random.uniform(0.1, 0.3)
                    seq[t, 48] = np.random.uniform(0.3, 0.6)
                    seq[t, 49] = 0.0
                    seq[t, 50] = 0.0

            else:  # Fall
                fall_start = np.random.randint(8, 18)
                for t in range(sequence_length):
                    if t < fall_start:
                        # Pre-fall: normal or slightly unstable
                        seq[t, :39] = _generate_upright_keypoints(t, noise_level)
                        seq[t, 39] = np.random.uniform(0, 0.2)
                        seq[t, 40] = 0.3 + np.random.normal(0, 0.01)
                        seq[t, 41] = np.random.normal(0, 0.01)
                        seq[t, 44] = np.random.uniform(0, 0.03)
                        seq[t, 45] = np.random.uniform(0.3, 0.5)
                        seq[t, 46] = np.random.uniform(0.6, 0.9)
                        seq[t, 49] = 0.0
                        seq[t, 50] = 0.0
                    elif t < fall_start + 5:
                        # During fall: rapid change
                        fall_progress = (t - fall_start) / 5
                        seq[t, :39] = _generate_falling_keypoints(t, fall_progress, noise_level)
                        seq[t, 39] = 0.2 + fall_progress * 0.6          # Angle increases
                        seq[t, 40] = 0.3 + fall_progress * 0.5          # Head drops
                        seq[t, 41] = 0.05 + fall_progress * 0.15        # High velocity
                        seq[t, 44] = 0.05 + fall_progress * 0.1         # High COM velocity
                        seq[t, 45] = 0.5 + fall_progress * 0.8          # Wider aspect ratio
                        seq[t, 46] = 0.7 - fall_progress * 0.5          # Height decreases
                        seq[t, 49] = 0.0
                        seq[t, 50] = 1.0 if fall_progress > 0.3 else 0.0
                    else:
                        # Post-fall: lying still
                        seq[t, :39] = _generate_lying_keypoints(t, noise_level)
                        seq[t, 39] = np.random.uniform(0.6, 0.9)        # High angle
                        seq[t, 40] = np.random.uniform(0.7, 0.9)        # Head near ground
                        seq[t, 41] = np.random.normal(0, 0.005)          # Almost no movement
                        seq[t, 44] = np.random.uniform(0, 0.01)
                        seq[t, 45] = np.random.uniform(1.0, 2.0)        # Very wide
                        seq[t, 46] = np.random.uniform(0.1, 0.3)        # Very short
                        post_fall_time = t - (fall_start + 5)
                        seq[t, 49] = min(post_fall_time / 15.0, 1.0)    # Increasing stillness
                        seq[t, 50] = 0.0

                    seq[t, 42:44] = [0.5, 0.5]
                    seq[t, 47] = np.random.uniform(0, 0.3)
                    seq[t, 48] = np.random.uniform(0.2, 0.8)

            all_X.append(seq)
            all_y.append(label)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    # Shuffle
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


def _generate_upright_keypoints(t, noise):
    """Generate keypoints for an upright standing/walking person."""
    kps = np.zeros(39, dtype=np.float32)
    sway = np.sin(t * 0.3) * 0.02
    # Simplified: place keypoints in rough upright skeleton positions
    # Nose
    kps[0], kps[1], kps[2] = 0.5 + sway, 0.2, 0.9
    # Left/Right Ear
    kps[3], kps[4], kps[5] = 0.48 + sway, 0.18, 0.8
    kps[6], kps[7], kps[8] = 0.52 + sway, 0.18, 0.8
    # Left/Right Shoulder
    kps[9], kps[10], kps[11] = 0.45 + sway, 0.3, 0.9
    kps[12], kps[13], kps[14] = 0.55 + sway, 0.3, 0.9
    # Left/Right Elbow
    kps[15], kps[16], kps[17] = 0.42, 0.45, 0.8
    kps[18], kps[19], kps[20] = 0.58, 0.45, 0.8
    # Left/Right Hip
    kps[21], kps[22], kps[23] = 0.47, 0.55, 0.9
    kps[24], kps[25], kps[26] = 0.53, 0.55, 0.9
    # Left/Right Knee
    kps[27], kps[28], kps[29] = 0.46, 0.7, 0.9
    kps[30], kps[31], kps[32] = 0.54, 0.7, 0.9
    # Left/Right Ankle
    kps[33], kps[34], kps[35] = 0.45, 0.88, 0.9
    kps[36], kps[37], kps[38] = 0.55, 0.88, 0.9

    kps += np.random.normal(0, noise, kps.shape).astype(np.float32)
    return kps


def _generate_tilting_keypoints(t, progress, noise):
    """Generate keypoints for a tilting/unstable person."""
    kps = _generate_upright_keypoints(t, noise)
    # Apply lateral tilt
    tilt = progress * 0.15
    # Shift upper body sideways
    for i in range(0, 21, 3):  # Upper body keypoints
        kps[i] += tilt
        kps[i + 1] += progress * 0.05
    return kps


def _generate_falling_keypoints(t, progress, noise):
    """Generate keypoints during a fall."""
    kps = _generate_upright_keypoints(t, noise)
    # Head drops, body rotates
    for i in range(0, 39, 3):
        kps[i] += progress * 0.1 * np.random.uniform(-1, 1)  # Spread x
        kps[i + 1] += progress * 0.3  # Drop y toward ground
    return kps


def _generate_lying_keypoints(t, noise):
    """Generate keypoints for a person lying on the ground."""
    kps = np.zeros(39, dtype=np.float32)
    # Person is horizontal near the bottom of frame
    base_y = 0.85 + np.random.uniform(-0.03, 0.03)
    for i in range(0, 39, 3):
        kps[i] = 0.3 + (i / 39) * 0.4 + np.random.normal(0, noise)
        kps[i + 1] = base_y + np.random.normal(0, 0.02)
        kps[i + 2] = 0.6 + np.random.uniform(0, 0.3)  # Lower visibility
    return kps
