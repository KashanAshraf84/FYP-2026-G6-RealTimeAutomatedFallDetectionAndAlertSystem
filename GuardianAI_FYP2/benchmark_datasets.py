"""
Fall Detection System - Benchmark Datasets Standardization
===========================================================
Standardizes public fall detection benchmarks into unified temporal sequence
arrays (N, sequence_length, num_features) with consistent label mapping:
  0: Normal (ADL: walking, sitting, bending, lying on bed)
  1: Warning (Pre-fall instability, stumbling, sudden descent)
  2: Fall (Impact and post-fall lying on ground)

Supported Benchmarks:
  1. UR Fall Detection Dataset (URFD)
     - Frontal (cam0) and overhead (cam1) RGB video sequences
     - Fall events and ADL (Activities of Daily Living)
  2. Le2i Fall Detection Dataset
     - 4 Realistic Scenarios: Home, Coffee Room, Office, Lecture Room
     - Frame-accurate annotation files (.txt) with fall start/end frames
  3. Multiple Cameras Fall Dataset (MCFD)
     - Multi-view synchronized camera recordings
"""

import os
import glob
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm

from fyp1_config import CONFIG, FeatureConfig, PoseConfig
from pose_estimator import PoseEstimator
from feature_extractor import FeatureExtractor, TemporalFeatureProcessor


LABEL_MAP = {"normal": 0, "warning": 1, "fall": 2}


class BenchmarkStandardizer:
    """
    Standardizes raw video benchmark datasets into unified PyTorch-ready .npz files.
    """

    def __init__(
        self,
        target_fps: int = 15,
        sequence_length: int = 30,
        stride: int = 5,
        pose_estimator: Optional[PoseEstimator] = None,
    ):
        self.target_fps = target_fps
        self.sequence_length = sequence_length
        self.stride = stride
        self.pose_estimator = pose_estimator

        self.fc = FeatureConfig(sequence_length=sequence_length, stride=stride)
        self.extractor = FeatureExtractor(self.fc, CONFIG.pose)
        self.processor = TemporalFeatureProcessor(self.fc)

    def _ensure_pose_estimator(self):
        if self.pose_estimator is None:
            print("Initializing YOLOv8-Pose Estimator...")
            self.pose_estimator = PoseEstimator(CONFIG.pose)

    def _extract_video_keypoints(self, video_path: str) -> np.ndarray:
        """Extract 13-point critical keypoints from a video file."""
        self._ensure_pose_estimator()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Unable to open video: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if src_fps <= 0 or np.isnan(src_fps):
            src_fps = 30.0

        step = max(1, int(round(src_fps / self.target_fps)))
        keypoints_list = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                kps, _ = self.pose_estimator.process_frame(frame)
                if kps is not None:
                    keypoints_list.append(kps)
                else:
                    # Missing detection: zero-fill (13, 3)
                    keypoints_list.append(np.zeros((13, 3), dtype=np.float32))

            frame_idx += 1

        cap.release()
        return np.array(keypoints_list, dtype=np.float32)

    def _build_features_from_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Convert keypoints into combined (raw + engineered) features."""
        self.extractor.reset()
        frame_features = []
        for kps in keypoints:
            raw = kps.flatten()  # 13 * 3 = 39
            eng = self.extractor.extract_frame_features(kps)  # 12
            combined = np.concatenate([raw, eng])  # 51
            frame_features.append(combined)
        return np.array(frame_features, dtype=np.float32)

    # -------------------------------------------------------------------------
    # 1. UR Fall Detection Dataset (URFD)
    # -------------------------------------------------------------------------

    def parse_urfd(self, dataset_dir: str, output_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse UR Fall Detection dataset directory.
        Expected structure:
          urfd_dataset/
            falls/
              fall-01-cam0-rgb.mp4
              fall-02-cam0-rgb.mp4
            adl/
              adl-01-cam0-rgb.mp4
              adl-02-cam0-rgb.mp4
        """
        print(f"\n[URFD] Processing dataset at: {dataset_dir}")
        all_seqs = []
        all_labels = []

        fall_files = glob.glob(os.path.join(dataset_dir, "falls", "*.mp4")) + \
                     glob.glob(os.path.join(dataset_dir, "falls", "*.avi"))
        adl_files = glob.glob(os.path.join(dataset_dir, "adl", "*.mp4")) + \
                    glob.glob(os.path.join(dataset_dir, "adl", "*.avi"))

        # Process ADL (Normal)
        for vf in tqdm(adl_files, desc="URFD ADL"):
            try:
                kps = self._extract_video_keypoints(vf)
                if len(kps) < self.sequence_length:
                    continue
                feats = self._build_features_from_keypoints(kps)
                labels = np.zeros(len(feats), dtype=np.int64)
                seqs, seq_lbls = self.processor.create_sequences(feats, labels)
                if len(seqs) > 0:
                    all_seqs.append(seqs)
                    all_labels.append(seq_lbls)
            except Exception as e:
                print(f"Error reading {vf}: {e}")

        # Process Falls
        for vf in tqdm(fall_files, desc="URFD Falls"):
            try:
                kps = self._extract_video_keypoints(vf)
                if len(kps) < self.sequence_length:
                    continue
                feats = self._build_features_from_keypoints(kps)
                N = len(feats)

                # URFD fall clips typically show 20-30% walking, fall around middle, then lying on floor
                labels = np.zeros(N, dtype=np.int64)
                fall_start = int(N * 0.35)
                fall_impact = int(N * 0.55)
                labels[fall_start:fall_impact] = 1  # warning / falling
                labels[fall_impact:] = 2            # fall / lying

                seqs, seq_lbls = self.processor.create_sequences(feats, labels)
                if len(seqs) > 0:
                    all_seqs.append(seqs)
                    all_labels.append(seq_lbls)
            except Exception as e:
                print(f"Error reading {vf}: {e}")

        return self._save_dataset_npz(all_seqs, all_labels, output_path, "URFD")

    # -------------------------------------------------------------------------
    # 2. Le2i Fall Detection Dataset
    # -------------------------------------------------------------------------

    def parse_le2i(self, dataset_dir: str, output_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse Le2i dataset directory.
        Scenarios: Home, Coffee_room, Office, Lecture_room.
        Each video has a matching .txt annotation file containing fall start and end frames.
        """
        print(f"\n[Le2i] Processing dataset at: {dataset_dir}")
        all_seqs = []
        all_labels = []

        scenarios = ["Home", "Coffee_room", "Office", "Lecture_room"]
        video_paths = []
        for s in scenarios:
            s_dir = os.path.join(dataset_dir, s)
            if os.path.isdir(s_dir):
                video_paths.extend(glob.glob(os.path.join(s_dir, "Videos", "*.avi")))
                video_paths.extend(glob.glob(os.path.join(s_dir, "*.avi")))

        if not video_paths:
            # Fallback search anywhere in dataset_dir
            video_paths = glob.glob(os.path.join(dataset_dir, "**", "*.avi"), recursive=True)

        print(f"Found {len(video_paths)} Le2i video clips.")

        for vf in tqdm(video_paths, desc="Le2i Videos"):
            try:
                # Find matching annotation file
                base = Path(vf).stem
                annot_path = os.path.join(os.path.dirname(vf), f"{base}.txt")
                if not os.path.isfile(annot_path):
                    # Check in Annotation folder
                    annot_path = os.path.join(Path(vf).parent.parent, "Annotation_files", f"{base}.txt")

                fall_start = None
                fall_end = None
                if os.path.isfile(annot_path):
                    with open(annot_path, "r") as f:
                        lines = [line.strip() for line in f if line.strip().isdigit()]
                        if len(lines) >= 2:
                            fall_start = int(lines[0])
                            fall_end = int(lines[1])

                kps = self._extract_video_keypoints(vf)
                if len(kps) < self.sequence_length:
                    continue
                feats = self._build_features_from_keypoints(kps)
                N = len(feats)
                labels = np.zeros(N, dtype=np.int64)

                if fall_start is not None and fall_end is not None:
                    # Convert original video frame numbers to sampled target_fps indices
                    cap = cv2.VideoCapture(vf)
                    src_fps = max(1.0, cap.get(cv2.CAP_PROP_FPS))
                    cap.release()
                    ratio = self.target_fps / src_fps

                    idx_start = min(N - 1, int(fall_start * ratio))
                    idx_end = min(N - 1, int(fall_end * ratio))
                    labels[idx_start:idx_end] = 1  # warning / descent
                    labels[idx_end:] = 2           # fall / lying

                seqs, seq_lbls = self.processor.create_sequences(feats, labels)
                if len(seqs) > 0:
                    all_seqs.append(seqs)
                    all_labels.append(seq_lbls)

            except Exception as e:
                print(f"Error processing {vf}: {e}")

        return self._save_dataset_npz(all_seqs, all_labels, output_path, "Le2i")

    # -------------------------------------------------------------------------
    # 3. Generic GitHub / Custom Dataset Parser
    # -------------------------------------------------------------------------

    def parse_generic(self, dataset_dir: str, output_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse ANY GitHub or custom fall dataset directory.
        Scans all video files (.mp4, .avi, .mov, .mkv) and categorizes into:
          - Falls: folders/files matching 'fall', 'falls', 'falling'
          - Normal/ADL: folders/files matching 'adl', 'normal', 'non_fall', 'non-fall', 'nonfall', etc.
          - Warning: folders/files matching 'warning', 'stumble', 'slip', 'trip'
        """
        print(f"\n[Generic Dataset] Parsing dataset from: {dataset_dir}")
        all_seqs = []
        all_labels = []

        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
        all_videos = []
        for root, _, files in os.walk(dataset_dir):
            for f in files:
                if Path(f).suffix.lower() in video_extensions:
                    all_videos.append(os.path.join(root, f))

        if not all_videos:
            raise FileNotFoundError(f"No video files found in {dataset_dir}")

        fall_videos = []
        normal_videos = []
        warning_videos = []

        fall_keywords = ["fall", "falls", "falling"]
        warning_keywords = ["warning", "stumble", "slip", "trip"]
        normal_keywords = ["adl", "normal", "non_fall", "non-fall", "nonfall", "walk", "sit", "stand", "bend"]

        for v in all_videos:
            rel = os.path.relpath(v, dataset_dir).lower()
            if any(k in rel for k in warning_keywords):
                warning_videos.append(v)
            elif any(k in rel for k in normal_keywords):
                normal_videos.append(v)
            elif any(k in rel for k in fall_keywords):
                fall_videos.append(v)
            else:
                normal_videos.append(v)

        print(f"  Found: {len(normal_videos)} Normal/ADL, {len(warning_videos)} Warning, {len(fall_videos)} Fall clips.")

        # Process Normal
        for vf in tqdm(normal_videos, desc="Normal / ADL Videos"):
            try:
                kps = self._extract_video_keypoints(vf)
                if len(kps) < self.sequence_length:
                    continue
                feats = self._build_features_from_keypoints(kps)
                labels = np.zeros(len(feats), dtype=np.int64)
                seqs, seq_lbls = self.processor.create_sequences(feats, labels)
                if len(seqs) > 0:
                    all_seqs.append(seqs)
                    all_labels.append(seq_lbls)
            except Exception as e:
                print(f"Error reading {vf}: {e}")

        # Process Warning
        for vf in tqdm(warning_videos, desc="Warning Videos"):
            try:
                kps = self._extract_video_keypoints(vf)
                if len(kps) < self.sequence_length:
                    continue
                feats = self._build_features_from_keypoints(kps)
                labels = np.ones(len(feats), dtype=np.int64)
                seqs, seq_lbls = self.processor.create_sequences(feats, labels)
                if len(seqs) > 0:
                    all_seqs.append(seqs)
                    all_labels.append(seq_lbls)
            except Exception as e:
                print(f"Error reading {vf}: {e}")

        # Process Fall
        for vf in tqdm(fall_videos, desc="Fall Videos"):
            try:
                kps = self._extract_video_keypoints(vf)
                if len(kps) < self.sequence_length:
                    continue
                feats = self._build_features_from_keypoints(kps)
                N = len(feats)
                labels = np.zeros(N, dtype=np.int64)
                fall_start = int(N * 0.35)
                fall_impact = int(N * 0.55)
                labels[fall_start:fall_impact] = 1
                labels[fall_impact:] = 2
                seqs, seq_lbls = self.processor.create_sequences(feats, labels)
                if len(seqs) > 0:
                    all_seqs.append(seqs)
                    all_labels.append(seq_lbls)
            except Exception as e:
                print(f"Error reading {vf}: {e}")

        return self._save_dataset_npz(all_seqs, all_labels, output_path, "Generic_Dataset")

    # -------------------------------------------------------------------------
    # Helper: Save Standardized NPZ
    # -------------------------------------------------------------------------

    def _save_dataset_npz(
        self,
        all_seqs: List[np.ndarray],
        all_labels: List[np.ndarray],
        output_path: str,
        benchmark_name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not all_seqs:
            raise RuntimeError(f"No sequences extracted for {benchmark_name}.")

        X = np.concatenate(all_seqs, axis=0)
        y = np.concatenate(all_labels, axis=0)

        # Normalize features
        X = self.processor.fit_normalize(X)
        norm_mean, norm_std = self.processor.get_normalization_params()

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.savez(
            output_path,
            X=X,
            y=y,
            norm_mean=norm_mean,
            norm_std=norm_std,
            benchmark=benchmark_name,
        )

        print(f"\n[{benchmark_name}] Dataset saved to: {output_path}")
        print(f"  Total Sequences: {len(X)}")
        for name, idx in LABEL_MAP.items():
            cnt = int((y == idx).sum())
            pct = (cnt / len(y) * 100) if len(y) > 0 else 0
            print(f"  {name.capitalize()}: {cnt:,} ({pct:.1f}%)")

        return X, y


# -----------------------------------------------------------------------------
# Synthetic Benchmark Generator for Immediate Team Testing
# -----------------------------------------------------------------------------

def generate_standard_benchmark_mock(
    output_path: str,
    num_normal: int = 400,
    num_warning: int = 150,
    num_fall: int = 200,
    benchmark_name: str = "Synthetic_Benchmark",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a realistic synthetic benchmark dataset matching the exact
    (N, 30, 51) feature shape for immediate pipeline validation and training.
    """
    from dataset import generate_synthetic_data
    print(f"\nGenerating {benchmark_name} with {num_normal} normal, {num_warning} warning, {num_fall} fall sequences...")
    X, y, norm_mean, norm_std = generate_synthetic_data(
        num_normal=num_normal,
        num_warning=num_warning,
        num_fall=num_fall,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez(
        output_path,
        X=X,
        y=y,
        norm_mean=norm_mean,
        norm_std=norm_std,
        benchmark=benchmark_name,
    )
    print(f"Mock benchmark saved to {output_path} (Shape: {X.shape})")
    return X, y
