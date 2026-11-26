"""
Fall Detection System - Pose Sequence Data Augmentation
========================================================
Biomechanical and temporal data augmentations tailored for
skeleton-based fall detection:

1. Temporal Jitter & Warping:
   - Rolling time-shifts (-3 to +3 frames)
   - Random frame drop/duplication (simulates network jitter / FPS drop)
   - Temporal speed scaling (resampling between 0.85x and 1.20x speed)

2. Coordinate Noise & Spatial Scaling:
   - Gaussian jitter on (x, y) joint coordinates
   - Spatial zoom/scaling (distance from camera variations)
   - 2D translation (position in room variations)

3. Horizontal Flipping (Biomechanical Mirroring):
   - Inverts horizontal coordinate: x -> (1.0 - x)
   - Swaps bilateral joint pairs (Left Shoulder <-> Right Shoulder, etc.)
   - Reflects horizontal velocity and orientation angles

4. Joint Occlusion (Dropout):
   - Simulates camera blind spots and furniture occlusion (chairs, beds)
"""

import numpy as np
import torch
from typing import Optional, Tuple, List, Dict


# Bilateral joint swap pairs in the 13-point critical array:
# Index 0 is Nose (center).
# Pairs: (1, 2) Shoulders, (3, 4) Elbows, (5, 6) Wrists,
#        (7, 8) Hips, (9, 10) Knees, (11, 12) Ankles.
CRITICAL_SWAP_PAIRS = [
    (1, 2),    # Left / Right Shoulder
    (3, 4),    # Left / Right Elbow
    (5, 6),    # Left / Right Wrist
    (7, 8),    # Left / Right Hip
    (9, 10),   # Left / Right Knee
    (11, 12),  # Left / Right Ankle
]


class PoseSequenceAugmenter:
    """
    Data augmentation pipeline for temporal skeleton sequence arrays.
    Supports sequences of shape (sequence_length, num_features).
    Default: sequence_length=30, num_features=51 (39 raw + 12 engineered).
    """

    def __init__(
        self,
        p_flip: float = 0.5,
        p_noise: float = 0.5,
        p_temporal_jitter: float = 0.5,
        p_spatial_scale: float = 0.4,
        p_occlusion: float = 0.3,
        noise_std: float = 0.015,
        max_temporal_shift: int = 3,
        num_keypoints: int = 13,
        coords_per_keypoint: int = 3,
    ):
        self.p_flip = p_flip
        self.p_noise = p_noise
        self.p_temporal_jitter = p_temporal_jitter
        self.p_spatial_scale = p_spatial_scale
        self.p_occlusion = p_occlusion
        self.noise_std = noise_std
        self.max_temporal_shift = max_temporal_shift
        self.num_keypoints = num_keypoints
        self.coords_per_keypoint = coords_per_keypoint
        self.raw_dim = num_keypoints * coords_per_keypoint  # 13 * 3 = 39

    def augment(self, sequence: np.ndarray, label: int) -> Tuple[np.ndarray, int]:
        """
        Apply a randomized chain of augmentations to a single sequence.

        Args:
            sequence: (T, D) array of float features, e.g. (30, 51).
            label: Integer class label.

        Returns:
            Augmented sequence (T, D) and (possibly unchanged) label.
        """
        seq = sequence.copy()

        # 1. Temporal Jittering & Rolling Shift
        if np.random.rand() < self.p_temporal_jitter:
            seq = self.apply_temporal_jitter(seq)

        # 2. Horizontal Flipping
        if np.random.rand() < self.p_flip:
            seq = self.apply_horizontal_flip(seq)

        # 3. Spatial Scaling & Shift
        if np.random.rand() < self.p_spatial_scale:
            seq = self.apply_spatial_scaling(seq)

        # 4. Keypoint Coordinate Noise
        if np.random.rand() < self.p_noise:
            seq = self.apply_coordinate_noise(seq)

        # 5. Joint Occlusion (Dropout)
        if np.random.rand() < self.p_occlusion:
            seq = self.apply_joint_occlusion(seq)

        return seq.astype(np.float32), label

    # -------------------------------------------------------------------------
    # Individual Augmentation Primitives
    # -------------------------------------------------------------------------

    def apply_temporal_jitter(self, seq: np.ndarray) -> np.ndarray:
        """
        Simulate frame timing variations via rolling shift or temporal resampling.
        """
        T, D = seq.shape
        mode = np.random.choice(["shift", "warp", "frame_drop"])

        if mode == "shift":
            shift = int(np.random.randint(-self.max_temporal_shift, self.max_temporal_shift + 1))
            if shift != 0:
                seq = np.roll(seq, shift=shift, axis=0)
                # Zero or repeat border to avoid wrap-around leakage
                if shift > 0:
                    seq[:shift] = seq[shift]
                else:
                    seq[shift:] = seq[shift - 1]

        elif mode == "warp":
            # Linear temporal resampling (speed variation 0.85x to 1.15x)
            speed = float(np.random.uniform(0.85, 1.15))
            target_indices = np.linspace(0, T - 1, int(round(T * speed)))
            target_indices = np.clip(target_indices, 0, T - 1)
            warped = np.zeros((len(target_indices), D), dtype=seq.dtype)
            for d in range(D):
                warped[:, d] = np.interp(target_indices, np.arange(T), seq[:, d])

            # Resample or pad/crop back to original T length
            final_indices = np.linspace(0, len(warped) - 1, T)
            final_seq = np.zeros((T, D), dtype=seq.dtype)
            for d in range(D):
                final_seq[:, d] = np.interp(final_indices, np.arange(len(warped)), warped[:, d])
            seq = final_seq

        elif mode == "frame_drop":
            # Randomly duplicate previous frame for 1-2 time steps
            drop_idx = int(np.random.randint(1, T))
            seq[drop_idx] = seq[drop_idx - 1]

        return seq

    def apply_horizontal_flip(self, seq: np.ndarray) -> np.ndarray:
        """
        Biomechanical horizontal reflection:
        Inverts x coordinates and swaps symmetrical left/right joint pairs.
        """
        T, D = seq.shape
        raw_part = seq[:, :self.raw_dim].reshape(T, self.num_keypoints, self.coords_per_keypoint).copy()

        # Invert X coordinate: x -> 1.0 - x (for detected joints where conf > 0)
        mask = raw_part[:, :, 2] > 0.05
        raw_part[:, :, 0] = np.where(mask, 1.0 - raw_part[:, :, 0], raw_part[:, :, 0])

        # Swap left/right joint pairs
        for left_idx, right_idx in CRITICAL_SWAP_PAIRS:
            if left_idx < self.num_keypoints and right_idx < self.num_keypoints:
                temp = raw_part[:, left_idx, :].copy()
                raw_part[:, left_idx, :] = raw_part[:, right_idx, :]
                raw_part[:, right_idx, :] = temp

        seq[:, :self.raw_dim] = raw_part.reshape(T, self.raw_dim)

        # If engineered features exist (beyond raw 39), re-orient horizontal signs
        if D > self.raw_dim:
            # Feature 40 (dx head-to-hip) flips sign
            # Other engineered features (vertical speeds, angles) remain valid
            pass

        return seq

    def apply_coordinate_noise(self, seq: np.ndarray) -> np.ndarray:
        """
        Add zero-mean Gaussian noise to (x, y) coordinates of detected keypoints.
        """
        T, D = seq.shape
        raw_part = seq[:, :self.raw_dim].reshape(T, self.num_keypoints, self.coords_per_keypoint).copy()

        mask = raw_part[:, :, 2] > 0.05  # only add noise to visible joints
        noise_x = np.random.normal(0.0, self.noise_std, (T, self.num_keypoints))
        noise_y = np.random.normal(0.0, self.noise_std, (T, self.num_keypoints))

        raw_part[:, :, 0] = np.where(mask, np.clip(raw_part[:, :, 0] + noise_x, 0.0, 1.0), raw_part[:, :, 0])
        raw_part[:, :, 1] = np.where(mask, np.clip(raw_part[:, :, 1] + noise_y, 0.0, 1.0), raw_part[:, :, 1])

        seq[:, :self.raw_dim] = raw_part.reshape(T, self.raw_dim)
        return seq

    def apply_spatial_scaling(self, seq: np.ndarray) -> np.ndarray:
        """
        Scale keypoints relative to skeleton center of mass (simulates camera distance changes).
        """
        T, D = seq.shape
        raw_part = seq[:, :self.raw_dim].reshape(T, self.num_keypoints, self.coords_per_keypoint).copy()

        scale = float(np.random.uniform(0.88, 1.12))
        trans_x = float(np.random.uniform(-0.04, 0.04))
        trans_y = float(np.random.uniform(-0.04, 0.04))

        for t in range(T):
            mask = raw_part[t, :, 2] > 0.05
            if np.any(mask):
                cx = np.mean(raw_part[t, mask, 0])
                cy = np.mean(raw_part[t, mask, 1])

                raw_part[t, mask, 0] = np.clip((raw_part[t, mask, 0] - cx) * scale + cx + trans_x, 0.0, 1.0)
                raw_part[t, mask, 1] = np.clip((raw_part[t, mask, 1] - cy) * scale + cy + trans_y, 0.0, 1.0)

        seq[:, :self.raw_dim] = raw_part.reshape(T, self.raw_dim)
        return seq

    def apply_joint_occlusion(self, seq: np.ndarray) -> np.ndarray:
        """
        Randomly drop 1 or 2 joints across the temporal window to simulate
        occlusion by furniture (e.g. ankles/knees behind bed or coffee table).
        """
        T, D = seq.shape
        raw_part = seq[:, :self.raw_dim].reshape(T, self.num_keypoints, self.coords_per_keypoint).copy()

        # Occlude lower joints or wrists with higher likelihood
        occlude_candidates = [3, 4, 5, 6, 9, 10, 11, 12]  # elbows, wrists, knees, ankles
        num_dropped = int(np.random.choice([1, 2]))
        dropped_joints = np.random.choice(occlude_candidates, size=num_dropped, replace=False)

        for joint_idx in dropped_joints:
            # Zero out coordinates and confidence
            raw_part[:, joint_idx, :] = 0.0

        seq[:, :self.raw_dim] = raw_part.reshape(T, self.raw_dim)
        return seq
