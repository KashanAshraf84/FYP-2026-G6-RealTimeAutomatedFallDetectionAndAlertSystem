"""
Fall Detection System - Feature Extractor (YOLO Edition)
======================================================
Implements user-provided heuristics for fall detection:
- Body Angle (Nose to Hip)
- head Drop Speed
- Ground Proximity Check
- Status Categories: NORMAL, UNSTABLE, PERSON LYING, FALL DETECTED
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple, List, Dict
from config import FeatureConfig, PoseConfig


class FeatureExtractor:
    """
    Extracts biomechanical features from YOLO pose keypoints.
    Implements a state machine for fall detection based on:
      - Vertical drop speed
      - Horizontal body angle
      - Proximity to ground
    """

    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        pose_config: Optional[PoseConfig] = None,
    ):
        self.fc = feature_config or FeatureConfig()
        self.pc = pose_config or PoseConfig()

        # Buffers for temporal analysis
        self.keypoint_buffer = deque(maxlen=self.fc.sequence_length)
        self.feature_buffer = deque(maxlen=self.fc.sequence_length)

        # Tracking state (from user logic)
        self._prev_head_y = None
        self._prev_head_y_rule = None
        self._fall_counter = 0
        self._lying_counter = 0
        
        # Telemetry for HUD
        self._last_angle = 90.0
        self._last_speed = 0.0

    def reset(self):
        """Clear all tracking state."""
        self.keypoint_buffer.clear()
        self.feature_buffer.clear()
        self._prev_head_y = None
        self._fall_counter = 0
        self._lying_counter = 0

    def extract_frame_features(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Extract per-frame engineered features from a single keypoint array.
        Compatible with YOLO keypoints (9 or 17 points).
        """
        features = []
        
        # Mapping to virtual pixel space for consistent rule-thresholds
        virtual_h = 480
        virtual_w = 640
        
        # Nose (0) and Hip midpoint (indices 7/8 in the 13-point critical array
        # -> left/right hip; raw YOLO indices 11/12)
        # Using the normalized coordinates for most math
        head = keypoints[0, :2]
        hip = (keypoints[7, :2] + keypoints[8, :2]) / 2 if len(keypoints) > 8 else head
        
        # --- 1. Body Orientation Angle ---
        dx = hip[0] - head[0]
        dy = hip[1] - head[1]
        body_angle = abs(np.degrees(np.arctan2(dy, dx)))
        features.append(body_angle / 180.0)

        # --- 2. Head Y (Normalized) ---
        features.append(head[1])

        # --- 3. Head Velocity ---
        head_y_px = head[1] * virtual_h
        if self._prev_head_y is not None:
            head_velocity = head_y_px - self._prev_head_y
        else:
            head_velocity = 0.0
        self._prev_head_y = head_y_px
        features.append(head_velocity / 100.0) # Scale for NN

        # --- 4. Aspect Ratio & Height ---
        # Approximate using min/max of keypoints
        xs = keypoints[:, 0]
        ys = keypoints[:, 1]
        w_norm = xs.max() - xs.min()
        h_norm = ys.max() - ys.min()
        aspect_ratio = w_norm / (h_norm + 1e-6)
        features.append(min(aspect_ratio / 2.0, 1.0))
        features.append(h_norm)

        # Fill remaining features to match config.model.input_size (51 total)
        # 39 raw (13*3) + 12 engineered
        # Current engineered count: 5. Need 7 more.
        for _ in range(7):
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def update(self, keypoints: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Keep feature sequence updated."""
        if keypoints is None:
            combined = np.zeros(self.fc.num_raw_features + self.fc.num_engineered_features, dtype=np.float32)
        else:
            raw = keypoints.flatten()
            engineered = self.extract_frame_features(keypoints)
            combined = np.concatenate([raw, engineered])

        self.keypoint_buffer.append(combined)

        if len(self.keypoint_buffer) >= self.fc.sequence_length:
            return np.array(list(self.keypoint_buffer), dtype=np.float32)
        return None

    def get_rule_based_status(self, keypoints: np.ndarray) -> Tuple[str, float]:
        """
        Implementation of the user-provided YOLO fall detection logic.
        """
        virtual_h = 480
        virtual_w = 640

        head = keypoints[0, :2] * [virtual_w, virtual_h]
        hip = (keypoints[7, :2] + keypoints[8, :2]) / 2 * [virtual_w, virtual_h]

        dx = hip[0] - head[0]
        dy = hip[1] - head[1]
        angle = abs(np.degrees(np.arctan2(dy, dx)))

        head_y = head[1]
        drop_speed = 0
        if self._prev_head_y_rule is not None:
            drop_speed = head_y - self._prev_head_y_rule
        self._prev_head_y_rule = head_y

        near_ground = head_y > virtual_h * 0.6
        is_lying = (angle < 50) and near_ground
        fall_trigger = (drop_speed > 20)

        # Store for telemetry
        self._last_angle = angle
        self._last_speed = drop_speed

        if fall_trigger:
            self._fall_counter += 1
        else:
            self._fall_counter = max(0, self._fall_counter - 1)

        if is_lying:
            self._lying_counter += 1
        else:
            self._lying_counter = 0

        status = "normal"
        confidence = 0.5

        if self._fall_counter >= 2:
            status = "fall"
            confidence = min(0.6 + self._fall_counter * 0.1, 1.0)
        elif self._lying_counter >= 5:
            status = "warning"
            confidence = 0.8
        elif angle < 70:
            status = "warning"
            confidence = 0.6
        else:
            status = "normal"
            confidence = 1.0

        return status, confidence


class TemporalFeatureProcessor:
    """Handles sequence creation and normalization for FYP-2 AI Training."""
    def __init__(self, feature_config=None):
        self.fc = feature_config or FeatureConfig()
        self._mean = None
        self._std = None

    def create_sequences(self, features, labels):
        X, y = [], []
        for i in range(0, len(features) - self.fc.sequence_length + 1, self.fc.stride):
            X.append(features[i : i + self.fc.sequence_length])
            # Majority vote for label
            vals, counts = np.unique(labels[i : i + self.fc.sequence_length], return_counts=True)
            y.append(vals[np.argmax(counts)])
        return np.array(X), np.array(y)

    def fit_normalize(self, X):
        flat = X.reshape(-1, X.shape[-1])
        self._mean = flat.mean(axis=0)
        self._std = flat.std(axis=0) + 1e-8
        return (X - self._mean) / self._std

    def get_normalization_params(self):
        return self._mean, self._std
