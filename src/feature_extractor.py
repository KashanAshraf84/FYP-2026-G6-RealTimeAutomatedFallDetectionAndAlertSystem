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
        self._prev_head_x_rule = None
        self._fall_counter = 0
        self._lying_counter = 0

        # Telemetry for HUD
        self._last_angle = 90.0
        self._last_speed = 0.0
        self._last_movement_speed = 0.0

    def reset(self):
        """Clear all tracking state."""
        self.keypoint_buffer.clear()
        self.feature_buffer.clear()
        self._prev_head_y = None
        self._prev_head_y_rule = None
        self._prev_head_x_rule = None
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

    def get_rule_based_status(self, keypoints: np.ndarray) -> Tuple[str, float, bool]:
        """
        Rule-based YOLO fall detection logic with hip-confidence validation.

        Fixes SRS defect #2: YOLO estimates hip positions even when hips are
        off-frame, corrupting the body-angle calculation.  We now check hip
        confidence and fall back to the shoulder midpoint when hips are not
        reliably visible.

        Returns (status, confidence, is_instant) — is_instant is True when
        "warning" was triggered by a single-frame speed jerk rather than the
        angle/lying rules, so the caller can surface it immediately instead
        of waiting on temporal smoothing.
        """
        virtual_h = 480
        virtual_w = 640

        head = keypoints[0, :2] * [virtual_w, virtual_h]

        # --- Hip confidence guard (SRS FR-2.8) ---
        l_hip_conf = keypoints[7, 2]
        r_hip_conf = keypoints[8, 2]
        HIP_CONF_THRESHOLD = 0.4

        if l_hip_conf >= HIP_CONF_THRESHOLD and r_hip_conf >= HIP_CONF_THRESHOLD:
            # Both hips visible — use true hip midpoint
            hip = (keypoints[7, :2] + keypoints[8, :2]) / 2 * [virtual_w, virtual_h]
            hips_reliable = True
        elif l_hip_conf >= HIP_CONF_THRESHOLD:
            hip = keypoints[7, :2] * [virtual_w, virtual_h]
            hips_reliable = True
        elif r_hip_conf >= HIP_CONF_THRESHOLD:
            hip = keypoints[8, :2] * [virtual_w, virtual_h]
            hips_reliable = True
        else:
            # Hips off-screen: use shoulder midpoint as lower anchor.
            # This underestimates the true body length but avoids garbage angles.
            l_sh_conf = keypoints[1, 2]
            r_sh_conf = keypoints[2, 2]
            if l_sh_conf >= HIP_CONF_THRESHOLD and r_sh_conf >= HIP_CONF_THRESHOLD:
                hip = (keypoints[1, :2] + keypoints[2, :2]) / 2 * [virtual_w, virtual_h]
            elif l_sh_conf >= HIP_CONF_THRESHOLD:
                hip = keypoints[1, :2] * [virtual_w, virtual_h]
            elif r_sh_conf >= HIP_CONF_THRESHOLD:
                hip = keypoints[2, :2] * [virtual_w, virtual_h]
            else:
                hip = head  # last resort: no anchor at all
            hips_reliable = False

        dx = hip[0] - head[0]
        dy = hip[1] - head[1]
        angle = abs(np.degrees(np.arctan2(dy, dx)))

        head_y = head[1]
        head_x = head[0]
        drop_speed = 0
        horiz_speed = 0
        if self._prev_head_y_rule is not None:
            drop_speed = head_y - self._prev_head_y_rule
        if self._prev_head_x_rule is not None:
            horiz_speed = head_x - self._prev_head_x_rule
        self._prev_head_y_rule = head_y
        self._prev_head_x_rule = head_x

        # Combined (any-direction) movement magnitude — catches sideways/
        # upward jerks that a vertical-only drop_speed would miss.
        movement_speed = float(np.hypot(drop_speed, horiz_speed))
        self._last_movement_speed = movement_speed

        near_ground = head_y > virtual_h * 0.6
        is_lying = (angle < 50) and near_ground and hips_reliable
        fall_trigger = (drop_speed > 20) and hips_reliable  # don't trigger fall if we can't see hips

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
        elif angle < 60 and hips_reliable:
            # Only flag warning from angle when posture leans beyond 30 degrees from vertical (angle < 60)
            status = "warning"
            confidence = 0.6

        else:
            status = "normal"
            confidence = 1.0 if hips_reliable else 0.7

        # --- Instant jerk override ---
        # A sudden movement below the fall drop-speed threshold still counts
        # as "warning" for this frame, even if it lasts a single frame and
        # would otherwise be voted out by temporal smoothing.
        is_instant = False
        if status != "fall" and movement_speed > self.fc.jerk_speed_threshold:
            status = "warning"
            confidence = max(confidence, 0.65)
            is_instant = True

        return status, confidence, is_instant


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
