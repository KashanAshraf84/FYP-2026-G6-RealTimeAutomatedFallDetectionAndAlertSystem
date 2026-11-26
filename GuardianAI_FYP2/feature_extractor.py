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
from fyp1_config import FeatureConfig, PoseConfig


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
        self._prev_hip_y_rule = None
        self._is_bending = False
        self._hips_elevated = True
        self._ground_counter = 0
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
        self._prev_hip_y_rule = None
        self._is_bending = False
        self._hips_elevated = True
        self._ground_counter = 0
        self._fall_counter = 0
        self._lying_counter = 0
        self._last_angle = 90.0
        self._last_speed = 0.0
        self._last_movement_speed = 0.0

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
        Clinical fall vs bending/fast-movement discriminator.
        Distinguishes intentional actions (bending, fast moving, sitting) from real falls:
          - Real Fall: Rapid simultaneous drop of head + hips, followed by floor collapse (lying).
          - Bending / Squatting: Head lowers while hips remain elevated and supported by legs.
          - Fast Movement: Head and hips move quickly without collapsing to the floor.
        """
        virtual_h = 480
        virtual_w = 640

        head = keypoints[0, :2] * [virtual_w, virtual_h]

        # --- Hip confidence guard ---
        l_hip_conf = keypoints[7, 2]
        r_hip_conf = keypoints[8, 2]
        HIP_CONF_THRESHOLD = 0.35

        if l_hip_conf >= HIP_CONF_THRESHOLD and r_hip_conf >= HIP_CONF_THRESHOLD:
            hip = (keypoints[7, :2] + keypoints[8, :2]) / 2 * [virtual_w, virtual_h]
            hips_reliable = True
        elif l_hip_conf >= HIP_CONF_THRESHOLD:
            hip = keypoints[7, :2] * [virtual_w, virtual_h]
            hips_reliable = True
        elif r_hip_conf >= HIP_CONF_THRESHOLD:
            hip = keypoints[8, :2] * [virtual_w, virtual_h]
            hips_reliable = True
        else:
            l_sh_conf = keypoints[1, 2]
            r_sh_conf = keypoints[2, 2]
            if l_sh_conf >= HIP_CONF_THRESHOLD and r_sh_conf >= HIP_CONF_THRESHOLD:
                hip = (keypoints[1, :2] + keypoints[2, :2]) / 2 * [virtual_w, virtual_h]
            elif l_sh_conf >= HIP_CONF_THRESHOLD:
                hip = keypoints[1, :2] * [virtual_w, virtual_h]
            elif r_sh_conf >= HIP_CONF_THRESHOLD:
                hip = keypoints[2, :2] * [virtual_w, virtual_h]
            else:
                hip = head
            hips_reliable = False

        dx = hip[0] - head[0]
        dy = hip[1] - head[1]
        angle = abs(np.degrees(np.arctan2(dy, dx)))

        head_y = head[1]
        head_x = head[0]
        hip_y = hip[1]

        drop_speed = 0.0
        horiz_speed = 0.0
        hip_drop_speed = 0.0

        if self._prev_head_y_rule is not None:
            drop_speed = head_y - self._prev_head_y_rule
        if self._prev_head_x_rule is not None:
            horiz_speed = head_x - self._prev_head_x_rule
        if getattr(self, "_prev_hip_y_rule", None) is not None:
            hip_drop_speed = hip_y - self._prev_hip_y_rule

        self._prev_head_y_rule = head_y
        self._prev_head_x_rule = head_x
        self._prev_hip_y_rule = hip_y

        movement_speed = float(np.hypot(drop_speed, horiz_speed))
        self._last_movement_speed = movement_speed
        self._last_angle = angle
        self._last_speed = drop_speed

        # Bounding box aspect ratio (width / height)
        xs = keypoints[:, 0] * virtual_w
        ys = keypoints[:, 1] * virtual_h
        bb_w = xs.max() - xs.min()
        bb_h = max(1.0, ys.max() - ys.min())
        aspect_ratio = bb_w / bb_h

        # Are hips elevated above floor level?
        # A standing/bending person has hips high (typically hip_y < 0.65 of frame).
        # In a real floor fall, hips collapse down (hip_y >= 0.65 of frame).
        self._hips_elevated = (hip_y < virtual_h * 0.65)

        # BENDING / SQUATTING CRITERIA:
        # Head tilts forward/downward (angle < 60), BUT hips remain elevated and supported by legs!
        # Head drops without hips collapsing.
        is_bending = False
        if self._hips_elevated:
            if angle < 60 or (drop_speed > 15 and hip_drop_speed < 12):
                is_bending = True
        self._is_bending = is_bending

        # GROUND / COLLAPSED CRITERIA (Real Fall Requirement):
        # In a real fall:
        # 1. Hips are near the floor (hip_y > virtual_h * 0.58)
        # 2. Head is near the floor (head_y > virtual_h * 0.58)
        # 3. Posture is prone/horizontal (angle < 45 or aspect_ratio > 0.85)
        is_ground_collapsed = (
            not self._hips_elevated
            and head_y > virtual_h * 0.58
            and (angle < 45 or aspect_ratio > 0.85)
            and hips_reliable
        )

        if is_ground_collapsed:
            self._ground_counter += 1
            self._lying_counter += 1
        else:
            self._ground_counter = max(0, self._ground_counter - 1)
            self._lying_counter = max(0, self._lying_counter - 1)

        # Real fall descent trigger requires BOTH head and center-of-gravity (hips) to drop rapidly.
        # Rapid drop of head alone without hip drop is just bending or ducking!
        rapid_descent = (drop_speed > 30 and hip_drop_speed > 15 and not is_bending)

        if rapid_descent:
            self._fall_counter += 1
        else:
            self._fall_counter = max(0, self._fall_counter - 1)

        status = "normal"
        confidence = 0.5
        is_instant = False

        # --- Clinical Decision Tree ---
        if is_bending:
            # Person is bending / tying shoes / leaning while standing
            status = "normal"
            confidence = 0.90
            self._fall_counter = 0

        elif (self._fall_counter >= 2 and is_ground_collapsed) or self._ground_counter >= 5:
            # Confirmed Fall: rapid descent + ground collapse OR sustained ground immobility
            status = "fall"
            confidence = min(0.75 + self._ground_counter * 0.05, 0.98)

        elif rapid_descent or (movement_speed > self.fc.jerk_speed_threshold and not is_bending):
            # Fast movement or stumble that has not collapsed to ground -> WARNING (Not Fall!)
            status = "warning"
            confidence = 0.70
            is_instant = True

        elif angle < 60 and hips_reliable and not self._hips_elevated:
            # Unsteady posture leaning near ground
            status = "warning"
            confidence = 0.65

        else:
            status = "normal"
            confidence = 0.95 if hips_reliable else 0.75

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
