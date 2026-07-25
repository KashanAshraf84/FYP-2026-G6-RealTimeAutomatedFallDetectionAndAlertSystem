"""
Fall Detection System - Pose Estimator (YOLOv8 Edition)
======================================================
YOLOv8-Pose based pose estimation with high-accuracy keypoint extraction.
"""

import cv2
import os
import numpy as np
from ultralytics import YOLO
from typing import Optional, Tuple, List, Dict
from config import PoseConfig, SystemConfig


class PoseEstimator:
    """
    Real-time human pose estimator using Ultralytics YOLOv8-Pose.
    Extracts 17 body landmarks and provides critical keypoints
    for fall detection analysis.
    """

    def __init__(self, config: Optional[PoseConfig] = None):
        self.config = config or PoseConfig()
        
        # Initialize YOLOv8-Pose Model
        from config import CONFIG
        model_path = CONFIG.yolo_model_path
        self.model = YOLO(model_path)
        
        self._prev_landmarks = None
        self._tracking_lost_count = 0
        self._max_lost_frames = 10

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Process a single frame using YOLOv8-Pose.
        """
        # Run inference
        results = self.model(frame, verbose=False, conf=self.config.min_detection_confidence)
        
        annotated = frame.copy()

        if len(results) > 0 and results[0].keypoints is not None:
            # We take the best detected person for this single-person estimator
            # results[0].keypoints.xyn is a tensor of normalized keypoints (N, 17, 3)
            # x, y, conf
            kp_data = results[0].keypoints.data.cpu().numpy()
            if kp_data.size == 0 or len(kp_data) == 0:
                return self._handle_lost_tracking(annotated)

            landmarks = kp_data[0] # (17, 3) -> [x, y, conf]
            
            self._tracking_lost_count = 0
            
            # Draw custom premium skeleton for YOLO keypoints
            annotated = self._draw_custom_skeleton(annotated, landmarks)

            # Extract critical keypoints in normalized coordinates [0, 1]
            h, w = frame.shape[:2]
            keypoints = self._extract_critical_keypoints(landmarks, (h, w))
            self._prev_landmarks = keypoints
            return keypoints, annotated
        else:
            return self._handle_lost_tracking(annotated)

    def _handle_lost_tracking(self, annotated):
        self._tracking_lost_count += 1
        if (self._prev_landmarks is not None and 
            self._tracking_lost_count <= self._max_lost_frames):
            return self._prev_landmarks, annotated
        return None, annotated

    def _draw_custom_skeleton(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Custom high-aesthetic skeleton visualizer for YOLOv8 (17 points)."""
        h, w = frame.shape[:2]
        
        # YOLO Connections (Center to extremities)
        # 0:nose, 5:l_shoulder, 6:r_shoulder, 11:l_hip, 12:r_hip, etc.
        connections = [
            (0, 5), (0, 6), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Upper body
            (5, 11), (6, 12), (11, 12), # Torso
            (11, 13), (13, 15), (12, 14), (14, 16) # Lower body
        ]

        def get_pos(idx):
            lm = landmarks[idx]
            return int(lm[0]), int(lm[1])

        # Draw Lines
        for start_idx, end_idx in connections:
            if landmarks[start_idx, 2] > 0.4 and landmarks[end_idx, 2] > 0.4:
                p1 = get_pos(start_idx)
                p2 = get_pos(end_idx)
                cv2.line(frame, p1, p2, (40, 40, 40), 4) # Outer glow
                cv2.line(frame, p1, p2, (255, 255, 255), 1) # Inner line

        # Draw Joints
        for i, lm in enumerate(landmarks):
            if lm[2] > 0.4:
                pos = get_pos(i)
                cv2.circle(frame, pos, 6, (78, 124, 254), -1) # Glow
                cv2.circle(frame, pos, 3, (255, 255, 255), -1)

        return frame

    def _extract_critical_keypoints(
        self, landmarks, frame_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Extract normalized critical keypoints from YOLO keypoints.
        Output format: [x, y, conf] normalized to [0, 1]
        """
        h, w = frame_shape
        keypoints = []

        for idx in self.config.CRITICAL_KEYPOINTS:
            lm = landmarks[idx]
            # YOLO returns absolute pixel coordinates
            nx = lm[0] / w
            ny = lm[1] / h
            keypoints.append([nx, ny, lm[2]]) 

        return np.array(keypoints, dtype=np.float32)

    def get_bounding_box(
        self, keypoints: np.ndarray, frame_shape: Tuple[int, int, int], padding: float = 0.1
    ) -> Optional[Tuple[int, int, int, int]]:
        """Compute bounding box around keypoints."""
        visible = keypoints[keypoints[:, 2] > 0.5]
        if len(visible) < 4:
            return None

        h, w, _ = frame_shape
        xs = visible[:, 0] * w
        ys = visible[:, 1] * h

        x1 = int(max(0, xs.min() - padding * w))
        y1 = int(max(0, ys.min() - padding * h))
        x2 = int(min(w, xs.max() + padding * w))
        y2 = int(min(h, ys.max() + padding * h))

        return (x1, y1, x2, y2)

    def release(self):
        """No explicit release needed for YOLO."""
        pass

    def __del__(self):
        self.release()


class MultiPersonPoseEstimator:
    """
    Multi-person pose estimation using YOLOv8 natively.
    """

    def __init__(self, config: Optional[PoseConfig] = None, max_persons: int = 5):
        self.config = config or PoseConfig()
        self.max_persons = max_persons
        from config import CONFIG
        self.model = YOLO(CONFIG.yolo_model_path)

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[List[Optional[np.ndarray]], np.ndarray]:
        """
        Detect multiple people and extract pose for each.
        """
        results = self.model(frame, verbose=False, conf=self.config.min_detection_confidence)
        
        annotated = frame.copy()
        all_keypoints = []

        if len(results) > 0 and results[0].keypoints is not None:
            kp_data = results[0].keypoints.data.cpu().numpy()
            h, w = frame.shape[:2]

            for i, landmarks in enumerate(kp_data[:self.max_persons]):
                # Draw skeleton for each
                annotated = self._draw_custom_skeleton(annotated, landmarks, i)
                
                # Extract normalized critical keypoints
                kps = []
                for idx in self.config.CRITICAL_KEYPOINTS:
                    lm = landmarks[idx]
                    kps.append([lm[0]/w, lm[1]/h, lm[2]])
                all_keypoints.append(np.array(kps))

        return all_keypoints, annotated

    def _draw_custom_skeleton(self, frame, landmarks, person_idx=0):
        """Basic skeleton drawing for multi-person view."""
        # Cycle colors
        color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)][person_idx % 4]
        
        connections = [(5, 6), (11, 12), (5, 11), (6, 12), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
        
        for p1, p2 in connections:
            if landmarks[p1, 2] > 0.4 and landmarks[p2, 2] > 0.4:
                cv2.line(frame, (int(landmarks[p1,0]), int(landmarks[p1,1])), 
                         (int(landmarks[p2,0]), int(landmarks[p2,1])), color, 2)
        return frame

    def release(self):
        pass
