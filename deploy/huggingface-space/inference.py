"""
Fall Detection System - Real-Time Inference Engine
===================================================
Combines pose estimation, feature extraction, and neural network
inference for real-time fall detection from video streams.
"""

import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from typing import Optional, Dict, Tuple

from config import SystemConfig, CONFIG
from pose_estimator import PoseEstimator
from feature_extractor import FeatureExtractor
from fall_detector_model import create_model
from alert_system import AlertSystem
from dataset import LABEL_NAMES


class FallDetectionInference:
    """
    Real-time fall detection inference engine.:

    Pipeline:
      1. Capture frame from camera/video
      2. Run pose estimation (MediaPipe)
      3. Extract temporal features
      4. Run neural network classification
      5. Combine with rule-based checks
      6. Trigger alerts if fall detected
    """

    def __init__(self, config: Optional[SystemConfig] = None, model_path: Optional[str] = None):
        self.config = config or CONFIG
        self.device = torch.device(self.config.get_device())

        # Initialize components
        print("Initializing Fall Detection System...")

        # 1. Pose Estimator
        self.pose_estimator = PoseEstimator(self.config.pose)
        print("  [OK] Pose Estimator ready")

        # 2. Neural Network Model
        self.model = self._load_model(model_path)
        print("  [OK] Neural Network ready")

        # 3. Alert System
        self.alert_system = AlertSystem(self.config.alert)
        print("  [OK] Alert System ready")

        # Single-person state tracking
        self.feature_extractor = FeatureExtractor(self.config.features, self.config.pose)
        self._status_history = deque(maxlen=10)

        # System state
        self._fps_buffer = deque(maxlen=30)
        self._frame_count = 0
        self._current_summary = {"status": "normal", "confidence": 0.0, "person_count": 0}

    def _load_model(self, model_path: Optional[str] = None) -> Optional[torch.nn.Module]:
        """Load trained model from checkpoint."""
        path = model_path or self.config.model_save_path

        # Check for best model first
        best_path = os.path.join(os.path.dirname(path), "fall_detector_best.pth")
        if os.path.exists(best_path):
            path = best_path

        if not os.path.exists(path):
            print(f"  ⚠ No trained model found at {path}")
            print("    Will use rule-based detection only.")
            return None

        try:
            model = create_model(self.config.model).to(self.device)
            checkpoint = torch.load(path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            return model
        except Exception as e:
            print(f"  ⚠ Failed to load model: {e}")
            return None

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single video frame for a single person.
        """
        self._frame_count += 1

        # --- Step 1: Pose Estimation ---
        keypoints, annotated_frame = self.pose_estimator.process_frame(frame)
        person_count = 1 if keypoints is not None else 0

        # --- Step 2: Detection Pipeline ---
        if keypoints is not None:
            # Feature extraction
            sequence = self.feature_extractor.update(keypoints)

            # Rule-Based Detection
            rule_status, rule_confidence = self.feature_extractor.get_rule_based_status(keypoints)

            # Neural Network Detection
            nn_status = "normal"
            nn_confidence = 0.0
            if sequence is not None and self.model is not None:
                nn_status, nn_confidence, _ = self._predict(sequence)

            # Fuse Predictions
            final_status, final_confidence = self._fuse_predictions(
                rule_status, rule_confidence,
                nn_status, nn_confidence,
            )

            # Temporal Smoothing
            final_status, final_confidence = self._smooth_predictions(
                final_status, final_confidence
            )

            # --- Step 3: Trigger Alerts ---
            if final_status == "fall" and final_confidence > self.config.confidence_threshold:
                self.alert_system.trigger_alert(
                    status=final_status,
                    confidence=final_confidence,
                    frame=frame
                )
        else:
            self.feature_extractor.update(None)
            final_status, final_confidence = "normal", 0.0

        self._current_summary = {
            "status": final_status,
            "confidence": round(final_confidence * 100, 2),
            "person_count": person_count,
            "angle": round(getattr(self.feature_extractor, "_last_angle", 90.0), 1),
            "speed": round(getattr(self.feature_extractor, "_last_speed", 0.0), 1)
        }

        # --- Step 4: Draw HUD ---
        annotated_frame = self._draw_hud(annotated_frame, final_status, final_confidence)
        
        return {
            "status": final_status,
            "confidence": round(final_confidence * 100, 2),
            "person_count": person_count,
            "frame": annotated_frame
        }

    def _feature_extractor_update_null(self):
        """Update feature extractor with null keypoints."""
        self.feature_extractor.update(None)

    @torch.no_grad()
    def _predict(self, sequence: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Run neural network inference on a feature sequence.

        Returns:
            (predicted_label_name, confidence, class_probabilities)
        """
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        predicted_class = int(probs.argmax())
        confidence = float(probs[predicted_class])
        status = LABEL_NAMES[predicted_class]

        return status, confidence, probs

    def _fuse_predictions(
        self,
        rule_status: str,
        rule_conf: float,
        nn_status: str,
        nn_conf: float,
        nn_weight: float = 0.7,
    ) -> Tuple[str, float]:
        """
        Fuse rule-based and neural network predictions.
        Neural network has higher weight when available.
        """
        if nn_conf == 0.0:
            # No neural network prediction available
            return rule_status, rule_conf

        # Priority fusion: if either detector says "fall", investigate
        status_priority = {"fall": 2, "warning": 1, "normal": 0}

        rule_priority = status_priority[rule_status]
        nn_priority = status_priority[nn_status]

        rule_weight = 1.0 - nn_weight

        if nn_priority >= rule_priority:
            final_status = nn_status
            final_conf = nn_conf * nn_weight + rule_conf * rule_weight
        else:
            # Rule-based detected higher severity
            if rule_conf > 0.8:
                final_status = rule_status
                final_conf = rule_conf * 0.5 + nn_conf * 0.5
            else:
                final_status = nn_status
                final_conf = nn_conf * nn_weight + rule_conf * rule_weight

        return final_status, min(final_conf, 1.0)

    def _smooth_predictions(
        self, status: str, confidence: float
    ) -> Tuple[str, float]:
        """
        Temporal smoothing to reduce flickering.
        """
        self._status_history.append((status, confidence))

        if len(self._status_history) < 3:
            return status, confidence

        recent = list(self._status_history)[-5:]
        status_counts = {}
        for s, c in recent:
            if s not in status_counts:
                status_counts[s] = {"count": 0, "total_conf": 0.0}
            status_counts[s]["count"] += 1
            status_counts[s]["total_conf"] += c

        if "fall" in status_counts and status_counts["fall"]["count"] >= 3:
            avg_conf = status_counts["fall"]["total_conf"] / status_counts["fall"]["count"]
            return "fall", avg_conf

        if "warning" in status_counts and status_counts["warning"]["count"] >= 2:
            if "fall" not in status_counts or status_counts["fall"]["count"] < 3:
                avg_conf = status_counts["warning"]["total_conf"] / status_counts["warning"]["count"]
                return "warning", avg_conf

        most_frequent = max(status_counts.items(), key=lambda x: x[1]["count"])
        avg_conf = most_frequent[1]["total_conf"] / most_frequent[1]["count"]
        return most_frequent[0], avg_conf

    def _draw_hud(
        self,
        frame: np.ndarray,
        status: str,
        confidence: float,
    ) -> np.ndarray:
        """Draw Advanced HUD for FYP-2."""
        h, w = frame.shape[:2]
        
        # Colors
        colors = {
            "normal": (0, 200, 0),
            "warning": (0, 200, 255),
            "fall": (0, 0, 255),
        }
        color = colors.get(status, (255, 255, 255))

        # --- 1. TOP STATUS BAR ---
        cv2.rectangle(frame, (10, 10), (300, 70), (0, 0, 0), -1)
        cv2.putText(
            frame, f"STATUS: {status.upper()}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )
        cv2.putText(
            frame, f"CONF: {confidence*100:.1f}%", (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        # --- 2. BIOMECHANICAL TELEMETRY PANEL (Right Side) ---
        panel_w = 200
        panel_x = w - panel_w - 10
        cv2.rectangle(frame, (panel_x, 80), (w - 10, 240), (20, 20, 20), -1)
        cv2.rectangle(frame, (panel_x, 80), (w - 10, 240), (100, 100, 100), 1)
        
        cv2.putText(frame, "TELEMETRY", (panel_x + 45, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.line(frame, (panel_x + 10, 115), (w - 20, 115), (60, 60, 60), 1)

        # Get values from feature extractor
        fe = self.feature_extractor
        # Note: We'd ideally store these in a result dict, but for now we pull from internal state
        # or calculate them here for the demo.
        # We'll use dummy or estimated values if not directly exposed.
        angle = getattr(fe, "_last_angle", 90.0) # We'll need to expose these
        speed = getattr(fe, "_last_speed", 0.0)

        cv2.putText(frame, f"ANGLE: {angle:4.1f} deg", (panel_x + 15, 145), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"SPEED: {speed:4.1f} px/f", (panel_x + 15, 175), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Stability Bar
        bar_w = 170
        bar_x = panel_x + 15
        stability = max(0, min(1, 1.0 - (speed / 50.0)))
        cv2.rectangle(frame, (bar_x, 205), (bar_x + bar_w, 215), (40, 40, 40), -1)
        cv2.rectangle(frame, (bar_x, 205), (bar_x + int(bar_w * stability), 215), (255, 150, 0), -1)
        cv2.putText(frame, "STABILITY", (bar_x, 230), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # --- 3. FPS & SYSTEM INFO ---
        if self.config.show_fps:
            self._fps_buffer.append(time.time())
            if len(self._fps_buffer) > 1:
                fps = len(self._fps_buffer) / (self._fps_buffer[-1] - self._fps_buffer[0] + 1e-6)
                cv2.putText(
                    frame, f"FPS: {fps:.0f}", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )

        return frame

    def run_on_video(self, source=None):
        """
        Run fall detection on a video source (camera or file).
        """
        source = source if source is not None else self.config.camera.source

        print(f"\nStarting fall detection on source: {source}")
        print("Press 'q' to quit, 'r' to reset tracking\n")

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Cannot open video source: {source}")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break

                # Process frame
                result = self.process_frame(frame)

                # Display
                cv2.imshow(self.config.window_name, result["frame"])

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"): break
                elif key == ord("r"):
                    self.feature_extractor.reset()
                    self._status_history.clear()
                    print("Tracking reset.")

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.pose_estimator.release()
            print("\nInference stopped.")

    def get_detection_result(self) -> Dict:
        """
        Get current system-wide detection result summary.
        """
        return self._current_summary

    def release(self):
        """Release all resources."""
        self.pose_estimator.release()


# Import datetime for screenshot naming
from datetime import datetime
