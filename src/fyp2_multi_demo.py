"""
FYP-2: Multi-Person Fall Detection Demo
=======================================
Advanced Prototype using YOLOv8-Pose & Temporal AI Fusion.
Handles multiple individuals simultaneously.
"""

import sys
import os
import cv2
import time
import argparse
import numpy as np
from collections import deque

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pose_estimator import MultiPersonPoseEstimator
from inference import FallDetectionInference
from config import SystemConfig, CONFIG

class PersonState:
    """Maintains detection state for a single person."""
    def __init__(self, config):
        # We reuse the inference logic but for a specific person
        self.inference = FallDetectionInference(config=config)
        # Point to the same model to save memory if needed, 
        # but here we'll just let each instance handle its own if fine.
        # However, to be efficient, we can share the model.
        
    def update(self, frame, keypoints):
        # Directly use keypoints for this person
        # We bypass the pose estimation in the internal inference since we have it
        if keypoints is None:
            return "normal", 0.0
            
        sequence = self.inference.feature_extractor.update(keypoints)
        rule_status, rule_conf, rule_is_instant = self.inference.feature_extractor.get_rule_based_status(keypoints)

        nn_status, nn_conf = "normal", 0.0
        if sequence is not None and self.inference.model is not None:
            nn_status, nn_conf, _ = self.inference._predict(sequence)

        status, conf = self.inference._fuse_predictions(rule_status, rule_conf, nn_status, nn_conf)
        status, conf = self.inference._smooth_predictions(status, conf)

        if rule_is_instant and status != "fall":
            status = "warning"
            conf = max(conf, rule_conf)

        return status, conf

def run_multi_demo(source=0):
    config = SystemConfig()
    pose_est = MultiPersonPoseEstimator(config.pose, max_persons=5)
    
    # Trackers for individuals (simple index-based for demo)
    trackers = [PersonState(config) for _ in range(5)]
    
    cap = cv2.VideoCapture(source)
    print("\n[*] FYP-2 MULTI-PERSON SYSTEM ACTIVE.")
    print("[*] Press 'q' to exit.\n")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Detect everyone
        all_keypoints, annotated = pose_est.process_frame(frame)
        
        # 2. Analyze each person
        for i, kps in enumerate(all_keypoints):
            if i >= len(trackers): break
            
            status, conf = trackers[i].update(frame, kps)
            
            # --- Draw individual status label near their head ---
            if kps is not None and len(kps) > 0:
                h, w = frame.shape[:2]
                head_x, head_y = int(kps[0,0] * w), int(kps[0,1] * h)
                
                color = (0, 255, 0)
                if status == "fall": color = (0, 0, 255)
                elif status == "warning": color = (0, 200, 255)
                
                label = f"P{i+1}: {status.upper()} ({conf*100:.0f}%)"
                cv2.rectangle(annotated, (head_x - 10, head_y - 40), 
                              (head_x + 180, head_y - 10), (0,0,0), -1)
                cv2.putText(annotated, label, (head_x, head_y - 18), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 3. System HUD
        cv2.rectangle(annotated, (10, 10), (280, 50), (0,0,0), -1)
        cv2.putText(annotated, f"FYP-2 MULTI-TRACKER | {len(all_keypoints)} DETECTED", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("FYP-2: Multi-Person Fall Detection", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="webcam")
    args = parser.parse_args()
    
    source = int(args.source) if args.source.isdigit() else args.source
    run_multi_demo(source)
