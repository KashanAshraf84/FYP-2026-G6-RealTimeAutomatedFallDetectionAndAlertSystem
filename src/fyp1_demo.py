"""
FYP-1: Real-Time Fall Detection System Demo
===========================================
Academic Prototype using MediaPipe Pose & Rule-Based Logic.
Designed for FYP-1 Demonstration.

Features:
- Real-time pose estimation
- Biomechanical heuristic detection (Orientation, Velocity, Crumple Ratio)
- Multi-Input support (Webcam / Random Video)
- Automated alert logging
"""

import sys
import os
import cv2
import random
import time
import argparse
import glob
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import FallDetectionInference
from config import CONFIG, SystemConfig

def get_random_video(video_dir):
    """Pick a random video from the specified directory."""
    extensions = ['*.mp4', '*.avi', '*.mkv', '*.mov', '*.webm']
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
    
    if not video_files:
        return None
    return random.choice(video_files)

def run_demo(source_type="webcam", video_folder=None):
    """Main demo execution loop."""
    config = SystemConfig()
    
    # Default video folder to absolute path in project dir
    if video_folder is None:
        video_folder = os.path.join(config.project_dir, "videos")
    elif not os.path.isabs(video_folder):
        video_folder = os.path.join(config.project_dir, video_folder)

    print("\n" + "="*50)
    
    # Initialize Inference Engine
    # Note: We can now use BOTH Rule-Based and AI for FYP-2
    detector = FallDetectionInference(config=config)
    
    # Mode state: "rule" or "ai" or "fused"
    # Starting with "fused" for FYP-2, but allowing toggle
    demo_mode = "fused"
    
    source = 0 # Default webcam
    
    if source_type == "random":
        video_path = get_random_video(video_folder)
        if video_path:
            source = video_path
            print(f"[*] Random Video selected: {os.path.basename(video_path)}")
        else:
            print(f"[!] No videos found in {video_folder}. Falling back to Webcam.")
            source = 0
    elif source_type.isdigit():
        source = int(source_type)
    elif os.path.exists(source_type):
        source = source_type
        print(f"[*] Processing Video: {os.path.basename(source_type)}")

    print("[*] Initializing Camera/Video Stream...")
    cap = cv2.VideoCapture(source)
    
    # Get FPS for timing control
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or video_fps > 120: video_fps = 30
    delay_ms = int(1000 / video_fps)

    print(f"[*] FYP-2 SYSTEM ACTIVE.")
    print(f"[*] Mode: {demo_mode.upper()} | Playback: {video_fps} FPS.")
    print("[*] Controls: 'q' to exit, 'm' to toggle mode, 'r' to reset buffers.\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # ... (video looping logic)
                if source_type == "random":
                    print("[*] Video finished. Picking another random video...")
                    video_path = get_random_video(video_folder)
                    if video_path:
                        cap.release()
                        cap = cv2.VideoCapture(video_path)
                        continue
                elif source_type != "webcam":
                    # Robust loop for single video files
                    print(f"[*] Reached end of video. Restarting demo: {os.path.basename(source)}")
                    cap.release()
                    cap = cv2.VideoCapture(source)
                    continue
                break

            # Process Frame
            # Note: We pass the mode to the detector if we want strict mode control
            # But the detector already fuses them by default.
            result = detector.process_frame(frame)
            processed_frame = result["frame"]
            status = result["status"]
            confidence = result["confidence"]

            # --- Premium UI Overlay ---
            h, w = processed_frame.shape[:2]
            
            # Status Banner
            banner_h = 60
            overlay = processed_frame.copy()
            
            # Banner color based on status
            color = (0, 0, 0)
            if status == "fall": color = (0, 0, 180) # Red
            elif status == "warning": color = (0, 150, 255) # Yellow/Orange
            else: color = (40, 150, 40) # Green
            
            cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
            cv2.addWeighted(overlay, 0.7, processed_frame, 0.3, 0, processed_frame)
            
            # Text Labels
            cv2.putText(processed_frame, f"FYP-2 TRANSITION | MODE: {demo_mode.upper()}", (15, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.putText(processed_frame, f"STATUS: {status.upper()}", (15, banner_h - 10), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.putText(processed_frame, f"CONFIDENCE: {confidence}%", (w - 220, banner_h - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            # Display
            cv2.imshow("Advanced Demo: AI Fall Detection (FYP-2)", processed_frame)

            # Keyboard Controls
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                # Toggle mode logic
                if demo_mode == "fused": demo_mode = "rule"
                elif demo_mode == "rule": demo_mode = "ai"
                else: demo_mode = "fused"
                print(f"[*] Switched Mode to: {demo_mode.upper()}")
            elif key == ord('r'):
                detector.feature_extractor.reset()
                print("[*] Tracking Buffers Reset.")

    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.release()
        print("\n[*] Demo Terminated Successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FYP-1 Fall Detection Demo")
    parser.add_argument("--source", type=str, default="webcam", 
                        help="Input source: 'webcam', 'random', or path to file")
    parser.add_argument("--folder", type=str, default="videos", 
                        help="Folder for random video selection")
    
    args = parser.parse_args()
    run_demo(source_type=args.source, video_folder=args.folder)
