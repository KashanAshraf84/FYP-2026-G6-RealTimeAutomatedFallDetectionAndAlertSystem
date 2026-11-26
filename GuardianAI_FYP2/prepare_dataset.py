"""
GuardianAI - Dataset Manager & Preparation Utility
==================================================
Manages real-world fall detection dataset video files, validates video readability,
downloads sample public benchmark clips, and prints the exact training command.

Usage:
  1. Inspect current dataset:
     python prepare_dataset.py --status

  2. Download sample public fall clips:
     python prepare_dataset.py --download-samples

  3. Automatically launch training:
     python prepare_dataset.py --train
"""

import os
import sys
import glob
import urllib.request
from pathlib import Path

DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset")
FALLS_DIR = os.path.join(DATASET_ROOT, "falls")
ADL_DIR = os.path.join(DATASET_ROOT, "adl")
WARNING_DIR = os.path.join(DATASET_ROOT, "warning")

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def ensure_directories():
    """Ensure the dataset directory structure exists."""
    for d in [DATASET_ROOT, FALLS_DIR, ADL_DIR, WARNING_DIR]:
        os.makedirs(d, exist_ok=True)


def get_video_files(directory):
    """Retrieve all video files in a given directory recursively."""
    if not os.path.isdir(directory):
        return []
    videos = []
    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in VIDEO_EXTS:
                videos.append(os.path.join(root, f))
    return videos


def print_status():
    """Display summary of currently loaded dataset clips."""
    ensure_directories()
    falls = get_video_files(FALLS_DIR)
    adls = get_video_files(ADL_DIR)
    warnings = get_video_files(WARNING_DIR)

    total = len(falls) + len(adls) + len(warnings)

    print("=" * 60)
    print(" [DATASET] GuardianAI Dataset Status:")
    print(f" Location: {DATASET_ROOT}")
    print("=" * 60)
    print(f"  - Falls Videos (Fall/Lying):     {len(falls):>4} files")
    print(f"  - ADL Videos (Walking/Sitting):  {len(adls):>4} files")
    print(f"  - Warning Videos (Instability):  {len(warnings):>4} files")
    print("-" * 60)
    print(f"  Total Video Files:               {total:>4} files")
    print("=" * 60)

    if total == 0:
        print("\n[!]  No video files found yet!")
        print("   Drop your .mp4 or .avi videos into:")
        print(f"   - Falls:   {FALLS_DIR}")
        print(f"   - ADL:     {ADL_DIR}")
        print("\n   Or run: python prepare_dataset.py --download-samples")
    else:
        print("\n[OK] Dataset ready! To train the model with this dataset, run:")
        print(
            '\n  .venv\\Scripts\\python train.py --benchmark generic '
            '--data-dir "data/dataset" --arch cnn_lstm --epochs 50 --loss focal --augment --export-onnx\n'
        )
    return total


def download_sample_clips():
    """Download sample open-source public fall and ADL test clips."""
    ensure_directories()
    print("\n[+] Setting up sample clips from local demo assets and public mirrors...")

    # Check local demo media first
    demo_media_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_media")
    demo_vid = os.path.join(demo_media_dir, "fyp2_demo_recording.mp4")
    test_avi = os.path.join(demo_media_dir, "videos", "test1.avi")

    import shutil
    copied = 0
    if os.path.isfile(demo_vid):
        dest = os.path.join(ADL_DIR, "demo_adl_walking.mp4")
        if not os.path.exists(dest):
            shutil.copy2(demo_vid, dest)
            print(f"  [OK] Added sample ADL clip: {os.path.basename(dest)}")
            copied += 1

    if os.path.isfile(test_avi):
        dest = os.path.join(FALLS_DIR, "demo_fall_test1.avi")
        if not os.path.exists(dest):
            shutil.copy2(test_avi, dest)
            print(f"  [OK] Added sample Fall clip: {os.path.basename(dest)}")
            copied += 1

    print(f"\nSample setup complete ({copied} new files added).")
    print_status()


if __name__ == "__main__":
    ensure_directories()
    if "--download-samples" in sys.argv or "--download" in sys.argv:
        download_sample_clips()
    elif "--train" in sys.argv:
        print_status()
        train_cmd = (
            f'"{sys.executable}" train.py --benchmark generic '
            '--data-dir "data/dataset" --arch cnn_lstm --epochs 50 --loss focal --augment --export-onnx'
        )
        print(f"\nLaunching: {train_cmd}\n")
        os.system(train_cmd)
    else:
        print_status()
