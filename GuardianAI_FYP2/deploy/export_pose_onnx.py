"""
Export YOLOv8n-Pose to static ONNX files for the live server (weights/pose_onnx/)
=================================================================================
One file per input shape used by ai/pose_backend.py:
  480x640  4:3 sources (laptop webcams)
  384x640  16:9 sources
  640x640  anything else (letterboxed into a square)
These are the shapes Ultralytics itself uses for those sources (minimal
stride-32 rectangle), so results match the PyTorch path used for evaluation
while 4:3 / 16:9 frames cost 25% / 40% less than a 640x640 square.
Needs the onnx package on the Python path (only for exporting).
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "weights" / "pose_onnx"
SHAPES = [(480, 640), (384, 640), (640, 640)]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for h, w in SHAPES:
        src = YOLO(str(PROJECT / "weights" / "yolov8n-pose.pt")).export(format="onnx", imgsz=[h, w], opset=17, simplify=True)
        dst = OUT / f"yolov8n-pose-{h}x{w}.onnx"
        shutil.move(src, dst)
        print(f"{dst.name}  {dst.stat().st_size / 1e6:.1f} MB")
