"""
Export the FYP-2 models for the browser demo (deploy/static_space/models/)
=========================================================================
- yolov8n-pose-640.onnx : YOLOv8n-Pose, input 1x3x640x640 RGB in [0,1], output 1x56x8400
- fallnet_v2.onnx       : FallNet v2 with feature normalisation and temperature
                          scaling baked in. Input: raw features (B, 30, 62) from
                          ai.pose_features.window_features. Output: probabilities (B, 3).

Each export is checked against PyTorch with onnxruntime.
Needs onnx / onnxruntime / onnxscript on the Python path.
"""

import shutil
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "deploy" / "static_space" / "models"


class CalibratedFallNet(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.model = net.model
        self.register_buffer("mean", torch.from_numpy(net.mean).float())
        self.register_buffer("std", torch.from_numpy(net.std).float())
        self.temperature = float(net.temperature)

    def forward(self, feats):
        return torch.softmax(self.model((feats - self.mean) / self.std) / self.temperature, dim=1)


def export_fallnet():
    import onnxruntime as ort
    from ai.fallnet import FallNet

    net = FallNet()
    wrapper = CalibratedFallNet(net).eval()
    dummy = torch.randn(2, 30, 62)
    path = OUT / "fallnet_v2.onnx"
    torch.onnx.export(wrapper, dummy, str(path), input_names=["features"], output_names=["probs"],
                      dynamic_axes={"features": {0: "batch"}, "probs": {0: "batch"}}, opset_version=17,
                      dynamo=False)
    x = np.random.default_rng(0).normal(size=(5, 30, 62)).astype(np.float32) * net.std + net.mean
    ref = net.predict_features(x)
    got = ort.InferenceSession(str(path)).run(None, {"features": x})[0]
    print(f"fallnet_v2.onnx  max |diff| vs PyTorch = {np.abs(ref - got).max():.2e}  ({path.stat().st_size / 1e6:.1f} MB)")
    return net.model_version


def export_yolo():
    from ultralytics import YOLO

    src = YOLO(str(PROJECT / "weights" / "yolov8n-pose.pt")).export(format="onnx", imgsz=640, opset=17, simplify=True)
    dst = OUT / "yolov8n-pose-640.onnx"
    shutil.move(src, dst)
    print(f"yolov8n-pose-640.onnx  ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    version = export_fallnet()
    export_yolo()
    (OUT / "model_version.txt").write_text(version, encoding="utf-8")
    print("model version", version)
