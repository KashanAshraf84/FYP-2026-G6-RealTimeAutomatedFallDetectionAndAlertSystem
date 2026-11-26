"""
Pose Backends for the live server
=================================
YOLOv8n-Pose inference with a choice of runtime, all returning exactly what
ai.pose_features.run_pose returns: (keypoints (n, 17, 3) in pixels, boxes
(n, 5) x1, y1, x2, y2, confidence) with Ultralytics' default post-processing
(confidence 0.25, NMS IoU 0.7, at most 300 people).

  directml  ONNX Runtime + DirectML on any DirectX 12 GPU. On a laptop with a
            second (NVIDIA / AMD) GPU this keeps inference off the Intel chip
            that draws the screen, which is what makes the UI stay smooth.
  openvino  Intel iGPU or CPU through OpenVINO, with the GPU queue throttled
            so a CPU core does not spin while waiting for the GPU.
  torch     Ultralytics / PyTorch, the reference used for training extraction.

Input shape: like Ultralytics, frames are letterboxed into the smallest
stride-32 rectangle (480x640 for 4:3, 384x640 for 16:9) instead of a 640x640
square, which costs 25-40% less and matches the evaluated PyTorch path.

Selection (environment variables, read by create_pose_backend):
  GUARDIAN_POSE_BACKEND = auto | directml | openvino | torch     (default auto)
  GUARDIAN_POSE_DEVICE  = DirectML adapter index (0, 1, ...) or OpenVINO device (GPU.0, CPU)
"auto" times every available option once on this machine (a few seconds at
the first start), keeps the fastest, and remembers it in weights/pose_backend.json.
"""

import json
import logging
import math
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from ai.pose_features import DET_CONF, NUM_KEYPOINTS, YOLO_IMGSZ

log = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent.parent
YOLO_WEIGHTS = PROJECT_DIR / "weights" / "yolov8n-pose.pt"
ONNX_DIR = PROJECT_DIR / "weights" / "pose_onnx"
CHOICE_FILE = PROJECT_DIR / "weights" / "pose_backend.json"
SHAPES = ((480, 640), (384, 640), (640, 640))  # (height, width) with an exported ONNX file
NMS_IOU = 0.7
MAX_DET = 300


def pick_shape(h, w, size=YOLO_IMGSZ):
    """Ultralytics' minimal stride-32 rectangle for an h x w frame, or the square if not exported."""
    r = min(size / h, size / w)
    shape = (math.ceil(round(h * r) / 32) * 32, math.ceil(round(w * r) / 32) * 32)
    return shape if shape in SHAPES else (size, size)


def letterbox(bgr, shape):
    """BGR frame -> (1, 3, H, W) float32 RGB in [0, 1] (grey padding 114) plus the mapping back."""
    H, W = shape
    h, w = bgr.shape[:2]
    r = min(H / h, W / w)
    nw, nh = round(w * r), round(h * r)
    top, left = round((H - nh) / 2 - 0.1), round((W - nw) / 2 - 0.1)
    if (nh, nw) == (H, W):
        canvas = bgr if (h, w) == (H, W) else cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        canvas = np.full((H, W, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255.0
    return tensor, (r, left, top, w, h)


def decode(output, mapping, conf=DET_CONF):
    """(1, 56, N) raw YOLOv8-pose output -> (keypoints px, boxes) in the source frame."""
    r, left, top, w, h = mapping
    pred = output[0]  # (56, N)
    keep = np.flatnonzero(pred[4] > conf)
    if keep.size == 0:
        return np.zeros((0, NUM_KEYPOINTS, 3), np.float32), np.zeros((0, 5), np.float32)
    p = pred[:, keep]
    cx, cy, bw, bh, score = p[0], p[1], p[2], p[3], p[4]
    xywh = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), score.tolist(), conf, NMS_IOU)
    idx = np.array(idx, dtype=int).reshape(-1)
    idx = idx[np.argsort(-score[idx])][:MAX_DET]
    x1 = np.clip((xywh[idx, 0] - left) / r, 0, w)
    y1 = np.clip((xywh[idx, 1] - top) / r, 0, h)
    x2 = np.clip((xywh[idx, 0] + xywh[idx, 2] - left) / r, 0, w)
    y2 = np.clip((xywh[idx, 1] + xywh[idx, 3] - top) / r, 0, h)
    boxes = np.stack([x1, y1, x2, y2, score[idx]], axis=1).astype(np.float32)
    kp = p[5:, idx].T.reshape(len(idx), NUM_KEYPOINTS, 3).copy()
    kp[..., 0] = np.clip((kp[..., 0] - left) / r, 0, w)
    kp[..., 1] = np.clip((kp[..., 1] - top) / r, 0, h)
    return kp.astype(np.float32), boxes


def onnx_path(shape):
    return ONNX_DIR / f"yolov8n-pose-{shape[0]}x{shape[1]}.onnx"


class _ShapedBackend:
    """Common part: one compiled model per input shape, created on first use."""

    def __init__(self):
        self._models = {}

    def _model(self, shape):
        if shape not in self._models:
            self._models[shape] = self._load(shape)
        return self._models[shape]

    def prepare(self, shapes=SHAPES[:2]):
        """Compile the common shapes up front so the first frame of a source does not wait."""
        for shape in shapes:
            self._model(shape)

    def detect(self, bgr):
        shape = pick_shape(*bgr.shape[:2])
        tensor, mapping = letterbox(bgr, shape)
        return decode(self._run(self._model(shape), tensor), mapping)


class DirectMLPose(_ShapedBackend):
    """ONNX Runtime + DirectML on one DirectX 12 adapter (index as in dxgi_adapters())."""

    name = "directml"

    def __init__(self, device=0):
        super().__init__()
        import onnxruntime as ort

        if "DmlExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("onnxruntime-directml is not installed")
        self._ort = ort
        self.device = int(device)
        adapter = dxgi_adapters().get(self.device)
        self.label = f"DirectML · {adapter or f'GPU {self.device}'}"

    def _load(self, shape):
        ort = self._ort
        options = ort.SessionOptions()
        options.enable_mem_pattern = False                      # required by the DirectML provider
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")  # no busy-waiting CPU threads
        session = ort.InferenceSession(
            str(onnx_path(shape)), options,
            providers=[("DmlExecutionProvider", {"device_id": self.device})],
        )
        return session, session.get_inputs()[0].name

    def _run(self, model, tensor):
        session, name = model
        return session.run(None, {name: tensor})[0]


class OpenVinoPose(_ShapedBackend):
    """OpenVINO on an Intel GPU (preferred) or the CPU."""

    name = "openvino"

    def __init__(self, device=None):
        super().__init__()
        import openvino as ov

        self._core = ov.Core()
        self.device = device or _pick_openvino_device(self._core)
        self.label = f"OpenVINO · {self._core.get_property(self.device, 'FULL_DEVICE_NAME')}"

    def _load(self, shape):
        config = {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": str(PROJECT_DIR / "weights" / "ov_cache")}
        if self.device.startswith("GPU"):
            # Sleep instead of spinning a CPU core while the GPU works, and let the desktop go first.
            config.update({"GPU_QUEUE_THROTTLE": "LOW", "MODEL_PRIORITY": "LOW"})
        else:
            config["INFERENCE_NUM_THREADS"] = max(1, min(4, (os.cpu_count() or 2) // 2))  # leave cores for the UI
        compiled = self._core.compile_model(self._core.read_model(str(onnx_path(shape))), self.device, config)
        return compiled.create_infer_request()

    def _run(self, request, tensor):
        request.infer({0: tensor})
        return request.get_output_tensor(0).data


class TorchPose:
    """Ultralytics / PyTorch reference backend."""

    name = "torch"

    def __init__(self):
        from ultralytics import YOLO

        self.model = YOLO(str(YOLO_WEIGHTS))
        self.label = "PyTorch CPU (Ultralytics)"

    def prepare(self, shapes=()):
        pass

    def detect(self, bgr):
        from ai.pose_features import run_pose

        return run_pose(self.model, bgr)


def _pick_openvino_device(core):
    for dev in core.available_devices:
        if dev.startswith("GPU") and "Intel" in core.get_property(dev, "FULL_DEVICE_NAME"):
            return dev
    return "CPU"


def dxgi_adapters():
    """{index: name} of DirectX adapters in DXGI order (the order DirectML device ids use). {} if unavailable."""
    try:
        import ctypes
        import uuid
        from ctypes import wintypes

        class Desc(ctypes.Structure):
            _fields_ = [("Description", wintypes.WCHAR * 128), ("VendorId", wintypes.UINT), ("DeviceId", wintypes.UINT),
                        ("SubSysId", wintypes.UINT), ("Revision", wintypes.UINT), ("DedicatedVideoMemory", ctypes.c_size_t),
                        ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                        ("AdapterLuid", ctypes.c_int64)]

        def method(obj, index, *argtypes):
            vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtable[index])

        iid = (ctypes.c_byte * 16).from_buffer_copy(uuid.UUID("7b7166ec-21c7-44ae-b21a-c9ae321ae369").bytes_le)
        factory = ctypes.c_void_p()
        if ctypes.WinDLL("dxgi").CreateDXGIFactory(ctypes.byref(iid), ctypes.byref(factory)) != 0:
            return {}
        names, i = {}, 0
        while i < 8:
            adapter = ctypes.c_void_p()
            if method(factory, 7, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))(factory, i, ctypes.byref(adapter)) != 0:
                break
            desc = Desc()
            if method(adapter, 8, ctypes.POINTER(Desc))(adapter, ctypes.byref(desc)) == 0:
                names[i] = desc.Description
            method(adapter, 2)(adapter)  # Release
            i += 1
        method(factory, 2)(factory)
        return names
    except Exception:
        return {}


def _time_backend(backend, runs=4):
    """Cost of one 640x480 frame after warm-up: median wall milliseconds plus CPU milliseconds."""
    frame = np.full((480, 640, 3), 90, np.uint8)
    cv2.rectangle(frame, (250, 120), (390, 460), (200, 180, 160), -1)
    backend.detect(frame)
    backend.detect(frame)
    times, cpu = [], time.process_time()
    for _ in range(runs):
        t = time.perf_counter()
        backend.detect(frame)
        times.append((time.perf_counter() - t) * 1000)
    # CPU time counts too: a backend that spins a core while the GPU works makes the laptop sluggish.
    return float(np.median(times)) + (time.process_time() - cpu) * 1000 / runs


def _candidates():
    """(kind, device, constructor) for everything this machine might run."""
    found = []
    try:
        import onnxruntime as ort

        if "DmlExecutionProvider" in ort.get_available_providers():
            adapters = dxgi_adapters()
            for index in (adapters or {0: "", 1: ""}):
                if "Basic Render" not in adapters.get(index, ""):  # software rasteriser
                    found.append(("directml", index, lambda i=index: DirectMLPose(i)))
    except Exception as exc:
        log.info("DirectML not available: %s", exc)
    try:
        import openvino as ov

        core = ov.Core()
        for dev in core.available_devices:
            if dev == "CPU" or "Intel" in core.get_property(dev, "FULL_DEVICE_NAME"):
                found.append(("openvino", dev, lambda d=dev: OpenVinoPose(d)))
    except Exception as exc:
        log.info("OpenVINO not available: %s", exc)
    return found


def _build(kind, device):
    if kind == "directml":
        return DirectMLPose(device if device is not None else 0)
    if kind == "openvino":
        return OpenVinoPose(str(device) if device is not None else None)
    return TorchPose()


def _auto_select():
    """Fastest working backend on this machine, remembered in weights/pose_backend.json."""
    host = platform.node()
    try:
        saved = json.loads(CHOICE_FILE.read_text(encoding="utf-8"))
        if saved.get("host") == host:
            backend = _build(saved["kind"], saved["device"])
            backend.prepare()
            return backend
    except Exception:
        pass  # no file, another machine, or the saved choice no longer works: measure again

    timings, best = {}, None
    for kind, device, make in _candidates():
        try:
            backend = make()
            ms = _time_backend(backend)
        except Exception as exc:
            log.info("Pose backend %s %s skipped: %s", kind, device, str(exc)[:120])
            continue
        timings[backend.label] = round(ms, 1)
        # A second GPU (DirectML index >= 1) does not draw the screen: prefer it unless clearly slower.
        score = ms * (0.77 if kind == "directml" and device >= 1 else 1.0)
        if best is None or score < best[0]:
            best = (score, kind, device, backend)
    if best is None:
        return TorchPose()
    _, kind, device, backend = best
    log.warning("Pose backends timed (wall + CPU ms per frame): %s -> using %s", timings, backend.label)
    try:
        CHOICE_FILE.write_text(json.dumps({"host": host, "kind": kind, "device": device, "label": backend.label,
                                           "timings_ms": timings}, indent=2), encoding="utf-8")
    except OSError:
        pass
    backend.prepare()
    return backend


def create_pose_backend():
    """Backend chosen by GUARDIAN_POSE_BACKEND / GUARDIAN_POSE_DEVICE, falling back to PyTorch."""
    choice = os.environ.get("GUARDIAN_POSE_BACKEND", "auto").lower()
    device = os.environ.get("GUARDIAN_POSE_DEVICE") or None
    try:
        if choice == "auto" and device is None:
            backend = _auto_select()
        elif choice == "torch":
            backend = TorchPose()
        else:
            kind = "openvino" if choice in ("auto", "openvino") else choice
            backend = _build(kind, int(device) if kind == "directml" and device is not None else device)
            backend.prepare()
        backend.detect(np.zeros((480, 640, 3), np.uint8))  # warm-up
    except Exception as exc:
        if choice not in ("auto",):
            raise
        log.warning("Accelerated pose backend unavailable (%s); using PyTorch", exc)
        backend = TorchPose()
    log.warning("Pose backend: %s", backend.label)
    return backend
