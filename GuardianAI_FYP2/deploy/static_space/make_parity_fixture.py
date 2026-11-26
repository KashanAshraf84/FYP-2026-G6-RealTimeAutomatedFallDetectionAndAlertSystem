"""
Fixtures for test_parity.mjs (not deployed).
1. Cached YOLO detections of a few clips replayed at 5 FPS through the Python
   PersonTracker + FallNet (same logic as ai.adapters.TeamDetector.predict),
   recording each frame's IDs and probabilities.
2. Two real frames: the 640x640 letterboxed input tensor plus what Ultralytics
   detects on the original frame, to check the browser's YOLO decoding.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from ai.fallnet import FallNet
from ai.pose_features import partial_view, run_pose
from ai.tracking import PersonTracker
from training.extract_keypoints import CACHE_DIR
from training.gmdcsa import load_clips

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
clips = {c.clip_id: c for c in load_clips()}
net = FallNet()

replays = []
for cid in ["S3-fall-03", "S1-fall-05", "S2-adl-18", "S4-fall-03", "S2-adl-22", "S3-fall-01"]:
    with np.load(CACHE_DIR / f"{cid}.npz") as d:
        times, det_frame, kps, boxes, H = d["times"], d["det_frame"], d["kps"], d["boxes"], int(d["height"])
    bins = np.floor(times * 5 + 1e-6)
    keep = np.r_[True, bins[1:] != bins[:-1]]
    tracker = PersonTracker()
    frames = []
    for f in np.flatnonzero(keep):
        rows = np.flatnonzero(det_frame == f)
        t = float(times[f])
        tracker.update(t, kps[rows], boxes[rows], H)
        people = tracker.active(t)
        expected = []
        for p in people:
            w = p.history.window(t)
            probs = None if w is None or partial_view(w, p.box[3], H) else net.predict_windows(w[None])[0].tolist()
            expected.append({"id": p.track_id, "probs": probs})
        frames.append({"t": t, "kps": kps[rows].tolist(), "boxes": boxes[rows].tolist(), "expected": expected})
    replays.append({"clip": cid, "H": H, "frames": frames})
(OUT / "replays.json").write_text(json.dumps(replays))

from ultralytics import YOLO
from ai.pose_features import YOLO_IMGSZ
yolo = YOLO("weights/yolov8n-pose.pt")
frames = []
for cid, t in [("S1-fall-05", 1.0), ("S2-adl-18", 3.0)]:
    cap = cv2.VideoCapture(str(clips[cid].path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, bgr = cap.read()
    cap.release()
    bgr = cv2.resize(bgr, (640, round(bgr.shape[0] * 640 / bgr.shape[1])))  # browser-sized frame
    h, w = bgr.shape[:2]
    r = min(YOLO_IMGSZ / h, YOLO_IMGSZ / w)
    nw, nh = round(w * r), round(h * r)
    top, left = round((YOLO_IMGSZ - nh) / 2 - 0.1), round((YOLO_IMGSZ - nw) / 2 - 0.1)
    canvas = np.full((YOLO_IMGSZ, YOLO_IMGSZ, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    name = f"frame_{cid}.bin"
    tensor.tofile(OUT / name)
    k, b = run_pose(yolo, bgr)
    frames.append({"file": name, "w": w, "h": h, "kps": k.tolist(), "boxes": b.tolist()})
(OUT / "frames.json").write_text(json.dumps(frames))
print("fixtures written:", len(replays), "replays,", sum(len(r["frames"]) for r in replays), "frames;", len(frames), "YOLO frames")
