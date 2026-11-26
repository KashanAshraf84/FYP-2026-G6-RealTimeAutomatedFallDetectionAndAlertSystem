"""
Pose Extraction (run once)
==========================
Runs YOLOv8-Pose over every GMDCSA-24 clip on a fixed 15 Hz time grid and
caches ALL person detections per sampled frame in data/keypoints/<clip_id>.npz.
Training never re-runs YOLO: changing labels, features or the model only
needs this cache. The run is resumable; finished clips are skipped.

    .venv\\Scripts\\python -m training.extract_keypoints
    .venv\\Scripts\\python -m training.extract_keypoints --force    # rebuild all
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

from ai.pose_features import DET_CONF, SAMPLE_HZ, YOLO_IMGSZ, run_pose
from training.gmdcsa import PROJECT_DIR, load_clips

CACHE_DIR = PROJECT_DIR / "data" / "keypoints"
CACHE_VERSION = 1
YOLO_WEIGHTS = PROJECT_DIR / "weights" / "yolov8n-pose.pt"


def extract_clip(model, video_path):
    """Return the cache arrays for one video sampled at SAMPLE_HZ."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    period = 1.0 / SAMPLE_HZ
    next_t = 0.0
    idx = -1
    times, det_frame, kps_all, boxes_all = [], [], [], []
    while cap.grab():
        idx += 1
        t = idx / fps
        # Half a source frame of tolerance keeps 15 FPS sources at every frame
        # and 30 FPS sources at every second frame.
        if t < next_t - 0.5 / fps:
            continue
        while next_t <= t + 0.5 / fps:
            next_t += period
        ok, frame = cap.retrieve()
        if not ok:
            break
        kps, boxes = run_pose(model, frame)
        frame_no = len(times)
        times.append(t)
        for k, b in zip(kps, boxes):
            det_frame.append(frame_no)
            kps_all.append(k)
            boxes_all.append(b)
    cap.release()

    return {
        "times": np.asarray(times, np.float32),
        "det_frame": np.asarray(det_frame, np.int32),
        "kps": np.asarray(kps_all, np.float32).reshape(-1, 17, 3),
        "boxes": np.asarray(boxes_all, np.float32).reshape(-1, 5),
        "width": np.int32(width),
        "height": np.int32(height),
        "fps": np.float32(fps),
        "src_frames": np.int32(idx + 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Re-extract clips that are already cached")
    args = parser.parse_args()

    from ultralytics import YOLO

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    clips = load_clips()
    model = YOLO(str(YOLO_WEIGHTS))
    print(f"[extract] {len(clips)} clips -> {CACHE_DIR} (sample {SAMPLE_HZ} Hz, conf {DET_CONF}, imgsz {YOLO_IMGSZ})", flush=True)

    manifest = {}
    started = time.time()
    done_now = 0
    for i, clip in enumerate(clips, 1):
        out = CACHE_DIR / f"{clip.clip_id}.npz"
        if out.exists() and not args.force:
            with np.load(out) as cached:
                if int(cached["cache_version"]) == CACHE_VERSION:
                    manifest[clip.clip_id] = json.loads(str(cached["summary"]))
                    continue

        t0 = time.time()
        data = extract_clip(model, clip.path)
        n = len(data["times"])
        with_person = len(np.unique(data["det_frame"])) if len(data["det_frame"]) else 0
        summary = {
            "video": str(clip.path.relative_to(PROJECT_DIR)),
            "sampled_frames": n,
            "frames_with_person": with_person,
            "detections": int(len(data["det_frame"])),
            "duration_s": round(float(data["src_frames"] / data["fps"]), 2),
            "fps": round(float(data["fps"]), 2),
            "resolution": f"{int(data['width'])}x{int(data['height'])}",
        }
        tmp = CACHE_DIR / f"{clip.clip_id}.tmp.npz"
        np.savez_compressed(
            tmp, **data,
            cache_version=np.int32(CACHE_VERSION),
            sample_hz=np.int32(SAMPLE_HZ),
            det_conf=np.float32(DET_CONF),
            imgsz=np.int32(YOLO_IMGSZ),
            summary=json.dumps(summary),
        )
        os.replace(tmp, out)
        manifest[clip.clip_id] = summary
        done_now += 1

        elapsed = time.time() - started
        remaining = sum(
            1 for c in clips[i:] if not (CACHE_DIR / f"{c.clip_id}.npz").exists() or args.force
        )
        eta = elapsed / done_now * remaining
        print(
            f"[{i:3d}/{len(clips)}] {clip.clip_id:<14} {n:4d} frames, person in {with_person:4d} "
            f"({time.time() - t0:5.1f}s)  elapsed {elapsed / 60:5.1f} min, ETA {eta / 60:5.1f} min",
            flush=True,
        )

    (CACHE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(m["sampled_frames"] for m in manifest.values())
    person = sum(m["frames_with_person"] for m in manifest.values())
    print(f"[extract] done: {len(manifest)} clips, {total} sampled frames, person detected in "
          f"{person} ({person / max(total, 1):.1%}); {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
