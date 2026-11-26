"""
Pose Features (shared by training and live inference)
=====================================================
Everything that turns YOLOv8-Pose output into model input lives here, so the
training pipeline (training/) and the Django adapter (ai/fallnet.py) cannot
drift apart.

Pipeline:
  1. run_pose()        YOLOv8-Pose -> all persons' 17 COCO keypoints (pixels) + boxes
  2. SubjectSelector   picks one subject per frame (box continuity, else largest)
  3. resample_window() last WINDOW_S seconds of timestamped keypoints -> SEQ_LEN
                       evenly spaced steps (makes 5 FPS browser input and 15 FPS
                       training data look the same to the model)
  4. window_features()  (SEQ_LEN, 17, 3) -> (SEQ_LEN, NUM_FEATURES), translation-
                       and scale-normalised by the subject's body size

Coordinates are divided by frame HEIGHT on both axes so the body's aspect
ratio is preserved for 4:3 and 16:9 sources.
"""

import numpy as np

FEATURE_VERSION = "pose-seq-v2"

# Sampling / windowing (identical in training and inference)
SAMPLE_HZ = 15            # training cache sampling rate
WINDOW_S = 2.0            # seconds of motion the model sees
SEQ_LEN = 30              # resampled steps per window (WINDOW_S * SAMPLE_HZ)
MIN_HISTORY_S = 1.5       # minimum pose history before a prediction is made

# YOLOv8-Pose call parameters (identical in training and inference)
DET_CONF = 0.25
YOLO_IMGSZ = 640

NUM_KEYPOINTS = 17
# COCO keypoints: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows,
# 9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles (left/right).
HEAD = [0, 1, 2, 3, 4]
SHOULDERS = [5, 6]
HIPS = [11, 12]
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
VISIBLE_CONF = 0.3
VELOCITY_CLIP = 8.0       # body-sizes per second


def run_pose(model, bgr):
    """
    YOLOv8-Pose on one BGR frame -> (keypoints (n,17,3) px, boxes (n,5) x1,y1,x2,y2,conf).
    `model` is an Ultralytics YOLO model, or any ai.pose_backend backend with a detect() method.
    """
    if hasattr(model, "detect"):
        return model.detect(bgr)
    result = model(bgr, verbose=False, conf=DET_CONF, imgsz=YOLO_IMGSZ)[0]
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, NUM_KEYPOINTS, 3), np.float32), np.zeros((0, 5), np.float32)
    kps = result.keypoints.data.cpu().numpy().astype(np.float32)
    boxes = np.concatenate(
        [result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()[:, None]], axis=1
    ).astype(np.float32)
    return kps, boxes


def to_height_units(kps_px, frame_height):
    """Pixel keypoints -> frame-height units (x and y both divided by the frame height)."""
    out = np.array(kps_px, dtype=np.float32, copy=True)
    out[..., :2] /= float(frame_height)
    return out


def _iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(box) + area(boxes) - inter + 1e-9)


class SubjectSelector:
    """
    Picks one subject per frame: the detection that overlaps the previous
    choice (IoU >= min_iou within max_gap_s), otherwise the box with the
    largest area x confidence. Single-subject policy; multi-person tracking
    would replace this class.
    """

    def __init__(self, max_gap_s: float = 1.0, min_iou: float = 0.3):
        self.max_gap_s = max_gap_s
        self.min_iou = min_iou
        self.reset()

    def reset(self):
        self.prev_box = None
        self.prev_t = None

    def select(self, boxes, t):
        """Return the index of the chosen detection in `boxes` (n,5), or None."""
        if len(boxes) == 0:
            return None
        idx = None
        if self.prev_box is not None and t - self.prev_t <= self.max_gap_s:
            ious = _iou(self.prev_box, boxes[:, :4])
            if ious.max() >= self.min_iou:
                idx = int(ious.argmax())
        if idx is None:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx = int(np.argmax(area * boxes[:, 4]))
        self.prev_box = boxes[idx, :4].copy()
        self.prev_t = t
        return idx


def resample_window(times, kps, t_end, window_s=WINDOW_S, steps=SEQ_LEN):
    """
    Linearly interpolate timestamped keypoints (n,17,3) onto `steps` evenly
    spaced instants ending at t_end. Instants outside the observed range hold
    the nearest observation. This makes the model independent of the capture
    frame rate (browser ~5 FPS, worker 5-15 FPS, training 15 Hz).
    """
    times = np.asarray(times, dtype=np.float64)
    kps = np.asarray(kps, dtype=np.float32)
    grid = t_end - window_s + window_s * (np.arange(1, steps + 1) / steps)
    flat = kps.reshape(len(kps), -1)
    if len(times) == 1:
        return np.repeat(kps, steps, axis=0)
    i1 = np.clip(np.searchsorted(times, grid, side="right"), 1, len(times) - 1)
    i0 = i1 - 1
    t0, t1 = times[i0], times[i1]
    w = np.clip(np.where(t1 > t0, (grid - t0) / np.maximum(t1 - t0, 1e-9), 0.0), 0.0, 1.0)
    out = flat[i0] * (1.0 - w)[:, None] + flat[i1] * w[:, None]
    return out.reshape(steps, *kps.shape[1:]).astype(np.float32)


def _weighted_point(xy, conf, idx):
    w = conf[:, idx] + 1e-3
    return (xy[:, idx] * w[..., None]).sum(axis=1) / w.sum(axis=1)[:, None]


FEATURE_NAMES = (
    [f"rel_{axis}_{k}" for k in range(NUM_KEYPOINTS) for axis in ("x", "y")]
    + ["hip_dx", "hip_dy", "head_dx", "head_dy"]
    + ["hip_vx", "hip_vy", "head_vy"]
    + ["torso_sin", "torso_cos"]
    + ["bbox_log_aspect", "bbox_height"]
    + [f"conf_{k}" for k in range(NUM_KEYPOINTS)]
)
NUM_FEATURES = len(FEATURE_NAMES)  # 62


def window_features(win):
    """
    (SEQ_LEN, 17, 3) keypoints in frame-height units -> (SEQ_LEN, NUM_FEATURES).

    Units: positions and displacements are in body-size units (median
    keypoint-box diagonal over the window), velocities in body-sizes per
    second, the torso angle is measured from image vertical.
    """
    xy = win[..., :2].astype(np.float32)
    conf = win[..., 2].astype(np.float32)
    steps = len(win)

    hip = _weighted_point(xy, conf, HIPS)
    shoulder = _weighted_point(xy, conf, SHOULDERS)
    head = _weighted_point(xy, conf, HEAD)

    # Body size: keypoint bounding box over confident joints (all joints if < 3 are confident).
    vis = conf >= VISIBLE_CONF
    enough = vis.sum(axis=1) >= 3
    mask = np.where(enough[:, None], vis, True)
    x, y = xy[..., 0], xy[..., 1]
    bw = np.max(np.where(mask, x, -np.inf), axis=1) - np.min(np.where(mask, x, np.inf), axis=1)
    bh = np.max(np.where(mask, y, -np.inf), axis=1) - np.min(np.where(mask, y, np.inf), axis=1)
    scale = max(float(np.median(np.hypot(bw, bh))), 0.05)

    rel = (xy - hip[:, None, :]) / scale
    hip_traj = (hip - hip[0]) / scale
    head_traj = (head - head[0]) / scale

    # Velocities are clipped: real falls stay within ~+-5 body-sizes/s, larger
    # values come from keypoint jitter (measured on GMDCSA-24, p99 ~ 4.6).
    dt = WINDOW_S / steps
    hip_v = np.clip(np.gradient(hip, axis=0) / dt / scale, -VELOCITY_CLIP, VELOCITY_CLIP)
    head_vy = np.clip(np.gradient(head[:, 1]) / dt / scale, -VELOCITY_CLIP, VELOCITY_CLIP)

    torso = shoulder - hip
    angle = np.arctan2(torso[:, 0], -torso[:, 1])  # 0 = upright, +-pi/2 = horizontal

    feats = np.concatenate(
        [
            rel.reshape(steps, -1),
            hip_traj,
            head_traj,
            hip_v,
            head_vy[:, None],
            np.sin(angle)[:, None],
            np.cos(angle)[:, None],
            np.log((bw + 1e-3) / (bh + 1e-3))[:, None],
            (bh / scale)[:, None],
            conf,
        ],
        axis=1,
    )
    return feats.astype(np.float32)


LEGS = [13, 14, 15, 16]
HEAD_SHOULDERS = [0, 1, 2, 3, 4, 5, 6]


def partial_view(win, box_bottom, frame_height):
    """
    True for an upper-body view: the person is cut off by the bottom of the frame
    (sitting or standing close to a webcam), knees and ankles are not visible, the
    head and shoulders are. The classifier was trained on whole bodies and calls
    such views "fall", so they are reported as "unknown / partial view" instead.
    """
    conf = win[..., 2]
    return bool(
        box_bottom >= 0.97 * frame_height
        and np.median(conf[:, LEGS].mean(axis=1)) < 0.2
        and np.median(conf[:, HEAD_SHOULDERS].mean(axis=1)) >= 0.3
    )


def window_evidence(win):
    """Interpretable measurements of the newest frame in a window (for the incident page)."""
    xy, conf = win[..., :2], win[..., 2]
    hip = _weighted_point(xy, conf, HIPS)
    shoulder = _weighted_point(xy, conf, SHOULDERS)
    torso = shoulder[-1] - hip[-1]
    tilt = float(np.degrees(np.arctan2(abs(torso[0]), -torso[1])))
    dt = WINDOW_S / len(win)
    hip_vy = float(np.gradient(hip[:, 1])[-1] / dt)
    drop = float(hip[-1, 1] - hip[:, 1].min())
    return {
        "torso_tilt_from_vertical_deg": round(tilt, 1),
        "hip_vertical_velocity_frame_heights_per_s": round(hip_vy, 3),
        "hip_drop_in_window_frame_heights": round(drop, 3),
        "mean_keypoint_confidence": round(float(conf[-1].mean()), 3),
    }
