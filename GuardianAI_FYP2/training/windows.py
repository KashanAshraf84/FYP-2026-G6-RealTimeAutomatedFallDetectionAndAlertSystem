"""
Training Windows
================
Cached YOLO poses + GMDCSA-24 annotations -> labelled 2-second windows.

Window label (from the annotated fall interval, seconds of overlap with the
window [t_end - span, t_end]):
    fall    : overlap >= 50% of the window   (person is falling / has fallen)
    warning : 0.25 s (scaled) <= overlap < 50%   (fall onset, descent starting)
    normal  : otherwise (every ADL window and the pre-fall part of fall clips)

Augmentation is applied to RAW keypoints before features and normalisation
(mirroring, camera roll, aspect jitter, keypoint noise, occlusion, lower
capture FPS, dropped detections, playback-speed changes).
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from ai.fallnet import LABELS
from ai.pose_features import (
    FLIP_PAIRS,
    HIPS,
    MIN_HISTORY_S,
    WINDOW_S,
    SubjectSelector,
    resample_window,
    to_height_units,
    window_features,
)
from training.extract_keypoints import CACHE_DIR
from training.gmdcsa import Clip

FALL_FRACTION = 0.5
WARNING_MIN_S = 0.25
WINDOW_STRIDE = 3  # every 3rd pose sample at 15 Hz = 0.2 s


@dataclass
class Track:
    """The selected subject's poses for one clip."""
    clip: Clip
    times: np.ndarray       # (n,) seconds, frames where the subject was detected
    kps: np.ndarray         # (n, 17, 3) frame-height units
    duration: float
    sampled_frames: int


def load_track(clip: Clip) -> Track:
    with np.load(CACHE_DIR / f"{clip.clip_id}.npz") as d:
        times, det_frame, kps, boxes = d["times"], d["det_frame"], d["kps"], d["boxes"]
        height = int(d["height"])
        duration = float(d["src_frames"] / d["fps"])
    selector = SubjectSelector()
    sel_t, sel_k = [], []
    for f, t in enumerate(times):
        rows = np.flatnonzero(det_frame == f)
        idx = selector.select(boxes[rows], float(t))
        if idx is not None:
            sel_t.append(float(t))
            sel_k.append(to_height_units(kps[rows[idx]], height))
    return Track(
        clip=clip,
        times=np.asarray(sel_t, np.float64),
        kps=np.asarray(sel_k, np.float32).reshape(-1, 17, 3),
        duration=duration,
        sampled_frames=len(times),
    )


def fall_overlap(clip: Clip, start: float, end: float, duration: float) -> float:
    total = 0.0
    for a, b in clip.fall_intervals:
        b = duration if b is None else b
        total += max(0.0, min(end, b) - max(start, a))
    return total


def window_label(clip: Clip, t_end: float, span: float, duration: float) -> int:
    overlap = fall_overlap(clip, t_end - span, t_end, duration)
    if overlap >= FALL_FRACTION * span:
        return 2
    if overlap >= WARNING_MIN_S * span / WINDOW_S:
        return 1
    return 0


def window_index(tracks: List[Track], stride: int = WINDOW_STRIDE):
    """(track_idx, t_end, label) for every window with enough pose history."""
    index = []
    for ti, tr in enumerate(tracks):
        if len(tr.times) < 2:
            continue
        valid = np.flatnonzero(tr.times - tr.times[0] >= MIN_HISTORY_S)
        for i in valid[::stride]:
            t_end = float(tr.times[i])
            index.append((ti, t_end, window_label(tr.clip, t_end, WINDOW_S, tr.duration)))
    return index


# -----------------------------------------------------------------------------
# Augmentation on raw keypoints
# -----------------------------------------------------------------------------

def _mirror(win):
    out = win.copy()
    out[..., 0] = -out[..., 0]
    for a, b in FLIP_PAIRS:
        out[:, [a, b]] = out[:, [b, a]]
    return out


def _rotate(win, deg):
    out = win.copy()
    centre = win[:, HIPS, :2].mean(axis=(0, 1))
    th = np.radians(deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
    out[..., :2] = (win[..., :2] - centre) @ rot.T + centre
    return out


def augmented_window(tr: Track, t_end: float, rng: np.random.Generator):
    """Random raw-keypoint window around t_end -> (window, span_seconds, t_end)."""
    lo = tr.times[0] + MIN_HISTORY_S
    t_end = float(np.clip(t_end + rng.uniform(-0.1, 0.1), lo, tr.times[-1]))
    # Snap to the newest real sample, as live inference does.
    t_end = float(tr.times[np.searchsorted(tr.times, t_end, side="right") - 1])

    span = WINDOW_S * (rng.uniform(0.8, 1.25) if rng.random() < 0.3 else 1.0)
    m = (tr.times >= t_end - span - 1.0) & (tr.times <= t_end)
    ts, ks = tr.times[m], tr.kps[m]

    if rng.random() < 0.6 and len(ts) > 2:  # lower capture FPS (browser ~5 FPS)
        fps = rng.uniform(4.0, 15.0)
        bins = np.floor((ts - ts[0] + rng.uniform(0, 1 / fps)) * fps)
        keep = np.r_[True, bins[1:] != bins[:-1]]
        keep[-1] = True
        ts, ks = ts[keep], ks[keep]
    if rng.random() < 0.2 and len(ts) > 4:  # missed detections
        keep = rng.random(len(ts)) > rng.uniform(0.1, 0.25)
        keep[-1] = True
        ts, ks = ts[keep], ks[keep]

    win = resample_window(ts, ks, t_end, window_s=span)

    if rng.random() < 0.5:
        win = _mirror(win)
    if rng.random() < 0.3:
        win = _rotate(win, rng.uniform(-10, 10))
    if rng.random() < 0.3:
        win[..., 0] *= rng.uniform(0.9, 1.1)
    if rng.random() < 0.5:
        win[..., :2] += rng.normal(0, rng.uniform(0.002, 0.01), win[..., :2].shape).astype(np.float32)
    if rng.random() < 0.3:  # occlusion: some joints unreliable for the whole window
        joints = rng.choice(17, size=int(rng.integers(1, 4)), replace=False)
        win[:, joints, 2] *= rng.uniform(0.0, 0.3)
        win[:, joints, :2] += rng.normal(0, 0.03, (len(win), len(joints), 2)).astype(np.float32)
    return win, span, t_end


class WindowDataset(Dataset):
    """Feature windows for a set of tracks; random augmentation when `augment`."""

    def __init__(self, tracks, index, mean, std, augment=False, seed=0):
        self.tracks = tracks
        self.index = index
        self.mean = mean
        self.std = std
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ti, t_end, label = self.index[i]
        tr = self.tracks[ti]
        if self.augment:
            win, span, t_end = augmented_window(tr, t_end, self.rng)
            label = window_label(tr.clip, t_end, span, tr.duration)
        else:
            win = resample_window(tr.times, tr.kps, t_end)
        x = (window_features(win) - self.mean) / self.std
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(label, dtype=torch.long)


def raw_features(tracks, index):
    """Un-augmented feature windows (N, SEQ_LEN, NUM_FEATURES) and labels (N,)."""
    feats = np.stack([window_features(resample_window(tracks[ti].times, tracks[ti].kps, t)) for ti, t, _ in index])
    labels = np.array([lbl for _, _, lbl in index], dtype=np.int64)
    return feats.astype(np.float32), labels


def label_counts(index):
    counts = np.bincount([lbl for _, _, lbl in index], minlength=len(LABELS))
    return {LABELS[i]: int(c) for i, c in enumerate(counts)}
