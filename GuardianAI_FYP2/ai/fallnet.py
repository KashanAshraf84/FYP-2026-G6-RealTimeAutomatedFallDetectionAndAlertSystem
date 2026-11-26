"""
FallNet v2 Runtime
==================
Loads a checkpoint written by training/train_fallnet.py and turns a stream of
timestamped pose detections into normal / warning / fall probabilities.

Used by:
  - ai.adapters.TeamDetector (live Django inference, YOLO frames -> Result)
  - training/train_fallnet.py (offline replay of cached poses at 5 / 15 FPS,
    so the reported clip-level metrics exercise this exact code path)
"""

from collections import deque
from pathlib import Path

import numpy as np
import torch

from ai.pose_features import (
    FEATURE_VERSION,
    MIN_HISTORY_S,
    NUM_FEATURES,
    WINDOW_S,
    SubjectSelector,
    resample_window,
    to_height_units,
    window_features,
)

LABELS = ["normal", "warning", "fall"]
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "weights" / "fallnet_v2.pt"


def build_model(model_config: dict):
    """Instantiate an architecture from fall_detector_model.py with a stored config."""
    from fyp1_config import ModelConfig
    from fall_detector_model import create_model

    return create_model(ModelConfig(**model_config))


class FallNet:
    """Temporal pose classifier plus the normalisation it was trained with."""

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, device="cpu"):
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        if ckpt.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"Checkpoint feature version {ckpt.get('feature_version')!r} does not match "
                f"ai.pose_features {FEATURE_VERSION!r}; retrain or use the matching code."
            )
        self.device = torch.device(device)
        self.model = build_model(ckpt["model_config"]).to(self.device).eval()
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.mean = ckpt["norm_mean"].float().numpy()
        self.std = ckpt["norm_std"].float().numpy()
        # Temperature scaling fitted on validation data so scores match the app's confidence gate.
        self.temperature = float(ckpt.get("temperature", 1.0))
        self.model_version = ckpt["model_version"]
        self.meta = {k: v for k, v in ckpt.items() if k not in ("model_state_dict", "norm_mean", "norm_std")}

    @classmethod
    def from_parts(cls, model, mean, std, temperature=1.0, model_version="in-training"):
        """Wrap an in-memory model (used during cross-validation)."""
        self = cls.__new__(cls)
        self.device = next(model.parameters()).device
        self.model = model.eval()
        self.mean, self.std = mean, std
        self.temperature = float(temperature)
        self.model_version = model_version
        self.meta = {}
        return self

    @torch.no_grad()
    def predict_windows(self, windows):
        """(B, SEQ_LEN, 17, 3) keypoint windows -> (B, 3) class probabilities."""
        feats = np.stack([window_features(w) for w in windows])
        return self.predict_features(feats)

    @torch.no_grad()
    def predict_features(self, feats):
        """(B, SEQ_LEN, NUM_FEATURES) raw features -> (B, 3) class probabilities."""
        assert feats.shape[-1] == NUM_FEATURES
        x = torch.from_numpy(((feats - self.mean) / self.std).astype(np.float32)).to(self.device)
        return torch.softmax(self.model(x) / self.temperature, dim=1).cpu().numpy()


class PoseHistory:
    """
    One person's recent timestamped poses (frame-height units). If the person
    is briefly not detected (e.g. lying behind a bed), the last window is
    reused for up to HOLD_S seconds.
    """

    HOLD_S = 1.0
    KEEP_S = WINDOW_S + 1.0

    def __init__(self):
        self.times = deque()
        self.poses = deque()

    def clear(self):
        self.times.clear()
        self.poses.clear()

    def add(self, t, pose):
        self.times.append(float(t))
        self.poses.append(pose)
        while self.times and self.times[0] < t - self.KEEP_S:
            self.times.popleft()
            self.poses.popleft()

    def window(self, now):
        """Keypoint window ending at the newest pose, or None if not enough history."""
        if not self.times or now - self.times[-1] > self.HOLD_S:
            return None
        if self.times[-1] - self.times[0] < MIN_HISTORY_S:
            return None
        return resample_window(np.array(self.times), np.stack(self.poses), self.times[-1])

    @property
    def warming_up(self):
        return bool(self.times) and self.times[-1] - self.times[0] < MIN_HISTORY_S


class SubjectStream:
    """
    Single-subject temporal state (the policy the model was trained and
    evaluated with): SubjectSelector picks the main person, PoseHistory keeps
    their poses. The live adapter uses ai.tracking.PersonTracker instead,
    which keeps one PoseHistory per tracked person.
    """

    def __init__(self):
        self.selector = SubjectSelector()
        self.history = PoseHistory()
        self.last_box = None

    def reset(self):
        self.selector.reset()
        self.history.clear()
        self.last_box = None

    def push(self, t, kps_px, boxes, frame_height):
        """Add one frame's detections; returns the chosen detection index or None."""
        idx = self.selector.select(boxes, t)
        if idx is not None:
            self.history.add(t, to_height_units(kps_px[idx], frame_height))
            self.last_box = boxes[idx]
        return idx

    def window(self, now):
        return self.history.window(now)

    @property
    def warming_up(self):
        return self.history.warming_up
