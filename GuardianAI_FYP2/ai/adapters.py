"""
AI Adapters for GuardianAI
===========================
- UnconfiguredDetector  : honest default stub; camera preview works, no model loaded.
- TeamDetector          : FYP-2 model. YOLOv8-Pose -> multi-person tracker (IDs P1, P2, ...)
                          -> FallNet v2 per person (weights/fallnet_v2.pt, training/train_fallnet.py).
- LegacyFYP1Detector    : FYP1's FallDetectionInference (rules + old CNN-LSTM), kept for comparison.

To activate the real model, set the environment variable:
    GUARDIAN_DETECTOR=ai.adapters.TeamDetector
"""

import numpy as np
from .contracts import Result, Prediction
from .pose_features import WINDOW_S, partial_view, run_pose, window_evidence


class UnconfiguredDetector:
    """Honest default: camera preview works, but no fall classifier is loaded."""

    def predict(self, frame, *, camera_id, timestamp):
        return Result(
            [],
            "unconfigured",
            ready=False,
            message="No trained model connected. Preview only.",
        )

    def reset(self, camera_id):
        pass


class TeamDetector:
    """
    FYP-2 fall-detection adapter.

    Per frame: YOLOv8-Pose finds every person, ai.tracking.PersonTracker gives
    each one a stable ID (P1, P2, ...) and their own pose history, and
    FallNet v2 classifies each person's last 2 seconds of motion (resampled by
    timestamp, so the browser's ~5 FPS and the worker's 5-15 FPS both match
    training). One Prediction is returned per person; the application's
    confidence gate, consecutive-frame confirmation and cooldown are applied
    per person afterwards by monitor/services.ingest.
    """

    MAX_PEOPLE = 30  # ai.runtime.validate limit

    def __init__(self):
        # Import here so Django can start without torch/ultralytics when unused.
        from .fallnet import DEFAULT_CHECKPOINT, LABELS, FallNet
        from .pose_backend import create_pose_backend

        import torch

        torch.set_num_threads(2)  # the classifier is tiny; more threads only steal CPU from the UI
        self._labels = LABELS
        self._net = FallNet(DEFAULT_CHECKPOINT)
        # Fastest runtime on this machine (DirectML / OpenVINO GPU / CPU), see ai/pose_backend.py.
        self._pose = create_pose_backend()
        self.pose_backend = getattr(self._pose, "label", "PyTorch")
        self._trackers = {}
        self._id_counters = {}  # survive reset() so a camera never reuses an ID within a run
        self.model_version = self._net.model_version

    def _tracker(self, camera_id):
        from itertools import count
        from .tracking import PersonTracker

        if camera_id not in self._trackers:
            counter = self._id_counters.setdefault(camera_id, count(1))
            self._trackers[camera_id] = PersonTracker(next_id=lambda: f"P{next(counter)}")
        return self._trackers[camera_id]

    def predict(self, frame, *, camera_id, timestamp):
        kps, boxes = run_pose(self._pose, _pil_to_bgr(frame))
        tracker = self._tracker(camera_id)
        tracker.update(timestamp, kps, boxes, frame.height)
        people = tracker.active(timestamp)[: self.MAX_PEOPLE]
        if not people:
            return Result([], self.model_version, message="No person detected.")

        windows = [person.history.window(timestamp) for person in people]
        partial = {i for i, w in enumerate(windows) if w is not None and partial_view(w, people[i].box[3], frame.height)}
        ready = [i for i, w in enumerate(windows) if w is not None and i not in partial]
        probs = dict(zip(ready, self._net.predict_windows(np.stack([windows[i] for i in ready])))) if ready else {}

        predictions = []
        for i, person in enumerate(people):
            bbox = _normalized_box(person.box, frame.width, frame.height)
            if i in partial:
                predictions.append(Prediction(
                    track_id=person.track_id,
                    label="unknown",
                    confidence=0.0,
                    bbox=bbox,
                    evidence={"status": "partial_view",
                              "note": "Only head and shoulders are visible. Move back so hips and legs are in view."},
                ))
                continue
            if i not in probs:
                predictions.append(Prediction(
                    track_id=person.track_id,
                    label="unknown",
                    confidence=0.0,
                    bbox=bbox,
                    evidence={"status": "warming_up", "history_needed_s": 1.5},
                ))
                continue
            p = probs[i]
            best = int(np.argmax(p))
            predictions.append(Prediction(
                track_id=person.track_id,
                label=self._labels[best],
                confidence=float(min(max(p[best], 0.0), 1.0)),
                bbox=bbox,
                evidence={
                    **window_evidence(windows[i]),
                    "class_probabilities": {name: round(float(v), 4) for name, v in zip(self._labels, p)},
                    "window_s": WINDOW_S,
                    "pose_age_s": round(float(timestamp - person.last_t), 2),
                    "tracked_for_s": round(float(timestamp - person.first_t), 1),
                    "classifier": self.model_version,
                },
            ))
        if partial and len(partial) == len(predictions):
            message = "Body not fully visible: move back so hips and legs are in view."
        elif all(p.label == "unknown" for p in predictions):
            message = "Collecting motion history."
        else:
            message = f"{len(predictions)} person(s) tracked · {self.pose_backend}"
        return Result(predictions, self.model_version, message=message[:500])

    def reset(self, camera_id):
        """Called by ai.runtime after a >3 s gap in this camera's frames."""
        self._trackers.pop(camera_id, None)


class LegacyFYP1Detector:
    """
    FYP1 fall-detection adapter (kept for comparison).

    Wraps FYP1's FallDetectionInference (YOLOv8-Pose + bidirectional CNN-LSTM)
    and maps its single-person output to the Guardian Result / Prediction contract.

    Frame lifecycle
    ---------------
    Guardian calls predict() with a PIL RGB Image.  We convert it to an OpenCV
    BGR ndarray (what FYP1 expects), run process_frame(), and build a single
    Prediction from the result.  A stable track_id of "person-1" is used because
    FYP1 is a single-person pipeline.

    Thread safety
    -------------
    ai.runtime serialises calls to this detector per-process, so no extra locking
    is needed here.
    """

    MODEL_VERSION = "fyp1-cnn-lstm-v1"

    def __init__(self):
        # Import here so Django can start without torch/ultralytics on the path
        # when running with UnconfiguredDetector.
        from fyp1_config import SystemConfig
        from inference import FallDetectionInference

        # Build a headless config — no GUI popups or sound in the server process
        cfg = SystemConfig()
        cfg.alert.headless = True
        cfg.alert.enable_sound = False
        cfg.alert.enable_email = False

        self._engine = FallDetectionInference(config=cfg)

    # ------------------------------------------------------------------
    # Detector protocol
    # ------------------------------------------------------------------

    def predict(self, frame, *, camera_id, timestamp):
        """
        Run fall detection on a single PIL RGB frame.

        Parameters
        ----------
        frame : PIL.Image.Image  (RGB, up to 1920x1080)
        camera_id : int
        timestamp : float  (unix seconds)

        Returns
        -------
        ai.contracts.Result
        """
        # PIL RGB  ->  NumPy HWC BGR  (what OpenCV / FYP1 expects)
        bgr = _pil_to_bgr(frame)

        result = self._engine.process_frame(bgr)

        status = result["status"]          # "normal" | "warning" | "fall"
        confidence = result["confidence"]  # 0-100 from FYP1

        # Guardian contract: confidence must be in [0, 1]
        conf_norm = min(max(confidence / 100.0, 0.0), 1.0)

        person_count = result.get("person_count", 0)

        # Build bounding box if a person was detected
        bbox = None
        if person_count > 0:
            bbox = _bbox_from_engine(self._engine, frame.width, frame.height)

        # Evidence dictionary — biomechanical telemetry for the Event detail page
        evidence = {
            "angle_deg": result.get("angle", None),
            "drop_speed_px_per_frame": result.get("speed", None),
            "person_count": person_count,
            "source": "fyp1-cnn-lstm",
        }

        # Only emit a prediction when a person is visible or non-normal status
        predictions = []
        if person_count > 0 or status != "normal":
            label = status if status in {"normal", "warning", "fall"} else "unknown"
            predictions = [
                Prediction(
                    track_id="person-1",
                    label=label,
                    confidence=conf_norm,
                    bbox=bbox,
                    evidence=evidence,
                )
            ]

        return Result(
            predictions=predictions,
            model_version=self.MODEL_VERSION,
            ready=True,
            message="",
        )

    def reset(self, camera_id):
        """Clear temporal state when the camera feed has been idle for > 3 s."""
        self._engine.feature_extractor.reset()
        self._engine._status_history.clear()
        self._engine._last_logged_status = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_bgr(pil_image) -> np.ndarray:
    """Convert a PIL RGB Image to an OpenCV-compatible BGR NumPy array."""
    rgb = np.array(pil_image, dtype=np.uint8)
    return rgb[:, :, ::-1].copy()


def _normalized_box(box, img_w: int, img_h: int):
    """Pixel (x1, y1, x2, y2[, conf]) -> normalized box in [0, 1], or None if degenerate."""
    if box is None:
        return None
    x1, x2 = (min(max(float(v) / img_w, 0.0), 1.0) for v in (box[0], box[2]))
    y1, y2 = (min(max(float(v) / img_h, 0.0), 1.0) for v in (box[1], box[3]))
    if x1 >= x2 or y1 >= y2:
        return None
    return (x1, y1, x2, y2)


def _bbox_from_engine(engine, img_w: int, img_h: int):
    """
    Derive a normalized bounding box from the pose keypoints held in the
    FYP1 feature extractor's keypoint buffer.

    Returns a (x1, y1, x2, y2) tuple with values in [0, 1], or None.
    """
    try:
        buf = engine.feature_extractor.keypoint_buffer
        if not buf:
            return None
        # Last frame's combined feature vector; first 39 values are raw keypoints
        # (13 keypoints x 3 values: x_norm, y_norm, confidence)
        last = np.array(buf[-1][:39]).reshape(13, 3)
        visible = last[last[:, 2] > 0.4]
        if len(visible) < 2:
            return None
        x1 = float(visible[:, 0].min())
        y1 = float(visible[:, 1].min())
        x2 = float(visible[:, 0].max())
        y2 = float(visible[:, 1].max())
        x1, y1, x2, y2 = (
            max(0.0, min(1.0, x1)),
            max(0.0, min(1.0, y1)),
            max(0.0, min(1.0, x2)),
            max(0.0, min(1.0, y2)),
        )
        if x1 >= x2 or y1 >= y2:
            return None
        return (x1, y1, x2, y2)
    except Exception:
        return None
