import json, math, os, threading, time
from dataclasses import asdict
from django.conf import settings
from django.utils.module_loading import import_string
from .contracts import Result

_lock = threading.Lock()
_detector = None
_path = None
_last = {}
_cost = None  # smoothed seconds of model time per frame

# Pacing: every live camera is told how long to wait before its next frame so that all
# cameras together keep the model busy at most DUTY of the time. One camera gets ~6
# frames/s; more cameras share the budget fairly instead of saturating the machine.
# Fall detection weakens below ~5 frames/s per camera (replay of GMDCSA-24: 78 falls
# caught at 5 FPS, 73 at 4 FPS, 70 at 3 FPS), so the budget is not set lower than needed.
DUTY = float(os.environ.get("GUARDIAN_AI_DUTY", "0.65"))
MIN_INTERVAL_MS, MAX_INTERVAL_MS = 160, 500
ACTIVE_WINDOW_S = 2.5


def _get_detector():
    global _detector, _path
    if _detector is None or _path != settings.DETECTOR_CLASS:
        _detector = import_string(settings.DETECTOR_CLASS)()
        _path = settings.DETECTOR_CLASS
        _last.clear()
    return _detector


def warm_up():
    """Load the detector before the first frame arrives (called in a thread at server start)."""
    with _lock:
        _get_detector()


def predict(frame, camera_id):
    global _cost
    with _lock:
        detector = _get_detector()
        now = time.time()
        if now - _last.get(camera_id, 0) > 3:
            detector.reset(camera_id)
        started = time.perf_counter()
        result = detector.predict(frame, camera_id=camera_id, timestamp=now)
        spent = time.perf_counter() - started
        _cost = spent if _cost is None else 0.8 * _cost + 0.2 * spent
        _last[camera_id] = now
        validate(result)
        return result


def next_frame_ms():
    """Milliseconds a camera should wait before sending its next frame (see DUTY above)."""
    now = time.time()
    active = sum(1 for seen in _last.values() if now - seen < ACTIVE_WINDOW_S) or 1
    wait = active * (_cost or 0.05) * 1000 / DUTY
    return int(min(MAX_INTERVAL_MS, max(MIN_INTERVAL_MS, wait)))


def validate(result):
    if not isinstance(result, Result) or type(result.ready) is not bool:
        raise ValueError("Detector must return Result with a boolean ready value")
    if (
        not isinstance(result.model_version, str)
        or not 1 <= len(result.model_version) <= 100
    ):
        raise ValueError("model_version must be 1-100 characters")
    if not isinstance(result.message, str) or len(result.message) > 500:
        raise ValueError("message must be at most 500 characters")
    if len(result.predictions) > 30:
        raise ValueError("Maximum 30 persons per frame")
    seen = set()
    for p in result.predictions:
        if (
            not isinstance(p.track_id, str)
            or not 1 <= len(p.track_id) <= 80
            or p.track_id in seen
        ):
            raise ValueError("Unique stable track IDs of 1-80 characters required")
        seen.add(p.track_id)
        if p.label not in {"normal", "warning", "fall", "unknown"}:
            raise ValueError("Invalid label")
        if (
            isinstance(p.confidence, bool)
            or not isinstance(p.confidence, (float, int))
            or not math.isfinite(p.confidence)
            or not 0 <= p.confidence <= 1
        ):
            raise ValueError("Confidence must be finite and in [0,1]")
        if p.bbox is not None:
            if len(p.bbox) != 4 or any(
                not isinstance(x, (int, float))
                or not math.isfinite(x)
                or not 0 <= x <= 1
                for x in p.bbox
            ):
                raise ValueError("Invalid normalized bounding box")
            if p.bbox[0] >= p.bbox[2] or p.bbox[1] >= p.bbox[3]:
                raise ValueError("Inverted bounding box")
        if not isinstance(p.evidence, dict):
            raise ValueError("Evidence must be an object")
    if len(json.dumps(asdict(result), allow_nan=False)) > 32000:
        raise ValueError("Result exceeds 32 KB")
