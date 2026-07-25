"""
GuardianAI - Cloud Demonstration Server (Hugging Face Spaces)
==============================================================
Runs the identical detection pipeline as the local edge application, adapted
for a headless cloud container:

  * Frames arrive from the VISITOR'S browser (getUserMedia) as base64 JPEG,
    rather than being read from a webcam attached to the server. A cloud
    container has no camera, so the capture stage moves to the client.

  * Desktop alert channels (buzzer, OpenCV popup, OS notification) are
    disabled via the `headless` configuration profile. The browser performs
    audible and visual alerting instead. Console output and JSON event
    logging are unaffected.

  * Detection state is isolated per visitor. Several people may open the demo
    at once, and a shared temporal buffer would let one visitor's frames
    corrupt another's 30-frame sequence.

The detection logic itself - pose estimation, feature engineering, rule
engine, CNN-LSTM inference, fusion and temporal smoothing - is unmodified and
imported directly from the same modules the local application uses.
"""

import os
import base64
import time
import threading

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

from config import SystemConfig
from inference import FallDetectionInference


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

PORT = int(os.environ.get("PORT", 7860))
LOG_DIR = os.environ.get("GUARDIANAI_LOG_DIR", "/tmp/logs")

# Each session holds its own detector (~200-300 MB). Cap concurrency so the
# free-tier container cannot be exhausted, and evict idle sessions.
MAX_SESSIONS = int(os.environ.get("GUARDIANAI_MAX_SESSIONS", 6))
SESSION_TTL_SECONDS = 180

app = Flask(__name__, static_folder="static")

_sessions = {}          # session_id -> {"detector": ..., "last_seen": float}
_sessions_lock = threading.Lock()


def _build_config() -> SystemConfig:
    """Configuration for the headless cloud profile."""
    config = SystemConfig()
    config.alert.headless = True        # no display, no audio device
    config.alert.log_dir = LOG_DIR
    config.alert.enable_email = False
    return config


def _get_detector(session_id: str):
    """Return this visitor's detector, creating one if capacity allows.

    Returns (detector, error_message). A non-None error means the demo is at
    capacity and the caller should surface that to the visitor.
    """
    now = time.time()

    with _sessions_lock:
        # Evict idle sessions first so capacity is reclaimed automatically.
        stale = [
            sid for sid, s in _sessions.items()
            if now - s["last_seen"] > SESSION_TTL_SECONDS
        ]
        for sid in stale:
            try:
                _sessions[sid]["detector"].release()
            except Exception:
                pass
            del _sessions[sid]
            print(f"[session] evicted idle session {sid[:8]}")

        session = _sessions.get(session_id)
        if session is not None:
            session["last_seen"] = now
            return session["detector"], None

        if len(_sessions) >= MAX_SESSIONS:
            return None, (
                "The demo is at capacity right now "
                f"({MAX_SESSIONS} concurrent visitors). Please try again shortly."
            )

        print(f"[session] creating detector for session {session_id[:8]}")
        detector = FallDetectionInference(config=_build_config())
        _sessions[session_id] = {"detector": detector, "last_seen": now}
        return detector, None


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


@app.route("/api/health", methods=["GET"])
def health():
    with _sessions_lock:
        active = len(_sessions)
    return jsonify({
        "status": "healthy",
        "active_sessions": active,
        "max_sessions": MAX_SESSIONS,
        "mode": "cloud-browser-camera",
    })


@app.route("/api/frame", methods=["POST"])
def process_frame():
    """Run one browser-captured frame through the detection pipeline."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session")
    image_b64 = data.get("image")

    if not session_id:
        return jsonify({"error": "Missing session identifier"}), 400
    if not image_b64:
        return jsonify({"error": "Missing image data"}), 400

    detector, capacity_error = _get_detector(session_id)
    if capacity_error:
        return jsonify({"error": capacity_error, "at_capacity": True}), 503

    try:
        # Strip an optional "data:image/jpeg;base64," prefix.
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        buffer = np.frombuffer(base64.b64decode(image_b64), np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        started = time.time()
        result = detector.process_frame(frame)
        elapsed_ms = (time.time() - started) * 1000

        ok, encoded = cv2.imencode(
            ".jpg", result["frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        )
        if not ok:
            return jsonify({"error": "Could not encode annotated frame"}), 500

        summary = detector.get_detection_result()
        events = detector.alert_system.get_event_summary()

        return jsonify({
            "status": result["status"],
            "confidence": result["confidence"],
            "person_count": result["person_count"],
            "angle": summary.get("angle", 90.0),
            "speed": summary.get("speed", 0.0),
            "falls": events.get("falls", 0),
            "warnings": events.get("warnings", 0),
            "processing_time_ms": round(elapsed_ms, 1),
            "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
        })

    except Exception as exc:
        print(f"[error] frame processing failed: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    """Clear one visitor's temporal buffers without affecting other visitors."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session")

    with _sessions_lock:
        session = _sessions.get(session_id)

    if session is None:
        return jsonify({"message": "No active session to reset"})

    detector = session["detector"]
    detector.feature_extractor.reset()
    detector._status_history.clear()
    return jsonify({"message": "Tracking state reset"})


if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    print("=" * 60)
    print("  GuardianAI - Cloud Demonstration Server")
    print("  Detection pipeline: identical to the local edge application")
    print("  Camera source: visitor's browser (getUserMedia)")
    print("  Desktop alert channels: disabled (headless profile)")
    print("=" * 60)
    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)
