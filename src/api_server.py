"""
Fall Detection System - Flask REST API for Cloud Deployment
============================================================
Provides REST endpoints for fall detection inference,
suitable for deployment on cloud platforms.
"""

import os
import io
import base64
import time
import numpy as np
import cv2
from flask import Flask, request, jsonify, Response, send_from_directory
from typing import Optional

from config import SystemConfig
from inference import FallDetectionInference


app = Flask(__name__)

# Global detector instance
detector: Optional[FallDetectionInference] = None

# Tracks whether the last read from the camera succeeded, so /status can
# report real camera health instead of assuming it's always connected.
camera_state = {"connected": False}


def get_detector() -> FallDetectionInference:
    """Lazy initialization of the detector."""
    global detector
    if detector is None:
        config = SystemConfig()
        detector = FallDetectionInference(config=config)
    return detector


@app.route("/")
def index():
    """Serve the dashboard UI."""
    return send_from_directory("static", "index.html")


@app.route("/static/<path:path>")
def send_static(path):
    """Serve static assets."""
    return send_from_directory("static", path)


def generate_frames():
    """Video streaming generator function.

    Reconnects the camera on read failure instead of ending the stream
    permanently, and always releases the capture handle on exit (including
    when the client disconnects) so a paused/resumed feed can reopen the
    device cleanly.
    """
    det = get_detector()
    source = det.config.camera.source
    cam_cfg = det.config.camera

    def _open_camera():
        c = cv2.VideoCapture(source)
        # Force the configured capture resolution/fps and a 1-frame buffer.
        # Without this, the OS/driver default resolution can be far larger
        # than what the detection pipeline needs, and an unbounded internal
        # buffer means a frame that arrives faster than it can be processed
        # just queues up — the feed drifts further behind real time the
        # longer it runs, which is what "laggy" looks like.
        c.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.width)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.height)
        c.set(cv2.CAP_PROP_FPS, cam_cfg.fps)
        c.set(cv2.CAP_PROP_BUFFERSIZE, cam_cfg.buffer_size)
        return c

    cap = _open_camera()
    camera_state["connected"] = cap.isOpened()
    consecutive_failures = 0
    max_consecutive_failures = 15  # ~a couple seconds before a full reopen
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

    try:
        while True:
            success, frame = cap.read()

            if not success:
                consecutive_failures += 1
                camera_state["connected"] = False
                if consecutive_failures >= max_consecutive_failures:
                    cap.release()
                    time.sleep(1.0)
                    cap = _open_camera()
                    camera_state["connected"] = cap.isOpened()
                    consecutive_failures = 0
                else:
                    time.sleep(0.1)
                continue

            consecutive_failures = 0
            camera_state["connected"] = True

            # Process frame through detection pipeline
            result = det.process_frame(frame)
            annotated_frame = result["frame"]

            # Encode as JPEG (quality 80 trims transfer size with no visible
            # difference on a webcam feed, and cuts encode time per frame)
            ret, buffer = cv2.imencode('.jpg', annotated_frame, jpeg_params)
            frame_bytes = buffer.tobytes()

            # Yield the output frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()
        camera_state["connected"] = False


@app.route("/video_feed")
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": get_detector().model is not None,
        "timestamp": time.time(),
    })


@app.route("/detect", methods=["POST"])
def detect_fall():
    """
    Detect fall from an uploaded image/frame.

    Accepts:
        - multipart/form-data with 'image' file
        - JSON with 'image' as base64 encoded string

    Returns:
        {
            "status": "normal" / "warning" / "fall",
            "confidence": percentage,
            "keypoints_detected": bool,
            "processing_time_ms": float
        }
    """
    start_time = time.time()

    try:
        # Parse input
        if request.content_type and "multipart/form-data" in request.content_type:
            file = request.files.get("image")
            if file is None:
                return jsonify({"error": "No image file provided"}), 400
            img_bytes = file.read()
        elif request.is_json:
            data = request.get_json()
            img_b64 = data.get("image")
            if img_b64 is None:
                return jsonify({"error": "No image data provided"}), 400
            img_bytes = base64.b64decode(img_b64)
        else:
            return jsonify({"error": "Unsupported content type"}), 400

        # Decode image
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Run detection
        det = get_detector()
        result = det.process_frame(frame)

        processing_time = (time.time() - start_time) * 1000

        return jsonify({
            "status": result["status"],
            "confidence": result["confidence"],
            "keypoints_detected": result["keypoints_detected"],
            "processing_time_ms": round(processing_time, 2),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/detect/batch", methods=["POST"])
def detect_fall_batch():
    """
    Process multiple frames in sequence (simulating a video clip).

    Accepts JSON:
        {
            "frames": [base64_img_1, base64_img_2, ...],
        }

    Returns:
        {
            "results": [
                {"status": ..., "confidence": ...},
                ...
            ],
            "final_status": ...,
            "processing_time_ms": ...
        }
    """
    start_time = time.time()

    try:
        data = request.get_json()
        frames_b64 = data.get("frames", [])

        if not frames_b64:
            return jsonify({"error": "No frames provided"}), 400

        det = get_detector()
        results = []

        for img_b64 in frames_b64:
            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                result = det.process_frame(frame)
                results.append({
                    "status": result["status"],
                    "confidence": result["confidence"],
                })

        processing_time = (time.time() - start_time) * 1000
        final = det.get_detection_result()

        return jsonify({
            "results": results,
            "final_status": final["status"],
            "final_confidence": final["confidence"],
            "num_frames": len(results),
            "processing_time_ms": round(processing_time, 2),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/status", methods=["GET"])
def get_status():
    """Get current detection status."""
    det = get_detector()
    result = det.get_detection_result()
    summary = det.alert_system.get_event_summary()

    return jsonify({
        "current_status": result,
        "event_summary": summary,
        "camera_connected": camera_state["connected"],
        "alerts_muted": det.alert_system.is_muted(),
    })


@app.route("/alerts/mute", methods=["POST"])
def toggle_mute():
    """Mute/unmute the buzzer and popup alert. Events are still logged."""
    det = get_detector()
    data = request.get_json(silent=True) or {}
    muted = data.get("muted")
    if muted is None:
        muted = not det.alert_system.is_muted()
    det.alert_system.set_muted(bool(muted))
    return jsonify({"muted": det.alert_system.is_muted()})


@app.route("/reset", methods=["POST"])
def reset_tracking():
    """Reset tracking state."""
    det = get_detector()
    det.feature_extractor.reset()
    det._status_history.clear()
    det._last_logged_status = None
    return jsonify({"message": "Tracking state reset"})


@app.route("/events", methods=["GET"])
def get_events():
    """Recent detection_events rows (every normal/warning/fall status change)."""
    det = get_detector()
    limit = request.args.get("limit", default=50, type=int)
    return jsonify({"events": det.database.get_recent_events(limit=limit)})


@app.route("/alerts", methods=["GET"])
def get_alerts():
    """Recent alerts rows (real, cooldown-gated alerts that fired)."""
    det = get_detector()
    limit = request.args.get("limit", default=50, type=int)
    return jsonify({"alerts": det.database.get_recent_alerts(limit=limit)})


@app.route("/alerts/<int:alert_id>/acknowledge", methods=["POST"])
def acknowledge_alert(alert_id):
    """Mark an alert as acknowledged — the explicit user-action -> DB-write demo."""
    det = get_detector()
    found = det.database.acknowledge_alert(alert_id)
    if not found:
        return jsonify({"error": "No such alert"}), 404
    return jsonify({"message": "Alert acknowledged", "id": alert_id})


def run_api(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """Start the Flask API server."""
    print(f"\nStarting Fall Detection Web Dashboard at http://localhost:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)


if __name__ == "__main__":
    run_api(port=5000, debug=True)


