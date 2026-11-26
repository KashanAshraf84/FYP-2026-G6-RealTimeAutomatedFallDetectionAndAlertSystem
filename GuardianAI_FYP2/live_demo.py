"""
GuardianAI - Live Multi-Person Demo
===================================
Runs the same detector as the Django app (ai.adapters.TeamDetector) on a
webcam or video file and draws every person's box with their ID and status.
Alerts follow the app's default rule: fall score >= 0.80 on 3 consecutive
processed frames for the same person.

    .venv\\Scripts\\python live_demo.py                                  # laptop webcam
    .venv\\Scripts\\python live_demo.py --source path\\to\\clip.mp4         # video file
    .venv\\Scripts\\python live_demo.py --source clip.mp4 --save out.mp4 --no-window

Press q to quit.
"""

import argparse
import time

import cv2
import numpy as np
from PIL import Image

from ai.adapters import TeamDetector

# BGR versions of the web UI colours (static/monitor.js)
COLORS = {
    "fall": (145, 107, 255),
    "warning": (90, 198, 249),
    "normal": (176, 221, 67),
    "unknown": (184, 168, 160),
}


class AlertRule:
    """Per-person confirmation streak, mirroring monitor/services.ingest defaults."""

    def __init__(self, confidence=0.80, frames=3, cooldown_s=30.0):
        self.confidence, self.frames, self.cooldown_s = confidence, frames, cooldown_s
        self.streak, self.last_alert = {}, {}

    def update(self, result, t):
        """Returns the track IDs that raise a new alert at time t."""
        fired = []
        present = {p.track_id for p in result.predictions}
        for tid in list(self.streak):
            if tid not in present:
                self.streak[tid] = 0
        for p in result.predictions:
            hit = p.label == "fall" and p.confidence >= self.confidence
            self.streak[p.track_id] = self.streak.get(p.track_id, 0) + 1 if hit else 0
            last = self.last_alert.get(p.track_id)
            if self.streak[p.track_id] >= self.frames and (last is None or t - last >= self.cooldown_s):
                self.last_alert[p.track_id] = t
                fired.append(p.track_id)
        return fired


def draw_overlay(frame, result, alerting=()):
    """Draw each person's box and 'P1 · NORMAL 98%' tag; banner while someone is alerting."""
    h, w = frame.shape[:2]
    top = 0
    if alerting:
        top = 34
        cv2.rectangle(frame, (0, 0), (w, top), COLORS["fall"], -1)
        cv2.putText(frame, "FALL ALERT: " + ", ".join(sorted(alerting)), (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for p in result.predictions:
        if p.bbox is None:
            continue
        color = COLORS.get(p.label, COLORS["unknown"])
        x1, y1, x2, y2 = (int(p.bbox[0] * w), int(p.bbox[1] * h), int(p.bbox[2] * w), int(p.bbox[3] * h))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if p.label == "fall" else 2)
        tag = f"{p.track_id} - " + ("WARMING UP" if p.label == "unknown" else f"{p.label.upper()} {p.confidence:.0%}")
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = y1 - th - 10 if y1 - th - 10 >= top else max(y1, top)  # above the box, else just inside it
        cv2.rectangle(frame, (x1, ty), (x1 + tw + 10, ty + th + 10), color, -1)
        cv2.putText(frame, tag, (x1 + 5, ty + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (24, 19, 17), 2)
    cv2.putText(frame, f"{len(result.predictions)} person(s)", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="Webcam index (0) or a video file path")
    parser.add_argument("--fps", type=float, default=5.0, help="Frames analysed per second (browser rate is ~5)")
    parser.add_argument("--width", type=int, default=640, help="Resize frames to this width, like the browser")
    parser.add_argument("--save", help="Write the annotated frames to this .mp4 file")
    parser.add_argument("--no-window", action="store_true", help="Do not open a preview window")
    args = parser.parse_args()

    live = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if live else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source {args.source!r}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    detector = TeamDetector()
    rule = AlertRule()
    print(f"Model {detector.model_version} | source {args.source} | analysing {args.fps:g} FPS. Press q to quit.")
    writer, alert_until, next_t, idx, started = None, {}, 0.0, -1, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            t = time.time() - started if live else idx / src_fps
            if t < next_t:
                continue
            next_t = t + 1.0 / args.fps
            scale = min(args.width / frame.shape[1], 1.0)
            frame = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector.predict(image, camera_id=0, timestamp=t)
            for tid in rule.update(result, t):
                alert_until[tid] = t + 3.0
                print(f"[{t:6.1f}s] FALL ALERT {tid}")
            alerting = [tid for tid, until in alert_until.items() if t <= until]
            draw_overlay(frame, result, alerting)
            if args.save:
                if writer is None:
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                             (frame.shape[1], frame.shape[0]))
                writer.write(frame)
            if not args.no_window:
                cv2.imshow("GuardianAI live demo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
