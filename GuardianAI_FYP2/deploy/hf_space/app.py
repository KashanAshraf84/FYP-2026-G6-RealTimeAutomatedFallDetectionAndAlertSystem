"""
GuardianAI FYP-2 - Online Demo (Hugging Face Space)
===================================================
Runs the same detector as the Django app (ai.adapters.TeamDetector):
YOLOv8-Pose -> multi-person tracker (P1, P2, ...) -> FallNet v2 per person.

Two modes:
  - Video: upload a clip; every person is tracked and classified at 5 FPS and
    an annotated video plus an alert log are returned.
  - Webcam: live frames from the browser, analysed on the server.

Frames are processed in memory and not stored.
"""

import os
import tempfile
import time
import uuid

import cv2
import gradio as gr
import imageio_ffmpeg
import numpy as np
from PIL import Image

from ai.adapters import TeamDetector
from live_demo import AlertRule, draw_overlay

DETECTOR = TeamDetector()
ANALYSE_FPS = 5.0
MAX_SECONDS = 60
SESSIONS = {}  # webcam session id -> (camera id, AlertRule, alert_until)


def _camera_id():
    return uuid.uuid4().int % 10**9


def _status_rows(result, t):
    return [[f"{t:.1f}", p.track_id, p.label, f"{p.confidence:.0%}" if p.label != "unknown" else "-"]
            for p in result.predictions]


def analyse_video(path, progress=gr.Progress()):
    if not path:
        raise gr.Error("Upload a video first.")
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    camera = _camera_id()
    rule, alert_until, log = AlertRule(), {}, []
    out_path = os.path.join(tempfile.mkdtemp(), "guardianai_result.mp4")
    writer = None
    idx, next_t = -1, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            t = idx / src_fps
            if t > MAX_SECONDS:
                break
            if t < next_t:
                continue
            next_t = t + 1.0 / ANALYSE_FPS
            scale = min(640 / frame.shape[1], 1.0)
            frame = cv2.resize(frame, (round(frame.shape[1] * scale) // 2 * 2, round(frame.shape[0] * scale) // 2 * 2))
            result = DETECTOR.predict(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                      camera_id=camera, timestamp=t)
            for tid in rule.update(result, t):
                alert_until[tid] = t + 3.0
                log.append([f"{t:.1f}", tid, "FALL ALERT", ""])
            draw_overlay(frame, result, [k for k, v in alert_until.items() if t <= v])
            if writer is None:
                writer = imageio_ffmpeg.write_frames(out_path, (frame.shape[1], frame.shape[0]), fps=ANALYSE_FPS,
                                                     codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2)
                writer.send(None)
            writer.send(np.ascontiguousarray(frame[:, :, ::-1]))
            if total:
                progress(min(idx / total, 1.0), desc="Analysing")
    finally:
        cap.release()
        if writer is not None:
            writer.close()
        DETECTOR.reset(camera)
    if writer is None:
        raise gr.Error("Could not read any frames from this video.")
    alerts = [r for r in log if r[2] == "FALL ALERT"]
    summary = (f"**{len(alerts)} fall alert(s)**: " + ", ".join(f"{r[1]} at {r[0]} s" for r in alerts)
               if alerts else "**No fall alert.** Everyone stayed normal (or the fall did not last 3 analysed frames).")
    if t > MAX_SECONDS:
        summary += f"\n\nOnly the first {MAX_SECONDS} s were analysed."
    return out_path, summary, log or [["-", "-", "no alerts", "-"]]


def analyse_webcam(image, session):
    if image is None:
        return None, "Waiting for the webcam...", session
    if session not in SESSIONS:
        session = session or uuid.uuid4().hex
        SESSIONS[session] = (_camera_id(), AlertRule(), {})
    camera, rule, alert_until = SESSIONS[session]
    t = time.time()
    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    scale = min(640 / frame.shape[1], 1.0)
    frame = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
    result = DETECTOR.predict(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), camera_id=camera, timestamp=t)
    for tid in rule.update(result, t):
        alert_until[tid] = t + 3.0
    draw_overlay(frame, result, [k for k, v in alert_until.items() if t <= v])
    people = ", ".join(f"{p.track_id}: {p.label.upper()}" + (f" {p.confidence:.0%}" if p.label != "unknown" else "")
                       for p in result.predictions) or result.message
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), people, session


ABOUT = f"""
# GuardianAI - FYP-2 Fall Detection (multi-person)
**Pipeline:** YOLOv8n-Pose → multi-person tracker (IDs P1, P2, …) → FallNet v2 (CNN-LSTM, last 2 s of motion per person)
→ alert when a person's fall score is ≥ 80% on 3 consecutive analysed frames.
Labels: **NORMAL** · **WARNING** (fall starting) · **FALL**.

Measured on people the model never saw (leave-one-subject-out, GMDCSA-24, 5 FPS replay): **69/79 falls detected (87%)**,
**74/81 normal-activity clips without a false alarm (91%)**, median alert 1.7 s after the fall starts.
Known weak spots: falls onto a bed, deliberately lying down quickly, push-ups.
Model `{DETECTOR.model_version}` · CPU · frames are analysed in memory and not stored. FYP demo, not a medical device.
"""

with gr.Blocks(title="GuardianAI FYP-2") as demo:
    gr.Markdown(ABOUT)
    with gr.Tab("Analyse a video"):
        with gr.Row():
            video_in = gr.Video(label="Upload a fall / daily-activity clip (max 60 s analysed)", sources=["upload"])
            video_out = gr.Video(label="Result: boxes, IDs and status")
        run = gr.Button("Analyse", variant="primary")
        summary = gr.Markdown()
        table = gr.Dataframe(headers=["time (s)", "person", "event", ""], label="Alerts", interactive=False)
        run.click(analyse_video, video_in, [video_out, summary, table], concurrency_limit=1)
    with gr.Tab("Live webcam"):
        gr.Markdown("Allow camera access. Frames are sent to this Space for analysis (about 3–5 per second on the free CPU).")
        session = gr.State("")
        with gr.Row():
            cam = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
            cam_out = gr.Image(label="GuardianAI view")
        people = gr.Textbox(label="People")
        cam.stream(analyse_webcam, [cam, session], [cam_out, people, session],
                   stream_every=0.25, concurrency_limit=1, time_limit=300)

if __name__ == "__main__":
    demo.queue().launch()
