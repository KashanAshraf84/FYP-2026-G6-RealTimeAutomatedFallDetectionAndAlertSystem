---
title: GuardianAI Fall Detection
emoji: 🚨
colorFrom: indigo
colorTo: purple
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Real-time AI fall detection running entirely in your browser
---

# GuardianAI — Real-Time Automated Fall Detection and Alert System

Live demonstration of a final-year project (FYP-2026-G6, BS Computer Science)
that detects human falls in real time from an ordinary camera feed.

**Everything runs inside your browser.** No video, image or frame is ever
uploaded — there is no server-side component at all.

## How to use

1. Press **Start Camera**. Allow **camera** access, and allow **notifications**
   if you want alerts while you are working in another application.
2. Wait for the models to download (~16 MB, once — then cached).
3. Stand back so your **head, torso and hips** are all in frame; the detector
   measures the angle between your nose and your hip midpoint.
4. Lower yourself toward the floor and watch the status card move
   `NORMAL` → `WARNING` → `FALL`.

### Alerts while you are in another application

On a fall the demo raises a **native OS notification** — the browser equivalent
of the `plyer` desktop notification in the local application — so the alert is
visible even when this tab is behind another window. Grant the notification
permission when prompted; the sidebar shows whether it is enabled.

Detection continues while the tab is in the background, but browsers throttle
hidden tabs, so throughput drops until you return to the tab. The **Mute**
button suppresses the tone and the notification together, matching the local
application's behaviour, while event logging continues.

## How it works

| Stage | Technology |
|---|---|
| Pose estimation | YOLOv8-Pose (nano), ONNX, 480×480 input — 17 keypoints, 13 retained |
| Feature engineering | Body orientation angle, head drop speed, ground proximity, aspect ratio |
| Rule engine | Explainable biomechanical state machine with counter-based hysteresis |
| Neural classifier | CNN-LSTM over a 30-frame window (~1 s), ONNX |
| Decision fusion | 0.7 neural / 0.3 rules, with a severity-priority override |
| Temporal smoothing | Majority vote — 3 of the last 5 frames for a fall verdict |
| Runtime | ONNX Runtime Web (WebGPU where available, WASM otherwise) |

The **body angle** and **drop speed** on the overlay are the actual evidence
behind each verdict. Surfacing them is a deliberate Explainable-AI decision: in
a clinical setting an automated alert must be auditable, not merely accurate.

## Relationship to the production system

GuardianAI is designed as an **edge** application — normally a small computer
attached to the camera, so video never leaves the premises. This demo preserves
that property exactly, by moving inference into the browser instead of a
server.

| | Production (edge) | This demo (browser) |
|---|---|---|
| Inference location | Local machine beside the camera | Your browser tab |
| Video leaves the device | Never | Never |
| Models | PyTorch (`.pt` / `.pth`) | ONNX, same trained weights |
| Alerts | Buzzer, popup, OS notification, e-mail, JSON log | Audio tone + on-screen alert |
| Throughput | 25–30 FPS | Depends on your CPU/GPU; lower on WASM |

The detection logic in `pipeline.js` is a line-for-line port of the Python
modules `feature_extractor.py` and `inference.py`, using the identical
thresholds from `config.py`.

## Known limitation

The rule engine's `fall` verdict currently triggers on descent speed alone,
without the ground-proximity and body-angle confirmation the specification
requires. A rapid movement close to the camera can therefore cause a false
positive, and someone already motionless on the floor may not register. This
defect is reproduced faithfully here rather than hidden, is documented in the
project's design document (SDD §12.2) and requirements specification
(SRS v3 §9.4), and is the first scheduled task of FYP-2.

## Project team

| Name | Roll No | Role |
|---|---|---|
| Kashan M. Ashraf | BSCS-01007 | Video Processing + Testing |
| Umer Ahmed | BSCS-01011 | AI Model + Development |
| Imran | BSCS-01014 | UI Design + Documentation |
