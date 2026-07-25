---
title: GuardianAI Fall Detection
emoji: 🚨
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Real-time AI fall detection using YOLOv8-Pose and CNN-LSTM
---

# GuardianAI — Real-Time Automated Fall Detection and Alert System

Live demonstration of a final-year project (FYP-2026-G6, BS Computer Science)
that detects human falls in real time from an ordinary camera feed.

## How to use this demo

1. Press **Start Camera** and allow camera access when prompted.
2. Stand back far enough that your **head, torso and hips** are all in frame —
   the detector measures the angle between your nose and your hip midpoint.
3. Lower yourself toward the floor to simulate a fall and watch the status
   card change from `NORMAL` → `WARNING` → `FALL`.

## How it works

| Stage | Technology |
|---|---|
| Pose estimation | YOLOv8-Pose (nano) — 17 keypoints, 13 retained for analysis |
| Feature engineering | Body orientation angle, head drop speed, ground proximity, aspect ratio |
| Rule engine | Explainable biomechanical state machine with counter-based hysteresis |
| Neural classifier | CNN-LSTM over a 30-frame temporal window (~1 s at 30 FPS) |
| Decision fusion | 0.7 neural / 0.3 rules, with a severity-priority override |
| Temporal smoothing | Majority vote — 3 of the last 5 frames for a fall verdict |

The **body angle** and **drop speed** shown on the overlay are the actual
evidence behind each verdict. Exposing them is a deliberate Explainable-AI
design decision: in a clinical setting an automated alert has to be auditable,
not just accurate.

## Differences from the production system

GuardianAI is designed as an **edge** application: it normally runs on a small
computer physically attached to the camera, so that video never leaves the
premises. This cloud demo necessarily inverts that so anyone can try it from a
browser:

| | Production (edge) | This demo (cloud) |
|---|---|---|
| Camera | Attached to the machine running the detector | Your browser, streamed to the server |
| Video handling | Never leaves the building | Frames sent to this Space for inference |
| Alerts | Buzzer, popup, OS notification, e-mail, JSON log | Browser tone + on-screen alert |
| Throughput | 25–30 FPS | Lower — each frame makes a network round trip |

Frames are processed in memory and are not recorded.

## Known limitation

The rule engine's `fall` verdict currently triggers on descent speed alone,
without the ground-proximity and body-angle confirmation the specification
requires. A rapid movement close to the camera can therefore produce a false
positive, and someone already motionless on the floor may not register. This
is documented in the project's design document and is the first scheduled
task of FYP-2.

## Project team

| Name | Roll No | Role |
|---|---|---|
| Kashan M. Ashraf | BSCS-01007 | Video Processing + Testing |
| Umer Ahmed | BSCS-01011 | AI Model + Development |
| Imran | BSCS-01014 | UI Design + Documentation |
