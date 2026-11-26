---
title: GuardianAI FYP-2 Monitoring Workspace
emoji: 🛡️
colorFrom: purple
colorTo: indigo
sdk: static
pinned: false
license: cc-by-nc-4.0
short_description: Fall detection workspace running fully in the browser
---

# GuardianAI — FYP-2 browser workspace

The GuardianAI monitoring application (sign-in, monitor wall, cameras, event history and review, notifications,
reports, settings) with the fall detector running entirely in your browser. Open `index.html`, sign in with a demo
account (`kashan` / `123` as administrator, `caregiver` / `123` as caregiver), open a camera and start
your webcam or choose a video file.

Pipeline per frame: YOLOv8n-Pose (ONNX Runtime Web, WebGPU when available) → multi-person tracker (IDs P1, P2, …)
→ FallNet v2 (CNN-LSTM over each person's last 2 s of motion) → NORMAL / WARNING / FALL → an incident when a
person's fall score is ≥ 80% on 3 consecutive analysed frames, the same rule as the Django server.

Incidents, settings and accounts are stored in your browser only; no frame or file is uploaded. Held-out results
(leave-one-subject-out on GMDCSA-24, 5 FPS): 69/79 falls detected, 74/81 normal clips without a false alarm.
Final Year Project (FYP-2026-G6) demo; not a medical device.
