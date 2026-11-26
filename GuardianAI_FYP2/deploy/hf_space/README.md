---
title: GuardianAI FYP-2 Fall Detection
emoji: 🛡️
colorFrom: purple
colorTo: indigo
sdk: gradio
python_version: "3.11"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Multi-person fall detection (YOLOv8-Pose + CNN-LSTM)
---

# GuardianAI — FYP-2 online demo

Upload a video or use the webcam. Every person gets a box with a stable ID (P1, P2, …) and a live status:
NORMAL, WARNING (fall starting) or FALL. A fall alert is raised when a person's fall score is at least 80%
on 3 consecutive analysed frames, the same rule as the GuardianAI web application.

- Pose: YOLOv8n-Pose (Ultralytics)
- Tracking: multi-person tracker with ghost filtering (`ai/tracking.py`)
- Classifier: FallNet v2, CNN-LSTM over the last 2 s of each person's motion, trained on GMDCSA-24
- Held-out results (leave-one-subject-out, 5 FPS): 69/79 falls detected, 74/81 normal clips without a false alarm

Final Year Project demo; not a medical device.
