# Real-Time Automated Fall Detection and Alert System
### For Elderly and Patient Safety in Low-Resource Environments

---

## Group ID
FYP-2026-G6

## Team Members
|       Name       |   Roll No   |              Role                |
|      ------      |  ---------  |             ------               |
| Kashan M. Ashraf | BSCS-01007  |   Video Processing + Testing     |
|     Umer Ahmed   | BSCS-01011  |    AI Model + Development        |
|       Imran      | BSCS-01014  |    UI Design + Documentation     |

---

## Problem Statement
In Pakistan, elderly care and patient monitoring are severely limited due to staff
shortages and the absence of smart healthcare infrastructure. Falls in homes and
hospitals frequently go unnoticed, leading to serious and often life-threatening
injuries. This project proposes an affordable AI-based system — **GuardianAI** —
that detects falls in real time using an ordinary camera feed, processes video
entirely on local hardware, and immediately alerts caregivers through multiple
channels.

---

## Implemented Features
- [✔] Real-time person detection and pose estimation using **YOLOv8-Pose**
- [✔] Biomechanical feature engineering (body angle, drop speed, ground proximity)
- [✔] Hybrid decision engine — explainable rule-based state machine **+** CNN-LSTM temporal classifier, fused
- [✔] Multi-channel alerting: audible buzzer, on-screen popup, native OS notification, optional e-mail, JSON event logging with JPEG capture
- [✔] Live web dashboard — MJPEG video feed, real-time telemetry, live system-health indicators
- [✔] Operator controls — pause/resume feed, mute/unmute alerts, reset tracking
- [✔] Camera disconnect detection and automatic reconnection
- [✔] Offline training pipeline (dataset preprocessing, CNN-LSTM/LSTM/Transformer training, ONNX export)
- [ ] Multi-person tracking integrated into the live dashboard *(implemented standalone, not yet wired into the main pipeline — FYP-2)*
- [ ] Browsable event history / live threshold configuration UI *(navigation present, intentionally disabled — FYP-2)*
- [ ] Automated test suite *(FYP-2)*

> **Note on scope accuracy:** an earlier revision of this checklist listed MediaPipe as
> the pose backend and YOLOv8 as not-yet-implemented. That was backwards — the project
> migrated from MediaPipe to YOLOv8-Pose during development, for better multi-person
> robustness and occlusion tolerance. See `architecture/Updated_Architecture_v1.0.pdf`
> for the full before/after rationale.

---

## Setup / Execution Instructions

**Requirements:**
- Python 3.9+ (tested on 3.14)
- A webcam or video file
- See `src/requirements.txt` for the full pinned dependency list (OpenCV, PyTorch, Ultralytics YOLOv8, Flask, plyer, etc.)

**Install:**
```bash
cd src
pip install -r requirements.txt
```

**Run the live dashboard (recommended):**
```bash
python api_server.py
```
Then open `http://127.0.0.1:5000` in a browser. This starts the Flask server, loads
the YOLOv8-Pose and CNN-LSTM models, and streams the annotated webcam feed with a
live telemetry dashboard.

**Alternative entry points (no web dashboard):**
```bash
python main.py detect              # CLI detection on webcam, OpenCV window
python main.py detect --source path/to/video.mp4
python fyp1_demo.py                # standalone single-person demo
python fyp2_multi_demo.py          # standalone multi-person demo
python main.py train --synthetic   # train on synthetic data (no dataset required)
```

Pre-trained weights (`yolov8n-pose.pt`, `models/fall_detector_best.pth`) are included,
so detection can be run immediately without training first.

---

## Current Project Status
🟢 **FYP-I Checkpoint-2 delivered.** End-to-end pipeline running at 25–30 FPS: pose
estimation, hybrid rule + neural fall classification, multi-channel alerting, and a
live web dashboard are all implemented and demonstrable. A known defect in the
rule-based fall-confirmation logic (see `docs/SDS_GuardianAI_v1.0.pdf`, §12.2) is
documented and scheduled as the first FYP-2 task.

---

## Documentation
| Document | Location | Description |
|---|---|---|
| Software Requirements Specification v3 | `docs/SRS_document_V3.pdf` | Current, authoritative requirements (supersedes V2) |
| Software Design Document v1.0 | `docs/SDS_GuardianAI_v1.0.pdf` | Full architecture, class model, data design, algorithms |
| Updated Architecture v1.0 | `architecture/Updated_Architecture_v1.0.pdf` | Checkpoint-1 → Checkpoint-2 architecture delta and rationale |

---

## Repository Structure
```
/docs           → SRS, SDD and other technical documentation
/architecture   → Architecture diagrams and the Checkpoint-2 update document
/src            → Application source code (Python + web dashboard)
/test           → Test cases and scenarios
/presentations  → Proposal and defence slides
/reports        → Progress and final reports
/evidence       → Screenshots, demo recordings, logs and testing evidence
/dataset        → Dataset references and preparation notes
/research       → Literature review and background reading
/meeting_logs   → Supervisor meeting records
```
