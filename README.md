# GuardianAI - Real-Time Automated Fall Detection and Alert System

Final Year Project (BS Computer Science)
Group ID: FYP-2026-G6

## What this project is about

In a lot of homes and hospitals in Pakistan, elderly patients aren't monitored
closely enough, so a fall can go unnoticed for a long time. GuardianAI tries to
fix that using just a normal camera - no wearable sensors needed.

It watches the camera feed, tracks the person's body pose, and decides whether
what's happening is normal, a warning sign, or an actual fall. If it detects a
fall, it immediately alerts a caregiver through sound, a popup, a notification,
email, and logs the event.

## Team

| Name | Roll No | Role |
|---|---|---|
| Kashan M. Ashraf | BSCS-01007 | Video Processing + Testing |
| Umer Ahmed | BSCS-01011 | AI Model + Development (Group Lead) |
| Imran | BSCS-01014 | UI Design + Documentation |

## How it works (short version)

1. **YOLOv8-Pose** detects the person and extracts body keypoints from each frame.
2. Those keypoints are turned into features like body angle, how fast the
   person is dropping, and how close they are to the ground.
3. A **hybrid classifier** (rule-based checks + a CNN-LSTM neural network)
   looks at the last ~1 second of motion and decides: normal, warning, or fall.
4. If it's a confirmed fall with high enough confidence, the alert system
   fires (buzzer, popup, OS notification, email) and the event gets saved to
   a SQLite database.

## How to run it

```bash
cd src
pip install -r requirements.txt
python api_server.py
```

Then open `http://127.0.0.1:5000` in your browser. This starts the web
dashboard with the live camera feed, pose overlay, current status, and alert
history.

Model weights are already included in the repo, so you don't need to train
anything before running it.

### Other ways to run it

```bash
python main.py detect                     # plain OpenCV window instead of the dashboard
python main.py detect --source video.mp4  # run on a video file instead of a webcam
python main.py train --synthetic          # train on generated synthetic data
```

## Live demo (no install needed)

There's also a browser-only version that runs entirely client-side (no
server, nothing uploaded) using ONNX Runtime Web:

**https://kashan84-guardianai.static.hf.space**

## Project structure

```
src/            main application code (Python + web dashboard)
docs/           SRS and SDS documents
architecture/   architecture diagrams
deploy/         Hugging Face demo versions (browser + Docker)
presentations/  slides used for the checkpoints
evidence/       screenshots and demo evidence
database/       DB schema and seed script
```

## Known limitations

- The fall-detection rule currently reacts mostly to how fast someone drops,
  not their actual body posture. This means a quick hand or head movement
  can occasionally register as a "fall", and a person who goes down slowly
  might not get flagged. This is a known issue, documented in the SDS
  (section 12.2), and the fix is planned for FYP-2.
- Multi-person detection works as a standalone demo but isn't wired into the
  live dashboard yet.
- No automated test suite yet.

## Status

FYP-I Checkpoint-2 completed. Pose estimation, hybrid fall classification,
multi-channel alerting, and the live dashboard are all working and were
demonstrated at Checkpoint-2.

## Documents

| Document | Location |
|---|---|
| Software Requirements Specification v3 | `docs/SRS_document_V3.pdf` |
| Software Design Document v1.0 | `docs/SDS_GuardianAI_v1.0.pdf` |
| Updated Architecture v1.0 | `architecture/Updated_Architecture_v1.0.pdf` |
