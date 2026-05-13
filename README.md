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
injuries. This project proposes an affordable AI-based system that detects falls
in real time using existing camera feeds, processes video locally, and immediately
alerts caregivers.

---

## Implemented Features
- [ ] Real-time person detection using YOLOv8
- [✔] Body pose estimation using MediaPipe
- [✔] Rule-based fall detection (Normal / Warning / Fall)
- [ ] Audio and on-screen alert on fall detection
- [ ] Event logging with timestamps
- [ ] Live dashboard UI

---

## Setup / Execution Instructions
> To be updated as development progresses.

**Requirements:**
- Python 3.9+
- OpenCV
- MediaPipe
- Ultralytics YOLOv8
- TensorFlow / Keras (FYP-2)

**Run:**
```bash
pip install -r requirements.txt
python main.py
```

---

## Current Project Status
🟡 FYP-1 In Progress — Rule-based detection pipeline under development.

---

## Repository Structure
```
/docs           → Technical documentation
/src            → Source code
/test           → Test cases and scenarios
/presentations  → Proposal and defence slides
/reports        → Progress and final reports
/evidence       → Testing evidence and accuracy results
```
