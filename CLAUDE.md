# CLAUDE.md — GuardianAI Project Context

Context for AI assistants working on this project. Read this first.

---

## 1. What this is

**GuardianAI — Real-Time Automated Fall Detection and Alert System**
A Final Year Project (FYP) for BS Computer Science. It detects human falls in
real time from an ordinary camera feed using YOLOv8-Pose plus a hybrid
rule-based + CNN-LSTM classifier, and alerts caregivers through multiple
channels. Designed for low-resource elderly-care environments in Pakistan,
where falls often go unnoticed.

| | |
|---|---|
| Group ID | FYP-2026-G6 |
| GitHub | https://github.com/KashanAshraf84/FYP-2026-G6-RealTimeAutomatedFallDetectionAndAlertSystem |
| Live demo | https://kashan84-guardianai.static.hf.space |
| Hugging Face account | `Kashan84` |

### Team

| Name | Roll No | Role |
|---|---|---|
| Kashan M. Ashraf | BSCS-01007 | Video Processing + Testing — **the usual operator of this repo** |
| Umer Ahmed | BSCS-01011 | AI Model + Development — **Group Lead** (signs off documents) |
| Imran | BSCS-01014 | UI Design + Documentation |

---

## 2. ⚠️ Two copies of the code — read this before editing

The code exists in **two places** and they are **not** symlinked:

| Path | Role |
|---|---|
| `C:\Users\kashan.ashraf\Desktop\Kashan_FYP\FYP1\` | **Working directory.** Not a git repo. This is where the app is actually run and edited, and where `logs/`, `models/` and the webcam demos live. |
| `...\FYP-2026-G6-RealTimeAutomatedFallDetectionAndAlertSystem\src\` | **Committed copy** inside the git repo. |

**When you change application code, edit `FYP1/` and then copy the changed
files into `src/`.** Otherwise the repo and the running app drift apart.

```powershell
Copy-Item "$fyp1\<changed>.py" "$repo\src\<changed>.py" -Force
```

A third partial copy of the pipeline lives in `deploy/huggingface-space/`
(server variant) and a **JavaScript port** in
`deploy/huggingface-static/pipeline.js`. If detection logic or a threshold in
`config.py` changes, the JS port must be updated to match, or the live demo
will diverge from the real system.

---

## 3. Environment and gotchas

- **OS:** Windows 11, **PowerShell 5.1**. No `&&` chaining — use `;` or `if ($?) { }`.
- **Python 3.14.6** at `C:\Users\kashan.ashraf\AppData\Local\Programs\Python\Python314`.
- **`pip` is NOT on PATH.** Always use `python -m pip`. Plain `pip` fails with
  `CommandNotFoundException`.
- `node` v24 is available (used to test the JS port).
- Chrome is used headless to render HTML documents to PDF.
- Git shows `LF will be replaced by CRLF` warnings on every add — normal, ignore.

---

## 4. Running the project

```powershell
cd C:\Users\kashan.ashraf\Desktop\Kashan_FYP\FYP1
python -m pip install -r requirements.txt
python api_server.py          # → http://127.0.0.1:5000   (main dashboard)
```

Other entry points:

```powershell
python main.py detect                     # CLI, OpenCV window
python main.py detect --source video.mp4
python main.py train --synthetic          # trains without a dataset
python main.py export --model models/fall_detector_best.pth --output models/fall_detector.onnx
python fyp1_demo.py                       # single-person demo
python fyp2_multi_demo.py                 # multi-person demo (standalone only)
```

Pre-trained weights are committed (`yolov8n-pose.pt`, `models/fall_detector_best.pth`),
so detection runs without training first.

---

## 5. Architecture

```
frame → YOLOv8-Pose (17 keypoints → 13 critical, normalised to [0,1])
      → feature vector (51-dim = 39 raw + 12 engineered, 5 active + 7 reserved)
      → 30-frame sliding buffer (~1 s at 30 FPS)
      → rule engine  ──┐
      → CNN-LSTM     ──┴→ fusion (0.7 NN / 0.3 rules, severity override)
      → temporal smoothing (fall = 3 of last 5, warning = 2 of 5)
      → alert if status=fall AND confidence>0.7 AND 30 s cooldown elapsed
```

| File | Responsibility |
|---|---|
| `config.py` | **All** thresholds/paths as dataclasses; `CONFIG` singleton. No magic numbers elsewhere. |
| `pose_estimator.py` | `PoseEstimator` (single), `MultiPersonPoseEstimator` (up to 5, standalone only) |
| `feature_extractor.py` | `FeatureExtractor` — biomechanics + sequence buffer; `TemporalFeatureProcessor` (training) |
| `fall_detector_model.py` | `FallDetectorLSTM` / `FallDetectorCNNLSTM` (default) / `FallDetectorTransformer` + `create_model()` factory |
| `inference.py` | `FallDetectionInference` — the orchestrator; fusion, smoothing, HUD |
| `alert_system.py` | `AlertSystem` — console, popup, buzzer, OS notification, e-mail, JSON log |
| `api_server.py` | Flask REST + MJPEG stream |
| `static/` | Dashboard (vanilla HTML/CSS/JS, no build step) |
| `train.py`, `dataset.py` | Offline training pipeline |

### Key numbers

Upright angle `70°` · lying angle `50°` · ground proximity `head_y > 60% of 480` ·
drop speed `> 20 px/frame` · fall counter `≥2` · lying counter `≥5` ·
sequence `30 frames` · fusion NN weight `0.7` · rule override `>0.8` ·
alert gate `0.7` · cooldown `30 s` · popup `3000 ms`.

**Critical keypoint indices** (into YOLO's 17): `[0,5,6,7,8,9,10,11,12,13,14,15,16]`.
Within the resulting 13-element array: **index 0 = nose, indices 7 and 8 = left/right hip.**
The torso anchor is the **midpoint of 7 and 8** — not a single hip, and definitely not index 3.

---

## 6. 🔴 Known open defects — do not "rediscover" these

**1. Fall rule ignores posture (highest priority, NOT fixed).**
In `feature_extractor.get_rule_based_status()`, the `fall` verdict is reachable
**only** through `drop_speed > 20`. The `is_lying` predicate (which encodes body
angle + ground proximity) can escalate no further than `warning`. Consequences:

- A fast head/hand movement near the camera can register `fall` while upright.
- Someone already motionless on the floor produces no velocity spike, so never reaches `fall`.

SRS FR-3.1 specifies the *correct* behaviour (speed **confirmed by** ground
proximity and angle < 50°). The requirement is right; the implementation is not.
Documented in SDS §12.2 and SRS v3 §9.4. **First scheduled FYP-2 task.**
The JS port in `pipeline.js` deliberately reproduces this defect so the demo
matches the real system.

**2. Hip keypoint confidence not validated** (SRS FR-2.8, not implemented).
YOLO emits estimated hip positions even when hips are out of frame; those feed
the angle calculation unchecked, causing false `warning` at close camera range.

**3. Other gaps:** multi-person not wired into the live dashboard; inference-time
feature normalisation not applied (train/serve skew); no automated tests; Flask
binds `0.0.0.0` with no authentication.

---

## 7. Bugs already fixed — don't reintroduce

| Fix | Why it matters |
|---|---|
| `requirements.txt`: removed `mediapipe` (unused) and `smtplib-attachment` (**not a real package** — broke `pip install` entirely); added `ultralytics` | Install was impossible before |
| `feature_extractor.py`: hip was `keypoints[3]` = **left elbow** | Corrupted the core body-angle signal |
| `api_server.py`: `threaded=True, use_reloader=False` | Reloader spuriously restarted the server and tore down the camera mid-stream; single-threaded server let the MJPEG stream starve `/status` |
| `static/app.js`: `updateTime()` / `addLogItem()` were **called but never defined** | Threw on page load, killing the entire polling loop — every dashboard number was frozen |
| `style.css`: `rgba(2ffa, 165, 2, …)` invalid | Warning status card had no colour |
| `alert_system.py`: buzzer moved onto a thread | `winsound.Beep()` blocks ~600 ms, stalling video on every alert |
| `generate_frames()`: camera retry + reopen + `finally` release | A single failed read used to end the stream permanently |
| `pipeline.js`: clamp keypoints to frame bounds | Ultralytics clamps; without it, subjects cut off by the frame skew the angle |

---

## 8. Deployment

**Live:** https://kashan84-guardianai.static.hf.space

⚠️ **Hugging Face free tier only allows `static` Spaces.** Docker and Gradio
Spaces return `402 Payment Required` without PRO. Do not attempt a Docker Space.

| Folder | Status |
|---|---|
| `deploy/huggingface-static/` | **Deployed.** Runs entirely in the browser via ONNX Runtime Web. `pipeline.js` is a line-for-line JS port of the Python detection logic, verified identical on a 45-frame synthetic sequence (angles 10°–90°). No video leaves the client — this preserves the project's privacy thesis. |
| `deploy/huggingface-space/` | **Built and tested, not deployed.** Docker/Flask variant, ~8.5 fps, per-visitor session isolation. Needs a paid host. |

Redeploy (requires a fresh HF Write token — the user revokes tokens after use):

```powershell
hf auth login --token <token>
python -c "from huggingface_hub import upload_folder; upload_folder(repo_id='Kashan84/guardianai', repo_type='space', folder_path='deploy/huggingface-static')"
```

The `headless` flag on `AlertConfig` (default `False`) suppresses buzzer, popup
and OS notification on hosts with no desktop/audio — this keeps **one codebase**
for both local and cloud rather than a fork.

---

## 9. Documents

All written as HTML and rendered to PDF with headless Chrome; the `.html` is the
editable source, so **edit the HTML and regenerate — never hand-edit a PDF.**

| Document | Path |
|---|---|
| SRS v3 (current) | `docs/SRS_document_V3.pdf` / `.html` |
| SRS v2 (superseded, kept for audit) | `docs/SRS_document_V2.docx` |
| Software Design Document v1.0 | `docs/SDS_GuardianAI_v1.0.pdf` / `.html` |
| Updated Architecture v1.0 | `architecture/Updated_Architecture_v1.0.pdf` / `.html` |

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --headless --disable-gpu `
  --no-pdf-header-footer --print-to-pdf="docs\X.pdf" "file:///C:/full/path/docs/X.html"
```

Diagrams are **hand-written inline SVG** (no Mermaid — SRS v2's Mermaid block
never rendered and appeared as unreadable text in the PDF).

---

## 10. Academic context

**FYP-I marks:** Proposal 30 + Checkpoint-1 30 + Checkpoint-2 30 + Supervisor 10 = 100.

**Checkpoint-2 deliverables** (deadline: first Thursday after End-Term exams):

| Deliverable | Status |
|---|---|
| Software Design Document | ✅ `docs/SDS_GuardianAI_v1.0.pdf` |
| Updated Architecture | ✅ `architecture/Updated_Architecture_v1.0.pdf` |
| Prototype / MVP Demonstration | ✅ source in `src/` + live demo URL |
| Initial Implementation Evidence | ❌ **`evidence/` is still empty — the user is doing this themselves** (screenshots, demo video, sample logs, test results) |

⚠️ **The FYP coordinator supplied document templates that no AI session has
seen.** The documents follow IEEE 830/1016/29148 structure instead. They must be
checked against the official templates before submission — flag this if asked
about document format.

---

## 11. Working preferences observed

- The user is learning this codebase (a friend wrote the original). Explain *why*,
  not just *what*.
- Be honest about defects rather than hiding them — known issues are deliberately
  documented in the SRS/SDS/README and even reproduced in the live demo. This has
  been treated as a strength for the viva, not a weakness.
- Verify claims by running things (the app, exports, PDF renders, the JS port
  cross-check) rather than asserting they work.
- Don't push to GitHub or deploy without being asked.
- The user pastes HF tokens when needed and revokes them afterwards — remind them.

### Repo state note

Commits are made **locally and not pushed** unless explicitly requested. Legacy
`src/*.rar` archives (one ~106 MB) from before the plain-source migration are
still tracked and now redundant; removal was offered and not yet actioned.
