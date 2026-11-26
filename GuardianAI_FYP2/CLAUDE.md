# CLAUDE.md — GuardianAI (FYP-2)

## Project summary
GuardianAI is a Django web application for camera monitoring and fall-incident review, with an AI fall detector
(YOLOv8-Pose -> multi-person tracker -> FallNet v2 CNN-LSTM). It is a university Final Year Project (FYP-2026-G6)
that **may later be piloted in a hospital**. Treat patient/people imagery and incident data as sensitive.
It is not a certified medical device and makes no validated accuracy claims beyond `docs/MODEL_CARD.md`.

Two front-ends exist:
- **Django app** (this repo root): server-side detector, SQLite/PostgreSQL, roles, audit log.
- **Static browser app** (`deploy/static_space/`): a standalone single-page copy with the detector running in the browser (ONNX Runtime Web), published to a public Hugging Face Space. Its data lives only in the visitor's browser.

More detail: `docs/BUSINESS_REQUIREMENTS.md`, `docs/TECHNICAL_OVERVIEW.md`, `docs/DATABASE.md`, `docs/KNOWN_ISSUES.md`.
Pre-existing human docs (do not duplicate, keep in sync only if asked): `README.md`, `docs/ARCHITECTURE.md`, `docs/AI_HANDOFF.md`, `docs/ACCEPTANCE.md`, `docs/MODEL_CARD.md`, `training/README.md`, `ROADMAP_FYP.md`.

## Tech stack
- Python (tested on 3.14.6, Windows 10/11), Django 5.2, Pillow. Plain Django templates + vanilla CSS/JS (no Node build, no CDN).
- DB: SQLite (default, WAL) or PostgreSQL 14+ (`GUARDIAN_DB=postgres`, driver `psycopg[binary]`).
- AI (optional install, `requirements-ai.txt`): torch, ultralytics (YOLOv8n-pose), OpenCV, numpy, scikit-learn, OpenVINO, onnxruntime-directml.
- Browser app: ONNX Runtime Web (WebGPU/WASM), plain JS modules, Node `.mjs` tests.
- Hosting of demo: Hugging Face static Space (`deploy/publish_static.py`, needs `HF_TOKEN`).
- Lint: ruff (`requirements-dev.txt`).

## Folder structure
```
config/            Django settings, root URLs, WSGI
monitor/           Main app: models, views, accounts (auth/users), services (alert logic), forms, tests, migrations
  management/commands/  camera_worker.py (USB/RTSP/file), seed_demo.py
ai/                Detector contract, runtime (lazy load, validate, pacing), adapters, tracking, pose backends, fallnet, evaluate
training/          FallNet v2 pipeline (extract_keypoints, gmdcsa, windows, train_fallnet) + results/
templates/ static/ Django HTML templates and CSS/JS
deploy/            export_onnx*.py, publish_*.py, hf_space/ (old Gradio-style demo), static_space/ (browser SPA + models)
data/              dataset annotations (CSV), keypoints cache (npz), example_predictions.csv (invented, not real)
weights/           fallnet_v2.pt, yolov8n-pose.pt, pose_onnx/ (per-machine pose_backend.json is local)
docs/              Documentation
instance/          (generated, git-ignored) SQLite DB + .secret   <- currently absent on disk
*.py in repo root  LEGACY FYP1 engine (see rules)
```

## How to run / build / test (Windows PowerShell, from repo root)
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt          # core only
.venv\Scripts\python -m pip install -r requirements-ai.txt       # needed for TeamDetector
.venv\Scripts\python manage.py migrate
.venv\Scripts\python manage.py seed_demo
.venv\Scripts\python manage.py createsuperuser                   # or: first registered account becomes Administrator
$env:DJANGO_DEBUG='1'; .venv\Scripts\python manage.py runserver 127.0.0.1:8000 --noreload
```
- `start.ps1` / `run.bat` do the same (and install `requirements-ai.txt` every run).
- Camera worker: `manage.py camera_worker --camera <id> --fps 5` with `GUARDIAN_CAMERA_SOURCE` set.
- Checks/tests: `manage.py check`; `manage.py test monitor` (36 tests; needs DB CREATEDB rights on PostgreSQL); `python -m unittest ai.test_tracking`; `ruff check .`
- Browser app tests: `node deploy/static_space/test_parity.mjs`, `test_browser.mjs` (parity with Python tracker/features).
- Retrain model: `python -m training.extract_keypoints` then `python -m training.train_fallnet` (overwrites `weights/fallnet_v2.pt`; see rules).
- Settings come from environment variables. `config/settings.py` auto-loads `.env` (no `.env` currently exists; `.env.example` documents keys).

## Coding conventions (observed)
- Python: PEP 8-style, Black-like formatting in `monitor/` and `ai/` (ruff F checks pass), double quotes, type hints sparse.
- Views are function-based; permission via `@login_required`, `@staff` (Admin or Operator), `@admin_required` (Admin only).
- Business logic lives in `monitor/services.py`; views stay thin. Every state-changing action writes an `Audit` row.
- Detector adapters implement `predict(frame, *, camera_id, timestamp) -> Result` and `reset(camera_id)` (`ai/contracts.py`). Frames are PIL **RGB**; convert to BGR for OpenCV/YOLO.
- Preprocessing is shared between training and inference (`ai/pose_features.py`, `training/windows.py`). Never fork it.
- UI text and docs must not invent metrics, delivery status or accuracy claims (project-wide honesty rule).
- Heavy ML imports stay lazy so Django starts without torch.

## Rules to follow in EVERY session
1. **Do not modify existing source, config, weights, datasets or databases unless the user explicitly asks.** Documentation-only tasks must touch only docs.
2. **PostgreSQL database `guardianai` (and any Postgres DB the user names) is PROTECTED.** Never run `flush`, `migrate` backwards, `loaddata`, `dumpdata` to overwrite, `reset_db`, DROP/TRUNCATE/DELETE, or `manage.py test` against it without explicit approval. `manage.py test` creates a separate `test_` DB; confirm settings point where intended first. See `docs/DATABASE.md`.
3. Never read, print, copy or commit `.env`, `instance/.secret`, SMTP passwords, `HF_TOKEN` or RTSP URLs. Never send them to external services.
4. Do not create or run migrations (`makemigrations`/`migrate`) without asking; show the migration first.
5. Root-level FYP1 files are **LEGACY — do not touch**: `alert_system.py augmentations.py benchmark_datasets.py database.py dataset.py fall_detector_model.py feature_extractor.py focal_loss.py fyp1_config.py inference.py live_demo.py pose_estimator.py prepare_dataset.py train.py`. (Exception: `focal_loss.py` is imported by `training/train_fallnet.py`; `inference.py`+`fyp1_config.py` are used by `LegacyFYP1Detector`.) Do not "clean up" or refactor them.
6. Do not retrain or overwrite `weights/fallnet_v2.pt` / ONNX models or `data/keypoints/`; model card hashes and `deploy/static_space/models/` must stay consistent. Any model change needs a new version name and updated `docs/MODEL_CARD.md`.
7. Do not publish to Hugging Face (`deploy/publish_*.py`) or send email/notifications without explicit approval.
8. Never start cameras or send video frames anywhere without being asked. Raw frames must stay in memory only (no disk storage).
9. Keep the Django app and `deploy/static_space/` behaviour in step (alert rule: fall score >= 0.80 on 3 consecutive frames, 30 s cooldown) when changing either; run the parity test.
10. Ask the user (AskUserQuestion) when requirements, data ownership, or deployment targets are unclear. Do not assume.
11. Windows + PowerShell environment: use `;` not `&&`, `.venv\Scripts\python` for Python.
12. No git repository exists in this folder: before any large edit, tell the user and suggest a backup/`git init`.
