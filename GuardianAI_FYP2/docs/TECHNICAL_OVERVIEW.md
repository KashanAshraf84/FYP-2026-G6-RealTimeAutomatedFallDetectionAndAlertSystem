# Technical Overview — GuardianAI

## 1. Architecture
Monolithic Django app with an in-process AI detector.

```
Browser webcam/video --JPEG POST (~5 FPS, CSRF+session)--> /cameras/<id>/frame/ --\
Server worker (USB/RTSP/file via OpenCV, camera_worker) -------------------------+--> ai.runtime.predict (global lock)
                                                                                      -> detector adapter (TeamDetector)
                                                                                      -> monitor.services.ingest (confirm, cooldown)
                                                                                      -> Event + Delivery (+ optional SMTP)
Wall/pages poll GET /api/status/ every 3 s; snapshot preview GET /cameras/<id>/snapshot/
```
- Detector chosen by `GUARDIAN_DETECTOR` (default in settings: `ai.adapters.TeamDetector`; `.env.example` suggests `UnconfiguredDetector`). Loaded lazily; a warm-up thread starts when `runserver` runs (`monitor/apps.py`).
- Inference is serialized by a process-wide lock; state (trackers, `LIVE_FRAMES`, login-lockout cache) is **per process**, so only a single web process is supported.
- Pacing: `next_ms` returned to browsers keeps model duty <= `GUARDIAN_AI_DUTY` (0.65), interval clamped 160–500 ms.

## 2. Modules
| Module | Role |
|---|---|
| `config/settings.py` | Env-driven settings, auto-loads `.env`, per-install secret in `instance/.secret`, DB selection |
| `monitor/models.py` | Camera, DetectionSettings (singleton pk=1), TrackState, Event, Delivery, Profile, PasswordResetCode, Audit |
| `monitor/views.py` | Pages + frame/status/snapshot/demo/export/report endpoints |
| `monitor/accounts.py` | Registration, login lockout, reset codes, own account, admin user management |
| `monitor/services.py` | `ingest()` (confirmation, cooldown, TrackState) and `notify()` |
| `monitor/auth_backends.py`, `signals.py`, `context.py`, `forms.py` | Username-or-email auth, auto Profile creation, template context, validation |
| `ai/contracts.py`, `runtime.py` | `Result`/`Prediction` contract; lazy loading, reset after 3 s gap, strict validation (<=30 people, <=32 KB) |
| `ai/adapters.py` | `UnconfiguredDetector`, `TeamDetector` (FYP-2), `LegacyFYP1Detector` |
| `ai/pose_backend.py` | Benchmarks OpenVINO / DirectML / torch once; caches choice in `weights/pose_backend.json` |
| `ai/pose_features.py`, `tracking.py`, `fallnet.py` | Pose parsing, 62-feature windows, multi-person tracker, FallNet v2 loader (`weights_only=True`) |
| `ai/evaluate.py` | Clip-level metrics from CSV |
| `training/*` | GMDCSA-24 annotation parsing, keypoint extraction, LOSO training of Bi-LSTM/CNN-LSTM/Transformer |
| `deploy/static_space/` | Browser SPA: `app.js`, `engine.js`, `pipeline.js` (JS port of tracker/features), `store.js` (localStorage data + demo accounts), ONNX + ORT wasm files |
| `deploy/hf_space/` | Older Python (Gradio-style) Space; superseded by static_space (assumed) |
| Root `*.py` (FYP1) | Legacy engine: MediaPipe-era config, own SQLite (`detection_events`, `alerts`), sound/desktop alerts. Do not modify |

## 3. Routes
| Path | Method | Access | Purpose |
|---|---|---|---|
| `/login/ /logout/ /register/` | | public | Auth |
| `/password/forgot/ /password/reset/` | | public | Reset by code |
| `/account/` | GET/POST | signed in | E-mail/password |
| `/users/`, `/users/<id>/` | GET/POST | Admin | User management |
| `/` | GET | signed in | Monitor wall |
| `/api/status/` | GET | signed in | Cameras, counts, open alerts |
| `/cameras/`, `/cameras/<id>/` | GET | signed in | List / detail |
| `/cameras/new/`, `/cameras/<id>/edit/` | GET/POST | Staff | Camera form |
| `/cameras/<id>/frame/` | POST | Staff | JPEG ingestion |
| `/cameras/<id>/snapshot/` | GET | signed in | Latest frame (memory, <= 6 s old) |
| `/cameras/<id>/demo/` | POST | Staff | Demo incident |
| `/events/`, `/events/<id>/`, `/events/export/` | GET/POST | signed in | List, review, CSV |
| `/reports/?days=1|7|30`, `/notifications/` | GET | signed in | Reports, delivery log |
| `/settings/` | GET/POST | Staff | Thresholds, e-mail |
| `/admin/` | | Django staff | Admin site |

Frame errors: 400 invalid image, 409 disabled/worker source, 415 wrong content type, 503 inference failure.

## 4. Third-party services and runtimes
- **SMTP** (optional; Gmail app password suggested in `.env.example`); console backend by default.
- **Hugging Face Hub / Spaces**: hosts the public static demo; publish needs `HF_TOKEN`. `deploy/publish_space.py` also targets HF.
- **PostgreSQL** (optional).
- **Pretrained model**: Ultralytics YOLOv8n-pose (AGPL-3.0 license consideration for any redistribution/commercial use — unconfirmed for this project).
- **Dataset**: GMDCSA-24 (third-party licence; raw videos excluded from git).
- No external AI API calls; inference is local.

## 5. Key dependencies
Django 5.2.x, Pillow, torch 2.14, torchvision, ultralytics 8.4, numpy, opencv, scikit-learn, scipy, matplotlib, openvino, onnxruntime-directml (Windows), psycopg 3, plyer (legacy), ruff. Locks: `requirements-lock.txt`, `requirements-ai-lock.txt` (Python 3.14.6). Browser: onnxruntime-web (vendored in `deploy/static_space/ort/`).

## 6. AI pipeline details
YOLOv8n-pose (imgsz 640, conf 0.25) -> tracker (stable IDs P1..) -> last 2.0 s of poses resampled by timestamp to 30 steps -> 62 features/step -> FallNet v2 (CNN-LSTM, 224k params, temperature T=0.365) -> normal / warning / fall. Needs >= 1.5 s history. Model version string `fallnet-v2-cnnlstm-9385bb3524`. Weights SHA-256 in `docs/MODEL_CARD.md`.

## 7. Data flow and privacy
- Frames: browser -> server memory -> detector -> discarded; only latest JPEG per camera kept in memory (`LIVE_FRAMES`), never on disk; visible to any signed-in user via `/snapshot/`.
- Stored: Event (score, evidence JSON, model version), Delivery, Audit, TrackState (purged after 1 h), Camera telemetry.
- RTSP URLs/credentials come only from environment variable, never DB/UI.
- Security: CSRF, session cookies HttpOnly + Lax, `X_FRAME_OPTIONS=DENY`, 2 MB upload cap. No HTTPS/secure-cookie settings configured (local use only).

## 8. Operations
Single process (`runserver --noreload`), SQLite WAL or PostgreSQL, one producer per camera, manual worker restart, no job queue (SMTP is synchronous, 10 s timeout). `manage.py check --deploy` has not been satisfied (see KNOWN_ISSUES).
