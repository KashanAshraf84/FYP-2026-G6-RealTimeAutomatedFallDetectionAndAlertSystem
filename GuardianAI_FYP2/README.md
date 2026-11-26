# GuardianAI — FYP-2

A runnable Django application for camera monitoring, fall-incident review, and AI-model integration.
Built afresh from the FYP-2 design reference; FYP1 was consulted for domain and model context.

**Status:** UI and application workflows implemented. A fall classifier trained on the real GMDCSA-24 videos (`weights/fallnet_v2.pt`) is connected through `ai.adapters.TeamDetector` (the default in `config/settings.py` and `.env`). Training pipeline and subject-disjoint results: [training/README.md](training/README.md) and [docs/MODEL_CARD.md](docs/MODEL_CARD.md). Set `GUARDIAN_DETECTOR=ai.adapters.UnconfiguredDetector` for the preview-only stub that creates no detections.

## Start on Windows

Open PowerShell in this folder. Python 3.12 is a practical starting point for the AI environment; check your chosen ML packages' Python support. The web application was tested with Python 3.14.6.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python manage.py migrate
.venv\Scripts\python manage.py seed_demo
.venv\Scripts\python manage.py createsuperuser
$env:DJANGO_DEBUG = '1'
.venv\Scripts\python manage.py runserver 127.0.0.1:8000 --noreload
```

Open **http://127.0.0.1:8000** and sign in with the account you just created. No shared/default password is included. `seed_demo` creates four named camera configurations, not fabricated events. Rerunning it does not duplicate those names.

After the first setup, `powershell -ExecutionPolicy Bypass -File .\start.ps1` runs the same local setup and server steps. This bypass applies only to that script process. Alternatively use the explicit commands above.

macOS/Linux: use `python3 -m venv .venv`, `.venv/bin/python` in place of the Windows interpreter path, and `export DJANGO_DEBUG=1`.

## Database

SQLite (a file in `instance/`) is the zero-setup default. For PostgreSQL, set these in `.env` and run `manage.py migrate`:

```text
GUARDIAN_DB=postgres
GUARDIAN_DB_NAME=guardianai
GUARDIAN_DB_USER=guardian
GUARDIAN_DB_PASSWORD=<password>
GUARDIAN_DB_HOST=127.0.0.1
GUARDIAN_DB_PORT=5432
GUARDIAN_PG_HOME=<folder holding pgsql\ and data\>   # optional: start.ps1 starts this server when it is not running
```

Any PostgreSQL 14+ works (installer from postgresql.org, or the zip binaries with `initdb`). Create a login role with
`CREATEDB` (needed by `manage.py test`) and a database it owns. To move existing SQLite data:
`set PYTHONUTF8=1`, `manage.py dumpdata --natural-foreign --natural-primary -e contenttypes -e auth.permission -e sessions -o dump.json`
with SQLite, then switch `.env`, `manage.py migrate`, `manage.py loaddata dump.json`. Look at the data with pgAdmin 4
(Servers > Register > host `localhost`, port `5432`, database `guardianai`, user `guardian`), or `psql -h localhost -U guardian guardianai`.
The tables are `auth_user`, `monitor_profile`, `monitor_camera`, `monitor_event`, `monitor_delivery`, `monitor_audit`,
`monitor_detectionsettings`, `monitor_trackstate` and `monitor_passwordresetcode`.

## Accounts and roles

- **First run:** open http://127.0.0.1:8000, click **Create an account**. The first account becomes the **Administrator**; later sign-ups start as **Caregiver**.
- **Roles** (changed only by an Administrator under **Users**):

  | Role | Can do |
  |---|---|
  | Administrator | Everything, plus the Users dashboard (roles, activate/deactivate, create, delete, send reset codes) |
  | Operator | Cameras, start sources, detection settings |
  | Caregiver | Monitor wall, review incidents, reports |

- **Sign in** with username **or** e-mail; "Keep me signed in" lasts 14 days. Five wrong passwords lock that account on that computer for 15 minutes.
- **Forgot password:** enter username or e-mail; a 6-digit code goes to the account's e-mail (valid 15 minutes, 5 tries). With the default console e-mail backend the code is printed in the server window; configure Gmail/SMTP in `.env` (see `.env.example`) to send real e-mails.
- **My account:** change e-mail and password. Set `GUARDIAN_REGISTRATION_APPROVAL=1` to require an Administrator to activate new sign-ups. All account changes go to the audit log.

## Try the working flows

1. Open **Cameras**. Add/edit a source, its location, and enabled state (administrator only).
2. Open a **Browser webcam** source and click **Start camera**. Allow browser camera permission yourself. Use localhost or HTTPS. Click **Stop** to release the camera. No camera starts automatically.
3. For **Local video playback**, choose a browser-supported video and click **Start analysis**. Sampled JPEG frames go to this Django server; the complete video file stays on your device.
4. With `TeamDetector`, every person gets a box with a stable ID (`P1`, `P2`, …). Each ID shows `WARMING UP` for the first ~1.5 s, then `NORMAL` / `WARNING` / `FALL` with a score, and has its own entry in the evidence panel. Alerts are confirmed per person. The AI stack (`requirements-ai.txt`) must be installed. With the stub adapter the UI says **Preview only**.
   Without the web app: `.venv\Scripts\python live_demo.py` (laptop webcam) or `--source clip.mp4 --save out.mp4` draws the same ID boxes in an OpenCV window.
   **Speed and several cameras:** at the first start the server times every runtime this computer has (DirectML on each GPU, OpenVINO on an Intel GPU or the CPU) and keeps the lightest one (`ai/pose_backend.py`, remembered in `weights/pose_backend.json`; delete that file to measure again). On the development laptop that is OpenVINO on the Intel UHD chip at about 21 ms a frame. Several webcam cameras can be live at once: start each on its camera page (pick the device if you have more than one); they all show on the Monitor wall and keep being analysed while you use other pages, until you press Stop. The server tells each camera how long to wait before its next frame so the model is busy at most 65% of the time (`GUARDIAN_AI_DUTY`): one or two cameras get about 6 frames/s each, four about 4 (measured on the development laptop with the server at 45% of one CPU core). Detection is best at 5 frames/s or more, so prefer fewer simultaneous cameras on a slow computer. Set `GUARDIAN_POSE_BACKEND=torch|openvino|directml` to force a runtime.
   **Close-up view:** a person cut off by the bottom of the frame with knees and ankles out of view is labelled `PARTIAL VIEW` and cannot raise an alert, because the model needs the whole body. Stand 2-3 m from the camera.
5. Click **Create demo incident** to exercise the review workflow without a model. Demo incidents are labelled, excluded from reports, and never emailed.
6. Open **Event history**, review the event, change it to Acknowledged / Resolved / False alarm, and save notes. Filter records and export CSV.
7. **Settings** controls the real confidence gate, consecutive-frame confirmation and per-camera/per-person cooldown. **Reports** counts stored real incidents by local calendar date.
8. Use `/admin/` with your superuser account to create caregiver accounts. Non-staff users can view/review; staff can also change cameras/settings and send frames. User provisioning remains a superuser responsibility.

## Live browser workspace

**https://kashan84-guardianai-fyp2.static.hf.space/index.html** — the same workspace (sign-in, monitor wall, cameras, event history and review, notifications, reports, settings) rebuilt as a single-page app in [deploy/static_space/](deploy/static_space/), with the detector running in the browser (ONNX Runtime Web: WebGPU when available, else WebAssembly; a 480 px pose model is used when the device is slow). Demo accounts: `kashan` / `123` (administrator) and `caregiver` / `123`. Incidents, settings and accounts stay in the visitor's browser; no frame is uploaded. Server camera workers (USB/RTSP) exist only in the Django deployment. The JavaScript tracker and features were checked against the Python modules (`deploy/static_space/test_parity.mjs`: identical IDs and labels on 229 frames). Publish with `deploy/publish_static.py` after `hf auth login`.

## What's included

- Responsive light/dark monitoring UI; no Node build or CDN needed.
- Django sessions, password hashing, CSRF protection, role checks, and SQLite migrations.
- Camera configuration, live telemetry with stale-source detection, webcam/video input, several live cameras on the monitor wall.
- SQLite by default, PostgreSQL by configuration.
- Independent server worker for USB/RTSP/file sources.
- Typed, validated model adapter interface and separate AI module.
- Persistent fall incidents, evidence JSON, human review, filtered CSV export, audit records.
- Dashboard notifications, opt-in browser notifications while a page is open, optional SMTP submission logging.
- Sequence-level evaluation utility with confusion matrix and per-class precision/recall/F1.
- Automated tests and detailed AI integration guide.

## Project map

```text
config/                     Django configuration, URLs, WSGI
monitor/models.py           Cameras, events, delivery records, settings, track state
monitor/services.py         Fall confirmation, cooldown, notifications
monitor/views.py            Pages and authenticated endpoints
monitor/forms.py            Server-side validation
monitor/migrations/         Versioned database schema
monitor/management/commands/ Camera worker and seed command
ai/contracts.py             Frame-to-prediction interface
ai/adapters.py              Default adapter and teammate implementation slot
ai/runtime.py               Lazy model loading, reset, output validation
ai/evaluate.py              Held-out prediction metrics
static/                     CSS and browser JavaScript
templates/                 Django page templates
data/example_predictions.csv Illustrative evaluation input, not a dataset
docs/AI_HANDOFF.md          Start here for model work
docs/ARCHITECTURE.md        Data flow and API reference
docs/ACCEPTANCE.md          Verification results and manual checklist
docs/MODEL_CARD_TEMPLATE.md Model evidence template
weights/                   Local weights, excluded from version control
instance/                  Generated local database and secret, not for sharing
```

For the exact tested core dependency set, install `requirements-lock.txt` instead of `requirements.txt`. Optional AI packages remain separate.

## Run checks

```powershell
.venv\Scripts\python manage.py check
.venv\Scripts\python manage.py test monitor
.venv\Scripts\python -m unittest ai.test_tracking
.venv\Scripts\python -m ai.evaluatedata/example_predictions.csv --outputinstance/example-metrics.json
```

The example CSV is invented solely to show the evaluation format. Its metrics are **not GuardianAI performance**.

## Important boundaries

This is a local FYP development handoff, not a deployment certified for unattended care. There are no validated model accuracy claims. No old FYP1 weights were silently reused.

- Model training, multi-person identity tracking, pose skeletons, and explanation features belong inside the teammate's adapter. The UI already renders returned person boxes and evidence.
- Browser analysis samples at at most roughly 5 FPS, reduced by inference/network time. It is not a 30 FPS temporal-model pipeline. Match training/inference sampling before evaluating.
- The wall displays telemetry, not a multi-camera video mosaic. Browser video appears on its source page; RTSP worker video is not streamed back to the browser.
- Raw frames are not retained. Face blurring, evidence clips and retention controls are therefore not offered as pretend-working settings.
- Warnings appear as model telemetry; the current service persists confirmed fall alerts only.
- SMTP is off by default; console output is explicitly labelled `console_only`. SMTP success means submitted, not delivered. SMS, GPIO buzzers, escalation and background mobile push remain future work.
- SQLite and in-process inference target a small local demo. Run one producer per camera. Do not scale to multiple web/worker processes per source without centralizing tracker state and job ownership. See architecture notes.

Before sharing, send the clean ZIP or source folder excluding `.venv`, `instance`, private datasets, credentials and unapproved recordings. The supplied ZIP already excludes local test data.

Framework references: [Django 5.2 documentation](https://docs.djangoproject.com/en/5.2/) · [Django deployment checklist](https://docs.djangoproject.com/en/5.2/howto/deployment/checklist/).
