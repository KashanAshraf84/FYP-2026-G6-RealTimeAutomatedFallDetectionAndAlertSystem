# Architecture and operations

## Why Django

This handoff needs authentication, permissions, persisted incidents, validated forms and schema migrations as well as Python model integration. Django supplies those pieces in one maintainable project. Templates and plain CSS/JavaScript keep frontend setup independent of Node and independent of the ML environment. Core dependencies are Django 5.2 and Pillow; heavyweight ML packages are optional.

## Data flow

Browser webcam / video -> sampled JPEG POST with session + CSRF -> Pillow RGB frame -> detector adapter -> validated predictions -> per-person confirmation/cooldown -> SQLite event -> dashboard record + optional email.

Server USB / trusted RTSP / video -> OpenCV management command -> Pillow RGB -> same detector and ingestion service.

Wall polling reads `/api/status/` every three seconds. Sources are offline after ten seconds without frames. Person counts are only what the adapter returns. Browser video is local to the source page; workers publish telemetry, not MJPEG video. No frame bytes or original files are stored.

## API routes

All application routes require sign-in. Browser POSTs require CSRF. Invalid/missing identifiers produce 404.

| Method | Path | Purpose | Permission |
|---|---|---|---|
| GET | `/api/status/` | Camera heartbeat, current model output, incident counts and latest open alerts | Signed in |
| POST | `/cameras/<id>/frame/` | Raw `image/jpeg`, <=1.5 MB and <=1920×1080 | Staff, enabled browser/video source |
| POST | `/cameras/<id>/demo/` | Manual labelled sample, throttled to one per source per 10 seconds | Staff |
| GET/POST | `/cameras/new/`, `/cameras/<id>/edit/` | Validated camera forms | Staff |
| GET/POST | `/events/<id>/` | Evidence and review outcome | Signed in |
| GET | `/events/` | Search/filter/page through incidents | Signed in |
| GET | `/events/export/` | CSV, same filters; spreadsheet formula prefixes escaped | Signed in |
| GET/POST | `/settings/` | Confidence, confirmation frames, cooldown, email preferences | Staff |
| GET | `/reports/?days=7` | Real events over 1/7/30 local calendar days | Signed in |

Frame response: `predictions`, `model_version`, `ready`, `message`, measured server `latency_ms`, and newly recorded `event_ids`. Latency includes ingestion and synchronous notification attempts; it is not camera-to-caregiver latency. Error responses include `error`, with 400 invalid image, 409 disabled/wrong source, 415 wrong content type, or 503 inference failure. A session expiry redirects to login; the UI detects it and stops capture.

## Database and audit

`Camera` retains current telemetry; `TrackState` retains confirmation/cooldown per camera and person. Track records older than one hour are cleaned during ingestion. `Event` retains model evidence and current human review. `Delivery` records notification attempts. `DetectionSettings` uses singleton row ID 1. `Audit` records camera/settings edits, demo creation and review changes. Audit entries are available through Django admin; login events and every frame are not audit entries.

Use `makemigrations` when changing models, commit the migration, then run `migrate` on teammates' machines. Do not send your local database as source code. The source automatically generates a per-install secret in `instance/.secret` unless `DJANGO_SECRET_KEY` is set. Keep it stable on that installation so sessions remain valid.

## Worker startup

Create/select a source of type **Server camera worker**. Use its numeric ID:

```powershell
.venv\Scripts\python -m pip install -r requirements-ai.txt
$env:GUARDIAN_CAMERA_SOURCE = '0' # server USB camera index
# Or set a trusted RTSP URL or local file path in this environment variable.
$env:GUARDIAN_DETECTOR = 'ai.adapters.TeamDetector'
.venv\Scripts\python manage.py camera_worker --camera 2 --fps 5
```

The source environment variable keeps RTSP credentials out of the database and UI. The application does not print the URL, although external OpenCV backends may emit their own diagnostics. Keep worker logs private. The web UI does not accept arbitrary server file paths or URLs. Stop the worker with Ctrl+C. It releases the capture and marks the camera offline. Network reads may block in OpenCV; after disconnect/EOF the current worker exits, so manually restart it after resolving the source. Automatic reconnect and process supervision are future work.

The database is SQLite by default and PostgreSQL when `GUARDIAN_DB=postgres` (README, "Database"); with PostgreSQL `select_for_update` takes real row locks. Only run one producer per camera. On browser sources, keep capture in a single tab; the server does not arbitrate source ownership. SQLite is suitable for the local FYP setup, not concurrent fleet-scale writers. `select_for_update` does not give row locks on SQLite. For expansion, use PostgreSQL, one owned inference worker per source, a queue, stable timestamps, shared tracker lifecycle and bounded backpressure. Web worker processes have separate Python model instances/state; a multi-process WSGI deployment is not supported for stateful inference without those changes.

## Email

The console backend and disabled email are defaults. Enabling email in Settings with the console backend produces `console_only`, not a delivered claim. To submit SMTP mail, set server environment variables:

```powershell
$env:EMAIL_BACKEND='django.core.mail.backends.smtp.EmailBackend'
$env:EMAIL_HOST='your-smtp-host'
$env:EMAIL_PORT='587'
$env:EMAIL_USE_TLS='1'
$env:EMAIL_HOST_USER='your-username'
$env:EMAIL_HOST_PASSWORD='your-app-password'
$env:DEFAULT_FROM_EMAIL='your-approved-sender'
```

Restart the application and worker. Set an approved recipient and enable email in Settings. Real confirmed falls then attempt SMTP; demos never do. A transport exception records `failed` while keeping the incident. A successful SMTP call records `submitted`, because it cannot prove recipient delivery. Current delivery is synchronous, can add up to the configured 10-second network timeout, and has no retry queue. Move it to a durable background queue before higher-throughput operation. No external messages were sent during project verification.

Browser notifications are opt-in on that browser and only run while a signed-in page stays open. Polling considers the latest six open incidents; bursts, immediate review in another session or suspended tabs can miss browser notices. The persistent event log is authoritative. SMS, GPIO buzzer, background web push and escalation are not implemented.

## Local development and future deployment

Use the supplied localhost development command. Never expose `runserver` or DEBUG on a public host. `.env.example` documents variables; it is not automatically loaded. PowerShell `$env:` settings affect that terminal process and its children.

Before a deployment: configure an actual application server, static-file hosting, HTTPS, secure session/CSRF cookies, allowed hosts, explicit secret management, database backups, authentication rate limiting, access/audit requirements, inference source ownership, notification retries and approved data handling. Run `manage.py check --deploy` against that deployment configuration. Those deployment settings are intentionally not presented as already configured.

## UI changes from the artifact

Retained the GuardianAI name, violet/light palette, dark preview tiles and monitoring-first navigation. Replaced temporary design controls with native accessible forms, persistent review state, role checks, genuine source states, responsive layouts and a theme switch. Removed invented accuracy, delivery, subject count and latency claims, fake RTSP tests, nonfunctional privacy toggles and the simulated phone screen. The same responsive application serves narrow screens.
