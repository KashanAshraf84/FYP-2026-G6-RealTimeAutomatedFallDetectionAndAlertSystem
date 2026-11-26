# Known Issues, TODOs and Risky Areas

No `TODO/FIXME` comments exist in the code. The items below come from reading the code and the project's own docs.
Severity: **H**igh / **M**edium / **L**ow. Nothing here has been fixed (analysis only).

## A. Security and privacy
| # | Sev | Issue |
|---|---|---|
| S1 | H | `DJANGO_DEBUG` defaults to **1** (`settings.py`). Any deployment that forgets to set it to 0 runs in DEBUG. |
| S2 | M | Public Hugging Face demo uses hard-coded accounts `kashan`/`123` (admin) and `caregiver`/`123` in `deploy/static_space/store.js` and in both READMEs. Intentional demo-only risk; **never reuse these credentials** anywhere real. |
| S3 | M | No HTTPS-related settings (`SECURE_*`, secure cookies, HSTS); `manage.py check --deploy` would fail. `ALLOWED_HOSTS` default includes `testserver`. |
| S4 | M | `/cameras/<id>/snapshot/` serves the latest live JPEG (identifiable people) to **any** signed-in user, including Caregivers and self-registered accounts (open registration by default, approval off). |
| S5 | M | Self-registration is open by default (`GUARDIAN_REGISTRATION_APPROVAL=0`) and grants Caregiver access to incident data. |
| S6 | M | Login lockout and reset throttle use Django's default per-process local-memory cache: resets on restart, not shared across workers; keyed by username+IP only. |
| S7 | L | Password-reset and registration reveal little, but reset flow always redirects (good); username enumeration via register form is possible. |
| S8 | L | Reset code stored as HMAC keyed with `SECRET_KEY`; losing `instance/.secret` invalidates pending codes and all sessions. |
| S9 | L | `.env` is parsed by hand in `settings.py` and again in `start.ps1`; quoting rules differ between them. |

## B. Fragile / architectural
| # | Sev | Issue |
|---|---|---|
| F1 | H | State is per process (trackers, `LIVE_FRAMES`, cache). Multi-process WSGI or several workers will break tracking, previews and lockouts. Documented as unsupported. |
| F2 | M | One global inference lock: all cameras serialize on one model; throughput drops with more cameras. |
| F3 | M | Camera worker does not auto-reconnect; exits on EOF/disconnect; no supervisor. Two workers for one camera are not prevented. Worker frames do not feed the wall snapshot. |
| F4 | M | Email is synchronous inside the frame request (up to 10 s), no retry or queue; "submitted" is not delivery. A slow SMTP stalls the camera pipeline. |
| F5 | M | `ingest()` runs `select_for_update`; on SQLite this gives no row locks (relies on `IMMEDIATE` transactions). Concurrent cameras can still contend ("database is locked" mitigated by 20 s timeout). |
| F6 | M | Warm-up thread starts only when `"runserver" in sys.argv`; under other servers (gunicorn/waitress) the first frame pays model load time, and failures are silently swallowed. |
| F7 | M | `TIME_ZONE = "UTC"`: "today"/report days are UTC days, not the hospital's local days, despite labels saying "local". |
| F8 | M | `TeamDetector` default in settings means a missing/incompatible torch/ultralytics install gives HTTP 503 on every frame (while `.env.example` suggests the stub). Settings vs `.env.example` defaults disagree. |
| F9 | L | `pose_backend.json` caches the fastest runtime per machine; copying `weights/` to another PC may select a wrong backend until deleted. `kernel.errors.txt` in root records OpenVINO/Intel GPU kernel header errors (cause not documented). |
| F10 | L | `start.ps1` reinstalls `requirements-ai.txt` (large) on every launch and does not stop on seed failure. `requirements-ai.txt` contains mojibake characters in comments (encoding issue, harmless). |
| F11 | L | Pinned for Python 3.14.6 / Windows only; README says Python 3.12 is "practical" for the AI stack. Portability untested. |

## C. Functional gaps and product risks
- Model quality: 8.6% of normal clips raise a false alarm; 10 of 79 falls missed (5 FPS). Only 4 subjects, staged falls, one dataset. Weak on partial views, multiple people overlap, slow slides, camera angles unseen in training. "Warning" class F1 is only 0.60.
- Only falls are persisted; warnings are not. Event model allows `warning` but nothing creates it.
- Roadmap items **not implemented**: threaded RTSP reader, auto-reconnect, RTSP URL in UI, MJPEG preview, post-fall immobility check, bed/sofa exclusion zones, ROC/PR report in UI, SMS/buzzer/push/escalation.
- No retention policy, evidence clip, or face blurring; no consent/privacy mechanism needed for hospital pilot.
- Only one alert e-mail recipient; no on-call or escalation.
- Caregivers cannot start browser cameras (frame endpoint is staff-only) though the README describes Caregiver as "Monitor wall"; README also says camera editing is "administrator only" while Operators may do it. Docs and code disagree.
- Browser alerts only fire while a page is open; polling looks at the latest 6 open incidents, so bursts can be missed.
- `data/example_predictions.csv` is invented demo data; its metrics are not performance.

## D. Code hygiene / documentation drift
- Twelve legacy FYP1 files in the repo root (MediaPipe-era config, separate SQLite schema, `database.py` mentions a `database/schema.sql` that does not exist). Marked "do not touch"; `LegacyFYP1Detector` still imports `inference.py` and `fyp1_config.py`.
- `deploy/hf_space/` (older demo) vs `deploy/static_space/` (current): status of the older one unclear. `deploy/static_space/app.css` is a copy of `static/app.css` (overwritten by `publish_static.py`), so edits to the copy are lost.
- `data/dataset/README.md` has broken characters (`alls/`, `dl/`, "GuardianAI_Merged" root) from lost escapes/old naming.
- README test instruction `python -m ai.evaluatedata/example_predictions.csv --outputinstance/...` is missing spaces (typo).
- `docs/ACCEPTANCE.md` says 18 tests (21 Sep 2026); code now has 36 + 13 tracking tests. Docs are outdated.
- JavaScript tracker/features duplicate the Python ones; they must be kept identical (parity test covers 229 frames only).
- `ROADMAP_FYP.md` sprint dates run to 2026-11-25 to 2026-12; project date today is 2026-11-24, so final testing and the paper deadline are imminent.
- No git repository in the folder (no history, no backup) and no CI. Large binaries (ONNX/WASM, ~85 MB in static_space) would bloat a repo.
- AGPL/licence questions: Ultralytics YOLO is AGPL-3.0; GMDCSA-24 has its own licence; static_space README declares `cc-by-nc-4.0`. Not reviewed.

## E. Things not verified
No physical webcam/RTSP device tested (per ACCEPTANCE.md); no run of tests was done while writing these docs; no live DB was inspected (none present).
