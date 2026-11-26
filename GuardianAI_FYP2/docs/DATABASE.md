# Database — GuardianAI

## PROTECTED databases
| Database | Engine | Status | Rule |
|---|---|---|---|
| **PostgreSQL `guardianai`** (user `guardian`, default host `127.0.0.1:5432`; actual host/credentials NOT in the repo and unconfirmed) | PostgreSQL | **PROTECTED** (confirmed by project owner) | Never delete, flush, drop, truncate, restore over, reset, or run destructive/backward migrations. Never read or print its password. Any schema change needs explicit approval and a backup first. |

Not protected (per owner): local SQLite files `instance/guardianai.db` / `instance/guardian.sqlite3`. They do not currently exist on disk. Still avoid deleting them without asking, because they may hold local demo accounts and incidents.

Other stores (not databases, but also do not wipe): `instance/.secret` (Django secret; losing it invalidates sessions and stored reset-code hashes), `weights/pose_backend.json`, `data/keypoints/` (expensive cache, ~20 min to rebuild).

Caution: `manage.py test` on PostgreSQL creates and destroys `test_guardianai`, not `guardianai`, but only if settings/env point where you expect. Check `GUARDIAN_DB*` env vars before running tests.

## Engine selection (`config/settings.py`)
- Default: SQLite, file `instance/guardian.sqlite3` if it exists, else `instance/guardianai.db`; override with `GUARDIAN_SQLITE_PATH`. WAL, 20 s timeout, IMMEDIATE transactions.
- PostgreSQL when `GUARDIAN_DB=postgres` (`GUARDIAN_DB_NAME/USER/PASSWORD/HOST/PORT`; `CONN_MAX_AGE=60`). `start.ps1` can start a local PG server from `GUARDIAN_PG_HOME`.
- Migrations: `monitor/migrations/0001_initial.py`, `0002_accounts.py`.

## Tables (Django app `monitor` + Django built-ins)
Built-ins used: `auth_user`, `django_session`, `django_admin_log`, `django_content_type`, `auth_permission`, `auth_group`.

| Table | Key fields | Notes |
|---|---|---|
| `monitor_camera` | name(80), location(100), kind(browser/video/worker), enabled, last_seen, last_status(24), last_result JSON | Live telemetry stored on the row |
| `monitor_detectionsettings` | confidence 0.5–1 (0.80), consecutive_frames 1–15 (3), cooldown_seconds 1–600 (30), email_enabled, email_recipient | Singleton, pk=1 via `current()` |
| `monitor_trackstate` | camera FK (CASCADE), track_id(80), streak, last_seen, last_alert | Unique (camera, track_id); rows older than 1 h purged on ingest |
| `monitor_event` | camera FK (**PROTECT**), created_at (indexed), track_id, label(fall/warning), confidence, model_version, evidence JSON, is_demo (indexed), status(open/acknowledged/resolved/false_alarm), updated_by FK (SET_NULL), updated_at, notes | Ordered newest first. Core business record |
| `monitor_delivery` | event FK (CASCADE), created_at, channel(20), status(30), detail(250) | Notification attempts (dashboard / email) |
| `monitor_profile` | user 1:1 (CASCADE), role(admin/operator/caregiver) | `save()` syncs `is_staff`/`is_superuser` |
| `monitor_passwordresetcode` | user FK (CASCADE), code_hash (HMAC-SHA256 with SECRET_KEY), created_at, attempts, used | 15 min life, 5 attempts |
| `monitor_audit` | created_at, actor FK (SET_NULL), action(80), detail(250) | Append-only by convention (no UI to delete) |

## Relationships
```
auth_user 1--1 monitor_profile
auth_user 1--* monitor_passwordresetcode
auth_user 1--* monitor_audit (actor, nullable)
auth_user 1--* monitor_event (updated_by, nullable)
monitor_camera 1--* monitor_event (PROTECT)
monitor_camera 1--* monitor_trackstate (CASCADE)
monitor_event  1--* monitor_delivery (CASCADE)
```
Deleting a user keeps events/audit (FK set to NULL). Deleting a camera cascades to TrackState but is blocked if events exist.

## Sensitive data held
Usernames, e-mails, password hashes, session data, incident evidence (pose-derived numbers; no images), audit trail. Identifiable video is **not** stored.

## Other (legacy / non-Django) stores
- **Legacy FYP1 SQLite** (`database.py`, default path `data/guardianai.db` in `fyp1_config.py`): tables `detection_events(id, timestamp, status, confidence, angle, speed, person_count)` and `alerts(id, event_id->detection_events, timestamp, status, confidence, person_id, acknowledged)`. Used only by the legacy engine; not by the Django app. `data/*.db` is git-ignored. File currently absent. Note it is a different file name pattern from Django's but similar (`guardianai.db`) — **path collision risk**: Django's fallback is `instance/guardianai.db`, legacy is `data/guardianai.db`; keep them separate.
- **Browser app** (`deploy/static_space/store.js`): localStorage only, per visitor.

## Seed data
`manage.py seed_demo` creates 4 cameras (Bed bay 1–4, Corridor east, Day room, Bed bay 7) and the settings row. Idempotent. No users or events are seeded.

## Backup / migration notes
SQLite -> PostgreSQL: `dumpdata --natural-foreign --natural-primary -e contenttypes -e auth.permission -e sessions` (with `PYTHONUTF8=1`), switch env, `migrate`, `loaddata`. Do this only with approval and a backup of the target.
