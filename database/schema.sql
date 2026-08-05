-- GuardianAI - Database Schema
-- ==============================
-- Standalone copy of the schema embedded in database.py (kept identical on
-- purpose - this file is for repo visibility / manual inspection, the
-- application itself runs the same statements from database.py on startup.

CREATE TABLE IF NOT EXISTS detection_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('normal', 'warning', 'fall')),
    confidence REAL NOT NULL,
    angle REAL,
    speed REAL,
    person_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER REFERENCES detection_events(id),
    timestamp TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL NOT NULL,
    person_id INTEGER NOT NULL DEFAULT 0,
    acknowledged INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_detection_events_timestamp ON detection_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
