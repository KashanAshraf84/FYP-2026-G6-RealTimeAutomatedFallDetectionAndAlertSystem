"""
Fall Detection System - Persistence Layer
==========================================
SQLite storage for detection history and fired alerts. A single file
database (no server process) fits the project's edge-deployment thesis —
a caregiver's device shouldn't need a database server running alongside it.

Schema (see database/schema.sql for the standalone copy):
  detection_events — one row per status change (normal/warning/fall)
  alerts           — one row per alert that actually fired (cooldown-gated),
                      linked back to the detection_event that caused it
"""

import os
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict


SCHEMA = """
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
"""


class Database:
    """Thin wrapper around a single SQLite file. Each call opens and closes
    its own short-lived connection — simplest option for a low-traffic
    single-process app, and avoids holding a lock across the video loop."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def log_detection_event(
        self,
        status: str,
        confidence: float,
        angle: Optional[float] = None,
        speed: Optional[float] = None,
        person_count: int = 0,
    ) -> int:
        """Insert a detection_events row and return its id."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO detection_events "
                "(timestamp, status, confidence, angle, speed, person_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), status, confidence, angle, speed, person_count),
            )
            return cur.lastrowid

    def log_alert(
        self,
        status: str,
        confidence: float,
        person_id: int = 0,
        event_id: Optional[int] = None,
    ) -> int:
        """Insert an alerts row (one real, cooldown-gated alert) and return its id."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO alerts (event_id, timestamp, status, confidence, person_id, acknowledged) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                (event_id, datetime.now().isoformat(), status, confidence, person_id),
            )
            return cur.lastrowid

    def acknowledge_alert(self, alert_id: int) -> bool:
        """Mark an alert as acknowledged. Returns False if no such alert exists."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,)
            )
            return cur.rowcount > 0

    def get_recent_events(self, limit: int = 50) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM detection_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
