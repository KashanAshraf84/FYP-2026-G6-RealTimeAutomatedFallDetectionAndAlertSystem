"""
GuardianAI - Database Seed Script
===================================
Inserts a handful of sample detection_events and alerts so the database has
inspectable structure before the first live camera session (e.g. for the
coordinator to see table contents without waiting for a real fall). Safe to
run multiple times -- it only adds rows, it does not clear existing data.

Usage:
    python database/seed.py
"""

import os
import sys
from datetime import datetime, timedelta

# config.py/database.py live one level up in the FYP1 working copy, but under
# src/ in the git repo copy -- add both so this script runs from either.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))

from config import CONFIG
from database import Database


def seed():
    db = Database(CONFIG.database_path)
    now = datetime.now()

    # A plausible timeline: normal -> a brief warning (lean) -> back to
    # normal -> a fall that triggers a real alert -> recovery.
    sample_events = [
        (now - timedelta(minutes=12), "normal", 0.97, 88.0, 1.2, 1),
        (now - timedelta(minutes=9), "warning", 0.62, 55.0, 8.4, 1),
        (now - timedelta(minutes=8), "normal", 0.95, 84.0, 0.9, 1),
        (now - timedelta(minutes=5), "fall", 0.91, 41.0, 62.0, 1),
        (now - timedelta(minutes=3), "warning", 0.8, 47.0, 3.1, 1),
        (now - timedelta(minutes=1), "normal", 0.98, 87.0, 1.0, 1),
    ]

    event_ids = []
    for ts, status, confidence, angle, speed, person_count in sample_events:
        with db._connect() as conn:
            cur = conn.execute(
                "INSERT INTO detection_events "
                "(timestamp, status, confidence, angle, speed, person_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ts.isoformat(), status, confidence, angle, speed, person_count),
            )
            event_ids.append(cur.lastrowid)

    # One fall alert linked to the fall event above, left unacknowledged so
    # the "Acknowledge" button in the dashboard has something to demonstrate.
    fall_event_id = event_ids[3]
    with db._connect() as conn:
        conn.execute(
            "INSERT INTO alerts (event_id, timestamp, status, confidence, person_id, acknowledged) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (fall_event_id, (now - timedelta(minutes=5)).isoformat(), "fall", 0.91, 0),
        )

    print(f"Seeded {len(sample_events)} detection_events and 1 alert into {CONFIG.database_path}")


if __name__ == "__main__":
    seed()
