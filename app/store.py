"""SQLite-backed persistence for return cases.

Cases are stored as JSON blobs keyed by id, with status and order id
denormalized into columns for querying. Uses stdlib sqlite3 to keep the
dependency footprint small.
"""
from __future__ import annotations

import sqlite3
import threading

from .models import ReturnCase, ReturnStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS return_cases (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    status TEXT NOT NULL,
    tracking_number TEXT,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cases_status ON return_cases(status);
CREATE INDEX IF NOT EXISTS idx_cases_tracking ON return_cases(tracking_number);
"""


class ReturnStore:
    def __init__(self, path: str = ":memory:"):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)

    def save(self, case: ReturnCase) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO return_cases (id, order_id, status, tracking_number, body) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET status=excluded.status, "
                "tracking_number=excluded.tracking_number, body=excluded.body",
                (
                    case.id,
                    case.order_id,
                    case.status.value,
                    case.label_tracking_number,
                    case.model_dump_json(),
                ),
            )
            self._conn.commit()

    def get(self, case_id: str) -> ReturnCase | None:
        row = self._conn.execute(
            "SELECT body FROM return_cases WHERE id = ?", (case_id,)
        ).fetchone()
        return ReturnCase.model_validate_json(row[0]) if row else None

    def get_by_tracking(self, tracking_number: str) -> ReturnCase | None:
        row = self._conn.execute(
            "SELECT body FROM return_cases WHERE tracking_number = ?",
            (tracking_number,),
        ).fetchone()
        return ReturnCase.model_validate_json(row[0]) if row else None

    def list(self, status: ReturnStatus | None = None) -> list[ReturnCase]:
        if status:
            rows = self._conn.execute(
                "SELECT body FROM return_cases WHERE status = ? ORDER BY rowid",
                (status.value,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT body FROM return_cases ORDER BY rowid"
            ).fetchall()
        return [ReturnCase.model_validate_json(r[0]) for r in rows]
