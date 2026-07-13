"""SQLite-backed persistence.

Return cases, registered orders, customer profiles, idempotency keys, and
the payment outbox. Documents are stored as JSON blobs with the queryable
fields denormalized into columns.

Thread-safety: a single connection is shared across request/worker threads
(and the review board's parallel graph nodes), and SQLite connections are
not safe for concurrent cursor use — so EVERY query, reads included, goes
through the store lock. Case saves additionally use optimistic locking:
every write must present the version it read, and the version increments
on each successful write, so concurrent read-modify-write cycles cannot
silently clobber each other.
"""
from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from .models import CustomerProfile, Order, ReturnCase, ReturnStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS return_cases (
    id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    status TEXT NOT NULL,
    tracking_number TEXT,
    version INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cases_status ON return_cases(status);
CREATE INDEX IF NOT EXISTS idx_cases_order ON return_cases(order_id);
CREATE INDEX IF NOT EXISTS idx_cases_tracking ON return_cases(tracking_number);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    body TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    body TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS idempotency_keys (
    key TEXT PRIMARY KEY,
    case_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS payment_outbox (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    kind TEXT NOT NULL,           -- refund | store_credit
    amount REAL NOT NULL,
    status TEXT NOT NULL,          -- pending | executed | failed
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TEXT NOT NULL,
    executed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_outbox_status ON payment_outbox(status);
"""


class ConcurrencyError(Exception):
    """Case was modified by another writer since it was read."""


class ReturnStore:
    def __init__(self, path: str = ":memory:"):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)

    # -- locked query helpers ------------------------------------------------

    def _fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def _fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # -- return cases (optimistic locking) --------------------------------

    def save(self, case: ReturnCase) -> None:
        """Insert or CAS-update; bumps case.version on success."""
        with self._lock:
            expected = case.version
            case.version = expected + 1
            body = case.model_dump_json()
            cur = self._conn.execute(
                "UPDATE return_cases SET order_id=?, status=?, tracking_number=?, "
                "version=?, updated_at=?, body=? WHERE id=? AND version=?",
                (
                    case.order_id,
                    case.status.value,
                    case.label_tracking_number,
                    case.version,
                    case.updated_at.isoformat(),
                    body,
                    case.id,
                    expected,
                ),
            )
            if cur.rowcount == 0:
                exists = self._conn.execute(
                    "SELECT 1 FROM return_cases WHERE id=?", (case.id,)
                ).fetchone()
                if exists:
                    case.version = expected  # roll back the local bump
                    self._conn.rollback()
                    raise ConcurrencyError(
                        f"case {case.id} was modified concurrently "
                        f"(expected version {expected})"
                    )
                self._conn.execute(
                    "INSERT INTO return_cases "
                    "(id, order_id, status, tracking_number, version, updated_at, body) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        case.id,
                        case.order_id,
                        case.status.value,
                        case.label_tracking_number,
                        case.version,
                        case.updated_at.isoformat(),
                        body,
                    ),
                )
            self._conn.commit()

    def get(self, case_id: str) -> ReturnCase | None:
        row = self._fetchone("SELECT body FROM return_cases WHERE id = ?", (case_id,))
        return ReturnCase.model_validate_json(row["body"]) if row else None

    def get_by_tracking(self, tracking_number: str) -> ReturnCase | None:
        row = self._fetchone(
            "SELECT body FROM return_cases WHERE tracking_number = ?",
            (tracking_number,),
        )
        return ReturnCase.model_validate_json(row["body"]) if row else None

    def list(self, status: ReturnStatus | None = None) -> list[ReturnCase]:
        if status:
            rows = self._fetchall(
                "SELECT body FROM return_cases WHERE status = ? ORDER BY rowid",
                (status.value,),
            )
        else:
            rows = self._fetchall("SELECT body FROM return_cases ORDER BY rowid")
        return [ReturnCase.model_validate_json(r["body"]) for r in rows]

    def list_by_order(self, order_id: str) -> list[ReturnCase]:
        rows = self._fetchall(
            "SELECT body FROM return_cases WHERE order_id = ? ORDER BY rowid",
            (order_id,),
        )
        return [ReturnCase.model_validate_json(r["body"]) for r in rows]

    def status_counts(self) -> dict[str, int]:
        rows = self._fetchall(
            "SELECT status, COUNT(*) AS n FROM return_cases GROUP BY status"
        )
        return {r["status"]: r["n"] for r in rows}

    # -- order / customer registry -----------------------------------------

    def register_order(self, order: Order, customer: CustomerProfile) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO orders (order_id, customer_id, body) VALUES (?, ?, ?) "
                "ON CONFLICT(order_id) DO UPDATE SET customer_id=excluded.customer_id, "
                "body=excluded.body",
                (order.order_id, order.customer_id, order.model_dump_json()),
            )
            self._conn.execute(
                "INSERT INTO customers (customer_id, body) VALUES (?, ?) "
                "ON CONFLICT(customer_id) DO UPDATE SET body=excluded.body",
                (customer.customer_id, customer.model_dump_json()),
            )
            self._conn.commit()

    def get_order(self, order_id: str) -> Order | None:
        row = self._fetchone("SELECT body FROM orders WHERE order_id = ?", (order_id,))
        return Order.model_validate_json(row["body"]) if row else None

    def get_customer(self, customer_id: str) -> CustomerProfile | None:
        row = self._fetchone(
            "SELECT body FROM customers WHERE customer_id = ?", (customer_id,)
        )
        return CustomerProfile.model_validate_json(row["body"]) if row else None

    # -- idempotency ---------------------------------------------------------

    def idempotency_lookup(self, key: str) -> str | None:
        row = self._fetchone(
            "SELECT case_id FROM idempotency_keys WHERE key = ?", (key,)
        )
        return row["case_id"] if row else None

    def idempotency_record(self, key: str, case_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO idempotency_keys (key, case_id) VALUES (?, ?)",
                (key, case_id),
            )
            self._conn.commit()

    # -- payment outbox --------------------------------------------------------

    def outbox_enqueue(self, case_id: str, kind: str, amount: float) -> str:
        entry_id = f"pay_{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._conn.execute(
                "INSERT INTO payment_outbox (id, case_id, kind, amount, status, created_at) "
                "VALUES (?, ?, ?, ?, 'pending', ?)",
                (entry_id, case_id, kind, amount, datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()
        return entry_id

    def outbox_pending(self) -> list[dict]:
        rows = self._fetchall(
            "SELECT * FROM payment_outbox WHERE status = 'pending' ORDER BY created_at"
        )
        return [dict(r) for r in rows]

    def outbox_mark(self, entry_id: str, status: str, error: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE payment_outbox SET status=?, attempts=attempts+1, last_error=?, "
                "executed_at=CASE WHEN ?='executed' THEN ? ELSE executed_at END WHERE id=?",
                (
                    status,
                    error,
                    status,
                    datetime.now(timezone.utc).isoformat(),
                    entry_id,
                ),
            )
            self._conn.commit()

    def outbox_totals(self) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(amount), 0) AS total "
                "FROM payment_outbox WHERE status = 'executed'"
            ).fetchone()
            pending = self._conn.execute(
                "SELECT COUNT(*) AS n FROM payment_outbox WHERE status = 'pending'"
            ).fetchone()
        return {
            "executed_count": row["n"],
            "executed_total": row["total"],
            "pending_count": pending["n"],
        }
