"""Idempotency, optimistic locking, and payment outbox tests."""
import pytest

from app.models import ReturnStatus
from app.store import ConcurrencyError, ReturnStore
from app.orchestrator import ReturnsOrchestrator
from tests.conftest import make_request, register_order


def test_idempotent_create_returns_same_case(orch):
    register_order(orch, quantity=1)
    first = orch.create_return(make_request(), idempotency_key="idem-1")
    replay = orch.create_return(make_request(), idempotency_key="idem-1")
    assert replay.id == first.id
    assert len(orch.store.list()) == 1


def test_different_keys_create_different_cases(orch):
    register_order(orch, quantity=2)
    a = orch.create_return(make_request(), idempotency_key="idem-a")
    b = orch.create_return(make_request(), idempotency_key="idem-b")
    assert a.id != b.id


def test_optimistic_locking_detects_conflict(orch):
    register_order(orch)
    case = orch.create_return(make_request())
    stale = orch.store.get(case.id)
    fresh = orch.store.get(case.id)
    fresh.add_event("system", "touch")
    orch.store.save(fresh)  # bumps version
    stale.add_event("system", "conflicting-write")
    with pytest.raises(ConcurrencyError):
        orch.store.save(stale)


def test_version_survives_roundtrip(orch):
    register_order(orch)
    case = orch.create_return(make_request())
    loaded = orch.store.get(case.id)
    assert loaded.version == case.version > 0


class FlakyPayments:
    """Fails the first attempt, succeeds after."""

    def __init__(self):
        self.attempts = 0
        self.refunds = []
        self.credits = []

    def refund(self, case, amount, idempotency_key):
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("payment gateway timeout")
        self.refunds.append((case.id, amount, idempotency_key))

    def issue_store_credit(self, case, amount, idempotency_key):
        self.credits.append((case.id, amount, idempotency_key))


def test_outbox_records_before_payment_and_retries():
    payments = FlakyPayments()
    orch = ReturnsOrchestrator(ReturnStore(":memory:"), payments=payments)
    register_order(orch)
    case = orch.create_return(make_request())
    orch.carrier_update(case.label_tracking_number, "delivered")
    case = orch.record_inspection(case.id, passed=True, agent="wh-1")

    # First attempt failed, but the intent + case state are durable.
    assert case.status is ReturnStatus.REFUNDED
    assert payments.refunds == []
    pending = orch.store.outbox_pending()
    assert len(pending) == 1
    assert pending[0]["attempts"] == 1
    assert "timeout" in pending[0]["last_error"]

    # Retry executes it exactly once, with the outbox id as idempotency key.
    result = orch.flush_outbox()
    assert result == {"executed": 1, "failed": 0}
    assert len(payments.refunds) == 1
    assert payments.refunds[0][2] == pending[0]["id"]
    assert orch.store.outbox_pending() == []


def test_outbox_totals(orch):
    register_order(orch, quantity=2, unit_price=40.0)
    for key in ("k1", "k2"):
        case = orch.create_return(make_request(quantity=1), idempotency_key=key)
        orch.carrier_update(case.label_tracking_number, "delivered")
        orch.record_inspection(case.id, passed=True, agent="wh-1")
    totals = orch.store.outbox_totals()
    assert totals["executed_count"] == 2
    assert totals["executed_total"] == 80.0
    assert totals["pending_count"] == 0
