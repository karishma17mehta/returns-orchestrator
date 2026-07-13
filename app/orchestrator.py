"""Return lifecycle orchestration.

Owns the state machine and coordinates the side effects at each step
(decision engine, carrier label, refund/exchange execution). External
integrations (carrier, payments) are injected so real adapters can replace
the built-in stubs.

State machine:

    requested ──▶ approved ──▶ label_issued ──▶ in_transit ──▶ received
        │             ▲                                            │
        ├──▶ manual_review ──▶ rejected                       inspecting
        └──▶ rejected                                              │
                          ┌────────────────┬──────────────────────┤
                          ▼                ▼                      ▼
                       refunded        exchanged / credited   closed_failed_inspection
"""
from __future__ import annotations

import uuid
from typing import Protocol

from . import policy as policy_engine
from .models import ReturnCase, ReturnRequestCreate, ReturnStatus, Resolution
from .policy import ReturnPolicy
from .store import ReturnStore

_TRANSITIONS: dict[ReturnStatus, set[ReturnStatus]] = {
    ReturnStatus.REQUESTED: {
        ReturnStatus.APPROVED,
        ReturnStatus.REJECTED,
        ReturnStatus.MANUAL_REVIEW,
    },
    ReturnStatus.MANUAL_REVIEW: {ReturnStatus.APPROVED, ReturnStatus.REJECTED},
    ReturnStatus.APPROVED: {ReturnStatus.LABEL_ISSUED},
    ReturnStatus.LABEL_ISSUED: {ReturnStatus.IN_TRANSIT, ReturnStatus.RECEIVED},
    ReturnStatus.IN_TRANSIT: {ReturnStatus.RECEIVED},
    ReturnStatus.RECEIVED: {ReturnStatus.INSPECTING},
    ReturnStatus.INSPECTING: {
        ReturnStatus.REFUNDED,
        ReturnStatus.EXCHANGED,
        ReturnStatus.CREDITED,
        ReturnStatus.CLOSED_FAILED_INSPECTION,
    },
}

_RESOLUTION_TO_STATUS = {
    Resolution.REFUND: ReturnStatus.REFUNDED,
    Resolution.EXCHANGE: ReturnStatus.EXCHANGED,
    Resolution.STORE_CREDIT: ReturnStatus.CREDITED,
}


class TransitionError(Exception):
    """Raised when a requested state transition is not allowed."""


class CarrierClient(Protocol):
    def create_label(self, case: ReturnCase) -> str: ...


class PaymentClient(Protocol):
    def refund(self, case: ReturnCase, amount: float) -> None: ...
    def issue_store_credit(self, case: ReturnCase, amount: float) -> None: ...


class StubCarrier:
    def create_label(self, case: ReturnCase) -> str:
        return f"TRK{uuid.uuid4().hex[:10].upper()}"


class StubPayments:
    def __init__(self) -> None:
        self.refunds: list[tuple[str, float]] = []
        self.credits: list[tuple[str, float]] = []

    def refund(self, case: ReturnCase, amount: float) -> None:
        self.refunds.append((case.id, amount))

    def issue_store_credit(self, case: ReturnCase, amount: float) -> None:
        self.credits.append((case.id, amount))


class ReturnsOrchestrator:
    def __init__(
        self,
        store: ReturnStore,
        policy: ReturnPolicy | None = None,
        carrier: CarrierClient | None = None,
        payments: PaymentClient | None = None,
    ):
        self.store = store
        self.policy = policy or ReturnPolicy()
        self.carrier = carrier or StubCarrier()
        self.payments = payments or StubPayments()

    # -- lifecycle entry point -------------------------------------------

    def create_return(self, request: ReturnRequestCreate) -> ReturnCase:
        case = ReturnCase(**request.model_dump())
        case.add_event("customer", "return_requested", f"reason={request.reason.value}")

        decision = policy_engine.evaluate(request, self.policy)
        case.decision_notes = decision.notes
        self._transition(case, decision.status, actor="system", detail="; ".join(decision.notes))

        if case.status is ReturnStatus.APPROVED:
            self._issue_label(case)
        self.store.save(case)
        return case

    # -- agent actions ----------------------------------------------------

    def review(self, case_id: str, approve: bool, agent: str, note: str | None = None) -> ReturnCase:
        case = self._require(case_id)
        target = ReturnStatus.APPROVED if approve else ReturnStatus.REJECTED
        self._transition(case, target, actor=f"agent:{agent}", detail=note)
        if approve:
            self._issue_label(case)
        self.store.save(case)
        return case

    def record_inspection(
        self, case_id: str, passed: bool, agent: str, note: str | None = None
    ) -> ReturnCase:
        case = self._require(case_id)
        if case.status is ReturnStatus.RECEIVED:
            self._transition(case, ReturnStatus.INSPECTING, actor=f"agent:{agent}")
        if case.status is not ReturnStatus.INSPECTING:
            raise TransitionError(
                f"cannot inspect case in status '{case.status.value}'"
            )
        if passed:
            self._settle(case, actor=f"agent:{agent}", note=note)
        else:
            self._transition(
                case,
                ReturnStatus.CLOSED_FAILED_INSPECTION,
                actor=f"agent:{agent}",
                detail=note or "inspection failed",
            )
        self.store.save(case)
        return case

    # -- carrier webhook ---------------------------------------------------

    def carrier_update(self, tracking_number: str, event: str) -> ReturnCase:
        case = self.store.get_by_tracking(tracking_number)
        if case is None:
            raise KeyError(f"no return with tracking number {tracking_number}")
        mapping = {
            "picked_up": ReturnStatus.IN_TRANSIT,
            "delivered": ReturnStatus.RECEIVED,
        }
        target = mapping.get(event)
        if target is None:
            case.add_event("carrier", f"tracking_{event}")
        else:
            self._transition(case, target, actor="carrier", detail=f"tracking event: {event}")
        self.store.save(case)
        return case

    # -- internals ----------------------------------------------------------

    def _issue_label(self, case: ReturnCase) -> None:
        tracking = self.carrier.create_label(case)
        case.label_tracking_number = tracking
        self._transition(case, ReturnStatus.LABEL_ISSUED, actor="system", detail=f"tracking={tracking}")

    def _settle(self, case: ReturnCase, actor: str, note: str | None) -> None:
        resolution = case.requested_resolution
        amount = case.total_value
        if resolution is Resolution.REFUND:
            self.payments.refund(case, amount)
            case.refund_amount = amount
        elif resolution is Resolution.STORE_CREDIT:
            self.payments.issue_store_credit(case, amount)
            case.refund_amount = amount
        # EXCHANGE: replacement order creation is left to the OMS integration.
        case.resolution = resolution
        self._transition(
            case,
            _RESOLUTION_TO_STATUS[resolution],
            actor=actor,
            detail=note or f"{resolution.value} for ${amount:.2f}",
        )

    def _transition(
        self,
        case: ReturnCase,
        target: ReturnStatus,
        actor: str,
        detail: str | None = None,
    ) -> None:
        allowed = _TRANSITIONS.get(case.status, set())
        if target not in allowed:
            raise TransitionError(
                f"invalid transition {case.status.value} -> {target.value}"
            )
        case.status = target
        case.add_event(actor, f"status_{target.value}", detail)

    def _require(self, case_id: str) -> ReturnCase:
        case = self.store.get(case_id)
        if case is None:
            raise KeyError(f"return case {case_id} not found")
        return case
