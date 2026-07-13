"""Return lifecycle orchestration.

Owns the state machine and coordinates the side effects at each step:
resolving requests against registered orders, the decision engine, carrier
labels, outbox-based refund execution, replacement orders, and customer
notifications. External integrations (carrier, payments, OMS, notifier)
are injected so real adapters can replace the built-in stubs.

State machine:

    requested ──▶ approved ──▶ label_issued ──▶ in_transit ──▶ received
        │             ▲             │  │                           │
        ├──▶ manual_review ──▶ rejected                       inspecting
        │         │             cancelled / expired                │
        └──▶ rejected                     ┌───────────┬───────────┤
                                          ▼           ▼           ▼
                                      refunded    exchanged/  closed_failed_
                                                  credited    inspection

Money movement is transactional: settlement writes a pending row to the
payment outbox and saves the case FIRST, then executes the payment and
marks the row — so a crash can lose an execution attempt (retryable via
flush_outbox) but can never move money without a record.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Protocol

from . import policy as policy_engine
from .models import (
    Order,
    ReturnCase,
    ReturnLine,
    ReturnRequestCreate,
    ReturnStatus,
    Resolution,
)
from .policy import PolicyCatalog, ReturnPolicy
from .store import ReturnStore

log = logging.getLogger("returns.orchestrator")

_TRANSITIONS: dict[ReturnStatus, set[ReturnStatus]] = {
    ReturnStatus.REQUESTED: {
        ReturnStatus.APPROVED,
        ReturnStatus.REJECTED,
        ReturnStatus.MANUAL_REVIEW,
        ReturnStatus.CANCELLED,
    },
    ReturnStatus.MANUAL_REVIEW: {
        ReturnStatus.APPROVED,
        ReturnStatus.REJECTED,
        ReturnStatus.CANCELLED,
    },
    ReturnStatus.APPROVED: {ReturnStatus.LABEL_ISSUED, ReturnStatus.CANCELLED},
    ReturnStatus.LABEL_ISSUED: {
        ReturnStatus.IN_TRANSIT,
        ReturnStatus.RECEIVED,
        ReturnStatus.CANCELLED,
        ReturnStatus.EXPIRED,
    },
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

# Statuses that no longer reserve returned quantity against the order.
_RELEASED_STATUSES = {
    ReturnStatus.REJECTED,
    ReturnStatus.CANCELLED,
    ReturnStatus.EXPIRED,
    ReturnStatus.CLOSED_FAILED_INSPECTION,
}


class TransitionError(Exception):
    """Raised when a requested state transition is not allowed."""


class NotFoundError(KeyError):
    """Raised when a referenced case/order/tracking number does not exist."""


class ValidationFailure(Exception):
    """Raised when a return request fails validation against the order."""


class CarrierClient(Protocol):
    def create_label(self, case: ReturnCase) -> str: ...


class PaymentClient(Protocol):
    def refund(self, case: ReturnCase, amount: float, idempotency_key: str) -> None: ...
    def issue_store_credit(
        self, case: ReturnCase, amount: float, idempotency_key: str
    ) -> None: ...


class OMSClient(Protocol):
    def create_replacement_order(self, case: ReturnCase) -> str: ...


class Notifier(Protocol):
    def send(self, case: ReturnCase, message: str) -> None: ...


class StubCarrier:
    def create_label(self, case: ReturnCase) -> str:
        return f"TRK{uuid.uuid4().hex[:10].upper()}"


class StubPayments:
    def __init__(self) -> None:
        self.refunds: list[tuple[str, float, str]] = []
        self.credits: list[tuple[str, float, str]] = []

    def refund(self, case: ReturnCase, amount: float, idempotency_key: str) -> None:
        self.refunds.append((case.id, amount, idempotency_key))

    def issue_store_credit(
        self, case: ReturnCase, amount: float, idempotency_key: str
    ) -> None:
        self.credits.append((case.id, amount, idempotency_key))


class StubOMS:
    def create_replacement_order(self, case: ReturnCase) -> str:
        return f"ord_repl_{uuid.uuid4().hex[:8]}"


class StubNotifier:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send(self, case: ReturnCase, message: str) -> None:
        self.sent.append((case.id, message))


class ReturnsOrchestrator:
    def __init__(
        self,
        store: ReturnStore,
        policy: ReturnPolicy | None = None,
        catalog: PolicyCatalog | None = None,
        carrier: CarrierClient | None = None,
        payments: PaymentClient | None = None,
        oms: OMSClient | None = None,
        notifier: Notifier | None = None,
        label_expiry_days: int = 21,
    ):
        self.store = store
        self.policy = policy or ReturnPolicy()
        self.catalog = catalog or PolicyCatalog()
        self.carrier = carrier or StubCarrier()
        self.payments = payments or StubPayments()
        self.oms = oms or StubOMS()
        self.notifier = notifier or StubNotifier()
        self.label_expiry_days = label_expiry_days

    # -- lifecycle entry point -------------------------------------------

    def create_return(
        self, request: ReturnRequestCreate, idempotency_key: str | None = None
    ) -> ReturnCase:
        if idempotency_key:
            existing_id = self.store.idempotency_lookup(idempotency_key)
            if existing_id:
                existing = self.store.get(existing_id)
                if existing:
                    return existing

        order, lines = self._resolve_against_order(request)
        customer = self.store.get_customer(order.customer_id)
        if customer is None:
            raise ValidationFailure(
                f"customer {order.customer_id} not found in registry"
            )

        case = ReturnCase(
            order_id=order.order_id,
            customer=customer,
            items=lines,
            reason=request.reason,
            requested_resolution=request.requested_resolution,
            item_condition=request.item_condition,
            has_receipt=request.has_receipt,
            order_date=order.order_date,
            comment=request.comment,
        )
        case.add_event("customer", "return_requested", f"reason={request.reason.value}")

        decision = policy_engine.evaluate(case, self.policy, self.catalog)
        case.decision_notes = decision.notes
        case.policy_snapshot = decision.policy_snapshot
        self._transition(case, decision.status, actor="system", detail="; ".join(decision.notes))

        if case.status is ReturnStatus.APPROVED:
            self._issue_label(case)
        self.store.save(case)
        if idempotency_key:
            self.store.idempotency_record(idempotency_key, case.id)
        log.info(
            "return created case=%s order=%s status=%s value=%.2f",
            case.id, case.order_id, case.status.value, case.total_value,
        )
        if case.status is ReturnStatus.REJECTED:
            self._notify(case, self._rejection_message(case))
        return case

    # -- agent actions ----------------------------------------------------

    def review(self, case_id: str, approve: bool, agent: str, note: str | None = None) -> ReturnCase:
        case = self._require(case_id)
        target = ReturnStatus.APPROVED if approve else ReturnStatus.REJECTED
        self._transition(case, target, actor=f"agent:{agent}", detail=note)
        if approve:
            self._issue_label(case)
        self.store.save(case)
        if not approve:
            self._notify(case, case.customer_message or self._rejection_message(case))
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
            self._notify(
                case,
                "Unfortunately your returned item did not pass inspection, so we "
                "are unable to issue a refund. Our team will contact you about "
                "next steps.",
            )
        return case

    def cancel(self, case_id: str, actor: str = "customer", note: str | None = None) -> ReturnCase:
        case = self._require(case_id)
        self._transition(case, ReturnStatus.CANCELLED, actor=actor, detail=note)
        self.store.save(case)
        return case

    # -- carrier webhook ---------------------------------------------------

    def carrier_update(self, tracking_number: str, event: str) -> ReturnCase:
        case = self.store.get_by_tracking(tracking_number)
        if case is None:
            raise NotFoundError(f"no return with tracking number {tracking_number}")
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

    # -- maintenance ---------------------------------------------------------

    def sweep_expired_labels(self) -> list[str]:
        """Expire label_issued cases whose label is older than the limit."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.label_expiry_days)
        expired = []
        for case in self.store.list(ReturnStatus.LABEL_ISSUED):
            if case.updated_at <= cutoff:
                self._transition(
                    case,
                    ReturnStatus.EXPIRED,
                    actor="system",
                    detail=f"label unused for {self.label_expiry_days} days",
                )
                self.store.save(case)
                self._notify(
                    case,
                    "Your return label expired because the item was not shipped. "
                    "If you still want to return it, please submit a new request.",
                )
                expired.append(case.id)
        return expired

    def flush_outbox(self) -> dict:
        """Execute pending payment-outbox entries; safe to call repeatedly."""
        executed, failed = 0, 0
        for entry in self.store.outbox_pending():
            case = self.store.get(entry["case_id"])
            try:
                if entry["kind"] == "refund":
                    self.payments.refund(case, entry["amount"], idempotency_key=entry["id"])
                else:
                    self.payments.issue_store_credit(
                        case, entry["amount"], idempotency_key=entry["id"]
                    )
                self.store.outbox_mark(entry["id"], "executed")
                executed += 1
            except Exception as e:  # keep flushing the rest
                log.warning("outbox entry %s failed: %s", entry["id"], e)
                self.store.outbox_mark(entry["id"], "pending", error=str(e))
                failed += 1
        return {"executed": executed, "failed": failed}

    # -- internals ----------------------------------------------------------

    def _resolve_against_order(
        self, request: ReturnRequestCreate
    ) -> tuple[Order, list[ReturnLine]]:
        order = self.store.get_order(request.order_id)
        if order is None:
            raise ValidationFailure(f"order {request.order_id} is not registered")

        by_sku = {item.sku: item for item in order.items}
        already_returned: dict[str, int] = {}
        for prior in self.store.list_by_order(order.order_id):
            if prior.status in _RELEASED_STATUSES:
                continue
            for line in prior.items:
                already_returned[line.sku] = already_returned.get(line.sku, 0) + line.quantity

        lines: list[ReturnLine] = []
        for req_line in request.lines:
            item = by_sku.get(req_line.sku)
            if item is None:
                raise ValidationFailure(
                    f"sku {req_line.sku} is not part of order {order.order_id}"
                )
            available = item.quantity - already_returned.get(req_line.sku, 0)
            if req_line.quantity > available:
                raise ValidationFailure(
                    f"sku {req_line.sku}: requested {req_line.quantity} but only "
                    f"{available} of {item.quantity} remain returnable"
                )
            lines.append(
                ReturnLine(
                    sku=item.sku,
                    name=item.name,
                    brand=item.brand,
                    category=item.category,
                    quantity=req_line.quantity,
                    unit_price=item.unit_price,
                    discount=item.discount,
                )
            )
        return order, lines

    def _issue_label(self, case: ReturnCase) -> None:
        tracking = self.carrier.create_label(case)
        case.label_tracking_number = tracking
        self._transition(case, ReturnStatus.LABEL_ISSUED, actor="system", detail=f"tracking={tracking}")

    def _settle(self, case: ReturnCase, actor: str, note: str | None) -> None:
        resolution = case.requested_resolution
        fee_pct = float(case.policy_snapshot.get("restock_fee_pct", 0.0))
        gross = case.total_value
        fee = round(gross * fee_pct, 2)
        amount = round(gross - fee, 2)
        case.restock_fee = fee
        case.resolution = resolution

        if resolution is Resolution.EXCHANGE:
            case.replacement_order_id = self.oms.create_replacement_order(case)
            detail = note or f"exchange, replacement order {case.replacement_order_id}"
            self._transition(case, ReturnStatus.EXCHANGED, actor=actor, detail=detail)
            self.store.save(case)
            self._notify(
                case,
                f"Your exchange is on its way — replacement order "
                f"{case.replacement_order_id} has been created.",
            )
            return

        kind = "refund" if resolution is Resolution.REFUND else "store_credit"
        case.refund_amount = amount
        detail = note or f"{resolution.value} for ${amount:.2f}" + (
            f" (after ${fee:.2f} restocking fee)" if fee else ""
        )
        self._transition(case, _RESOLUTION_TO_STATUS[resolution], actor=actor, detail=detail)
        # Record intent + state BEFORE moving money (outbox pattern).
        self.store.outbox_enqueue(case.id, kind, amount)
        self.store.save(case)
        self.flush_outbox()
        noun = "refund" if kind == "refund" else "store credit"
        self._notify(
            case,
            f"Good news — your return passed inspection and a {noun} of "
            f"${amount:.2f} has been issued.",
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
        log.info("case=%s %s -> %s by %s", case.id, case.status.value, target.value, actor)
        case.status = target
        case.add_event(actor, f"status_{target.value}", detail)

    def _notify(self, case: ReturnCase, message: str) -> None:
        try:
            self.notifier.send(case, message)
            case.add_event("system", "customer_notified", message[:200])
            self.store.save(case)
        except Exception as e:  # notification failure must not fail the flow
            log.warning("notify failed for case %s: %s", case.id, e)

    def _rejection_message(self, case: ReturnCase) -> str:
        reason = "; ".join(case.decision_notes) or "it does not meet the return policy"
        return (
            f"We're sorry — your return request for order {case.order_id} "
            f"could not be accepted: {reason}."
        )

    def _require(self, case_id: str) -> ReturnCase:
        case = self.store.get(case_id)
        if case is None:
            raise NotFoundError(f"return case {case_id} not found")
        return case
