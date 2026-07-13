"""Policy-driven decision engine.

Evaluates a new return request against merchant policy and returns a
decision: approve, reject, or route to manual review. Rules run in order;
the first terminal rule wins. Every decision records the rules that fired
so an agent can see *why* the system decided what it did.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .models import MERCHANT_FAULT_REASONS, ReturnRequestCreate, ReturnStatus


@dataclass
class ReturnPolicy:
    return_window_days: int = 30
    # Categories that can never be returned (health/hygiene, perishables...)
    excluded_categories: set[str] = field(
        default_factory=lambda: {"final_sale", "perishable", "gift_card", "intimates"}
    )
    # Above this order value, a human must look at it.
    manual_review_value_threshold: float = 500.0
    # Customers returning more than this fraction of their orders get reviewed.
    manual_review_return_rate: float = 0.5
    # A customer needs at least this many orders before return-rate checks apply.
    return_rate_min_orders: int = 3
    # Refunds at or below this value skip inspection-gated refunds ("keep it"
    # returns are a policy many large retailers use for cheap items).
    keep_it_threshold: float = 0.0


@dataclass
class Decision:
    status: ReturnStatus  # APPROVED, REJECTED, or MANUAL_REVIEW
    notes: list[str] = field(default_factory=list)


def evaluate(request: ReturnRequestCreate, policy: ReturnPolicy) -> Decision:
    notes: list[str] = []

    # 1. Excluded categories are a hard reject regardless of reason.
    excluded = [i for i in request.items if i.category in policy.excluded_categories]
    if excluded:
        skus = ", ".join(i.sku for i in excluded)
        notes.append(f"rejected: category not returnable ({skus})")
        return Decision(ReturnStatus.REJECTED, notes)

    # 2. Return window. Merchant-fault reasons bypass the window but go to
    #    a human, since the claim can't be verified automatically.
    order_date = request.order_date
    if order_date.tzinfo is None:
        order_date = order_date.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - order_date).days
    if age_days > policy.return_window_days:
        if request.reason in MERCHANT_FAULT_REASONS:
            notes.append(
                f"manual review: outside {policy.return_window_days}-day window "
                f"({age_days} days) but reason '{request.reason.value}' is merchant-fault"
            )
            return Decision(ReturnStatus.MANUAL_REVIEW, notes)
        notes.append(
            f"rejected: outside {policy.return_window_days}-day return window ({age_days} days)"
        )
        return Decision(ReturnStatus.REJECTED, notes)

    # 3. Fraud / abuse signals route to manual review.
    total = sum(i.total for i in request.items)
    if total > policy.manual_review_value_threshold:
        notes.append(
            f"manual review: value ${total:.2f} exceeds "
            f"${policy.manual_review_value_threshold:.2f} threshold"
        )
        return Decision(ReturnStatus.MANUAL_REVIEW, notes)

    customer = request.customer
    if (
        customer.lifetime_orders >= policy.return_rate_min_orders
        and customer.return_rate > policy.manual_review_return_rate
    ):
        notes.append(
            f"manual review: customer return rate {customer.return_rate:.0%} exceeds "
            f"{policy.manual_review_return_rate:.0%} threshold"
        )
        return Decision(ReturnStatus.MANUAL_REVIEW, notes)

    notes.append(f"auto-approved: within window ({age_days} days), no risk flags")
    return Decision(ReturnStatus.APPROVED, notes)
