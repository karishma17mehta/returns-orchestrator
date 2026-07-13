"""Domain models for the returns orchestrator."""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class ReturnStatus(str, enum.Enum):
    REQUESTED = "requested"
    MANUAL_REVIEW = "manual_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    LABEL_ISSUED = "label_issued"
    IN_TRANSIT = "in_transit"
    RECEIVED = "received"
    INSPECTING = "inspecting"
    REFUNDED = "refunded"
    EXCHANGED = "exchanged"
    CREDITED = "credited"
    CLOSED_FAILED_INSPECTION = "closed_failed_inspection"


class Resolution(str, enum.Enum):
    REFUND = "refund"
    EXCHANGE = "exchange"
    STORE_CREDIT = "store_credit"


class ReturnReason(str, enum.Enum):
    DEFECTIVE = "defective"
    WRONG_ITEM = "wrong_item"
    NOT_AS_DESCRIBED = "not_as_described"
    NO_LONGER_NEEDED = "no_longer_needed"
    BETTER_PRICE_FOUND = "better_price_found"
    SIZE_FIT = "size_fit"
    OTHER = "other"


# Reasons where the merchant is at fault: these bypass the return window
# and route to manual review instead of auto-rejection.
MERCHANT_FAULT_REASONS = {
    ReturnReason.DEFECTIVE,
    ReturnReason.WRONG_ITEM,
    ReturnReason.NOT_AS_DESCRIBED,
}


class LineItem(BaseModel):
    sku: str
    name: str
    category: str
    quantity: int = Field(ge=1)
    unit_price: float = Field(ge=0)

    @property
    def total(self) -> float:
        return self.quantity * self.unit_price


class CustomerProfile(BaseModel):
    customer_id: str
    lifetime_orders: int = 0
    lifetime_returns: int = 0

    @property
    def return_rate(self) -> float:
        if self.lifetime_orders == 0:
            return 0.0
        return self.lifetime_returns / self.lifetime_orders


class ReturnRequestCreate(BaseModel):
    order_id: str
    customer: CustomerProfile
    items: list[LineItem] = Field(min_length=1)
    reason: ReturnReason
    requested_resolution: Resolution = Resolution.REFUND
    order_date: datetime
    comment: str | None = None


class ReturnEvent(BaseModel):
    at: datetime = Field(default_factory=_now)
    actor: str  # "system", "agent:<id>", "carrier", "customer"
    event: str
    detail: str | None = None


class ReturnCase(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("ret"))
    order_id: str
    customer: CustomerProfile
    items: list[LineItem]
    reason: ReturnReason
    requested_resolution: Resolution
    order_date: datetime
    comment: str | None = None

    status: ReturnStatus = ReturnStatus.REQUESTED
    resolution: Resolution | None = None
    decision_notes: list[str] = Field(default_factory=list)
    label_tracking_number: str | None = None
    refund_amount: float | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    events: list[ReturnEvent] = Field(default_factory=list)

    @property
    def total_value(self) -> float:
        return sum(item.total for item in self.items)

    def add_event(self, actor: str, event: str, detail: str | None = None) -> None:
        self.events.append(ReturnEvent(actor=actor, event=event, detail=detail))
        self.updated_at = _now()
