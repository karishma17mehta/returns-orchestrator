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
    CANCELLED = "cancelled"
    EXPIRED = "expired"


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


class ItemCondition(str, enum.Enum):
    NEW = "new"
    TRIED_ON = "tried_on"
    WORN = "worn"
    DAMAGED = "damaged"


# Reasons where the merchant is at fault: these bypass the return window
# and route to manual review instead of auto-rejection. A DAMAGED condition
# with a merchant-fault reason is also reviewable rather than rejected.
MERCHANT_FAULT_REASONS = {
    ReturnReason.DEFECTIVE,
    ReturnReason.WRONG_ITEM,
    ReturnReason.NOT_AS_DESCRIBED,
}


# -- catalog-side records (registered by the merchant's systems) -------------

class OrderItem(BaseModel):
    sku: str
    name: str
    brand: str
    category: str
    quantity: int = Field(ge=1)
    unit_price: float = Field(ge=0)
    discount: float = Field(default=0.0, ge=0, description="per-unit discount applied")


class Order(BaseModel):
    order_id: str
    customer_id: str
    order_date: datetime
    items: list[OrderItem] = Field(min_length=1)


class CustomerProfile(BaseModel):
    customer_id: str
    lifetime_orders: int = 0
    lifetime_returns: int = 0
    loyalty_tier: str | None = None

    @property
    def return_rate(self) -> float:
        if self.lifetime_orders == 0:
            return 0.0
        return self.lifetime_returns / self.lifetime_orders


class OrderRegistration(BaseModel):
    """Payload for the internal order-ingest endpoint."""

    order: Order
    customer: CustomerProfile


# -- return request (references the registered order, carries no prices) -----

class ReturnLineRequest(BaseModel):
    sku: str
    quantity: int = Field(ge=1)


class ReturnRequestCreate(BaseModel):
    order_id: str
    lines: list[ReturnLineRequest] = Field(min_length=1)
    reason: ReturnReason
    requested_resolution: Resolution = Resolution.REFUND
    item_condition: ItemCondition = ItemCondition.NEW
    has_receipt: bool = True
    comment: str | None = None


class ReturnLine(BaseModel):
    """A requested line resolved against the registered order."""

    sku: str
    name: str
    brand: str
    category: str
    quantity: int
    unit_price: float
    discount: float = 0.0

    @property
    def refundable(self) -> float:
        return self.quantity * (self.unit_price - self.discount)


class ReturnEvent(BaseModel):
    at: datetime = Field(default_factory=_now)
    actor: str  # "system", "agent:<id>", "ai:<agent>", "carrier", "customer"
    event: str
    detail: str | None = None


class ReturnCase(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("ret"))
    order_id: str
    customer: CustomerProfile  # snapshot at request time, from the registry
    items: list[ReturnLine]
    reason: ReturnReason
    requested_resolution: Resolution
    item_condition: ItemCondition = ItemCondition.NEW
    has_receipt: bool = True
    order_date: datetime
    comment: str | None = None

    status: ReturnStatus = ReturnStatus.REQUESTED
    resolution: Resolution | None = None
    decision_notes: list[str] = Field(default_factory=list)
    # The exact policy terms this case was decided under (audit/reproducibility).
    policy_snapshot: dict = Field(default_factory=dict)
    label_tracking_number: str | None = None
    refund_amount: float | None = None
    restock_fee: float = 0.0
    replacement_order_id: str | None = None
    # Populated by the multi-agent review board (app/agents). Typed loosely
    # here to keep models.py free of an import cycle with the agents package.
    agent_assessments: list[dict] = Field(default_factory=list)
    customer_message: str | None = None
    version: int = 0  # optimistic-locking counter, managed by the store
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    events: list[ReturnEvent] = Field(default_factory=list)

    @property
    def total_value(self) -> float:
        return sum(item.refundable for item in self.items)

    def add_event(self, actor: str, event: str, detail: str | None = None) -> None:
        self.events.append(ReturnEvent(actor=actor, event=event, detail=detail))
        self.updated_at = _now()
