"""Shared fixtures: an orchestrator with a registered order/customer."""
import os

# Keep `import app.main` in tests from creating a real returns.db file.
os.environ.setdefault("RETURNS_DB", ":memory:")

from datetime import datetime, timedelta, timezone

import pytest

from app.models import (
    CustomerProfile,
    ItemCondition,
    Order,
    OrderItem,
    Resolution,
    ReturnLineRequest,
    ReturnReason,
    ReturnRequestCreate,
)
from app.orchestrator import ReturnsOrchestrator
from app.store import ReturnStore


def register_order(
    orch: ReturnsOrchestrator,
    *,
    order_id: str = "ord_1001",
    customer_id: str = "cus_1",
    days_ago: int = 5,
    brand: str = "H&M",
    category: str = "apparel",
    unit_price: float = 40.0,
    discount: float = 0.0,
    quantity: int = 1,
    orders: int = 10,
    returns: int = 1,
    extra_items: list[OrderItem] | None = None,
) -> Order:
    items = [
        OrderItem(
            sku="SKU-1", name="Denim jacket", brand=brand, category=category,
            quantity=quantity, unit_price=unit_price, discount=discount,
        )
    ] + (extra_items or [])
    order = Order(
        order_id=order_id,
        customer_id=customer_id,
        order_date=datetime.now(timezone.utc) - timedelta(days=days_ago),
        items=items,
    )
    customer = CustomerProfile(
        customer_id=customer_id, lifetime_orders=orders, lifetime_returns=returns
    )
    orch.store.register_order(order, customer)
    return order


def make_request(
    *,
    order_id: str = "ord_1001",
    sku: str = "SKU-1",
    quantity: int = 1,
    reason: ReturnReason = ReturnReason.SIZE_FIT,
    resolution: Resolution = Resolution.REFUND,
    condition: ItemCondition = ItemCondition.NEW,
    has_receipt: bool = True,
    comment: str | None = None,
) -> ReturnRequestCreate:
    return ReturnRequestCreate(
        order_id=order_id,
        lines=[ReturnLineRequest(sku=sku, quantity=quantity)],
        reason=reason,
        requested_resolution=resolution,
        item_condition=condition,
        has_receipt=has_receipt,
        comment=comment,
    )


@pytest.fixture
def orch() -> ReturnsOrchestrator:
    return ReturnsOrchestrator(ReturnStore(":memory:"))
