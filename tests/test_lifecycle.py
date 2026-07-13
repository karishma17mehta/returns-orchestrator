"""Cancellation, label expiry, and policy catalog tests."""
from datetime import datetime, timedelta, timezone

import pytest

from app.models import ItemCondition, ReturnStatus
from app.orchestrator import ReturnsOrchestrator, TransitionError
from app.policy import BrandPolicy, PolicyCatalog
from app.store import ReturnStore
from tests.conftest import make_request, register_order


def test_cancel_before_shipping(orch):
    register_order(orch)
    case = orch.create_return(make_request())
    assert case.status is ReturnStatus.LABEL_ISSUED
    case = orch.cancel(case.id, note="changed my mind")
    assert case.status is ReturnStatus.CANCELLED


def test_cancel_releases_quantity(orch):
    register_order(orch, quantity=1)
    case = orch.create_return(make_request())
    orch.cancel(case.id)
    again = orch.create_return(make_request())
    assert again.status is ReturnStatus.LABEL_ISSUED


def test_cannot_cancel_after_transit(orch):
    register_order(orch)
    case = orch.create_return(make_request())
    orch.carrier_update(case.label_tracking_number, "picked_up")
    with pytest.raises(TransitionError):
        orch.cancel(case.id)


def test_expiry_sweep(orch):
    register_order(orch)
    case = orch.create_return(make_request())
    # Backdate the case's last update past the expiry cutoff.
    stored = orch.store.get(case.id)
    stored.updated_at = datetime.now(timezone.utc) - timedelta(days=30)
    orch.store.save(stored)

    expired = orch.sweep_expired_labels()
    assert expired == [case.id]
    assert orch.store.get(case.id).status is ReturnStatus.EXPIRED
    assert any(cid == case.id for cid, _ in orch.notifier.sent)


def test_sweep_ignores_fresh_labels(orch):
    register_order(orch)
    orch.create_return(make_request())
    assert orch.sweep_expired_labels() == []


# -- policy catalog -------------------------------------------------------------

def test_catalog_brand_window_applies():
    catalog = PolicyCatalog(
        {("gucci", "bags"): BrandPolicy(return_window_days=14)}
    )
    orch = ReturnsOrchestrator(ReturnStore(":memory:"), catalog=catalog)
    register_order(orch, brand="Gucci", category="Bags", days_ago=20)
    case = orch.create_return(make_request())
    # 20 days is inside the merchant default (30) but past Gucci's 14 + 7 grace
    assert case.status is ReturnStatus.MANUAL_REVIEW or case.status is ReturnStatus.REJECTED
    assert case.policy_snapshot["window_days"] == 14


def test_catalog_final_sale_rejects():
    catalog = PolicyCatalog(
        {("h&m", "beauty"): BrandPolicy(return_window_days=30, final_sale=True)}
    )
    orch = ReturnsOrchestrator(ReturnStore(":memory:"), catalog=catalog)
    register_order(orch, brand="H&M", category="Beauty")
    case = orch.create_return(make_request())
    assert case.status is ReturnStatus.REJECTED
    assert any("final-sale" in n for n in case.decision_notes)


def test_catalog_restock_fee_applies():
    catalog = PolicyCatalog(
        {("h&m", "apparel"): BrandPolicy(return_window_days=30, restock_fee_pct=0.1)}
    )
    orch = ReturnsOrchestrator(ReturnStore(":memory:"), catalog=catalog)
    register_order(orch, unit_price=100.0)
    case = orch.create_return(make_request())
    orch.carrier_update(case.label_tracking_number, "delivered")
    case = orch.record_inspection(case.id, passed=True, agent="wh-1")
    assert case.restock_fee == 10.0
    assert case.refund_amount == 90.0


def test_catalog_from_users_csv():
    catalog = PolicyCatalog.from_csv("returns_triage_dataset/policies_map.csv")
    from app.policy import ReturnPolicy

    defaults = ReturnPolicy()
    assert catalog.resolve("Gucci", "Bags", defaults).return_window_days == 14
    assert catalog.resolve("Under Armour", "Tops", defaults).return_window_days == 60
    assert catalog.resolve("H&M", "Beauty", defaults).final_sale is True
    # Unknown combos fall back to merchant defaults.
    assert catalog.resolve("Nobody", "Nothing", defaults).return_window_days == 30
