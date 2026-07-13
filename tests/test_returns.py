from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app import main
from app.models import (
    CustomerProfile,
    LineItem,
    Resolution,
    ReturnReason,
    ReturnRequestCreate,
    ReturnStatus,
)
from app.orchestrator import ReturnsOrchestrator, StubPayments, TransitionError
from app.store import ReturnStore


def make_request(
    *,
    days_ago: int = 5,
    category: str = "apparel",
    unit_price: float = 40.0,
    reason: ReturnReason = ReturnReason.SIZE_FIT,
    resolution: Resolution = Resolution.REFUND,
    orders: int = 10,
    returns: int = 1,
) -> ReturnRequestCreate:
    return ReturnRequestCreate(
        order_id="ord_1001",
        customer=CustomerProfile(
            customer_id="cus_1", lifetime_orders=orders, lifetime_returns=returns
        ),
        items=[
            LineItem(
                sku="SKU-1", name="Denim jacket", category=category,
                quantity=1, unit_price=unit_price,
            )
        ],
        reason=reason,
        requested_resolution=resolution,
        order_date=datetime.now(timezone.utc) - timedelta(days=days_ago),
    )


@pytest.fixture
def orch() -> ReturnsOrchestrator:
    return ReturnsOrchestrator(ReturnStore(":memory:"), payments=StubPayments())


# -- decision engine ---------------------------------------------------------

def test_auto_approve_issues_label(orch):
    case = orch.create_return(make_request())
    assert case.status is ReturnStatus.LABEL_ISSUED
    assert case.label_tracking_number
    assert any("auto-approved" in n for n in case.decision_notes)


def test_excluded_category_rejected(orch):
    case = orch.create_return(make_request(category="perishable"))
    assert case.status is ReturnStatus.REJECTED


def test_outside_window_rejected(orch):
    case = orch.create_return(make_request(days_ago=45))
    assert case.status is ReturnStatus.REJECTED


def test_outside_window_defective_goes_to_review(orch):
    case = orch.create_return(
        make_request(days_ago=45, reason=ReturnReason.DEFECTIVE)
    )
    assert case.status is ReturnStatus.MANUAL_REVIEW


def test_high_value_goes_to_review(orch):
    case = orch.create_return(make_request(unit_price=750.0))
    assert case.status is ReturnStatus.MANUAL_REVIEW


def test_serial_returner_goes_to_review(orch):
    case = orch.create_return(make_request(orders=10, returns=8))
    assert case.status is ReturnStatus.MANUAL_REVIEW


def test_new_customer_not_flagged_for_return_rate(orch):
    # 1 order / 1 return is a 100% rate but too little history to flag.
    case = orch.create_return(make_request(orders=1, returns=1))
    assert case.status is ReturnStatus.LABEL_ISSUED


# -- lifecycle ----------------------------------------------------------------

def test_full_happy_path_refund(orch):
    case = orch.create_return(make_request(unit_price=40.0))
    tracking = case.label_tracking_number

    case = orch.carrier_update(tracking, "picked_up")
    assert case.status is ReturnStatus.IN_TRANSIT
    case = orch.carrier_update(tracking, "delivered")
    assert case.status is ReturnStatus.RECEIVED

    case = orch.record_inspection(case.id, passed=True, agent="wh-42")
    assert case.status is ReturnStatus.REFUNDED
    assert case.refund_amount == 40.0
    assert orch.payments.refunds == [(case.id, 40.0)]


def test_store_credit_resolution(orch):
    case = orch.create_return(make_request(resolution=Resolution.STORE_CREDIT))
    orch.carrier_update(case.label_tracking_number, "delivered")
    case = orch.record_inspection(case.id, passed=True, agent="wh-1")
    assert case.status is ReturnStatus.CREDITED
    assert orch.payments.credits == [(case.id, 40.0)]


def test_failed_inspection_no_refund(orch):
    case = orch.create_return(make_request())
    orch.carrier_update(case.label_tracking_number, "delivered")
    case = orch.record_inspection(case.id, passed=False, agent="wh-1", note="worn, tags removed")
    assert case.status is ReturnStatus.CLOSED_FAILED_INSPECTION
    assert case.refund_amount is None
    assert orch.payments.refunds == []


def test_manual_review_approval_flow(orch):
    case = orch.create_return(make_request(unit_price=750.0))
    case = orch.review(case.id, approve=True, agent="cs-7", note="loyal customer")
    assert case.status is ReturnStatus.LABEL_ISSUED


def test_manual_review_rejection(orch):
    case = orch.create_return(make_request(unit_price=750.0))
    case = orch.review(case.id, approve=False, agent="cs-7", note="suspected fraud")
    assert case.status is ReturnStatus.REJECTED


def test_invalid_transition_raises(orch):
    case = orch.create_return(make_request(category="perishable"))  # rejected
    with pytest.raises(TransitionError):
        orch.review(case.id, approve=True, agent="cs-1")


def test_cannot_refund_before_delivery(orch):
    case = orch.create_return(make_request())  # label issued, not delivered
    with pytest.raises(TransitionError):
        orch.record_inspection(case.id, passed=True, agent="wh-1")


def test_events_are_audit_trail(orch):
    case = orch.create_return(make_request())
    orch.carrier_update(case.label_tracking_number, "delivered")
    case = orch.record_inspection(case.id, passed=True, agent="wh-9")
    kinds = [e.event for e in case.events]
    assert kinds == [
        "return_requested",
        "status_approved",
        "status_label_issued",
        "status_received",
        "status_inspecting",
        "status_refunded",
    ]


# -- API ------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        main, "orchestrator", ReturnsOrchestrator(ReturnStore(":memory:"))
    )
    return TestClient(main.app)


def test_api_end_to_end(client):
    payload = make_request().model_dump(mode="json")
    r = client.post("/returns", json=payload)
    assert r.status_code == 201
    case = r.json()
    assert case["status"] == "label_issued"

    r = client.post(
        "/webhooks/carrier",
        json={"tracking_number": case["label_tracking_number"], "event": "delivered"},
    )
    assert r.status_code == 200

    r = client.post(
        f"/returns/{case['id']}/inspection",
        json={"passed": True, "agent": "wh-1"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "refunded"

    r = client.get("/returns", params={"status": "refunded"})
    assert len(r.json()) == 1


def test_api_404_and_409(client):
    assert client.get("/returns/ret_missing").status_code == 404
    payload = make_request(category="perishable").model_dump(mode="json")
    case = client.post("/returns", json=payload).json()
    r = client.post(
        f"/returns/{case['id']}/review",
        json={"approve": True, "agent": "cs-1"},
    )
    assert r.status_code == 409
