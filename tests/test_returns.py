"""Decision engine, lifecycle, and validation tests."""
import pytest

from app.models import ItemCondition, Resolution, ReturnReason, ReturnStatus
from app.orchestrator import TransitionError, ValidationFailure
from tests.conftest import make_request, register_order


# -- decision engine ---------------------------------------------------------

def test_auto_approve_issues_label(orch):
    register_order(orch)
    case = orch.create_return(make_request())
    assert case.status is ReturnStatus.LABEL_ISSUED
    assert case.label_tracking_number
    assert any("auto-approved" in n for n in case.decision_notes)
    assert case.policy_snapshot["window_days"] == 30
    assert case.policy_snapshot["engine_version"]


def test_worn_item_rejected(orch):
    register_order(orch)
    case = orch.create_return(make_request(condition=ItemCondition.WORN))
    assert case.status is ReturnStatus.REJECTED


def test_damaged_defective_goes_to_review(orch):
    register_order(orch)
    case = orch.create_return(
        make_request(condition=ItemCondition.DAMAGED, reason=ReturnReason.DEFECTIVE)
    )
    assert case.status is ReturnStatus.MANUAL_REVIEW


def test_outside_window_rejected(orch):
    register_order(orch, days_ago=45)
    case = orch.create_return(make_request())
    assert case.status is ReturnStatus.REJECTED


def test_grace_period_goes_to_review(orch):
    register_order(orch, days_ago=33)  # window 30, grace 7
    case = orch.create_return(make_request())
    assert case.status is ReturnStatus.MANUAL_REVIEW


def test_outside_window_defective_goes_to_review(orch):
    register_order(orch, days_ago=45)
    case = orch.create_return(make_request(reason=ReturnReason.DEFECTIVE))
    assert case.status is ReturnStatus.MANUAL_REVIEW


def test_missing_receipt_goes_to_review(orch):
    register_order(orch)
    case = orch.create_return(make_request(has_receipt=False))
    assert case.status is ReturnStatus.MANUAL_REVIEW


def test_high_value_goes_to_review(orch):
    register_order(orch, unit_price=750.0)
    case = orch.create_return(make_request())
    assert case.status is ReturnStatus.MANUAL_REVIEW


def test_serial_returner_goes_to_review(orch):
    register_order(orch, orders=10, returns=8)
    case = orch.create_return(make_request())
    assert case.status is ReturnStatus.MANUAL_REVIEW


def test_new_customer_not_flagged_for_return_rate(orch):
    register_order(orch, orders=1, returns=1)
    case = orch.create_return(make_request())
    assert case.status is ReturnStatus.LABEL_ISSUED


# -- order validation ------------------------------------------------------------

def test_unregistered_order_fails(orch):
    with pytest.raises(ValidationFailure, match="not registered"):
        orch.create_return(make_request(order_id="ord_ghost"))


def test_unknown_sku_fails(orch):
    register_order(orch)
    with pytest.raises(ValidationFailure, match="not part of order"):
        orch.create_return(make_request(sku="SKU-NOPE"))


def test_cannot_return_more_than_purchased(orch):
    register_order(orch, quantity=2)
    with pytest.raises(ValidationFailure, match="remain returnable"):
        orch.create_return(make_request(quantity=3))


def test_duplicate_return_blocked(orch):
    register_order(orch, quantity=1)
    orch.create_return(make_request())  # takes the only unit
    with pytest.raises(ValidationFailure, match="remain returnable"):
        orch.create_return(make_request())


def test_rejected_case_releases_quantity(orch):
    register_order(orch, quantity=1)
    rejected = orch.create_return(make_request(condition=ItemCondition.WORN))
    assert rejected.status is ReturnStatus.REJECTED
    case = orch.create_return(make_request())  # can try again
    assert case.status is ReturnStatus.LABEL_ISSUED


def test_partial_quantity_return(orch):
    register_order(orch, quantity=3, unit_price=40.0)
    case = orch.create_return(make_request(quantity=2))
    assert case.total_value == 80.0
    case2 = orch.create_return(make_request(quantity=1))
    assert case2.total_value == 40.0


# -- refund math -----------------------------------------------------------------

def test_discount_reduces_refund(orch):
    register_order(orch, unit_price=100.0, discount=25.0)
    case = orch.create_return(make_request())
    orch.carrier_update(case.label_tracking_number, "delivered")
    case = orch.record_inspection(case.id, passed=True, agent="wh-1")
    assert case.refund_amount == 75.0


# -- lifecycle ----------------------------------------------------------------

def test_full_happy_path_refund(orch):
    register_order(orch, unit_price=40.0)
    case = orch.create_return(make_request())
    tracking = case.label_tracking_number

    case = orch.carrier_update(tracking, "picked_up")
    assert case.status is ReturnStatus.IN_TRANSIT
    case = orch.carrier_update(tracking, "delivered")
    assert case.status is ReturnStatus.RECEIVED

    case = orch.record_inspection(case.id, passed=True, agent="wh-42")
    assert case.status is ReturnStatus.REFUNDED
    assert case.refund_amount == 40.0
    assert [(r[0], r[1]) for r in orch.payments.refunds] == [(case.id, 40.0)]


def test_store_credit_resolution(orch):
    register_order(orch)
    case = orch.create_return(make_request(resolution=Resolution.STORE_CREDIT))
    orch.carrier_update(case.label_tracking_number, "delivered")
    case = orch.record_inspection(case.id, passed=True, agent="wh-1")
    assert case.status is ReturnStatus.CREDITED
    assert [(c[0], c[1]) for c in orch.payments.credits] == [(case.id, 40.0)]


def test_exchange_creates_replacement_order(orch):
    register_order(orch)
    case = orch.create_return(make_request(resolution=Resolution.EXCHANGE))
    orch.carrier_update(case.label_tracking_number, "delivered")
    case = orch.record_inspection(case.id, passed=True, agent="wh-1")
    assert case.status is ReturnStatus.EXCHANGED
    assert case.replacement_order_id
    assert orch.payments.refunds == []


def test_failed_inspection_no_refund(orch):
    register_order(orch)
    case = orch.create_return(make_request())
    orch.carrier_update(case.label_tracking_number, "delivered")
    case = orch.record_inspection(case.id, passed=False, agent="wh-1", note="worn, tags removed")
    assert case.status is ReturnStatus.CLOSED_FAILED_INSPECTION
    assert case.refund_amount is None
    assert orch.payments.refunds == []


def test_manual_review_approval_flow(orch):
    register_order(orch, unit_price=750.0)
    case = orch.create_return(make_request())
    case = orch.review(case.id, approve=True, agent="cs-7", note="loyal customer")
    assert case.status is ReturnStatus.LABEL_ISSUED


def test_manual_review_rejection_notifies_customer(orch):
    register_order(orch, unit_price=750.0)
    case = orch.create_return(make_request())
    case = orch.review(case.id, approve=False, agent="cs-7", note="suspected fraud")
    assert case.status is ReturnStatus.REJECTED
    assert any(cid == case.id for cid, _ in orch.notifier.sent)


def test_invalid_transition_raises(orch):
    register_order(orch)
    case = orch.create_return(make_request(condition=ItemCondition.WORN))  # rejected
    with pytest.raises(TransitionError):
        orch.review(case.id, approve=True, agent="cs-1")


def test_cannot_refund_before_delivery(orch):
    register_order(orch)
    case = orch.create_return(make_request())  # label issued, not delivered
    with pytest.raises(TransitionError):
        orch.record_inspection(case.id, passed=True, agent="wh-1")


def test_refund_notification_sent(orch):
    register_order(orch)
    case = orch.create_return(make_request())
    orch.carrier_update(case.label_tracking_number, "delivered")
    orch.record_inspection(case.id, passed=True, agent="wh-9")
    messages = [m for cid, m in orch.notifier.sent if cid == case.id]
    assert any("refund" in m and "$40.00" in m for m in messages)


def test_events_are_audit_trail(orch):
    register_order(orch)
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
        "customer_notified",
    ]
