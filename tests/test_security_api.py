"""API-key auth, webhook signatures, and end-to-end API tests."""
import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from app import main
from app.orchestrator import ReturnsOrchestrator
from app.store import ReturnStore
from tests.conftest import make_request, register_order


@pytest.fixture
def client(monkeypatch):
    orch = ReturnsOrchestrator(ReturnStore(":memory:"))
    monkeypatch.setattr(main, "orchestrator", orch)
    return TestClient(main.app), orch


def _payload():
    return make_request().model_dump(mode="json")


# -- auth ---------------------------------------------------------------------

def test_no_keys_configured_means_open(client, monkeypatch):
    monkeypatch.delenv("RETURNS_API_KEYS", raising=False)
    api, orch = client
    register_order(orch)
    assert api.post("/returns", json=_payload()).status_code == 201


def test_missing_key_rejected(client, monkeypatch):
    monkeypatch.setenv("RETURNS_API_KEYS", "svc-key:service,ops-key:ops")
    api, _ = client
    assert api.post("/returns", json=_payload()).status_code == 401


def test_invalid_key_rejected(client, monkeypatch):
    monkeypatch.setenv("RETURNS_API_KEYS", "svc-key:service")
    api, _ = client
    r = api.post("/returns", json=_payload(), headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_service_key_cannot_review(client, monkeypatch):
    monkeypatch.setenv("RETURNS_API_KEYS", "svc-key:service,ops-key:ops")
    api, _ = client
    r = api.post(
        "/returns/ret_x/review",
        json={"approve": True, "agent": "cs-1"},
        headers={"X-API-Key": "svc-key"},
    )
    assert r.status_code == 403


def test_ops_key_implies_service(client, monkeypatch):
    monkeypatch.setenv("RETURNS_API_KEYS", "ops-key:ops")
    api, orch = client
    register_order(orch)
    r = api.post("/returns", json=_payload(), headers={"X-API-Key": "ops-key"})
    assert r.status_code == 201


# -- webhook signatures ------------------------------------------------------------

def test_webhook_signature_required_when_secret_set(client, monkeypatch):
    monkeypatch.setenv("CARRIER_WEBHOOK_SECRET", "shh")
    api, orch = client
    register_order(orch)
    case = orch.create_return(make_request())

    body = {"tracking_number": case.label_tracking_number, "event": "delivered"}
    r = api.post("/webhooks/carrier", json=body)
    assert r.status_code == 401

    raw = json.dumps(body).encode()
    sig = hmac.new(b"shh", raw, hashlib.sha256).hexdigest()
    r = api.post(
        "/webhooks/carrier",
        content=raw,
        headers={"X-Carrier-Signature": sig, "Content-Type": "application/json"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "received"


def test_webhook_bad_signature_rejected(client, monkeypatch):
    monkeypatch.setenv("CARRIER_WEBHOOK_SECRET", "shh")
    api, _ = client
    r = api.post(
        "/webhooks/carrier",
        json={"tracking_number": "TRK1", "event": "delivered"},
        headers={"X-Carrier-Signature": "deadbeef"},
    )
    assert r.status_code == 401


# -- API end to end ------------------------------------------------------------------

def test_api_end_to_end(client, monkeypatch):
    monkeypatch.delenv("RETURNS_API_KEYS", raising=False)
    monkeypatch.delenv("CARRIER_WEBHOOK_SECRET", raising=False)
    api, orch = client

    # Register the order through the internal endpoint.
    order = register_order(orch, order_id="ord_api")
    r = api.post("/returns", json=make_request(order_id="ord_api").model_dump(mode="json"))
    assert r.status_code == 201
    case = r.json()
    assert case["status"] == "label_issued"

    r = api.post(
        "/webhooks/carrier",
        json={"tracking_number": case["label_tracking_number"], "event": "delivered"},
    )
    assert r.status_code == 200

    r = api.post(
        f"/returns/{case['id']}/inspection", json={"passed": True, "agent": "wh-1"}
    )
    assert r.status_code == 200
    assert r.json()["status"] == "refunded"

    r = api.get("/metrics")
    assert r.status_code == 200
    m = r.json()
    assert m["cases_by_status"]["refunded"] == 1
    assert m["payments"]["executed_total"] == 40.0


def test_api_idempotency_header(client, monkeypatch):
    monkeypatch.delenv("RETURNS_API_KEYS", raising=False)
    api, orch = client
    register_order(orch, quantity=1)
    payload = _payload()
    r1 = api.post("/returns", json=payload, headers={"Idempotency-Key": "k-1"})
    r2 = api.post("/returns", json=payload, headers={"Idempotency-Key": "k-1"})
    assert r1.status_code == r2.status_code == 201
    assert r1.json()["id"] == r2.json()["id"]


def test_api_validation_maps_to_422(client, monkeypatch):
    monkeypatch.delenv("RETURNS_API_KEYS", raising=False)
    api, _ = client
    r = api.post("/returns", json=_payload())  # order not registered
    assert r.status_code == 422


def test_api_cancel(client, monkeypatch):
    monkeypatch.delenv("RETURNS_API_KEYS", raising=False)
    api, orch = client
    register_order(orch)
    case = orch.create_return(make_request())
    r = api.post(f"/returns/{case.id}/cancel", json={"note": "nvm"})
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"


def test_api_sweep_and_outbox_endpoints(client, monkeypatch):
    monkeypatch.delenv("RETURNS_API_KEYS", raising=False)
    api, _ = client
    assert api.post("/internal/sweep-expired").json() == {"expired": []}
    assert api.post("/internal/outbox/flush").json() == {"executed": 0, "failed": 0}
