"""FastAPI surface for the returns orchestrator."""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .models import ReturnCase, ReturnRequestCreate, ReturnStatus
from .orchestrator import ReturnsOrchestrator, TransitionError
from .store import ReturnStore

app = FastAPI(title="Returns Orchestrator", version="0.1.0")

_db_path = os.environ.get("RETURNS_DB", "returns.db")
orchestrator = ReturnsOrchestrator(ReturnStore(_db_path))


class ReviewBody(BaseModel):
    approve: bool
    agent: str
    note: str | None = None


class InspectionBody(BaseModel):
    passed: bool
    agent: str
    note: str | None = None


class CarrierWebhook(BaseModel):
    tracking_number: str
    event: str  # picked_up | delivered | anything else is logged only


@app.post("/returns", response_model=ReturnCase, status_code=201)
def create_return(body: ReturnRequestCreate) -> ReturnCase:
    return orchestrator.create_return(body)


@app.get("/returns", response_model=list[ReturnCase])
def list_returns(status: ReturnStatus | None = None) -> list[ReturnCase]:
    return orchestrator.store.list(status)


@app.get("/returns/{case_id}", response_model=ReturnCase)
def get_return(case_id: str) -> ReturnCase:
    case = orchestrator.store.get(case_id)
    if case is None:
        raise HTTPException(404, f"return {case_id} not found")
    return case


@app.post("/returns/{case_id}/review", response_model=ReturnCase)
def review_return(case_id: str, body: ReviewBody) -> ReturnCase:
    try:
        return orchestrator.review(case_id, body.approve, body.agent, body.note)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except TransitionError as e:
        raise HTTPException(409, str(e))


@app.post("/returns/{case_id}/inspection", response_model=ReturnCase)
def record_inspection(case_id: str, body: InspectionBody) -> ReturnCase:
    try:
        return orchestrator.record_inspection(case_id, body.passed, body.agent, body.note)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except TransitionError as e:
        raise HTTPException(409, str(e))


@app.post("/webhooks/carrier", response_model=ReturnCase)
def carrier_webhook(body: CarrierWebhook) -> ReturnCase:
    try:
        return orchestrator.carrier_update(body.tracking_number, body.event)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except TransitionError as e:
        raise HTTPException(409, str(e))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
