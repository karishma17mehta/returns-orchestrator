"""FastAPI surface for the returns orchestrator.

Environment:
  RETURNS_DB              SQLite path (default returns.db, :memory: for ephemeral)
  RETURNS_API_KEYS        "key:role,key:role" — roles: service, ops (unset = auth off)
  CARRIER_WEBHOOK_SECRET  HMAC secret for carrier webhooks (unset = check off)
  POLICY_CATALOG_CSV      per-brand/category policy map (policies_map.csv format)
  POLICY_CHUNKS_PKL       pickled chunk DataFrame for the policy retriever
  OPENAI_API_KEY          enables the multi-agent review board
"""
from __future__ import annotations

import logging
import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from .agents import OpenAIClient, ReviewBoard
from .agents.llm import LLMUnavailableError
from .models import (
    OrderRegistration,
    ReturnCase,
    ReturnRequestCreate,
    ReturnStatus,
)
from .orchestrator import ReturnsOrchestrator, TransitionError, ValidationFailure
from .policy import PolicyCatalog
from .security import require_role, verify_carrier_signature
from .store import ConcurrencyError, ReturnStore

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("returns.api")

app = FastAPI(title="Returns Orchestrator", version="0.3.0")


def _build_orchestrator() -> ReturnsOrchestrator:
    catalog = PolicyCatalog()
    catalog_csv = os.environ.get("POLICY_CATALOG_CSV")
    if catalog_csv:
        catalog = PolicyCatalog.from_csv(catalog_csv)
        log.info("policy catalog loaded from %s", catalog_csv)
    return ReturnsOrchestrator(
        ReturnStore(os.environ.get("RETURNS_DB", "returns.db")), catalog=catalog
    )


def _build_board(orch: ReturnsOrchestrator) -> ReviewBoard:
    retriever = None
    chunks_pkl = os.environ.get("POLICY_CHUNKS_PKL")
    if chunks_pkl:
        from .agents.retriever import LexicalPolicyRetriever

        retriever = LexicalPolicyRetriever.from_pickle(chunks_pkl)
        log.info("policy retriever loaded from %s", chunks_pkl)
    return ReviewBoard(orch, llm=OpenAIClient(), retriever=retriever)


orchestrator = _build_orchestrator()
review_board = _build_board(orchestrator)


def _http_wrap(fn):
    """Map domain errors to HTTP codes."""
    try:
        return fn()
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValidationFailure as e:
        raise HTTPException(422, str(e))
    except (TransitionError, ValueError) as e:
        raise HTTPException(409, str(e))
    except ConcurrencyError as e:
        raise HTTPException(409, str(e))
    except LLMUnavailableError as e:
        raise HTTPException(503, str(e))


# -- returns ------------------------------------------------------------------

@app.post(
    "/returns",
    response_model=ReturnCase,
    status_code=201,
    dependencies=[Depends(require_role("service"))],
)
def create_return(
    body: ReturnRequestCreate,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> ReturnCase:
    return _http_wrap(lambda: orchestrator.create_return(body, idempotency_key))


@app.get(
    "/returns",
    response_model=list[ReturnCase],
    dependencies=[Depends(require_role("service"))],
)
def list_returns(status: ReturnStatus | None = None) -> list[ReturnCase]:
    return orchestrator.store.list(status)


@app.get(
    "/returns/{case_id}",
    response_model=ReturnCase,
    dependencies=[Depends(require_role("service"))],
)
def get_return(case_id: str) -> ReturnCase:
    case = orchestrator.store.get(case_id)
    if case is None:
        raise HTTPException(404, f"return {case_id} not found")
    return case


class ReviewBody(BaseModel):
    approve: bool
    agent: str
    note: str | None = None


@app.post(
    "/returns/{case_id}/review",
    response_model=ReturnCase,
    dependencies=[Depends(require_role("ops"))],
)
def review_return(case_id: str, body: ReviewBody) -> ReturnCase:
    return _http_wrap(
        lambda: orchestrator.review(case_id, body.approve, body.agent, body.note)
    )


class InspectionBody(BaseModel):
    passed: bool
    agent: str
    note: str | None = None


@app.post(
    "/returns/{case_id}/inspection",
    response_model=ReturnCase,
    dependencies=[Depends(require_role("ops"))],
)
def record_inspection(case_id: str, body: InspectionBody) -> ReturnCase:
    return _http_wrap(
        lambda: orchestrator.record_inspection(
            case_id, body.passed, body.agent, body.note
        )
    )


class CancelBody(BaseModel):
    note: str | None = None


@app.post(
    "/returns/{case_id}/cancel",
    response_model=ReturnCase,
    dependencies=[Depends(require_role("service"))],
)
def cancel_return(case_id: str, body: CancelBody) -> ReturnCase:
    return _http_wrap(
        lambda: orchestrator.cancel(case_id, actor="customer", note=body.note)
    )


class AgentReviewBody(BaseModel):
    auto_apply: bool = True


@app.post(
    "/returns/{case_id}/agent-review",
    dependencies=[Depends(require_role("ops"))],
)
def agent_review(case_id: str, body: AgentReviewBody):
    """Run the multi-agent review board on a case in manual review."""
    return _http_wrap(
        lambda: review_board.review_case(case_id, auto_apply=body.auto_apply)
    )


# -- webhooks --------------------------------------------------------------

class CarrierWebhook(BaseModel):
    tracking_number: str
    event: str  # picked_up | delivered | anything else is logged only


@app.post(
    "/webhooks/carrier",
    response_model=ReturnCase,
    dependencies=[Depends(verify_carrier_signature)],
)
def carrier_webhook(body: CarrierWebhook) -> ReturnCase:
    return _http_wrap(
        lambda: orchestrator.carrier_update(body.tracking_number, body.event)
    )


# -- internal (merchant systems) ---------------------------------------------

@app.post(
    "/internal/orders",
    status_code=204,
    dependencies=[Depends(require_role("ops"))],
)
def register_order(body: OrderRegistration) -> None:
    orchestrator.store.register_order(body.order, body.customer)


@app.post(
    "/internal/outbox/flush",
    dependencies=[Depends(require_role("ops"))],
)
def flush_outbox() -> dict:
    return orchestrator.flush_outbox()


@app.post(
    "/internal/sweep-expired",
    dependencies=[Depends(require_role("ops"))],
)
def sweep_expired() -> dict:
    expired = orchestrator.sweep_expired_labels()
    return {"expired": expired}


# -- observability ---------------------------------------------------------------

@app.get("/metrics", dependencies=[Depends(require_role("service"))])
def metrics() -> dict:
    board_usage = getattr(review_board.llm, "usage", None)
    return {
        "cases_by_status": orchestrator.store.status_counts(),
        "payments": orchestrator.store.outbox_totals(),
        "llm": board_usage.snapshot() if board_usage else {},
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
