"""FastAPI surface for the returns orchestrator.

Configuration comes from app.config.Settings (environment or .env file):
RETURNS_DB, RETURNS_GRAPH_DB, POLICY_CATALOG_CSV, POLICY_CHUNKS_PKL,
LABEL_EXPIRY_DAYS, BOARD_CONFIDENCE_THRESHOLD, LOG_LEVEL — plus
RETURNS_API_KEYS / CARRIER_WEBHOOK_SECRET (read per-request in
app.security) and OPENAI_API_KEY / LANGSMITH_* (read by the SDKs).

Domain errors surface through registered exception handlers:
NotFoundError -> 404, ValidationFailure -> 422, TransitionError and
ConcurrencyError -> 409, LLMUnavailableError -> 503.
"""
from __future__ import annotations

import logging

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()  # export .env so the OpenAI/LangSmith SDKs see their keys

from .agents import OpenAIClient, ReviewBoard
from .agents.llm import LLMUnavailableError
from .config import get_settings
from .models import (
    OrderRegistration,
    ReturnCase,
    ReturnRequestCreate,
    ReturnStatus,
)
from .orchestrator import (
    NotFoundError,
    ReturnsOrchestrator,
    TransitionError,
    ValidationFailure,
)
from .policy import PolicyCatalog
from .security import require_role, verify_carrier_signature
from .store import ConcurrencyError, ReturnStore

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("returns.api")

app = FastAPI(title="Returns Orchestrator", version="0.4.0")

_ERROR_STATUS = {
    NotFoundError: 404,
    ValidationFailure: 422,
    TransitionError: 409,
    ConcurrencyError: 409,
    LLMUnavailableError: 503,
}

for exc_type, status_code in _ERROR_STATUS.items():
    def _handler(request: Request, exc: Exception, status_code=status_code):
        detail = str(exc.args[0]) if exc.args else str(exc)
        return JSONResponse(status_code=status_code, content={"detail": detail})

    app.add_exception_handler(exc_type, _handler)


def _build_orchestrator() -> ReturnsOrchestrator:
    catalog = PolicyCatalog()
    if settings.policy_catalog_csv:
        catalog = PolicyCatalog.from_csv(settings.policy_catalog_csv)
        log.info("policy catalog loaded from %s", settings.policy_catalog_csv)
    return ReturnsOrchestrator(
        ReturnStore(settings.returns_db),
        catalog=catalog,
        label_expiry_days=settings.label_expiry_days,
    )


def _build_board(orch: ReturnsOrchestrator) -> ReviewBoard:
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    retriever = None
    if settings.policy_chunks_pkl:
        from .agents.retriever import LexicalPolicyRetriever

        retriever = LexicalPolicyRetriever.from_pickle(settings.policy_chunks_pkl)
        log.info("policy retriever loaded from %s", settings.policy_chunks_pkl)
    # Durable graph state: crash mid-review resumes from the last completed
    # node; paused (interrupted) reviews survive restarts.
    checkpointer = SqliteSaver(
        sqlite3.connect(settings.returns_graph_db, check_same_thread=False)
    )
    return ReviewBoard(
        orch,
        llm=OpenAIClient(),
        retriever=retriever,
        checkpointer=checkpointer,
        confidence_threshold=settings.board_confidence_threshold,
    )


orchestrator = _build_orchestrator()
review_board = _build_board(orchestrator)


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
    return orchestrator.create_return(body, idempotency_key)


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
    return orchestrator.review(case_id, body.approve, body.agent, body.note)


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
    return orchestrator.record_inspection(case_id, body.passed, body.agent, body.note)


class CancelBody(BaseModel):
    note: str | None = None


@app.post(
    "/returns/{case_id}/cancel",
    response_model=ReturnCase,
    dependencies=[Depends(require_role("service"))],
)
def cancel_return(case_id: str, body: CancelBody) -> ReturnCase:
    return orchestrator.cancel(case_id, actor="customer", note=body.note)


class AgentReviewBody(BaseModel):
    auto_apply: bool = True


@app.post(
    "/returns/{case_id}/agent-review",
    dependencies=[Depends(require_role("ops"))],
)
def agent_review(case_id: str, body: AgentReviewBody):
    """Run the multi-agent review board on a case in manual review. If the
    verdict can't be auto-applied, the review pauses for a human decision
    (pending_human=true) — feed it via the resume endpoint."""
    return review_board.review_case(case_id, auto_apply=body.auto_apply)


class ResumeReviewBody(BaseModel):
    approve: bool
    agent: str
    note: str | None = None


@app.post(
    "/returns/{case_id}/agent-review/resume",
    dependencies=[Depends(require_role("ops"))],
)
def resume_agent_review(case_id: str, body: ResumeReviewBody):
    """Resume a paused review with the human decision."""
    return review_board.resume_review(
        case_id, approve=body.approve, agent=body.agent, note=body.note
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
    return orchestrator.carrier_update(body.tracking_number, body.event)


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
    return {"expired": orchestrator.sweep_expired_labels()}


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
