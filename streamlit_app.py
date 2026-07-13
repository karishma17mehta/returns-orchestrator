"""Streamlit demo dashboard for the Returns Orchestrator.

Runs the orchestrator + review board in-process (the same code the FastAPI
service exposes) against an ephemeral in-memory store, so the whole return
lifecycle is demoable end to end without standing up a server or a database.

The multi-agent board defaults to a FREE simulated LLM (deterministic,
grounded in the same policy facts the real agents use) so the demo costs
nothing. Flip "Board engine" to Live LLM to use the real OpenAI calls when
OPENAI_API_KEY is set.

    pip install -e ".[dev,demo]"
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make the `app` package importable no matter what directory Streamlit is
# launched from (local run, Streamlit Community Cloud, etc.).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from app.agents import ReviewBoard
from app.agents.specialists import AssessmentPayload
from app.agents.coordinator import LeadVerdictPayload
from app.models import (
    CustomerProfile,
    ItemCondition,
    Order,
    OrderItem,
    Resolution,
    ReturnLineRequest,
    ReturnReason,
    ReturnRequestCreate,
    ReturnStatus,
)
from app.orchestrator import ReturnsOrchestrator, TransitionError, ValidationFailure
from app.policy import PolicyCatalog, ReturnPolicy, policy_facts
from app.store import ReturnStore

ROOT = Path(__file__).resolve().parent
CATALOG_CSV = ROOT / "returns_triage_dataset" / "policies_map.csv"
CHUNKS_PKL = ROOT / "policy_index_baseline" / "df_chunks.pkl"

STATUS_EMOJI = {
    ReturnStatus.REQUESTED: "📝",
    ReturnStatus.MANUAL_REVIEW: "🧑‍⚖️",
    ReturnStatus.APPROVED: "✅",
    ReturnStatus.REJECTED: "⛔",
    ReturnStatus.LABEL_ISSUED: "🏷️",
    ReturnStatus.IN_TRANSIT: "🚚",
    ReturnStatus.RECEIVED: "📦",
    ReturnStatus.INSPECTING: "🔍",
    ReturnStatus.REFUNDED: "💸",
    ReturnStatus.EXCHANGED: "🔁",
    ReturnStatus.CREDITED: "🎟️",
    ReturnStatus.CLOSED_FAILED_INSPECTION: "❌",
    ReturnStatus.CANCELLED: "🚫",
    ReturnStatus.EXPIRED: "⌛",
}


# --------------------------------------------------------------------------
# Simulated LLM — deterministic, free. Mirrors the real agents' logic so the
# board flow is fully demoable with no API cost.
# --------------------------------------------------------------------------

class SimulatedLLM:
    """A stand-in LLMClient that returns plausible structured outputs derived
    from the same evidence the real agents see. No network, no cost."""

    class _Usage:
        # Keys must match UsageTracker (all numeric): the board computes a
        # numeric delta over them, so no string fields here.
        def snapshot(self):
            return {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    usage = _Usage()

    def complete(self, system: str, user: str, output_model):
        if "lead reviewer" in system:
            return self._lead(user)
        if "policy compliance" in system:
            return self._policy(user)
        if "fraud analyst" in system:
            return self._fraud(user)
        return self._cx(user)

    @staticmethod
    def _has_disqualifier(user: str) -> bool:
        if "HARD DISQUALIFIERS" not in user:
            return False
        tail = user.split("HARD DISQUALIFIERS", 1)[1]
        return "no hard disqualifier present" not in tail

    def _policy(self, user: str) -> AssessmentPayload:
        if self._has_disqualifier(user):
            return AssessmentPayload(
                recommendation="reject", confidence=0.9,
                rationale="The written policy does not permit this return: a hard "
                "disqualifier (window/condition/receipt/final-sale) is present.",
            )
        return AssessmentPayload(
            recommendation="approve", confidence=0.85,
            rationale="No hard disqualifier in the policy determination; the return "
            "is within the written policy.",
        )

    def _fraud(self, user: str) -> AssessmentPayload:
        if "return rate" in user.lower() and "exceeds" in user.lower():
            return AssessmentPayload(
                recommendation="escalate", confidence=0.6,
                rationale="Elevated return rate is worth a human look, though not "
                "conclusive abuse on its own.",
            )
        return AssessmentPayload(
            recommendation="approve", confidence=0.8,
            rationale="No strong abuse indicators: order history and item type look "
            "consistent with a genuine return.",
        )

    def _cx(self, user: str) -> AssessmentPayload:
        return AssessmentPayload(
            recommendation="approve", confidence=0.7,
            rationale="From a retention standpoint a goodwill approval protects the "
            "customer relationship; deferring to policy on hard limits.",
        )

    def _lead(self, user: str) -> LeadVerdictPayload:
        # Policy compliance carries the most weight.
        policy_rejects = "[policy_compliance] recommends reject" in user
        approves = user.count("recommends approve")
        if policy_rejects:
            return LeadVerdictPayload(
                decision="reject", confidence=0.85,
                rationale="Policy compliance found the return outside the written "
                "policy; goodwill does not override a hard policy limit.",
                customer_message="We're sorry — after review we're unable to accept "
                "this return under our policy. Please reach out if you have questions.",
            )
        if approves >= 2:
            return LeadVerdictPayload(
                decision="approve", confidence=0.85,
                rationale="No policy disqualifier and the board broadly supports the "
                "return; approving.",
                customer_message="Good news — your return has been approved! A prepaid "
                "label is on its way to your inbox.",
            )
        return LeadVerdictPayload(
            decision="escalate", confidence=0.5,
            rationale="The board is split; a human should make the final call.",
            customer_message="Thanks for your patience — your request needs a few more "
            "days of review by our team.",
        )


# --------------------------------------------------------------------------
# Orchestrator wiring (persists across Streamlit reruns)
# --------------------------------------------------------------------------

@st.cache_resource
def build_orchestrator() -> ReturnsOrchestrator:
    catalog = PolicyCatalog.from_csv(str(CATALOG_CSV)) if CATALOG_CSV.exists() else PolicyCatalog()
    orch = ReturnsOrchestrator(ReturnStore(":memory:"), catalog=catalog)
    _seed_demo_orders(orch)
    _seed_demo_cases(orch)
    return orch


def _retriever():
    if CHUNKS_PKL.exists():
        from app.agents.retriever import LexicalPolicyRetriever

        return LexicalPolicyRetriever.from_pickle(str(CHUNKS_PKL))
    return None


@st.cache_resource
def build_board(_orch: ReturnsOrchestrator, live: bool) -> ReviewBoard:
    # Cached so the LangGraph checkpointer persists across Streamlit reruns —
    # otherwise a review paused for a human (interrupt) would be lost on the
    # next interaction. (_orch is underscored to skip Streamlit hashing.)
    if live:
        from app.agents import OpenAIClient

        llm = OpenAIClient()
    else:
        llm = SimulatedLLM()
    return ReviewBoard(_orch, llm, retriever=_retriever())


def _board_pending(board: ReviewBoard, case_id: str) -> bool:
    """True if a board review for this case is paused awaiting a human."""
    state = board.graph.get_state({"configurable": {"thread_id": case_id}})
    return bool(state.next)


DEMO_ORDERS = [
    # (order_id, brand, category, name, price, days_ago, orders, returns, note)
    ("ORD-1001", "H&M", "Tops", "Cotton t-shirt", 24.0, 6, 8, 1, "in window, low risk"),
    ("ORD-1002", "Gucci", "Bags", "Marmont tote", 2200.0, 20, 4, 1, "luxury, past 14-day window"),
    ("ORD-1003", "Under Armour", "Activewear", "Training shorts", 45.0, 40, 12, 2, "UA 60-day window"),
    ("ORD-1004", "Zara", "Dresses", "Linen dress", 89.0, 12, 15, 9, "serial returner"),
    ("ORD-1005", "Shein", "Tops", "Graphic tee", 12.0, 8, 3, 0, "cheap, in window"),
]


def _seed_demo_orders(orch: ReturnsOrchestrator) -> None:
    for oid, brand, cat, name, price, days, norders, nreturns, _note in DEMO_ORDERS:
        order = Order(
            order_id=oid,
            customer_id=f"cus_{oid[-4:]}",
            order_date=datetime.now(timezone.utc) - timedelta(days=days),
            items=[OrderItem(sku=f"{oid}-A", name=name, brand=brand,
                             category=cat, quantity=1, unit_price=price)],
        )
        cust = CustomerProfile(customer_id=order.customer_id,
                               lifetime_orders=norders, lifetime_returns=nreturns)
        orch.store.register_order(order, cust)


def _seed_demo_cases(orch: ReturnsOrchestrator) -> None:
    """Pre-file a few returns so the Cases tab is populated on first load —
    one per interesting outcome (auto-approved, two manual reviews, rejected).
    (order_id, reason, condition, resolution, has_receipt)"""
    samples = [
        ("ORD-1001", ReturnReason.SIZE_FIT, ItemCondition.NEW, Resolution.REFUND, True),
        ("ORD-1002", ReturnReason.SIZE_FIT, ItemCondition.NEW, Resolution.REFUND, True),
        ("ORD-1004", ReturnReason.DEFECTIVE, ItemCondition.TRIED_ON, Resolution.REFUND, True),
        ("ORD-1003", ReturnReason.NO_LONGER_NEEDED, ItemCondition.WORN, Resolution.REFUND, True),
    ]
    for oid, reason, condition, resolution, receipt in samples:
        orch.create_return(ReturnRequestCreate(
            order_id=oid,
            lines=[ReturnLineRequest(sku=f"{oid}-A", quantity=1)],
            reason=reason, requested_resolution=resolution,
            item_condition=condition, has_receipt=receipt,
        ))


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="Returns Orchestrator", page_icon="📦", layout="wide")
orch = build_orchestrator()

st.title("📦 Returns Orchestrator")
st.caption(
    "Policy-driven returns automation with a multi-agent AI review board. "
    "This dashboard runs the same engine the API exposes, in-process."
)

# -- sidebar: controls + metrics ------------------------------------------
with st.sidebar:
    st.header("Board engine")
    key_present = bool(os.environ.get("OPENAI_API_KEY"))
    live = st.toggle(
        "Live LLM (uses OpenAI)", value=False,
        help="Off = free simulated board. On = real gpt-4o-mini calls; needs OPENAI_API_KEY.",
    )
    if live and not key_present:
        st.error("OPENAI_API_KEY not set — falling back to simulated board.")
        live = False
    st.caption("🟢 Live LLM" if live else "⚪ Simulated (free)")

    st.divider()
    st.header("Metrics")
    counts = orch.store.status_counts()
    totals = orch.store.outbox_totals()
    st.metric("Total returns", sum(counts.values()))
    c1, c2 = st.columns(2)
    c1.metric("Refunds issued", totals["executed_count"])
    c2.metric("Refund $", f"${totals['executed_total']:,.0f}")
    if counts:
        st.caption("By status")
        for s, n in sorted(counts.items()):
            st.write(f"{STATUS_EMOJI.get(ReturnStatus(s), '•')} {s.replace('_', ' ')}: **{n}**")

board = build_board(orch, live)

tab_new, tab_cases = st.tabs(["➕ New return", "📋 Cases & lifecycle"])

# -- new return -----------------------------------------------------------
with tab_new:
    st.subheader("File a return against a seeded order")
    st.caption("Five demo orders are pre-loaded (see the scenarios below).")

    order_map = {f"{o[0]} — {o[1]} {o[3]} (${o[4]:,.0f}) · {o[8]}": o for o in DEMO_ORDERS}
    with st.form("new_return"):
        pick = st.selectbox("Order", list(order_map.keys()))
        chosen = order_map[pick]
        col1, col2, col3 = st.columns(3)
        reason = col1.selectbox("Reason", [r.value for r in ReturnReason], index=5)
        condition = col2.selectbox("Item condition", [c.value for c in ItemCondition])
        resolution = col3.selectbox("Resolution", [r.value for r in Resolution])
        has_receipt = st.checkbox("Customer has receipt", value=True)
        comment = st.text_input("Customer comment (optional)")
        submitted = st.form_submit_button("Submit return", type="primary")

    if submitted:
        req = ReturnRequestCreate(
            order_id=chosen[0],
            lines=[ReturnLineRequest(sku=f"{chosen[0]}-A", quantity=1)],
            reason=ReturnReason(reason),
            requested_resolution=Resolution(resolution),
            item_condition=ItemCondition(condition),
            has_receipt=has_receipt,
            comment=comment or None,
        )
        try:
            case = orch.create_return(req)
            st.session_state["selected_case"] = case.id
            emoji = STATUS_EMOJI.get(case.status, "•")
            st.success(f"{emoji} Return **{case.id}** → **{case.status.value}**")
            st.write("**Why the engine decided this:**")
            for n in case.decision_notes:
                st.write(f"- {n}")
            snap = case.policy_snapshot
            st.info(
                f"Policy applied: **{snap.get('window_days')}-day** window "
                f"(+{snap.get('grace_days')}-day grace) · "
                f"receipt required: {snap.get('requires_receipt')}"
            )
            if case.status == ReturnStatus.MANUAL_REVIEW:
                st.warning("→ Routed to manual review. Open the **Cases** tab to run the AI board.")
        except (ValidationFailure, TransitionError) as e:
            st.error(str(e))


# -- cases & lifecycle ----------------------------------------------------
def _lifecycle_actions(case):
    """Render status-appropriate action buttons; return True if state changed."""
    changed = False
    cols = st.columns(4)
    s = case.status
    if s == ReturnStatus.MANUAL_REVIEW:
        pending = _board_pending(board, case.id)
        if not pending and cols[0].button("🤖 Run AI board", key=f"board_{case.id}"):
            with st.spinner("Convening the review board…"):
                st.session_state["board_outcome"] = board.review_case(case.id, auto_apply=True)
            changed = True

        def _finalize(approve: bool):
            # If the board paused this case for a human, resume that graph;
            # otherwise it's a straight human decision.
            if _board_pending(board, case.id):
                st.session_state["board_outcome"] = board.resume_review(
                    case.id, approve=approve, agent="demo-cs")
            else:
                orch.review(case.id, approve=approve, agent="demo-cs")

        if cols[1].button("✅ Approve (human)", key=f"appr_{case.id}"):
            _finalize(True); changed = True
        if cols[2].button("⛔ Reject (human)", key=f"rej_{case.id}"):
            _finalize(False); changed = True
    if s in (ReturnStatus.LABEL_ISSUED, ReturnStatus.IN_TRANSIT):
        if cols[0].button("🚚 Mark picked up", key=f"pick_{case.id}"):
            orch.carrier_update(case.label_tracking_number, "picked_up"); changed = True
        if cols[1].button("📦 Mark delivered", key=f"deliv_{case.id}"):
            orch.carrier_update(case.label_tracking_number, "delivered"); changed = True
    if s in (ReturnStatus.LABEL_ISSUED, ReturnStatus.APPROVED, ReturnStatus.MANUAL_REVIEW):
        if cols[3].button("🚫 Cancel", key=f"canc_{case.id}"):
            orch.cancel(case.id); changed = True
    if s in (ReturnStatus.RECEIVED, ReturnStatus.INSPECTING):
        if cols[0].button("🔍 Inspection PASS", key=f"pass_{case.id}"):
            orch.record_inspection(case.id, passed=True, agent="demo-wh"); changed = True
        if cols[1].button("❌ Inspection FAIL", key=f"fail_{case.id}"):
            orch.record_inspection(case.id, passed=False, agent="demo-wh"); changed = True
    return changed


with tab_cases:
    cases = orch.store.list()
    if not cases:
        st.info("No returns yet — file one in the **New return** tab.")
    else:
        labels = {
            f"{STATUS_EMOJI.get(c.status,'•')} {c.id} · {c.items[0].brand} "
            f"{c.items[0].name} · {c.status.value}": c.id
            for c in reversed(cases)
        }
        default = st.session_state.get("selected_case")
        keys = list(labels.keys())
        idx = next((i for i, k in enumerate(keys) if labels[k] == default), 0)
        pick = st.selectbox("Select a case", keys, index=idx)
        case = orch.store.get(labels[pick])

        left, right = st.columns([3, 2])
        with left:
            st.subheader(f"{STATUS_EMOJI.get(case.status,'•')} {case.status.value.replace('_',' ').title()}")
            item = case.items[0]
            st.write(f"**{item.brand} {item.name}** · ${item.unit_price:,.2f} · "
                     f"reason: {case.reason.value} · condition: {case.item_condition.value}")
            if case.refund_amount is not None:
                st.write(f"💸 Refund: **${case.refund_amount:,.2f}**"
                         + (f" (−${case.restock_fee:,.2f} restock)" if case.restock_fee else ""))
            if case.replacement_order_id:
                st.write(f"🔁 Replacement order: `{case.replacement_order_id}`")

            st.markdown("**Actions**")
            if _lifecycle_actions(case):
                st.rerun()

            st.markdown("**Audit trail**")
            for e in case.events:
                st.write(f"`{e.at:%H:%M:%S}` **{e.actor}** — {e.event}"
                         + (f": {e.detail}" if e.detail else ""))

        with right:
            outcome = st.session_state.get("board_outcome")
            if outcome and outcome.case_id == case.id:
                st.markdown("### 🤖 AI review board")
                verdict = outcome.verdict
                st.write(f"**Verdict:** {verdict.decision.value} · "
                         f"agreement confidence **{outcome.derived_confidence:.0%}**")
                if outcome.pending_human:
                    st.warning("⏸️ Paused for a human decision (not auto-applied).")
                elif outcome.applied:
                    st.success(f"Auto-applied by {outcome.applied_by}.")
                st.markdown("**Specialist assessments**")
                for a in outcome.assessments:
                    st.write(f"- **{a.agent}** → _{a.recommendation.value}_ "
                             f"({a.confidence:.0%}): {a.rationale}")
                st.markdown("**Lead rationale**")
                st.write(verdict.rationale)
                st.markdown("**Drafted customer message**")
                st.info(verdict.customer_message)
            else:
                facts = policy_facts(case, orch.policy, orch.catalog)
                st.markdown("### Policy determination")
                st.write(f"Item age: **{facts['age_days']} days** vs "
                         f"{facts['window_days']}-day window (+{facts['grace_days']} grace)")
                disq = facts["hard_disqualifiers"]
                if disq:
                    st.error("Hard disqualifiers:\n" + "\n".join(f"- {d}" for d in disq))
                else:
                    st.success("No hard policy disqualifier.")
                if case.status == ReturnStatus.MANUAL_REVIEW:
                    st.caption("Click **Run AI board** to see the multi-agent review.")
