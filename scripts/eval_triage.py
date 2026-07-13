#!/usr/bin/env python3
"""Evaluate the decision engine (and optionally the LLM review board)
against the labeled triage dataset.

    python scripts/eval_triage.py
    python scripts/eval_triage.py --board --sample 20   # needs OPENAI_API_KEY

Dataset labels map to engine decisions as:
    eligible      -> approved
    needs_review  -> manual_review
    ineligible    -> rejected

The rule-engine eval is deterministic and offline. --board runs the
multi-agent review board (real LLM calls) over a sample of rows the engine
sends to manual review, and reports how the board's verdicts split.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import (
    CustomerProfile,
    ItemCondition,
    ReturnCase,
    ReturnLine,
    ReturnReason,
    ReturnStatus,
)
from app.policy import PolicyCatalog, ReturnPolicy, evaluate

DATASET = Path(__file__).resolve().parent.parent / "returns_triage_dataset"

LABEL_TO_STATUS = {
    "eligible": ReturnStatus.APPROVED,
    "needs_review": ReturnStatus.MANUAL_REVIEW,
    "ineligible": ReturnStatus.REJECTED,
}

REASON_MAP = {
    "too small": ReturnReason.SIZE_FIT,
    "too large": ReturnReason.SIZE_FIT,
    "changed mind": ReturnReason.NO_LONGER_NEEDED,
    "defective": ReturnReason.DEFECTIVE,
    "wrong item": ReturnReason.WRONG_ITEM,
    "no reason": ReturnReason.OTHER,
}

CONDITION_MAP = {
    "new": ItemCondition.NEW,
    "tried-on": ItemCondition.TRIED_ON,
    "worn": ItemCondition.WORN,
    "damaged": ItemCondition.DAMAGED,
}


def load_rows() -> list[dict]:
    with open(DATASET / "triage_training.csv", newline="") as f:
        return list(csv.DictReader(f))


def load_customers() -> dict[str, dict]:
    with open(DATASET / "customers.csv", newline="") as f:
        return {r["customer_id"]: r for r in csv.DictReader(f)}


def row_to_case(row: dict, customers: dict[str, dict]) -> ReturnCase:
    """Build a resolved ReturnCase from a dataset row. The dataset gives
    days_since_purchase directly, so order_date is back-computed from now."""
    days = int(row["days_since_purchase"])
    cust_row = customers.get(row["customer_id"], {})
    # The dataset stores a return_rate fraction; express it as counts.
    rate = float(cust_row.get("return_rate", 0.1))
    lifetime_orders = 20
    return ReturnCase(
        order_id=str(row["order_id"]),
        customer=CustomerProfile(
            customer_id=str(row["customer_id"]),
            lifetime_orders=lifetime_orders,
            lifetime_returns=round(rate * lifetime_orders),
        ),
        items=[
            ReturnLine(
                sku=f"SKU-{row['order_item_id']}",
                name=f"{row['brand']} {row['category']}",
                brand=row["brand"],
                category=row["category"],
                quantity=1,
                unit_price=float(row["price"]),
                discount=float(row.get("discount") or 0.0),
            )
        ],
        reason=REASON_MAP[row["return_reason"].strip().lower()],
        requested_resolution="refund",
        item_condition=CONDITION_MAP[row["item_condition"].strip().lower()],
        has_receipt=row["receipt_flag"] == "1",
        order_date=datetime.now(timezone.utc) - timedelta(days=days),
    )


def run_rule_engine_eval(rows: list[dict], customers: dict) -> list[tuple[dict, ReturnStatus]]:
    catalog = PolicyCatalog.from_csv(str(DATASET / "policies_map.csv"))
    # Match the dataset's label logic: no risk thresholds in the labels,
    # so lift the value/return-rate review triggers out of the way.
    policy = ReturnPolicy(
        manual_review_value_threshold=float("inf"),
        manual_review_return_rate=1.1,
    )
    results = []
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    correct = 0
    for row in rows:
        case = row_to_case(row, customers)
        decision = evaluate(case, policy, catalog)
        predicted = decision.status
        expected = LABEL_TO_STATUS[row["label"]]
        confusion[(row["label"], predicted.value)] += 1
        if predicted is expected:
            correct += 1
        results.append((row, predicted))

    n = len(rows)
    print(f"\nRule engine vs labels: {correct}/{n} = {correct/n:.1%} agreement\n")
    statuses = ["approved", "manual_review", "rejected"]
    print(f"{'label ↓ / engine →':>20} " + " ".join(f"{s:>14}" for s in statuses))
    for label in ("eligible", "needs_review", "ineligible"):
        cells = " ".join(f"{confusion[(label, s)]:>14}" for s in statuses)
        print(f"{label:>20} {cells}")
    print()
    return results


def run_board_eval(results, customers, sample_n: int) -> None:
    from app.agents import OpenAIClient, ReviewBoard
    from app.agents.retriever import LexicalPolicyRetriever
    from app.orchestrator import ReturnsOrchestrator
    from app.store import ReturnStore

    review_rows = [
        (row, s) for row, s in results if s is ReturnStatus.MANUAL_REVIEW
    ]
    if not review_rows:
        print("no manual_review rows to sample for the board")
        return
    random.seed(7)
    sample = random.sample(review_rows, min(sample_n, len(review_rows)))

    chunks_pkl = Path(__file__).resolve().parent.parent / "policy_index_baseline/df_chunks.pkl"
    retriever = (
        LexicalPolicyRetriever.from_pickle(str(chunks_pkl))
        if chunks_pkl.exists()
        else None
    )
    store = ReturnStore(":memory:")
    orch = ReturnsOrchestrator(store)
    llm = OpenAIClient()
    board = ReviewBoard(orch, llm, retriever=retriever)

    verdicts = Counter()
    applied = 0
    for i, (row, _status) in enumerate(sample, 1):
        case = row_to_case(row, customers)
        case.status = ReturnStatus.MANUAL_REVIEW
        case.decision_notes = ["manual review (dataset eval)"]
        store.save(case)
        outcome = board.review_case(case.id, auto_apply=False)
        verdicts[outcome.verdict.decision.value] += 1
        if board._may_apply(outcome.verdict, outcome.assessments):
            applied += 1
        print(
            f"  [{i}/{len(sample)}] {row['brand']}/{row['category']} "
            f"${row['price']} label={row['label']} -> "
            f"board={outcome.verdict.decision.value} "
            f"(conf {outcome.verdict.confidence:.2f})"
        )

    print(f"\nBoard verdicts over {len(sample)} manual-review rows: {dict(verdicts)}")
    print(f"Would auto-apply: {applied}/{len(sample)}")
    print(f"LLM usage: {llm.usage.snapshot()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", action="store_true", help="also run the LLM review board")
    parser.add_argument("--sample", type=int, default=10, help="rows to sample for --board")
    args = parser.parse_args()

    rows = load_rows()
    customers = load_customers()
    print(f"{len(rows)} labeled rows, labels: {Counter(r['label'] for r in rows)}")
    results = run_rule_engine_eval(rows, customers)

    if args.board:
        run_board_eval(results, customers, args.sample)


if __name__ == "__main__":
    main()
