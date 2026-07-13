"""Offline regression gate for the rule engine against the labeled dataset.

Runs with no network/API key. Guards the two invariants that must never
regress: the engine must not auto-approve a policy-ineligible return, nor
auto-reject a genuinely eligible one, and overall agreement must stay above
a floor. The LLM board sweep is gated separately in CI (needs a key).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("pandas")  # only needed for the retriever, keep parity

import eval_triage  # noqa: E402


@pytest.fixture(scope="module")
def metrics():
    if not (eval_triage.DATASET / "triage_training.csv").exists():
        pytest.skip("triage dataset not present")
    rows = eval_triage.load_rows()
    customers = eval_triage.load_customers()
    _results, m = eval_triage.run_rule_engine_eval(rows, customers)
    return m


def test_no_false_approvals(metrics):
    # Auto-approving a policy-ineligible return moves money it shouldn't.
    assert metrics["false_approvals"] == 0


def test_no_false_rejections(metrics):
    # Auto-rejecting a genuinely eligible return loses a customer.
    assert metrics["false_rejections"] == 0


def test_accuracy_floor(metrics):
    assert metrics["accuracy"] >= 0.80, (
        f"rule-engine agreement dropped to {metrics['accuracy']:.3f}"
    )
