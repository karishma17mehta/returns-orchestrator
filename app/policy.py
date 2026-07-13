"""Policy catalog and decision engine.

`PolicyCatalog` resolves per-brand/per-category return terms (loadable from
a CSV like returns_triage_dataset/policies_map.csv); `ReturnPolicy` holds
the merchant-wide risk thresholds. `evaluate()` runs the ordered rules and
returns a decision plus a snapshot of the exact terms used, so every case
records what policy it was decided under.

Rules (first terminal rule wins):
 1. final-sale / excluded category            -> reject
 2. item worn or damaged                      -> reject, unless merchant-fault -> review
 3. outside window + 7-day grace              -> reject, unless merchant-fault -> review
    inside grace (window < age <= window+7)   -> manual review
 4. receipt required but missing              -> manual review
 5. value above threshold                     -> manual review
 6. customer return rate above threshold      -> manual review
 7. otherwise                                 -> approve
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .models import (
    MERCHANT_FAULT_REASONS,
    CustomerProfile,
    ItemCondition,
    ReturnCase,
    ReturnStatus,
)

ENGINE_VERSION = "2026-07-13.1"


@dataclass
class ReturnPolicy:
    """Merchant-wide defaults and risk thresholds."""

    return_window_days: int = 30
    grace_period_days: int = 7
    excluded_categories: set[str] = field(
        default_factory=lambda: {"final_sale", "perishable", "gift_card", "intimates"}
    )
    requires_receipt: bool = True
    restock_fee_pct: float = 0.0
    manual_review_value_threshold: float = 500.0
    manual_review_return_rate: float = 0.5
    return_rate_min_orders: int = 3


@dataclass(frozen=True)
class BrandPolicy:
    """Per-brand/category overrides resolved from the catalog."""

    return_window_days: int
    final_sale: bool = False
    requires_receipt: bool = True
    restock_fee_pct: float = 0.0
    notes: str = ""


class PolicyCatalog:
    def __init__(self, entries: dict[tuple[str, str], BrandPolicy] | None = None):
        self._entries = entries or {}

    @classmethod
    def from_csv(cls, path: str) -> "PolicyCatalog":
        """Load from a policies_map.csv (brand,category,return_window_days,
        final_sale,requires_receipt,restock_fee,notes)."""
        entries: dict[tuple[str, str], BrandPolicy] = {}
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                entries[(row["brand"].lower(), row["category"].lower())] = BrandPolicy(
                    return_window_days=int(row["return_window_days"]),
                    final_sale=row["final_sale"].strip().lower() == "true",
                    requires_receipt=row["requires_receipt"].strip().lower() == "true",
                    restock_fee_pct=float(row.get("restock_fee") or 0.0),
                    notes=row.get("notes", ""),
                )
        return cls(entries)

    def resolve(self, brand: str, category: str, defaults: ReturnPolicy) -> BrandPolicy:
        entry = self._entries.get((brand.lower(), category.lower()))
        if entry is not None:
            return entry
        return BrandPolicy(
            return_window_days=defaults.return_window_days,
            final_sale=category.lower() in defaults.excluded_categories,
            requires_receipt=defaults.requires_receipt,
            restock_fee_pct=defaults.restock_fee_pct,
            notes="merchant default",
        )


@dataclass
class Decision:
    status: ReturnStatus  # APPROVED, REJECTED, or MANUAL_REVIEW
    notes: list[str] = field(default_factory=list)
    policy_snapshot: dict = field(default_factory=dict)


def evaluate(case: ReturnCase, policy: ReturnPolicy, catalog: PolicyCatalog) -> Decision:
    """Evaluate a resolved return case. `case.items` must already be resolved
    against the registered order (brand/category/prices present)."""
    notes: list[str] = []
    customer: CustomerProfile = case.customer

    resolved = {
        (line.brand, line.category): catalog.resolve(line.brand, line.category, policy)
        for line in case.items
    }
    # The strictest terms across the case's lines govern the whole case.
    window = min(bp.return_window_days for bp in resolved.values())
    requires_receipt = any(bp.requires_receipt for bp in resolved.values())
    restock_fee_pct = max(bp.restock_fee_pct for bp in resolved.values())
    snapshot = {
        "engine_version": ENGINE_VERSION,
        "window_days": window,
        "grace_days": policy.grace_period_days,
        "requires_receipt": requires_receipt,
        "restock_fee_pct": restock_fee_pct,
        "value_threshold": policy.manual_review_value_threshold,
        "return_rate_threshold": policy.manual_review_return_rate,
        "terms": {
            f"{b}/{c}": {
                "window_days": bp.return_window_days,
                "final_sale": bp.final_sale,
                "notes": bp.notes,
            }
            for (b, c), bp in resolved.items()
        },
    }

    def decision(status: ReturnStatus) -> Decision:
        return Decision(status, notes, snapshot)

    # 1. Final-sale / excluded categories are a hard reject regardless of reason.
    final_sale = [
        line for line in case.items
        if resolved[(line.brand, line.category)].final_sale
    ]
    if final_sale:
        skus = ", ".join(line.sku for line in final_sale)
        notes.append(f"rejected: final-sale/non-returnable category ({skus})")
        return decision(ReturnStatus.REJECTED)

    # 2. Worn or damaged items are not resellable. Damage the merchant caused
    #    (defective, wrong item...) goes to a human instead.
    if case.item_condition in (ItemCondition.WORN, ItemCondition.DAMAGED):
        if case.reason in MERCHANT_FAULT_REASONS:
            notes.append(
                f"manual review: item reported {case.item_condition.value} with "
                f"merchant-fault reason '{case.reason.value}'"
            )
            return decision(ReturnStatus.MANUAL_REVIEW)
        notes.append(f"rejected: item condition '{case.item_condition.value}'")
        return decision(ReturnStatus.REJECTED)

    # 3. Return window with grace period.
    order_date = case.order_date
    if order_date.tzinfo is None:
        order_date = order_date.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - order_date).days
    if age_days > window + policy.grace_period_days:
        if case.reason in MERCHANT_FAULT_REASONS:
            notes.append(
                f"manual review: outside {window}-day window ({age_days} days) but "
                f"reason '{case.reason.value}' is merchant-fault"
            )
            return decision(ReturnStatus.MANUAL_REVIEW)
        notes.append(f"rejected: outside {window}-day return window ({age_days} days)")
        return decision(ReturnStatus.REJECTED)
    if age_days > window:
        notes.append(
            f"manual review: {age_days} days is past the {window}-day window "
            f"but within the {policy.grace_period_days}-day grace period"
        )
        return decision(ReturnStatus.MANUAL_REVIEW)

    # 4. Receipt.
    if requires_receipt and not case.has_receipt:
        notes.append("manual review: receipt required but not provided")
        return decision(ReturnStatus.MANUAL_REVIEW)

    # 5-6. Risk thresholds.
    total = case.total_value
    if total > policy.manual_review_value_threshold:
        notes.append(
            f"manual review: value ${total:.2f} exceeds "
            f"${policy.manual_review_value_threshold:.2f} threshold"
        )
        return decision(ReturnStatus.MANUAL_REVIEW)

    if (
        customer.lifetime_orders >= policy.return_rate_min_orders
        and customer.return_rate > policy.manual_review_return_rate
    ):
        notes.append(
            f"manual review: customer return rate {customer.return_rate:.0%} exceeds "
            f"{policy.manual_review_return_rate:.0%} threshold"
        )
        return decision(ReturnStatus.MANUAL_REVIEW)

    notes.append(f"auto-approved: within {window}-day window ({age_days} days), no risk flags")
    return decision(ReturnStatus.APPROVED)
