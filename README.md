# Returns Orchestrator

A policy-driven returns orchestration service for retail. It takes a return
request, decides automatically whether to approve, reject, or escalate it,
issues a carrier label, tracks the package via carrier webhooks, and settles
the case (refund, exchange, or store credit) after warehouse inspection —
with a full audit trail of every event.

## Lifecycle

```
requested ──▶ approved ──▶ label_issued ──▶ in_transit ──▶ received ──▶ inspecting
    │             ▲                                                        │
    ├──▶ manual_review ──▶ rejected                     ┌─────────────┬────┴────┐
    └──▶ rejected                                       ▼             ▼         ▼
                                                    refunded   exchanged/   closed_failed_
                                                               credited     inspection
```

## Decision engine

Rules run in order on every new request (see `app/policy.py`, all thresholds
configurable via `ReturnPolicy`):

1. **Excluded categories** (perishable, final sale, gift cards…) → reject
2. **Outside return window** (default 30 days) → reject, unless the reason is
   merchant-fault (defective, wrong item, not as described) → manual review
3. **High value** (default > $500) → manual review
4. **Serial returner** (return rate > 50% with ≥3 orders) → manual review
5. Otherwise → auto-approve and issue a return label

Every decision records the rules that fired in `decision_notes`.

## API

| Endpoint | What it does |
|---|---|
| `POST /returns` | Create a return; auto-decided immediately |
| `GET /returns?status=` | List cases, optionally by status |
| `GET /returns/{id}` | Case detail incl. event audit trail |
| `POST /returns/{id}/review` | Agent approves/rejects a manual-review case |
| `POST /returns/{id}/inspection` | Warehouse records inspection pass/fail |
| `POST /webhooks/carrier` | Carrier tracking events (`picked_up`, `delivered`) |

Carrier and payment integrations are injected `Protocol`s
(`app/orchestrator.py`) — the built-in stubs are swappable for real
EasyPost/Stripe/OMS adapters.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/uvicorn app.main:app --reload   # API + docs at http://localhost:8000/docs
.venv/bin/pytest                          # test suite
```

State persists to `returns.db` (override with the `RETURNS_DB` env var,
`:memory:` for ephemeral).
