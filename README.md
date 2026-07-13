# Returns Orchestrator

A policy-driven returns orchestration service for retail. Merchant systems
register orders and customer profiles; customers file returns against them;
the engine decides automatically whether to approve, reject, or escalate;
a multi-agent AI review board can work the escalation queue; carrier
webhooks track the package; and settlement (refund, exchange, or store
credit) executes through a payment outbox after warehouse inspection —
with a full audit trail of every event.

## Lifecycle

```
requested ──▶ approved ──▶ label_issued ──▶ in_transit ──▶ received ──▶ inspecting
    │             ▲          │ cancelled/expired                            │
    ├──▶ manual_review ──▶ rejected                     ┌─────────────┬────┴────┐
    └──▶ rejected                                       ▼             ▼         ▼
                                                    refunded   exchanged/   closed_failed_
                                                               credited     inspection
```

## Decision engine

Requests are validated against the registered order first (order exists,
SKUs belong to it, quantities not already returned — partial quantities are
fine). Then ordered rules run (`app/policy.py`), resolved per brand/category
from the policy catalog:

1. **Final-sale/excluded category** → reject
2. **Worn/damaged item** → reject, unless the reason is merchant-fault
   (defective, wrong item, not as described) → manual review
3. **Outside window + 7-day grace** → reject (merchant-fault → review);
   inside the grace period → manual review
4. **Missing receipt** (where required) → manual review
5. **High value** (default > $500) → manual review
6. **Serial returner** (>50% return rate, ≥3 orders) → manual review
7. Otherwise → auto-approve and issue a return label

Every case stores the rules that fired (`decision_notes`) and a
`policy_snapshot` of the exact terms it was decided under, so decisions
stay explainable after policies change.

Set `POLICY_CATALOG_CSV=returns_triage_dataset/policies_map.csv` to load
per-brand windows (luxury 14-day, Under Armour 60-day, Beauty final-sale…).

**Evaluated against the 4,000-row labeled triage dataset: 85.7% agreement,
zero false approvals, zero false rejections** — every disagreement is the
engine escalating a labeled-ineligible merchant-fault claim to a human
instead of auto-rejecting. Reproduce with `python scripts/eval_triage.py`
(add `--board --sample 20` to also run the LLM board; needs
`OPENAI_API_KEY`).

## Multi-agent review board (LangGraph)

Cases in `manual_review` can be worked by an LLM review board
(`app/agents/`), built as a **LangGraph `StateGraph`**:

```
                ┌─ fraud_analyst ──────┐
START ──fan-out─┼─ policy_compliance ──┼─▶ synthesize ─▶ decide ─▶ END
                └─ customer_experience ┘                    │
                                              interrupt(): pauses for a human
```

The three specialists run in parallel (one graph superstep); the **policy
compliance** agent is grounded in your indexed brand policy PDFs via
`POLICY_CHUNKS_PKL`. A **lead reviewer** node synthesizes a verdict plus a
drafted customer message.

**Durable execution:** graph state is checkpointed to SQLite
(`RETURNS_GRAPH_DB`), so a crash mid-review resumes from the last completed
node without re-spending tokens, and a paused review survives a restart.

**Human-in-the-loop:** when a verdict can't be auto-applied the graph
pauses at LangGraph's `interrupt()` (`pending_human: true`); an ops agent
resumes it via `POST /returns/{id}/agent-review/resume`, and their decision
flows through the normal state machine and audit trail.

**Guardrails:**
- The board only sees rule-engine escalations.
- Auto-apply is **asymmetric**: only approvals are ever auto-applied (a
  wrong approval costs one bounded refund; a wrong rejection costs a
  customer). Rejections and escalations always pause for a human.
- Auto-apply confidence is **derived from specialist vote agreement**, not
  the model's self-reported number — live evals showed uniform 0.95
  self-reports, so that signal is recorded but never gates anything. An
  approval auto-applies only when agreement clears the threshold (default
  0.75) and no specialist hard-disagrees.
- Customer free text is fenced as untrusted data in every prompt.

**Structured outputs:** every LLM call uses OpenAI structured outputs
(`chat.completions.parse` with a pydantic schema), so malformed JSON isn't
a failure mode — there is no hand-rolled parsing/repair. The `LLMClient` is
an injected protocol with retries/backoff; tests run on fakes, no API key
needed. Token/latency usage is reported per review and in `/metrics`. Set
`LANGSMITH_TRACING=true` (with `LANGSMITH_API_KEY`, `pip install '.[trace]'`)
to trace every call in LangSmith.

## Reliability

- **Idempotency**: send `Idempotency-Key` on `POST /returns`; replays
  return the original case.
- **Optimistic locking**: every case save is a compare-and-swap on a
  version counter; concurrent writers get a 409, not a lost update.
- **Payment outbox**: settlement records intent and case state *before*
  moving money; execution carries an idempotency key and failed attempts
  are retryable via `POST /internal/outbox/flush`. A crash can never move
  money without a record.

## Security

- `RETURNS_API_KEYS="key1:service,key2:ops"` — `service` creates/reads
  returns and cancels; `ops` additionally reviews, inspects, runs the
  board, and uses `/internal/*`. Unset = auth disabled (dev only).
- `CARRIER_WEBHOOK_SECRET` — carrier webhooks must carry an HMAC-SHA256
  signature of the raw body in `X-Carrier-Signature`.
- Customer history and prices come from the server-side registry, never
  from the request payload.

## API

| Endpoint | Role | What it does |
|---|---|---|
| `POST /internal/orders` | ops | Register an order + customer profile |
| `POST /returns` | service | Create a return (validated, auto-decided) |
| `GET /returns`, `GET /returns/{id}` | service | List/inspect cases |
| `POST /returns/{id}/cancel` | service | Cancel before the package ships |
| `POST /returns/{id}/review` | ops | Human approves/rejects an escalation |
| `POST /returns/{id}/agent-review` | ops | Run the multi-agent review board |
| `POST /returns/{id}/agent-review/resume` | ops | Resume a paused review with a human decision |
| `POST /returns/{id}/inspection` | ops | Warehouse pass/fail; triggers settlement |
| `POST /webhooks/carrier` | HMAC | Tracking events (`picked_up`, `delivered`) |
| `POST /internal/outbox/flush` | ops | Retry pending payments |
| `POST /internal/sweep-expired` | ops | Expire unused labels (default 21 days) |
| `GET /metrics` | service | Cases by status, payment totals, LLM usage |

Refunds are `(unit_price − discount) × qty` minus any per-policy restocking
fee; exchanges create a replacement order through the OMS integration; all
outcomes notify the customer through the Notifier integration. Carrier,
payments, OMS, and notifier are injected `Protocol`s with built-in stubs —
swap in EasyPost/Stripe/your OMS without touching workflow logic.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                          # 74 tests, no network needed
.venv/bin/uvicorn app.main:app --reload   # docs at http://localhost:8000/docs
```

Configuration comes from the environment or a `.env` file (see
`app/config.py`): `RETURNS_DB` (SQLite path, `:memory:` for ephemeral),
`RETURNS_GRAPH_DB` (LangGraph checkpoints), `POLICY_CATALOG_CSV`,
`POLICY_CHUNKS_PKL`, `LABEL_EXPIRY_DAYS`, `BOARD_CONFIDENCE_THRESHOLD`,
`LOG_LEVEL` — plus `RETURNS_API_KEYS` and `CARRIER_WEBHOOK_SECRET` (auth,
read per-request), `OPENAI_API_KEY` (enables the board), and
`LANGSMITH_TRACING` / `LANGSMITH_API_KEY` (optional tracing).
