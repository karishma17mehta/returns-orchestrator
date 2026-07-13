"""API authentication.

Two mechanisms:

- API keys with roles, from RETURNS_API_KEYS ("key1:service,key2:ops").
  `service` can create/read returns; `ops` can additionally review, inspect,
  cancel, run agent reviews, and hit internal endpoints (ops implies service).
- HMAC-SHA256 signatures on the carrier webhook, from CARRIER_WEBHOOK_SECRET.
  The carrier signs the raw request body; we compare digests in constant time.

If an env var is unset, that check is DISABLED and a warning is logged once —
convenient for local dev, never acceptable in production.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

log = logging.getLogger("returns.security")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_ROLE_IMPLIES = {"ops": {"ops", "service"}, "service": {"service"}}
_warned: set[str] = set()


def _load_keys() -> dict[str, str]:
    """Parse RETURNS_API_KEYS into {key: role}."""
    raw = os.environ.get("RETURNS_API_KEYS", "").strip()
    keys: dict[str, str] = {}
    if raw:
        for pair in raw.split(","):
            key, _, role = pair.strip().partition(":")
            if key:
                keys[key] = role or "service"
    return keys


def require_role(role: str):
    def dependency(api_key: str | None = Security(_api_key_header)) -> str:
        keys = _load_keys()
        if not keys:
            if "keys" not in _warned:
                _warned.add("keys")
                log.warning(
                    "RETURNS_API_KEYS is not set — API authentication is DISABLED"
                )
            return "dev"
        if api_key is None:
            raise HTTPException(401, "missing X-API-Key header")
        granted = keys.get(api_key)
        if granted is None:
            raise HTTPException(401, "invalid API key")
        if role not in _ROLE_IMPLIES.get(granted, {granted}):
            raise HTTPException(403, f"role '{granted}' may not perform this action")
        return granted

    return dependency


async def verify_carrier_signature(request: Request) -> None:
    secret = os.environ.get("CARRIER_WEBHOOK_SECRET", "")
    if not secret:
        if "webhook" not in _warned:
            _warned.add("webhook")
            log.warning(
                "CARRIER_WEBHOOK_SECRET is not set — webhook signature "
                "verification is DISABLED"
            )
        return
    provided = request.headers.get("X-Carrier-Signature", "")
    body = await request.body()
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(401, "invalid carrier webhook signature")
