"""Verify a Shimpz account session token against the `accounts` service.

This is how the capsule-driver scopes every op to the authenticated account (Capsule ownership). The
public store only FORWARDS the user's token (it holds no secret); THIS driver is the enforcer — it
verifies the token here and ties/authorizes Capsules by the returned account_id. Stdlib only.
"""

from __future__ import annotations

import http.client
import json
import os
from urllib.parse import urlparse

ACCOUNTS_URL = os.environ.get("SHIMPZ_ACCOUNTS_URL", "http://accounts:7079")


def verify(token: str) -> str | None:
    """Return the account_id for a valid token, else None. NEVER raises — accounts down → None → deny."""
    if not token:
        return None
    parsed = urlparse(ACCOUNTS_URL)
    try:
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 7079, timeout=10)
        conn.request("POST", "/v1/verify", json.dumps({"token": token}), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read() or b"{}")
        conn.close()
    except (OSError, json.JSONDecodeError):
        return None
    return data.get("account_id") if resp.status == 200 else None
