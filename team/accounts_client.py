"""Verify a Shimpz account session token against the `accounts` service.

This is how the team-driver scopes every op to the authenticated account (Team ownership). The
public store only FORWARDS the user's token (it holds no secret); THIS driver is the enforcer — it
verifies the token here and ties/authorizes Teams by the returned account_id. Stdlib only.

SELF-HOST PHONE-HOME: on shimpz.com's own Space this points at the internal `accounts` container; a
self-hosted Space instead sets SHIMPZ_ACCOUNTS_URL=https://shimpz.com/api/accounts — the SAME verify
call then validates the account against shimpz.com (the store's public passthrough), which is what
makes a marketplace install on a self-hosted Space require a real Shimpz account.
"""

from __future__ import annotations

import http.client
import json
import os
from contextlib import suppress
from urllib.parse import urlparse

ACCOUNTS_URL = os.environ.get("SHIMPZ_ACCOUNTS_URL", "http://accounts:7079")
VERIFY_TIMEOUT_SECONDS = 10


def verify(token: str) -> str | None:
    """Return the account_id for a valid token, else None. NEVER raises — accounts down → None → deny."""
    if not token:
        return None
    parsed = urlparse(ACCOUNTS_URL)
    https = parsed.scheme == "https"
    conn_cls = http.client.HTTPSConnection if https else http.client.HTTPConnection
    path = f"{parsed.path.rstrip('/')}/v1/verify"
    conn = None
    try:
        conn = conn_cls(parsed.hostname, parsed.port or (443 if https else 7079), timeout=VERIFY_TIMEOUT_SECONDS)
        conn.request("POST", path, json.dumps({"token": token}), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read() or b"{}")
    except OSError, ValueError, http.client.HTTPException:
        return None
    finally:
        if conn is not None:
            with suppress(OSError):
                conn.close()
    if resp.status != 200 or not isinstance(data, dict):
        return None
    account_id = data.get("account_id")
    return account_id if isinstance(account_id, str) and account_id else None
