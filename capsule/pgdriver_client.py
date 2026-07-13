"""Thin HTTP client for pg-driver — the capsule-driver requests a SCOPED database+role for a capsule.

The capsule-driver never holds SHIMPZ_PG_DSN (the Postgres superuser). It asks pg-driver, which owns
the superuser and exposes only create/drop, returning a least-privilege proj_<name> DSN. Same bearer +
token-file pattern the brain's own CLIs use. Stdlib only.
"""

from __future__ import annotations

import http.client
import json
import os
from pathlib import Path
from urllib.parse import urlparse

PGDRIVER_URL = os.environ.get("SHIMPZ_PGDRIVER_URL", "http://pg-driver:7072")
TOKEN_FILE = os.environ.get("SHIMPZ_PGDRIVER_TOKEN_FILE", "/run/shimpz-pgdriver/token")


class PgDriverError(Exception):
    """pg-driver refused or was unreachable — surfaced loudly, never a silent DB skip."""


def _call(path: str, payload: dict) -> dict:
    token = Path(TOKEN_FILE).read_text(encoding="utf-8").strip()
    parsed = urlparse(PGDRIVER_URL)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 7072, timeout=30)
    try:
        conn.request(
            "POST",
            path,
            json.dumps(payload),
            {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status != 200:
            raise PgDriverError(f"pg-driver {path} -> {resp.status}: {raw[:200]!r}")
        return json.loads(raw or b"{}")
    finally:
        conn.close()


def create_db(project: str) -> dict:
    """Provision (idempotent) a scoped DB+role for `project`; returns {database_url, created, ...}."""
    return _call("/v1/db/create", {"name": project})


def drop_db(project: str) -> dict:
    """Drop the scoped DB+role for `project`."""
    return _call("/v1/db/drop", {"name": project})
