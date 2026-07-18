#!/usr/local/bin/python3
"""Authenticated, bounded liveness probe for team-driver-local."""

from __future__ import annotations

import http.client
import json
from pathlib import Path

TOKEN_PATH = Path("/run/shimpz-local/token")


def main() -> int:
    try:
        token = TOKEN_PATH.read_text(encoding="ascii")
        if len(token) != 64:
            return 1
        connection = http.client.HTTPConnection("127.0.0.1", 7077, timeout=3)
        connection.request("GET", "/healthz", headers={"Authorization": f"Bearer {token}"})
        response = connection.getresponse()
        if response.status != 200 or response.getheader("Content-Type") != "application/json":
            return 1
        length = int(response.getheader("Content-Length", "0"))
        if not 1 <= length <= 1024:
            return 1
        payload = json.loads(response.read(length))
        return (
            0
            if isinstance(payload, dict)
            and set(payload) == {"status", "trace_id"}
            and payload["status"] == "ok"
            and isinstance(payload["trace_id"], str)
            and len(payload["trace_id"]) == 32
            else 1
        )
    except OSError, UnicodeError, ValueError, json.JSONDecodeError, http.client.HTTPException:
        return 1
    finally:
        if "connection" in locals():
            connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
