"""Structured audit log for every mutating capsule-driver operation.

Matches the repo-wide structlog JSON schema `logq` expects (ts/level/service/trace_id/msg/…extra).
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

AUDIT_PATH = Path(os.environ.get("SHIMPZ_CAPSULEDRIVER_AUDIT_LOG", "/var/log/capsule-driver/audit.jsonl"))
MAX_BYTES = 10 * 1024 * 1024
BACKUPS = 3


def _rotate() -> None:
    if not AUDIT_PATH.exists() or AUDIT_PATH.stat().st_size <= MAX_BYTES:
        return
    for i in range(BACKUPS - 1, 0, -1):
        src = AUDIT_PATH.with_name(f"{AUDIT_PATH.name}.{i}")
        dst = AUDIT_PATH.with_name(f"{AUDIT_PATH.name}.{i + 1}")
        if src.exists():
            src.replace(dst)
    AUDIT_PATH.replace(AUDIT_PATH.with_name(f"{AUDIT_PATH.name}.1"))


def log(
    op: str, capsule: str, *, result: str, trace_id: str | None = None, level: str | None = None, **extra: object
) -> str:
    """Emit one audit line; returns the trace_id (generated if absent) for the caller to echo back."""
    trace_id = trace_id or uuid.uuid4().hex
    event = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level": level or ("info" if result == "ok" else "warn"),
        "service": "capsule-driver",
        "trace_id": trace_id,
        "msg": f"{op} {capsule}: {result}",
        "op": op,
        "capsule": capsule,
        "result": result,
        **extra,
    }
    line = json.dumps(event, sort_keys=True)
    print(line, file=sys.stdout, flush=True)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _rotate()
    with AUDIT_PATH.open("a") as fh:
        fh.write(line + "\n")
    return trace_id
