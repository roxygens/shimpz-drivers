"""Small bounded audit journal that deliberately excludes bodies and credentials."""

from __future__ import annotations

import json
import os
import stat
import threading
import time
import uuid
from pathlib import Path

AUDIT_PATH = Path("/var/log/shimpz-local/audit.jsonl")
MAX_BYTES = 1024 * 1024
BACKUPS = 2
_LOCK = threading.Lock()


def _safe_file(path: Path) -> None:
    if not path.exists():
        return
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise RuntimeError("the local audit journal has unsafe metadata")


def _rotate(path: Path) -> None:
    if not path.exists() or path.stat().st_size <= MAX_BYTES:
        return
    oldest = path.with_name(f"{path.name}.{BACKUPS}")
    if oldest.exists():
        _safe_file(oldest)
        oldest.unlink()
    for index in range(BACKUPS - 1, 0, -1):
        source = path.with_name(f"{path.name}.{index}")
        if source.exists():
            _safe_file(source)
            source.replace(path.with_name(f"{path.name}.{index + 1}"))
    path.replace(path.with_name(f"{path.name}.1"))


def record(
    operation: str,
    *,
    result: str,
    team_id: str | None = None,
    assistant: str | None = None,
    detail: str | None = None,
) -> str:
    """Append one metadata-only event and return its correlation id."""
    trace_id = uuid.uuid4().hex
    event = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "service": "team-driver-local",
        "trace_id": trace_id,
        "operation": operation,
        "result": result,
    }
    if team_id is not None:
        event["team_id"] = team_id
    if assistant is not None:
        event["assistant"] = assistant
    if detail is not None:
        event["detail"] = detail
    encoded = (json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")

    with _LOCK:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _safe_file(AUDIT_PATH)
        _rotate(AUDIT_PATH)
        descriptor = os.open(AUDIT_PATH, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return trace_id
