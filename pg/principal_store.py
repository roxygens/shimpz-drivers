"""Hashed pg-driver principals scoped to one Capsule and its exact database set."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

STATE_PATH = Path(os.environ.get("SHIMPZ_PGDRIVER_PRINCIPALS_FILE", "/var/lib/pg-driver/principals.json"))
_lock = threading.RLock()


class PrincipalError(Exception):
    """A tenant principal is unknown or outside its registered database scope."""


class PrincipalStoreError(Exception):
    """The durable principal registry could not be read or committed."""


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _read() -> dict[str, dict[str, object]]:
    try:
        if not STATE_PATH.exists():
            return {}
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrincipalStoreError("principal registry could not be read") from exc
    if not isinstance(data, dict):
        raise PrincipalStoreError("principal registry is not a JSON object")
    for digest, record in data.items():
        if not isinstance(digest, str) or not isinstance(record, dict):
            raise PrincipalStoreError("principal registry contains an invalid record")
        capsule_id = record.get("capsule_id")
        databases = record.get("databases")
        if not isinstance(capsule_id, str) or not isinstance(databases, list):
            raise PrincipalStoreError("principal registry contains an invalid record")
        if not all(isinstance(database, str) for database in databases):
            raise PrincipalStoreError("principal registry contains an invalid database set")
        if not isinstance(record.get("retired", False), bool):
            raise PrincipalStoreError("principal registry contains an invalid retirement state")
    return data


def _write(data: dict[str, dict[str, object]]) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = STATE_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(STATE_PATH)
    except OSError as exc:
        raise PrincipalStoreError("principal registry could not be committed") from exc


def register(capsule_id: str, token: str, database: str) -> None:
    """Register or rotate exactly one principal for `capsule_id`; cleartext is never stored."""
    with _lock:
        data = _read()
        for digest, record in list(data.items()):
            if record.get("capsule_id") == capsule_id:
                del data[digest]
        data[_digest(token)] = {"capsule_id": capsule_id, "databases": [database], "retired": False}
        _write(data)


def databases(token: str, capsule_id: str, *, allow_retired: bool = False) -> frozenset[str]:
    with _lock:
        record = _read().get(_digest(token))
        if record is None or record.get("capsule_id") != capsule_id:
            raise PrincipalError("unknown principal or capsule scope mismatch")
        if record.get("retired", False) and not allow_retired:
            raise PrincipalError("Capsule principal is retired")
        values = record.get("databases")
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise PrincipalError("principal registry contains an invalid database set")
        return frozenset(values)


def add_database(token: str, capsule_id: str, database: str) -> None:
    with _lock:
        data = _read()
        digest = _digest(token)
        record = data.get(digest)
        if record is None or record.get("capsule_id") != capsule_id or record.get("retired", False):
            raise PrincipalError("unknown principal or capsule scope mismatch")
        values = record.get("databases")
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise PrincipalError("principal registry contains an invalid database set")
        record["databases"] = sorted({*values, database})
        _write(data)


def remove_database(token: str, capsule_id: str, database: str) -> None:
    with _lock:
        data = _read()
        digest = _digest(token)
        record = data.get(digest)
        if record is None or record.get("capsule_id") != capsule_id or record.get("retired", False):
            raise PrincipalError("unknown principal or capsule scope mismatch")
        values = record.get("databases")
        if not isinstance(values, list) or database not in values:
            raise PrincipalError("database is outside this principal's scope")
        record["databases"] = [value for value in values if value != database]
        _write(data)


def retire(token: str, capsule_id: str) -> None:
    """Keep an empty, idempotent drop proof until the controller finalizes runtime cleanup."""
    with _lock:
        data = _read()
        digest = _digest(token)
        record = data.get(digest)
        if record is None or record.get("capsule_id") != capsule_id:
            raise PrincipalError("unknown principal or capsule scope mismatch")
        values = record.get("databases")
        if not isinstance(values, list) or values:
            raise PrincipalError("cannot retire a principal with registered databases")
        record["retired"] = True
        _write(data)


def finalize(capsule_id: str) -> None:
    """Provisioner-authorized, retry-safe removal of this Capsule's retired principal proof."""
    with _lock:
        data = _read()
        matched = [digest for digest, record in data.items() if record.get("capsule_id") == capsule_id]
        for digest in matched:
            record = data[digest]
            if not record.get("retired", False) or record.get("databases"):
                raise PrincipalError("Capsule principal is still active")
        if matched:
            for digest in matched:
                del data[digest]
            _write(data)


def remove(token: str, capsule_id: str) -> None:
    with _lock:
        data = _read()
        digest = _digest(token)
        record = data.get(digest)
        if record is None or record.get("capsule_id") != capsule_id:
            raise PrincipalError("unknown principal or capsule scope mismatch")
        del data[digest]
        _write(data)
