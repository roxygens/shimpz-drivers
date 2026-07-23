"""Hashed pg-driver principals scoped to one Team and its exact database set."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
from pathlib import Path

STATE_PATH = Path(os.environ.get("SHIMPZ_PGDRIVER_PRINCIPALS_FILE", "/var/lib/pg-driver/principals.json"))
_lock = threading.RLock()
_DATABASE_NAMESPACE_RE = re.compile(r"[a-f0-9]{12}\Z")


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
        team_id = record.get("team_id")
        databases = record.get("databases")
        database_namespace = record.get("database_namespace")
        if (
            not isinstance(team_id, str)
            or not isinstance(databases, list)
            or not isinstance(database_namespace, str)
            or _DATABASE_NAMESPACE_RE.fullmatch(database_namespace) is None
        ):
            raise PrincipalStoreError("principal registry contains an invalid record")
        if not all(isinstance(database, str) for database in databases):
            raise PrincipalStoreError("principal registry contains an invalid database set")
        if not isinstance(record.get("retired", False), bool):
            raise PrincipalStoreError("principal registry contains an invalid retirement state")
    namespaces = [record["database_namespace"] for record in data.values()]
    if len(namespaces) != len(set(namespaces)):
        raise PrincipalStoreError("principal registry contains a duplicate database namespace")
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


def register(team_id: str, token: str, database: str) -> None:
    """Register or rotate exactly one principal for `team_id`; cleartext is never stored."""
    with _lock:
        data = _read()
        existing = [record for record in data.values() if record.get("team_id") == team_id]
        if len(existing) > 1:
            raise PrincipalStoreError("principal registry contains duplicate Team identities")
        if existing:
            database_namespace = existing[0]["database_namespace"]
        else:
            used = {record["database_namespace"] for record in data.values()}
            while (database_namespace := secrets.token_hex(6)) in used:
                pass
        for digest, record in list(data.items()):
            if record.get("team_id") == team_id:
                del data[digest]
        data[_digest(token)] = {
            "team_id": team_id,
            "databases": [database],
            "database_namespace": database_namespace,
            "retired": False,
        }
        _write(data)


def database_namespace(token: str, team_id: str) -> str:
    """Return the registry-assigned namespace only to its exact active principal."""
    with _lock:
        record = _read().get(_digest(token))
        if record is None or record.get("team_id") != team_id or record.get("retired", False):
            raise PrincipalError("unknown principal or team scope mismatch")
        value = record.get("database_namespace")
        if not isinstance(value, str) or _DATABASE_NAMESPACE_RE.fullmatch(value) is None:
            raise PrincipalError("principal registry contains an invalid database namespace")
        return value


def owns_database(team_id: str, database: str) -> bool:
    """Whether the durable registry assigns one exact database to this Team."""
    with _lock:
        matches = [record for record in _read().values() if record.get("team_id") == team_id]
        if len(matches) > 1:
            raise PrincipalStoreError("principal registry contains duplicate Team identities")
        if not matches:
            return False
        databases = matches[0].get("databases")
        if not isinstance(databases, list):
            raise PrincipalStoreError("principal registry contains an invalid database set")
        return database in databases


def databases(token: str, team_id: str, *, allow_retired: bool = False) -> frozenset[str]:
    with _lock:
        record = _read().get(_digest(token))
        if record is None or record.get("team_id") != team_id:
            raise PrincipalError("unknown principal or team scope mismatch")
        if record.get("retired", False) and not allow_retired:
            raise PrincipalError("Team principal is retired")
        values = record.get("databases")
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise PrincipalError("principal registry contains an invalid database set")
        return frozenset(values)


def add_database(token: str, team_id: str, database: str) -> None:
    with _lock:
        data = _read()
        digest = _digest(token)
        record = data.get(digest)
        if record is None or record.get("team_id") != team_id or record.get("retired", False):
            raise PrincipalError("unknown principal or team scope mismatch")
        values = record.get("databases")
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise PrincipalError("principal registry contains an invalid database set")
        record["databases"] = sorted({*values, database})
        _write(data)


def remove_database(token: str, team_id: str, database: str) -> None:
    with _lock:
        data = _read()
        digest = _digest(token)
        record = data.get(digest)
        if record is None or record.get("team_id") != team_id or record.get("retired", False):
            raise PrincipalError("unknown principal or team scope mismatch")
        values = record.get("databases")
        if not isinstance(values, list) or database not in values:
            raise PrincipalError("database is outside this principal's scope")
        record["databases"] = [value for value in values if value != database]
        _write(data)


def retire(token: str, team_id: str) -> None:
    """Keep an empty, idempotent drop proof until the controller finalizes runtime cleanup."""
    with _lock:
        data = _read()
        digest = _digest(token)
        record = data.get(digest)
        if record is None or record.get("team_id") != team_id:
            raise PrincipalError("unknown principal or team scope mismatch")
        values = record.get("databases")
        if not isinstance(values, list) or values:
            raise PrincipalError("cannot retire a principal with registered databases")
        record["retired"] = True
        _write(data)


def finalize(team_id: str) -> None:
    """Provisioner-authorized, retry-safe removal of this Team's retired principal proof."""
    with _lock:
        data = _read()
        matched = [digest for digest, record in data.items() if record.get("team_id") == team_id]
        for digest in matched:
            record = data[digest]
            if not record.get("retired", False) or record.get("databases"):
                raise PrincipalError("Team principal is still active")
        if matched:
            for digest in matched:
                del data[digest]
            _write(data)


def remove(token: str, team_id: str) -> None:
    with _lock:
        data = _read()
        digest = _digest(token)
        record = data.get(digest)
        if record is None or record.get("team_id") != team_id:
            raise PrincipalError("unknown principal or team scope mismatch")
        del data[digest]
        _write(data)
