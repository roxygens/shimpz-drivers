"""Hashed R2 driver principals scoped to exactly one Team."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
import stat
import threading
from pathlib import Path

STATE_PATH = Path(
    os.environ.get(
        "SHIMPZ_R2DRIVER_PRINCIPALS_FILE",
        "/var/lib/shimpz-r2principals/principals.json",
    )
)
STATE_VERSION = 2
MAX_STATE_BYTES = 1024 * 1024
MAX_PRINCIPALS = 4096

_TEAM_ID_RE = re.compile(r"^[a-z0-9_]{1,40}$")
_TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_ROOT_KEYS = {"version", "principals"}
_RECORD_KEYS = {"team_id", "status"}
_STATUSES = {"active", "retired", "finalized"}


class PrincipalError(Exception):
    """A bearer is unknown, retired, or outside its exact Team scope."""


class PrincipalStoreError(Exception):
    """The hashed principal registry could not be safely read or committed."""


def _team_id(value: object) -> str:
    if not isinstance(value, str) or _TEAM_ID_RE.fullmatch(value) is None:
        raise PrincipalError("team_id must match [a-z0-9_]{1,40}")
    return value


def _digest(token: object) -> str:
    if not isinstance(token, str) or _TOKEN_RE.fullmatch(token) is None:
        raise PrincipalError("principal token must be a 256-bit lowercase hex value")
    return hashlib.sha256(token.encode()).hexdigest()


def _ensure_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
        metadata = path.lstat()
    except OSError as exc:
        raise PrincipalStoreError("principal registry directory is unavailable") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise PrincipalStoreError("principal registry directory is not owner-only")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise PrincipalStoreError("principal registry directory could not be committed") from exc


def _validate_state(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _ROOT_KEYS or value.get("version") != STATE_VERSION:
        raise PrincipalStoreError("principal registry root is malformed")
    principals = value.get("principals")
    if not isinstance(principals, dict) or len(principals) > MAX_PRINCIPALS:
        raise PrincipalStoreError("principal registry set is malformed")
    live_teams: set[str] = set()
    for digest, record in principals.items():
        if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
            raise PrincipalStoreError("principal registry contains an invalid digest")
        if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
            raise PrincipalStoreError("principal registry contains an invalid record")
        team = record.get("team_id")
        try:
            team = _team_id(team)
        except PrincipalError as exc:
            raise PrincipalStoreError("principal registry contains an invalid Team") from exc
        status = record.get("status")
        if status not in _STATUSES:
            raise PrincipalStoreError("principal registry contains an invalid lifecycle status")
        if status != "finalized":
            if team in live_teams:
                raise PrincipalStoreError("principal registry contains duplicate live Team records")
            live_teams.add(team)
    return value


class PrincipalStore:
    def __init__(self, state_path: Path = STATE_PATH) -> None:
        self.state_path = Path(state_path)
        if not self.state_path.is_absolute():
            raise PrincipalStoreError("principal registry path must be absolute")
        self._lock = threading.RLock()

    def _read(self) -> dict[str, object]:
        _ensure_directory(self.state_path.parent)
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.state_path, flags)
        except FileNotFoundError:
            return {"version": STATE_VERSION, "principals": {}}
        except OSError as exc:
            raise PrincipalStoreError("principal registry could not be opened") from exc
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
                or metadata.st_size > MAX_STATE_BYTES
            ):
                raise PrincipalStoreError("principal registry has unsafe filesystem metadata")
            chunks: list[bytes] = []
            remaining = metadata.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if len(payload) != metadata.st_size:
                raise PrincipalStoreError("principal registry changed while it was read")
        except OSError as exc:
            raise PrincipalStoreError("principal registry could not be read") from exc
        finally:
            os.close(descriptor)
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PrincipalStoreError("principal registry is not valid JSON") from exc
        return _validate_state(value)

    def _write(self, state: dict[str, object]) -> None:
        payload = json.dumps(_validate_state(state), sort_keys=True, separators=(",", ":")).encode()
        if len(payload) > MAX_STATE_BYTES:
            raise PrincipalStoreError("principal registry exceeds its fixed byte limit")
        _ensure_directory(self.state_path.parent)
        temporary = self.state_path.parent / f".{self.state_path.name}.{secrets.token_hex(12)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = -1
        try:
            descriptor = os.open(temporary, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written < 1:
                    raise OSError("short principal registry write")
                remaining = remaining[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            temporary.replace(self.state_path)
            _fsync_directory(self.state_path.parent)
        except OSError as exc:
            raise PrincipalStoreError("principal registry could not be committed") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)

    def check_health(self) -> None:
        """Fail closed when the durable Team-principal registry cannot be trusted."""
        with self._lock:
            self._read()

    def provision(self, team_id: object, token: object) -> None:
        """Register or retry one lifecycle principal; cleartext is never persisted."""
        team = _team_id(team_id)
        digest = _digest(token)
        with self._lock:
            state = self._read()
            principals = state["principals"]
            if not isinstance(principals, dict):
                raise PrincipalStoreError("principal registry set is malformed")
            existing_for_digest = principals.get(digest)
            if isinstance(existing_for_digest, dict):
                if existing_for_digest.get("team_id") != team:
                    raise PrincipalError("principal token is already scoped to another Team")
                if existing_for_digest.get("status") == "active":
                    return
                raise PrincipalError("principal token lifecycle cannot be replayed")
            matching = [key for key, record in principals.items() if record.get("team_id") == team]
            if any(principals[key].get("status") != "finalized" for key in matching):
                raise PrincipalError("Team already has a live principal lifecycle")
            if len(principals) >= MAX_PRINCIPALS:
                raise PrincipalStoreError("principal registry capacity is exhausted")
            principals[digest] = {"team_id": team, "status": "active"}
            self._write(state)

    def resolve(self, token: object, team_id: object, *, allow_retired: bool = False) -> str:
        team = _team_id(team_id)
        digest = _digest(token)
        with self._lock:
            state = self._read()
            principals = state["principals"]
            if not isinstance(principals, dict):
                raise PrincipalStoreError("principal registry set is malformed")
            record = principals.get(digest)
            if not isinstance(record, dict) or record.get("team_id") != team:
                raise PrincipalError("unknown principal or Team scope mismatch")
            if record.get("status") == "finalized" or (record.get("status") == "retired" and not allow_retired):
                raise PrincipalError("Team principal is retired")
            return team

    @contextlib.contextmanager
    def authorized(self, token: object, team_id: object):
        """Hold the lifecycle gate so retire cannot complete while a scoped use starts or runs."""
        team = _team_id(team_id)
        digest = _digest(token)
        with self._lock:
            state = self._read()
            principals = state["principals"]
            if not isinstance(principals, dict):
                raise PrincipalStoreError("principal registry set is malformed")
            record = principals.get(digest)
            if not isinstance(record, dict) or record.get("team_id") != team or record.get("status") != "active":
                raise PrincipalError("unknown, retired, or Team-mismatched principal")
            yield team

    def retire(self, token: object, team_id: object) -> None:
        team = _team_id(team_id)
        digest = _digest(token)
        with self._lock:
            state = self._read()
            principals = state["principals"]
            if not isinstance(principals, dict):
                raise PrincipalStoreError("principal registry set is malformed")
            record = principals.get(digest)
            if not isinstance(record, dict) or record.get("team_id") != team:
                raise PrincipalError("unknown principal or Team scope mismatch")
            if record.get("status") == "finalized":
                raise PrincipalError("Team principal lifecycle is finalized")
            if record.get("status") == "retired":
                return
            record["status"] = "retired"
            self._write(state)

    def retire_team(self, team_id: object) -> None:
        """Retire the principal by provisioner-authorized Team identity, retry-safe."""
        team = _team_id(team_id)
        with self._lock:
            state = self._read()
            principals = state["principals"]
            if not isinstance(principals, dict):
                raise PrincipalStoreError("principal registry set is malformed")
            matching = [
                record
                for record in principals.values()
                if record.get("team_id") == team and record.get("status") != "finalized"
            ]
            if not matching or all(record.get("status") == "retired" for record in matching):
                return
            for record in matching:
                record["status"] = "retired"
            self._write(state)

    def assert_finalizable(self, team_id: object) -> None:
        """Reject an active lifecycle without mutating its replay history."""
        team = _team_id(team_id)
        with self._lock:
            state = self._read()
            principals = state["principals"]
            if not isinstance(principals, dict):
                raise PrincipalStoreError("principal registry set is malformed")
            if any(
                record.get("team_id") == team and record.get("status") == "active" for record in principals.values()
            ):
                raise PrincipalError("Team principal is still active")

    def finalize(self, team_id: object) -> None:
        team = _team_id(team_id)
        with self._lock:
            state = self._read()
            principals = state["principals"]
            if not isinstance(principals, dict):
                raise PrincipalStoreError("principal registry set is malformed")
            matching = [
                key
                for key, record in principals.items()
                if record.get("team_id") == team and record.get("status") != "finalized"
            ]
            if any(principals[key].get("status") != "retired" for key in matching):
                raise PrincipalError("Team principal is still active")
            if matching:
                for key in matching:
                    principals[key]["status"] = "finalized"
                self._write(state)


STORE = PrincipalStore()
