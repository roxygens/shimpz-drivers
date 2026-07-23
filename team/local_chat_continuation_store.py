"""Encrypted, short-lived local chat continuations.

Only routing metadata is plaintext. The paused Brain turn, frozen Team identity,
Power inputs, and human answer log stay inside an AES-256-GCM envelope bound to
the exact Team, challenge, suspension kind, release bindings, and generation.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import stat
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

STATE_PATH = Path("/var/lib/shimpz-local/chat-continuations/state/continuations.json")
KEY_PATH = Path("/var/lib/shimpz-local/chat-continuations/key/aes256.key")
SCHEMA_VERSION = 1
MAX_CONTINUATIONS = 32
MAX_PLAINTEXT_BYTES = 256 * 1024
MAX_STATE_BYTES = 12 * 1024 * 1024
MAX_BINDINGS = 64
MAX_BINDING_BYTES = 640
MAX_TTL_SECONDS = 900
_TEAM_ID = re.compile(r"[a-z0-9_]{1,40}\Z")
_CHALLENGE_ID = re.compile(r"[0-9a-f]{32}\Z")
_KINDS = frozenset({"accounts", "secrets", "input", "approval"})


class ContinuationStoreError(RuntimeError):
    """Encrypted continuation state is invalid or unavailable."""


class ContinuationNotFoundError(ContinuationStoreError):
    """The continuation is absent, expired, consumed, or owned by another challenge."""


@dataclass(frozen=True, slots=True, repr=False)
class StoredContinuation:
    team_id: str
    kind: str
    challenge_id: str
    expires_at: int
    generation: int
    bindings: tuple[str, ...]
    payload: bytes


def _team_id(value: object) -> str:
    if not isinstance(value, str) or _TEAM_ID.fullmatch(value) is None:
        raise ContinuationStoreError("continuation Team is invalid")
    return value


def _kind(value: object) -> str:
    if not isinstance(value, str) or value not in _KINDS:
        raise ContinuationStoreError("continuation kind is invalid")
    return value


def _challenge_id(value: object) -> str:
    if not isinstance(value, str) or _CHALLENGE_ID.fullmatch(value) is None:
        raise ContinuationStoreError("continuation challenge is invalid")
    return value


def _bindings(value: object) -> tuple[str, ...]:
    if not isinstance(value, Iterable) or isinstance(value, str | bytes | Mapping):
        raise ContinuationStoreError("continuation bindings are invalid")
    result: list[str] = []
    for item in value:
        if (
            len(result) == MAX_BINDINGS
            or not isinstance(item, str)
            or not item
            or item != item.strip()
            or not item.isprintable()
        ):
            raise ContinuationStoreError("continuation bindings are invalid")
        try:
            encoded = item.encode("utf-8")
        except UnicodeError as exc:
            raise ContinuationStoreError("continuation bindings are invalid") from exc
        if len(encoded) > MAX_BINDING_BYTES:
            raise ContinuationStoreError("continuation bindings are invalid")
        result.append(item)
    if not result or len(set(result)) != len(result):
        raise ContinuationStoreError("continuation bindings are invalid")
    return tuple(sorted(result))


def _aad(
    team_id: str,
    kind: str,
    challenge_id: str,
    expires_at: int,
    generation: int,
    bindings: tuple[str, ...],
) -> bytes:
    return json.dumps(
        [
            "shimpz-local-chat-continuation-v1",
            team_id,
            kind,
            challenge_id,
            expires_at,
            generation,
            list(bindings),
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")


def _empty_state() -> dict[str, object]:
    return {"schema": SCHEMA_VERSION, "records": {}}


def _read_private_file(path: Path, maximum: int, label: str) -> bytes | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ContinuationStoreError(f"{label} is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size > maximum
        ):
            raise ContinuationStoreError(f"{label} failed its ownership contract")
        payload = bytearray()
        while len(payload) <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > maximum:
            raise ContinuationStoreError(f"{label} exceeds its fixed byte limit")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _require_private_parent(path: Path, label: str) -> None:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ContinuationStoreError(f"{label} directory is unavailable") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ContinuationStoreError(f"{label} directory failed its ownership contract")


def _atomic_write(path: Path, payload: bytes, label: str) -> None:
    _require_private_parent(path.parent, label)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("short private write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        raise ContinuationStoreError(f"{label} could not be persisted") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            temporary.unlink()


def _decode_part(
    value: object,
    *,
    expected: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> bytes:
    if not isinstance(value, str) or len(value) > (MAX_PLAINTEXT_BYTES * 2) + 128:
        raise ContinuationStoreError("continuation envelope is malformed")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (TypeError, ValueError) as exc:
        raise ContinuationStoreError("continuation envelope is malformed") from exc
    if (
        (expected is not None and len(decoded) != expected)
        or (minimum is not None and len(decoded) < minimum)
        or (maximum is not None and len(decoded) > maximum)
    ):
        raise ContinuationStoreError("continuation envelope is malformed")
    return decoded


def _record(value: object, expected_team: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "team_id",
        "kind",
        "challenge_id",
        "expires_at",
        "generation",
        "bindings",
        "envelope",
    }:
        raise ContinuationStoreError("continuation record is malformed")
    team = _team_id(value["team_id"])
    kind = _kind(value["kind"])
    challenge = _challenge_id(value["challenge_id"])
    expires_at = value["expires_at"]
    generation = value["generation"]
    bindings = _bindings(value["bindings"])
    envelope = value["envelope"]
    if (
        team != expected_team
        or type(expires_at) is not int
        or not 1 <= expires_at < 2**63
        or type(generation) is not int
        or not 1 <= generation <= 2**31 - 1
        or not isinstance(envelope, dict)
        or set(envelope) != {"algorithm", "nonce", "ciphertext"}
        or envelope["algorithm"] != "AES-256-GCM"
    ):
        raise ContinuationStoreError("continuation record is malformed")
    _decode_part(envelope["nonce"], expected=12)
    _decode_part(
        envelope["ciphertext"],
        minimum=17,
        maximum=MAX_PLAINTEXT_BYTES + 16,
    )
    value["kind"] = kind
    value["challenge_id"] = challenge
    value["bindings"] = list(bindings)
    return value


def _no_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ContinuationStoreError("continuation state has duplicate fields")
        value[key] = item
    return value


def _decode_json(payload: bytes) -> object:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ContinuationStoreError("continuation state contains a non-finite number")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinuationStoreError("continuation state is not valid JSON") from exc


def _state(value: object) -> dict[str, object]:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "records"}
        or value["schema"] != SCHEMA_VERSION
        or not isinstance(value["records"], dict)
        or len(value["records"]) > MAX_CONTINUATIONS
    ):
        raise ContinuationStoreError("continuation state has an unsupported shape")
    for team, record in value["records"].items():
        _team_id(team)
        _record(record, team)
    return value


class EncryptedContinuationStore:
    """Atomically keep at most one encrypted continuation per Team."""

    def __init__(
        self,
        state_path: Path = STATE_PATH,
        key_path: Path = KEY_PATH,
        *,
        now: Callable[[], float] = time.time,
        capacity: int = MAX_CONTINUATIONS,
    ) -> None:
        self.state_path = Path(state_path)
        self.key_path = Path(key_path)
        if not self.state_path.is_absolute() or not self.key_path.is_absolute():
            raise ContinuationStoreError("continuation state and key paths must be absolute")
        try:
            state_parent = self.state_path.parent.resolve()
            key_parent = self.key_path.parent.resolve()
        except OSError as exc:
            raise ContinuationStoreError("continuation storage paths are unavailable") from exc
        if state_parent == key_parent:
            raise ContinuationStoreError("continuation keyring must be separate from encrypted state")
        if not callable(now) or type(capacity) is not int or not 1 <= capacity <= MAX_CONTINUATIONS:
            raise ValueError("continuation store configuration is invalid")
        self._now = now
        self._capacity = capacity
        self._lock = threading.RLock()

    def _read_state(self) -> dict[str, object]:
        payload = _read_private_file(self.state_path, MAX_STATE_BYTES, "continuation state")
        return _empty_state() if payload is None else _state(_decode_json(payload))

    def _write_state(self, state: Mapping[str, object]) -> None:
        validated = _state(dict(state))
        payload = json.dumps(
            validated,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        if len(payload) > MAX_STATE_BYTES:
            raise ContinuationStoreError("continuation state exceeds its fixed byte limit")
        _atomic_write(self.state_path, payload, "continuation state")

    def _key(self, *, allow_create: bool = False) -> bytes:
        payload = _read_private_file(self.key_path, 32, "continuation keyring")
        if payload is None:
            if not allow_create:
                raise ContinuationStoreError("continuation keyring is unavailable")
            payload = AESGCM.generate_key(bit_length=256)
            _atomic_write(self.key_path, payload, "continuation keyring")
        if len(payload) != 32:
            raise ContinuationStoreError("continuation keyring is invalid")
        return payload

    def put(
        self,
        team_id: object,
        kind: object,
        challenge_id: object,
        expires_at: object,
        bindings: object,
        payload: object,
    ) -> StoredContinuation:
        team = _team_id(team_id)
        suspension_kind = _kind(kind)
        challenge = _challenge_id(challenge_id)
        canonical_bindings = _bindings(bindings)
        now = int(self._now())
        if (
            type(expires_at) is not int
            or not now < expires_at <= now + MAX_TTL_SECONDS
            or not isinstance(payload, bytes)
            or not 1 <= len(payload) <= MAX_PLAINTEXT_BYTES
        ):
            raise ContinuationStoreError("continuation payload is invalid")
        with self._lock:
            state = self._read_state()
            records = state["records"]
            if not isinstance(records, dict):
                raise ContinuationStoreError("continuation state is malformed")
            previous = records.get(team)
            if previous is None and len(records) >= self._capacity:
                raise ContinuationStoreError("continuation capacity reached")
            generation = int(previous["generation"]) + 1 if isinstance(previous, dict) else 1
            nonce = os.urandom(12)
            ciphertext = AESGCM(self._key(allow_create=not records)).encrypt(
                nonce,
                payload,
                _aad(
                    team,
                    suspension_kind,
                    challenge,
                    expires_at,
                    generation,
                    canonical_bindings,
                ),
            )
            records[team] = {
                "team_id": team,
                "kind": suspension_kind,
                "challenge_id": challenge,
                "expires_at": expires_at,
                "generation": generation,
                "bindings": list(canonical_bindings),
                "envelope": {
                    "algorithm": "AES-256-GCM",
                    "nonce": base64.b64encode(nonce).decode("ascii"),
                    "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
                },
            }
            self._write_state(state)
            return StoredContinuation(
                team,
                suspension_kind,
                challenge,
                expires_at,
                generation,
                canonical_bindings,
                payload,
            )

    def _resolved(self, team: str, raw: object) -> StoredContinuation:
        record = _record(raw, team)
        kind = str(record["kind"])
        challenge = str(record["challenge_id"])
        expires_at = int(record["expires_at"])
        generation = int(record["generation"])
        bindings = tuple(record["bindings"])
        envelope = record["envelope"]
        if not isinstance(envelope, dict):
            raise ContinuationStoreError("continuation envelope is malformed")
        try:
            plaintext = AESGCM(self._key()).decrypt(
                _decode_part(envelope["nonce"], expected=12),
                _decode_part(
                    envelope["ciphertext"],
                    minimum=17,
                    maximum=MAX_PLAINTEXT_BYTES + 16,
                ),
                _aad(team, kind, challenge, expires_at, generation, bindings),
            )
        except InvalidTag as exc:
            raise ContinuationStoreError("continuation envelope authentication failed") from exc
        if not 1 <= len(plaintext) <= MAX_PLAINTEXT_BYTES:
            raise ContinuationStoreError("decrypted continuation is malformed")
        return StoredContinuation(
            team,
            kind,
            challenge,
            expires_at,
            generation,
            bindings,
            plaintext,
        )

    def active(self) -> tuple[StoredContinuation, ...]:
        with self._lock:
            state = self._read_state()
            records = state["records"]
            if not isinstance(records, dict):
                raise ContinuationStoreError("continuation state is malformed")
            now = int(self._now())
            expired = [team for team, item in records.items() if int(item["expires_at"]) <= now]
            for team in expired:
                records.pop(team)
            if expired:
                self._write_state(state)
            return tuple(self._resolved(team, records[team]) for team in sorted(records))

    def current(self, team_id: object) -> StoredContinuation | None:
        team = _team_id(team_id)
        return next((item for item in self.active() if item.team_id == team), None)

    def delete(self, team_id: object, challenge_id: object | None = None) -> bool:
        team = _team_id(team_id)
        expected = _challenge_id(challenge_id) if challenge_id is not None else None
        with self._lock:
            state = self._read_state()
            records = state["records"]
            if not isinstance(records, dict):
                raise ContinuationStoreError("continuation state is malformed")
            raw = records.get(team)
            if raw is None:
                return False
            record = _record(raw, team)
            if expected is not None and record["challenge_id"] != expected:
                raise ContinuationNotFoundError("continuation is unavailable")
            records.pop(team)
            self._write_state(state)
            return True

    def clear(self) -> int:
        with self._lock:
            state = self._read_state()
            records = state["records"]
            if not isinstance(records, dict):
                raise ContinuationStoreError("continuation state is malformed")
            removed = len(records)
            if removed:
                records.clear()
                self._write_state(state)
            return removed
