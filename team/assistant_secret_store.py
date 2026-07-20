"""Encrypted, write-only-at-the-boundary secrets for installed Assistants.

Secret values are controller data.  They never belong to the Assistant manifest,
the Brain prompt, Docker environment variables, command arguments, or public API
responses.  This store keeps one independently authenticated AES-GCM envelope per
``(Team, Assistant, secret)`` generation and exposes only bounded metadata.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import stat
import threading
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

STATE_PATH = Path("/var/lib/shimpz-local/assistant-secrets/state/secrets.json")
KEY_PATH = Path("/var/lib/shimpz-local/assistant-secrets/key/aes256.key")
MAX_STATE_BYTES = 2 * 1024 * 1024
MAX_SECRET_BYTES = 16 * 1024
MAX_SECRETS_PER_ASSISTANT = 32
MAX_TOTAL_RECORDS = 4096
_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_TEAM_ID = re.compile(r"[a-z0-9_]{1,40}")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


class AssistantSecretError(RuntimeError):
    """Assistant secret state is invalid or unavailable."""


class AssistantSecretValidationError(AssistantSecretError):
    """A caller supplied an invalid public identifier or secret value."""


class AssistantSecretMissingError(AssistantSecretError):
    """One or more declared secrets have not been configured."""

    def __init__(self, missing: Iterable[str]) -> None:
        self.missing = tuple(sorted(set(missing)))
        super().__init__("one or more Assistant secrets are not configured")


@dataclass(frozen=True, slots=True)
class SecretMetadata:
    """Public status; ``mask`` deliberately reveals bounded edge characters."""

    id: str
    configured: bool
    mask: str | None
    generation: int | None


def _canonical_id(value: object, scope: str) -> str:
    if not isinstance(value, str) or len(value) > 64 or _ID.fullmatch(value) is None:
        raise AssistantSecretValidationError(f"{scope} is invalid")
    return value


def _canonical_team_id(value: object) -> str:
    if not isinstance(value, str) or _TEAM_ID.fullmatch(value) is None:
        raise AssistantSecretValidationError("Team id is invalid")
    return value


def _canonical_secret(value: object) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise AssistantSecretValidationError("secret value is invalid")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        raise AssistantSecretValidationError("secret value is invalid") from exc
    if len(encoded) > MAX_SECRET_BYTES or not value.isprintable():
        raise AssistantSecretValidationError("secret value is invalid")
    return value


def mask_secret(value: str) -> str:
    """Return a deliberately lossy identifier with at most eight visible characters.

    Values shorter than eight characters disclose nothing. Longer values expose
    one to four characters at each edge, using broad length buckets. The result is
    still sensitive metadata and must not be treated as redaction for arbitrary
    logs or error messages.
    """
    length = len(value)
    if length < 8:
        return "••••"
    visible = 1 if length < 16 else 2 if length < 32 else 3 if length < 64 else 4
    return f"{value[:visible]}…{value[-visible:]}"


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _aad(team_id: str, assistant_id: str, secret_id: str, generation: int) -> bytes:
    return json.dumps(
        ["shimpz-assistant-secret-v1", team_id, assistant_id, secret_id, generation],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")


def _empty_state() -> dict[str, object]:
    return {"schema": 1, "teams": {}}


def _decode_part(
    value: object,
    *,
    expected: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> bytes:
    if not isinstance(value, str) or len(value) > (MAX_SECRET_BYTES * 2) + 128:
        raise AssistantSecretError("Assistant secret envelope is malformed")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise AssistantSecretError("Assistant secret envelope is malformed") from exc
    if (
        (expected is not None and len(decoded) != expected)
        or (minimum is not None and len(decoded) < minimum)
        or (maximum is not None and len(decoded) > maximum)
    ):
        raise AssistantSecretError("Assistant secret envelope is malformed")
    return decoded


def _read_private_file(path: Path, maximum: int, label: str) -> bytes | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise AssistantSecretError(f"{label} is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size > maximum
        ):
            raise AssistantSecretError(f"{label} failed its ownership contract")
        payload = bytearray()
        while len(payload) <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > maximum:
            raise AssistantSecretError(f"{label} exceeds its fixed byte limit")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _require_private_parent(path: Path, label: str) -> None:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise AssistantSecretError(f"{label} directory is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise AssistantSecretError(f"{label} directory failed its ownership contract")


def _atomic_write(path: Path, payload: bytes, label: str) -> None:
    _require_private_parent(path.parent, label)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    descriptor = -1
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
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
        raise AssistantSecretError(f"{label} could not be persisted") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            temporary.unlink()


def _validate_record(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"generation", "mask", "updated_at", "envelope"}:
        raise AssistantSecretError("Assistant secret state record is malformed")
    generation = value.get("generation")
    mask = value.get("mask")
    updated_at = value.get("updated_at")
    envelope = value.get("envelope")
    if (
        type(generation) is not int
        or generation < 1
        or not isinstance(mask, str)
        or not 1 <= len(mask) <= 9
        or not mask.isprintable()
        or not isinstance(updated_at, str)
        or _TIMESTAMP.fullmatch(updated_at) is None
        or not isinstance(envelope, dict)
        or set(envelope) != {"algorithm", "nonce", "ciphertext"}
        or envelope.get("algorithm") != "AES-256-GCM"
    ):
        raise AssistantSecretError("Assistant secret state record is malformed")
    _decode_part(envelope.get("nonce"), expected=12)
    _decode_part(
        envelope.get("ciphertext"),
        minimum=17,
        maximum=MAX_SECRET_BYTES + 16,
    )
    return value


def _validate_state(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"schema", "teams"} or value.get("schema") != 1:
        raise AssistantSecretError("Assistant secret state has an unsupported shape")
    teams = value.get("teams")
    if not isinstance(teams, dict):
        raise AssistantSecretError("Assistant secret state is malformed")
    records = 0
    for raw_team, raw_assistants in teams.items():
        try:
            _canonical_team_id(raw_team)
        except AssistantSecretValidationError as exc:
            raise AssistantSecretError("Assistant secret state is malformed") from exc
        if not isinstance(raw_assistants, dict):
            raise AssistantSecretError("Assistant secret state is malformed")
        for raw_assistant, raw_secrets in raw_assistants.items():
            try:
                _canonical_id(raw_assistant, "Assistant id")
            except AssistantSecretValidationError as exc:
                raise AssistantSecretError("Assistant secret state is malformed") from exc
            if not isinstance(raw_secrets, dict) or len(raw_secrets) > MAX_SECRETS_PER_ASSISTANT:
                raise AssistantSecretError("Assistant secret state is malformed")
            for raw_secret, raw_record in raw_secrets.items():
                try:
                    _canonical_id(raw_secret, "secret id")
                except AssistantSecretValidationError as exc:
                    raise AssistantSecretError("Assistant secret state is malformed") from exc
                _validate_record(raw_record)
                records += 1
                if records > MAX_TOTAL_RECORDS:
                    raise AssistantSecretError("Assistant secret state exceeds its record limit")
    return value


def _canonical_ids(values: object) -> tuple[str, ...]:
    if not isinstance(values, Iterable) or isinstance(values, str | bytes | Mapping):
        raise AssistantSecretValidationError("secret ids are invalid")
    canonical: list[str] = []
    seen: set[str] = set()
    for value in values:
        if len(canonical) == MAX_SECRETS_PER_ASSISTANT:
            raise AssistantSecretValidationError("secret ids are invalid")
        secret_id = _canonical_id(value, "secret id")
        if secret_id in seen:
            raise AssistantSecretValidationError("secret ids are invalid")
        canonical.append(secret_id)
        seen.add(secret_id)
    return tuple(canonical)


def _state_has_records(state: Mapping[str, object]) -> bool:
    teams = state.get("teams")
    if not isinstance(teams, dict):
        raise AssistantSecretError("Assistant secret state is malformed")
    for assistants in teams.values():
        if not isinstance(assistants, dict):
            raise AssistantSecretError("Assistant secret state is malformed")
        for records in assistants.values():
            if not isinstance(records, dict):
                raise AssistantSecretError("Assistant secret state is malformed")
            if records:
                return True
    return False


class AssistantSecretStore:
    def __init__(self, state_path: Path = STATE_PATH, key_path: Path = KEY_PATH) -> None:
        self.state_path = Path(state_path)
        self.key_path = Path(key_path)
        if not self.state_path.is_absolute() or not self.key_path.is_absolute():
            raise AssistantSecretError("Assistant secret state and key paths must be absolute")
        try:
            state_parent = self.state_path.parent.resolve()
            key_parent = self.key_path.parent.resolve()
        except OSError as exc:
            raise AssistantSecretError("Assistant secret storage paths are unavailable") from exc
        if state_parent == key_parent:
            raise AssistantSecretError("Assistant secret keyring must be separate from encrypted state")
        self._lock = threading.RLock()

    def _read_state(self) -> dict[str, object]:
        payload = _read_private_file(self.state_path, MAX_STATE_BYTES, "Assistant secret state")
        if payload is None:
            return _empty_state()
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AssistantSecretError("Assistant secret state is not valid JSON") from exc
        return _validate_state(value)

    def _write_state(self, state: Mapping[str, object]) -> None:
        validated = _validate_state(dict(state))
        payload = json.dumps(validated, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(payload) > MAX_STATE_BYTES:
            raise AssistantSecretError("Assistant secret state exceeds its fixed byte limit")
        _atomic_write(self.state_path, payload, "Assistant secret state")

    def _key(self, *, allow_create: bool = False) -> bytes:
        payload = _read_private_file(self.key_path, 32, "Assistant secret keyring")
        if payload is None:
            if not allow_create:
                raise AssistantSecretError("Assistant secret keyring is unavailable")
            payload = AESGCM.generate_key(bit_length=256)
            _atomic_write(self.key_path, payload, "Assistant secret keyring")
        if len(payload) != 32:
            raise AssistantSecretError("Assistant secret keyring is invalid")
        return payload

    @staticmethod
    def _record_set(
        state: dict[str, object],
        team_id: str,
        assistant_id: str,
        *,
        create: bool,
    ) -> dict[str, object]:
        teams = state["teams"]
        if not isinstance(teams, dict):
            raise AssistantSecretError("Assistant secret state is malformed")
        assistants = teams.get(team_id)
        if assistants is None:
            if not create:
                return {}
            assistants = {}
            teams[team_id] = assistants
        elif not isinstance(assistants, dict):
            raise AssistantSecretError("Assistant secret state is malformed")
        records = assistants.get(assistant_id)
        if records is None:
            if not create:
                return {}
            records = {}
            assistants[assistant_id] = records
        elif not isinstance(records, dict):
            raise AssistantSecretError("Assistant secret state is malformed")
        return records

    def put_many(self, team_id: object, assistant_id: object, values: object) -> tuple[SecretMetadata, ...]:
        team = _canonical_team_id(team_id)
        assistant = _canonical_id(assistant_id, "Assistant id")
        if not isinstance(values, Mapping) or not 1 <= len(values) <= MAX_SECRETS_PER_ASSISTANT:
            raise AssistantSecretValidationError("Assistant secret values are invalid")
        canonical: dict[str, str] = {}
        for raw_id, raw_value in values.items():
            secret_id = _canonical_id(raw_id, "secret id")
            if secret_id in canonical:
                raise AssistantSecretValidationError("Assistant secret values are invalid")
            canonical[secret_id] = _canonical_secret(raw_value)
        with self._lock:
            state = self._read_state()
            key = self._key(allow_create=not _state_has_records(state))
            records = self._record_set(state, team, assistant, create=True)
            now = _timestamp()
            for secret_id, value in canonical.items():
                previous = records.get(secret_id)
                generation = int(previous.get("generation", 0)) + 1 if isinstance(previous, dict) else 1
                nonce = os.urandom(12)
                ciphertext = AESGCM(key).encrypt(
                    nonce,
                    value.encode("utf-8"),
                    _aad(team, assistant, secret_id, generation),
                )
                records[secret_id] = {
                    "generation": generation,
                    "mask": mask_secret(value),
                    "updated_at": now,
                    "envelope": {
                        "algorithm": "AES-256-GCM",
                        "nonce": base64.b64encode(nonce).decode("ascii"),
                        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
                    },
                }
            self._write_state(state)
            return tuple(
                SecretMetadata(
                    secret_id,
                    True,
                    mask_secret(canonical[secret_id]),
                    int(records[secret_id]["generation"]),
                )
                for secret_id in sorted(canonical)
            )

    def resolve_many(self, team_id: object, assistant_id: object, secret_ids: object) -> dict[str, str]:
        team = _canonical_team_id(team_id)
        assistant = _canonical_id(assistant_id, "Assistant id")
        requested = _canonical_ids(secret_ids)
        if not requested:
            return {}
        with self._lock:
            state = self._read_state()
            records = self._record_set(state, team, assistant, create=False)
            missing = [secret_id for secret_id in requested if secret_id not in records]
            if missing:
                raise AssistantSecretMissingError(missing)
            key = self._key()
            resolved: dict[str, str] = {}
            for secret_id in requested:
                record = _validate_record(records[secret_id])
                envelope = record["envelope"]
                if not isinstance(envelope, dict):
                    raise AssistantSecretError("Assistant secret envelope is malformed")
                try:
                    plaintext = AESGCM(key).decrypt(
                        _decode_part(envelope.get("nonce"), expected=12),
                        _decode_part(envelope.get("ciphertext")),
                        _aad(team, assistant, secret_id, int(record["generation"])),
                    )
                except InvalidTag as exc:
                    raise AssistantSecretError("Assistant secret envelope authentication failed") from exc
                try:
                    resolved[secret_id] = _canonical_secret(plaintext.decode("utf-8"))
                except (UnicodeDecodeError, AssistantSecretValidationError) as exc:
                    raise AssistantSecretError("decrypted Assistant secret is malformed") from exc
            return resolved

    def metadata(self, team_id: object, assistant_id: object, declared_ids: object) -> tuple[SecretMetadata, ...]:
        team = _canonical_team_id(team_id)
        assistant = _canonical_id(assistant_id, "Assistant id")
        declared = _canonical_ids(declared_ids)
        with self._lock:
            state = self._read_state()
            records = self._record_set(state, team, assistant, create=False)
            return tuple(
                SecretMetadata(
                    secret_id,
                    secret_id in records,
                    str(records[secret_id]["mask"]) if secret_id in records else None,
                    int(records[secret_id]["generation"]) if secret_id in records else None,
                )
                for secret_id in declared
            )

    def retain_declared(self, team_id: object, assistant_id: object, declared_ids: object) -> bool:
        """Atomically discard encrypted records removed from an Assistant's new release."""
        team = _canonical_team_id(team_id)
        assistant = _canonical_id(assistant_id, "Assistant id")
        declared = set(_canonical_ids(declared_ids))
        with self._lock:
            state = self._read_state()
            teams = state["teams"]
            if not isinstance(teams, dict) or not isinstance(teams.get(team), dict):
                return False
            assistants = teams[team]
            records = self._record_set(state, team, assistant, create=False)
            obsolete = set(records) - declared
            if not obsolete:
                return False
            for secret_id in obsolete:
                records.pop(secret_id)
            if not records:
                assistants.pop(assistant, None)
            if not assistants:
                teams.pop(team, None)
            self._write_state(state)
            return True

    def delete_assistant(self, team_id: object, assistant_id: object) -> bool:
        team = _canonical_team_id(team_id)
        assistant = _canonical_id(assistant_id, "Assistant id")
        with self._lock:
            state = self._read_state()
            teams = state["teams"]
            if not isinstance(teams, dict) or not isinstance(teams.get(team), dict):
                return False
            assistants = teams[team]
            removed = assistants.pop(assistant, None) is not None
            if not assistants:
                teams.pop(team, None)
            if removed:
                self._write_state(state)
            return removed

    def delete_team(self, team_id: object) -> bool:
        team = _canonical_team_id(team_id)
        with self._lock:
            state = self._read_state()
            teams = state["teams"]
            if not isinstance(teams, dict):
                raise AssistantSecretError("Assistant secret state is malformed")
            removed = teams.pop(team, None) is not None
            if removed:
                self._write_state(state)
            return removed

    def delete_all(self) -> bool:
        """Atomically remove every encrypted record during an owned Space reset."""
        with self._lock:
            state = self._read_state()
            if not _state_has_records(state):
                return False
            self._write_state(_empty_state())
            return True
