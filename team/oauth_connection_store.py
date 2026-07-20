"""Encrypted, controller-owned OAuth tokens for Assistant connections.

Tokens never belong to an Assistant manifest, Brain prompt, process environment,
command argument, public API response, or log.  Each encrypted record is bound to
the exact Team, Assistant, connection, provider, scopes, expiry, status, and
generation through AES-GCM authenticated additional data (AAD).
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import oauth_providers
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

STATE_PATH = Path("/var/lib/shimpz-local/assistant-connections/state/connections.json")
KEY_PATH = Path("/var/lib/shimpz-local/assistant-connections/key/aes256.key")
MAX_STATE_BYTES = 4 * 1024 * 1024
MAX_TOKEN_BYTES = 16 * 1024
MAX_PLAINTEXT_BYTES = (MAX_TOKEN_BYTES * 2) + 2048
MAX_CONNECTIONS_PER_ASSISTANT = 16
MAX_TOTAL_RECORDS = 4096
MAX_ACCOUNT_ID_BYTES = 256
MAX_ACCOUNT_TEXT_BYTES = 512
REFRESH_WINDOW_SECONDS = 60
_TEAM_ID = re.compile(r"[a-z0-9_]{1,40}\Z")
_COMPONENT_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_STORED_STATUSES = frozenset({"connected", "reauthorization-required"})
StoredStatus = Literal["connected", "reauthorization-required"]
ConnectionStatus = Literal[
    "missing",
    "connected",
    "refresh-required",
    "reauthorization-required",
]


class OAuthConnectionStoreError(RuntimeError):
    """OAuth connection state is invalid, unavailable, or unauthentic."""


class OAuthConnectionValidationError(OAuthConnectionStoreError):
    """A caller supplied invalid OAuth connection data."""


class OAuthConnectionMissingError(OAuthConnectionStoreError):
    """The requested OAuth connection has not been configured."""


class OAuthConnectionReauthorizationError(OAuthConnectionStoreError):
    """A new provider authorization is required before this connection can run."""


@dataclass(frozen=True, slots=True)
class ConnectionAccount:
    id: str
    username: str | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class OAuthConnectionMetadata:
    """Bounded public inventory that never contains either OAuth token."""

    id: str
    provider: str
    scopes: tuple[str, ...]
    status: ConnectionStatus
    account: ConnectionAccount | None
    expires_at: int | None
    generation: int


@dataclass(frozen=True, slots=True, repr=False)
class _TokenGrant:
    access_token: str
    refresh_token: str | None
    scopes: tuple[str, ...]
    expires_at: int
    account: ConnectionAccount | None
    status: StoredStatus
    generation: int = 0


def _component_id(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) > 64 or _COMPONENT_ID.fullmatch(value) is None:
        raise OAuthConnectionValidationError(f"{label} is invalid")
    return value


def _team_id(value: object) -> str:
    if not isinstance(value, str) or _TEAM_ID.fullmatch(value) is None:
        raise OAuthConnectionValidationError("Team id is invalid")
    return value


def _bounded_text(
    value: object,
    label: str,
    maximum: int,
    *,
    optional: bool = False,
) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip() or not value.isprintable():
        raise OAuthConnectionValidationError(f"{label} is invalid")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        raise OAuthConnectionValidationError(f"{label} is invalid") from exc
    if len(encoded) > maximum:
        raise OAuthConnectionValidationError(f"{label} is invalid")
    return value


def _token(value: object, label: str, *, optional: bool = False) -> str | None:
    return _bounded_text(value, label, MAX_TOKEN_BYTES, optional=optional)


def _account(value: object) -> ConnectionAccount | None:
    if value is None:
        return None
    if isinstance(value, ConnectionAccount):
        raw_id, raw_username, raw_name = value.id, value.username, value.name
    elif isinstance(value, Mapping) and set(value) <= {"id", "username", "name"} and "id" in value:
        raw_id = value.get("id")
        raw_username = value.get("username")
        raw_name = value.get("name")
    else:
        raise OAuthConnectionValidationError("OAuth account is invalid")
    return ConnectionAccount(
        id=str(_bounded_text(raw_id, "OAuth account id", MAX_ACCOUNT_ID_BYTES)),
        username=_bounded_text(
            raw_username,
            "OAuth account username",
            MAX_ACCOUNT_TEXT_BYTES,
            optional=True,
        ),
        name=_bounded_text(raw_name, "OAuth account name", MAX_ACCOUNT_TEXT_BYTES, optional=True),
    )


def _stored_status(value: object) -> StoredStatus:
    if not isinstance(value, str) or value not in _STORED_STATUSES:
        raise OAuthConnectionValidationError("OAuth connection status is invalid")
    return value  # type: ignore[return-value]


def _intent(provider_id: object, scopes: object) -> tuple[str, tuple[str, ...]]:
    try:
        intent = oauth_providers.connection_intent(provider_id, scopes)
    except oauth_providers.OAuthProviderError as exc:
        raise OAuthConnectionValidationError("OAuth connection declaration is invalid") from exc
    return intent.provider.id, intent.scopes


def _token_set(
    value: object,
    expected_scopes: tuple[str, ...],
    now: int,
    account: object,
) -> _TokenGrant:
    try:
        access_token = value.access_token  # type: ignore[attr-defined]
        refresh_token = value.refresh_token  # type: ignore[attr-defined]
        raw_scopes = value.scopes  # type: ignore[attr-defined]
        expires_in = value.expires_in  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as exc:
        raise OAuthConnectionValidationError("OAuth token set is invalid") from exc
    if (
        not isinstance(raw_scopes, tuple)
        or raw_scopes != expected_scopes
        or type(expires_in) is not int
        or not 30 <= expires_in <= 31_536_000
    ):
        raise OAuthConnectionValidationError("OAuth token set is invalid")
    expiry = now + expires_in
    if not 1 <= expiry <= (2**53 - 1):
        raise OAuthConnectionValidationError("OAuth token expiry is invalid")
    return _TokenGrant(
        access_token=str(_token(access_token, "OAuth access token")),
        refresh_token=_token(refresh_token, "OAuth refresh token", optional=True),
        scopes=expected_scopes,
        expires_at=expiry,
        account=_account(account),
        status="connected",
    )


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _empty_state() -> dict[str, object]:
    return {"schema": 1, "teams": {}}


def _strict_json(payload: bytes) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise OAuthConnectionStoreError("OAuth connection state has duplicate fields")
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OAuthConnectionStoreError("OAuth connection state is not valid JSON") from exc


def _decode_part(
    value: object,
    *,
    expected: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> bytes:
    if not isinstance(value, str) or len(value) > (MAX_PLAINTEXT_BYTES * 2) + 128:
        raise OAuthConnectionStoreError("OAuth connection envelope is malformed")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise OAuthConnectionStoreError("OAuth connection envelope is malformed") from exc
    if (
        (expected is not None and len(decoded) != expected)
        or (minimum is not None and len(decoded) < minimum)
        or (maximum is not None and len(decoded) > maximum)
    ):
        raise OAuthConnectionStoreError("OAuth connection envelope is malformed")
    return decoded


def _read_private_file(path: Path, maximum: int, label: str) -> bytes | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise OAuthConnectionStoreError(f"{label} is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size > maximum
        ):
            raise OAuthConnectionStoreError(f"{label} failed its ownership contract")
        payload = bytearray()
        while len(payload) <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > maximum:
            raise OAuthConnectionStoreError(f"{label} exceeds its fixed byte limit")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _require_private_parent(path: Path, label: str) -> None:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise OAuthConnectionStoreError(f"{label} directory is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise OAuthConnectionStoreError(f"{label} directory failed its ownership contract")


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
        raise OAuthConnectionStoreError(f"{label} could not be persisted") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            temporary.unlink()


def _record_metadata(
    record: Mapping[str, object],
) -> tuple[str, tuple[str, ...], int, StoredStatus, int]:
    provider = record.get("provider")
    raw_scopes = record.get("scopes")
    expires_at = record.get("expires_at")
    status = record.get("status")
    generation = record.get("generation")
    try:
        canonical_provider, scopes = _intent(provider, raw_scopes)
        canonical_status = _stored_status(status)
    except OAuthConnectionValidationError as exc:
        raise OAuthConnectionStoreError("OAuth connection state record is malformed") from exc
    if (
        not isinstance(raw_scopes, list)
        or tuple(raw_scopes) != scopes
        or type(expires_at) is not int
        or not 1 <= expires_at <= (2**53 - 1)
        or type(generation) is not int
        or generation < 1
    ):
        raise OAuthConnectionStoreError("OAuth connection state record is malformed")
    return canonical_provider, scopes, expires_at, canonical_status, generation


def _validate_record(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "provider",
        "scopes",
        "expires_at",
        "status",
        "generation",
        "updated_at",
        "envelope",
    }:
        raise OAuthConnectionStoreError("OAuth connection state record is malformed")
    _record_metadata(value)
    updated_at = value.get("updated_at")
    envelope = value.get("envelope")
    if (
        not isinstance(updated_at, str)
        or _TIMESTAMP.fullmatch(updated_at) is None
        or not isinstance(envelope, dict)
        or set(envelope) != {"algorithm", "nonce", "ciphertext"}
        or envelope.get("algorithm") != "AES-256-GCM"
    ):
        raise OAuthConnectionStoreError("OAuth connection state record is malformed")
    _decode_part(envelope.get("nonce"), expected=12)
    _decode_part(envelope.get("ciphertext"), minimum=17, maximum=MAX_PLAINTEXT_BYTES + 16)
    return value


def _validate_state(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"schema", "teams"} or value.get("schema") != 1:
        raise OAuthConnectionStoreError("OAuth connection state has an unsupported shape")
    teams = value.get("teams")
    if not isinstance(teams, dict):
        raise OAuthConnectionStoreError("OAuth connection state is malformed")
    total = 0
    for raw_team, raw_assistants in teams.items():
        try:
            _team_id(raw_team)
        except OAuthConnectionValidationError as exc:
            raise OAuthConnectionStoreError("OAuth connection state is malformed") from exc
        if not isinstance(raw_assistants, dict):
            raise OAuthConnectionStoreError("OAuth connection state is malformed")
        for raw_assistant, raw_connections in raw_assistants.items():
            try:
                _component_id(raw_assistant, "Assistant id")
            except OAuthConnectionValidationError as exc:
                raise OAuthConnectionStoreError("OAuth connection state is malformed") from exc
            if not isinstance(raw_connections, dict) or len(raw_connections) > MAX_CONNECTIONS_PER_ASSISTANT:
                raise OAuthConnectionStoreError("OAuth connection state is malformed")
            for raw_connection, raw_record in raw_connections.items():
                try:
                    _component_id(raw_connection, "connection id")
                except OAuthConnectionValidationError as exc:
                    raise OAuthConnectionStoreError("OAuth connection state is malformed") from exc
                _validate_record(raw_record)
                total += 1
                if total > MAX_TOTAL_RECORDS:
                    raise OAuthConnectionStoreError("OAuth connection state exceeds its record limit")
    return value


def _has_records(state: Mapping[str, object]) -> bool:
    teams = state.get("teams")
    if not isinstance(teams, dict):
        raise OAuthConnectionStoreError("OAuth connection state is malformed")
    return any(
        bool(connections)
        for assistants in teams.values()
        if isinstance(assistants, dict)
        for connections in assistants.values()
        if isinstance(connections, dict)
    )


def _aad(
    team_id: str,
    assistant_id: str,
    connection_id: str,
    record: Mapping[str, object],
) -> bytes:
    provider, scopes, expires_at, status, generation = _record_metadata(record)
    return json.dumps(
        [
            "shimpz-oauth-connection-v1",
            team_id,
            assistant_id,
            connection_id,
            provider,
            list(scopes),
            expires_at,
            status,
            generation,
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")


def _declarations(value: object) -> dict[str, tuple[str, tuple[str, ...]]]:
    if not isinstance(value, Mapping) or len(value) > MAX_CONNECTIONS_PER_ASSISTANT:
        raise OAuthConnectionValidationError("OAuth connection declarations are invalid")
    declared: dict[str, tuple[str, tuple[str, ...]]] = {}
    for raw_id, raw_spec in value.items():
        connection_id = _component_id(raw_id, "connection id")
        if isinstance(raw_spec, Mapping) and set(raw_spec) == {"provider", "scopes"}:
            provider = raw_spec.get("provider")
            scopes = raw_spec.get("scopes")
        else:
            try:
                provider = raw_spec.provider  # type: ignore[attr-defined]
                scopes = raw_spec.scopes  # type: ignore[attr-defined]
            except (AttributeError, TypeError) as exc:
                raise OAuthConnectionValidationError("OAuth connection declarations are invalid") from exc
        declared[connection_id] = _intent(provider, scopes)
    return declared


def _declared_ids(value: object) -> tuple[str, ...]:
    values: Iterable[object]
    if isinstance(value, Mapping):
        values = value.keys()
    elif isinstance(value, Iterable) and not isinstance(value, str | bytes):
        values = value
    else:
        raise OAuthConnectionValidationError("OAuth connection ids are invalid")
    declared: list[str] = []
    seen: set[str] = set()
    for raw_id in values:
        if len(declared) == MAX_CONNECTIONS_PER_ASSISTANT:
            raise OAuthConnectionValidationError("OAuth connection ids are invalid")
        connection_id = _component_id(raw_id, "connection id")
        if connection_id in seen:
            raise OAuthConnectionValidationError("OAuth connection ids are invalid")
        declared.append(connection_id)
        seen.add(connection_id)
    return tuple(declared)


class OAuthConnectionStore:
    def __init__(
        self,
        state_path: Path = STATE_PATH,
        key_path: Path = KEY_PATH,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.state_path = Path(state_path)
        self.key_path = Path(key_path)
        if not self.state_path.is_absolute() or not self.key_path.is_absolute():
            raise OAuthConnectionStoreError("OAuth connection state and key paths must be absolute")
        try:
            state_parent = self.state_path.parent.resolve()
            key_parent = self.key_path.parent.resolve()
        except OSError as exc:
            raise OAuthConnectionStoreError("OAuth connection storage paths are unavailable") from exc
        if state_parent == key_parent:
            raise OAuthConnectionStoreError("OAuth connection keyring must be separate from encrypted state")
        if not callable(clock):
            raise OAuthConnectionStoreError("OAuth connection clock is invalid")
        self._clock = clock
        self._lock = threading.RLock()

    def _now(self) -> int:
        now = self._clock()
        if not isinstance(now, int | float) or isinstance(now, bool) or not 0 <= now <= (2**53 - 1):
            raise OAuthConnectionStoreError("OAuth connection clock is invalid")
        return int(now)

    def _read_state(self) -> dict[str, object]:
        payload = _read_private_file(self.state_path, MAX_STATE_BYTES, "OAuth connection state")
        return _empty_state() if payload is None else _validate_state(_strict_json(payload))

    def _write_state(self, state: Mapping[str, object]) -> None:
        validated = _validate_state(dict(state))
        payload = json.dumps(validated, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(payload) > MAX_STATE_BYTES:
            raise OAuthConnectionStoreError("OAuth connection state exceeds its fixed byte limit")
        _atomic_write(self.state_path, payload, "OAuth connection state")

    def _key(self, *, allow_create: bool = False) -> bytes:
        payload = _read_private_file(self.key_path, 32, "OAuth connection keyring")
        if payload is None:
            if not allow_create:
                raise OAuthConnectionStoreError("OAuth connection keyring is unavailable")
            payload = AESGCM.generate_key(bit_length=256)
            _atomic_write(self.key_path, payload, "OAuth connection keyring")
        if len(payload) != 32:
            raise OAuthConnectionStoreError("OAuth connection keyring is invalid")
        return payload

    @staticmethod
    def _records(
        state: dict[str, object],
        team_id: str,
        assistant_id: str,
        *,
        create: bool,
    ) -> dict[str, object]:
        teams = state["teams"]
        if not isinstance(teams, dict):
            raise OAuthConnectionStoreError("OAuth connection state is malformed")
        assistants = teams.get(team_id)
        if assistants is None:
            if not create:
                return {}
            assistants = {}
            teams[team_id] = assistants
        elif not isinstance(assistants, dict):
            raise OAuthConnectionStoreError("OAuth connection state is malformed")
        records = assistants.get(assistant_id)
        if records is None:
            if not create:
                return {}
            records = {}
            assistants[assistant_id] = records
        elif not isinstance(records, dict):
            raise OAuthConnectionStoreError("OAuth connection state is malformed")
        return records

    @staticmethod
    def _plaintext(grant: _TokenGrant) -> bytes:
        payload = json.dumps(
            {
                "access_token": grant.access_token,
                "refresh_token": grant.refresh_token,
                "account": (
                    None
                    if grant.account is None
                    else {
                        "id": grant.account.id,
                        "username": grant.account.username,
                        "name": grant.account.name,
                    }
                ),
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(payload) > MAX_PLAINTEXT_BYTES:
            raise OAuthConnectionValidationError("OAuth token set is too large")
        return payload

    @staticmethod
    def _decrypted(
        plaintext: bytes,
        provider: str,
        scopes: tuple[str, ...],
        expires_at: int,
        status: StoredStatus,
        generation: int,
    ) -> _TokenGrant:
        if len(plaintext) > MAX_PLAINTEXT_BYTES:
            raise OAuthConnectionStoreError("decrypted OAuth connection is malformed")
        value = _strict_json(plaintext)
        if not isinstance(value, dict) or set(value) != {"access_token", "refresh_token", "account"}:
            raise OAuthConnectionStoreError("decrypted OAuth connection is malformed")
        try:
            account = _account(value.get("account"))
            return _TokenGrant(
                access_token=str(_token(value.get("access_token"), "OAuth access token")),
                refresh_token=_token(value.get("refresh_token"), "OAuth refresh token", optional=True),
                scopes=scopes,
                expires_at=expires_at,
                account=account,
                status=status,
                generation=generation,
            )
        except OAuthConnectionValidationError as exc:
            raise OAuthConnectionStoreError("decrypted OAuth connection is malformed") from exc

    def _resolve_record(
        self,
        team: str,
        assistant: str,
        connection: str,
        record: object,
    ) -> _TokenGrant:
        validated = _validate_record(record)
        provider, scopes, expires_at, status, generation = _record_metadata(validated)
        envelope = validated["envelope"]
        if not isinstance(envelope, dict):
            raise OAuthConnectionStoreError("OAuth connection envelope is malformed")
        try:
            plaintext = AESGCM(self._key()).decrypt(
                _decode_part(envelope.get("nonce"), expected=12),
                _decode_part(envelope.get("ciphertext")),
                _aad(team, assistant, connection, validated),
            )
        except InvalidTag as exc:
            raise OAuthConnectionStoreError("OAuth connection envelope authentication failed") from exc
        return self._decrypted(plaintext, provider, scopes, expires_at, status, generation)

    def put(
        self,
        team_id: object,
        assistant_id: object,
        connection_id: object,
        provider: object,
        scopes: object,
        token_set: object,
        account: object = None,
    ) -> OAuthConnectionMetadata:
        """Encrypt one exchanged token set and atomically advance its generation."""
        team = _team_id(team_id)
        assistant = _component_id(assistant_id, "Assistant id")
        connection = _component_id(connection_id, "connection id")
        canonical_provider, canonical_scopes = _intent(provider, scopes)
        canonical = _token_set(token_set, canonical_scopes, self._now(), account)
        plaintext = self._plaintext(canonical)
        with self._lock:
            state = self._read_state()
            key = self._key(allow_create=not _has_records(state))
            records = self._records(state, team, assistant, create=True)
            if connection not in records and len(records) >= MAX_CONNECTIONS_PER_ASSISTANT:
                raise OAuthConnectionStoreError("OAuth connection capacity reached")
            previous = records.get(connection)
            generation = int(previous.get("generation", 0)) + 1 if isinstance(previous, dict) else 1
            record: dict[str, object] = {
                "provider": canonical_provider,
                "scopes": list(canonical.scopes),
                "expires_at": canonical.expires_at,
                "status": canonical.status,
                "generation": generation,
                "updated_at": _timestamp(),
                "envelope": {},
            }
            nonce = os.urandom(12)
            ciphertext = AESGCM(key).encrypt(
                nonce,
                plaintext,
                _aad(team, assistant, connection, record),
            )
            record["envelope"] = {
                "algorithm": "AES-256-GCM",
                "nonce": base64.b64encode(nonce).decode("ascii"),
                "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            }
            records[connection] = record
            self._write_state(state)
            return OAuthConnectionMetadata(
                connection,
                canonical_provider,
                canonical.scopes,
                "connected",
                canonical.account,
                canonical.expires_at,
                generation,
            )

    def resolve(
        self,
        team_id: object,
        assistant_id: object,
        connection_id: object,
        provider: object,
        scopes: object,
        refresh_callback: Callable[[str], object],
    ) -> str:
        """Return one bounded access token, refreshing once under a single-flight lock."""
        team = _team_id(team_id)
        assistant = _component_id(assistant_id, "Assistant id")
        connection = _component_id(connection_id, "connection id")
        canonical_provider, expected_scopes = _intent(provider, scopes)
        if not callable(refresh_callback):
            raise OAuthConnectionValidationError("OAuth refresh callback is invalid")
        with self._lock:
            state = self._read_state()
            records = self._records(state, team, assistant, create=False)
            if connection not in records:
                raise OAuthConnectionMissingError("OAuth connection is not configured")
            grant = self._resolve_record(team, assistant, connection, records[connection])
            stored_provider, stored_scopes, _, _, _ = _record_metadata(_validate_record(records[connection]))
            if stored_provider != canonical_provider or stored_scopes != expected_scopes:
                raise OAuthConnectionReauthorizationError(
                    "OAuth connection declaration changed; reauthorization is required"
                )
            if grant.status == "reauthorization-required":
                raise OAuthConnectionReauthorizationError("OAuth connection requires reauthorization")
            if grant.expires_at > self._now() + REFRESH_WINDOW_SECONDS:
                return grant.access_token
            if grant.refresh_token is None:
                raise OAuthConnectionReauthorizationError("OAuth connection requires reauthorization")
            refreshed = refresh_callback(grant.refresh_token)
            canonical = _token_set(refreshed, expected_scopes, self._now(), grant.account)
            self.put(
                team,
                assistant,
                connection,
                canonical_provider,
                expected_scopes,
                refreshed,
                grant.account,
            )
            return canonical.access_token

    def metadata(
        self,
        team_id: object,
        assistant_id: object,
        declarations: object,
    ) -> tuple[OAuthConnectionMetadata, ...]:
        """Return complete declared inventory, including missing connection rows."""
        team = _team_id(team_id)
        assistant = _component_id(assistant_id, "Assistant id")
        declared = _declarations(declarations)
        with self._lock:
            state = self._read_state()
            records = self._records(state, team, assistant, create=False)
            now = self._now()
            result: list[OAuthConnectionMetadata] = []
            for connection, (provider, scopes) in declared.items():
                if connection not in records:
                    result.append(
                        OAuthConnectionMetadata(
                            connection,
                            provider,
                            scopes,
                            "missing",
                            None,
                            None,
                            0,
                        )
                    )
                    continue
                record = _validate_record(records[connection])
                stored_provider, stored_scopes, expires_at, status, generation = _record_metadata(record)
                grant = self._resolve_record(team, assistant, connection, record)
                if stored_provider != provider or stored_scopes != scopes:
                    result.append(
                        OAuthConnectionMetadata(
                            connection,
                            provider,
                            scopes,
                            "reauthorization-required",
                            None,
                            expires_at,
                            generation,
                        )
                    )
                    continue
                projected: ConnectionStatus = status
                if status == "connected" and expires_at <= now:
                    projected = "refresh-required" if grant.refresh_token is not None else "reauthorization-required"
                result.append(
                    OAuthConnectionMetadata(
                        connection,
                        provider,
                        scopes,
                        projected,
                        grant.account,
                        expires_at,
                        generation,
                    )
                )
            return tuple(result)

    def retain_declared(
        self,
        team_id: object,
        assistant_id: object,
        declared_ids: object,
    ) -> bool:
        """Atomically discard connections removed from a new Assistant release."""
        team = _team_id(team_id)
        assistant = _component_id(assistant_id, "Assistant id")
        declared = set(_declared_ids(declared_ids))
        with self._lock:
            state = self._read_state()
            teams = state["teams"]
            if not isinstance(teams, dict) or not isinstance(teams.get(team), dict):
                return False
            assistants = teams[team]
            records = self._records(state, team, assistant, create=False)
            obsolete = set(records) - declared
            if not obsolete:
                return False
            for connection in obsolete:
                records.pop(connection)
            if not records:
                assistants.pop(assistant, None)
            if not assistants:
                teams.pop(team, None)
            self._write_state(state)
            return True

    def delete_connection(
        self,
        team_id: object,
        assistant_id: object,
        connection_id: object,
    ) -> bool:
        team = _team_id(team_id)
        assistant = _component_id(assistant_id, "Assistant id")
        connection = _component_id(connection_id, "connection id")
        with self._lock:
            state = self._read_state()
            teams = state["teams"]
            if not isinstance(teams, dict) or not isinstance(teams.get(team), dict):
                return False
            assistants = teams[team]
            records = self._records(state, team, assistant, create=False)
            removed = records.pop(connection, None) is not None
            if removed and not records:
                assistants.pop(assistant, None)
            if removed and not assistants:
                teams.pop(team, None)
            if removed:
                self._write_state(state)
            return removed

    def revoke_then_delete(
        self,
        team_id: object,
        assistant_id: object,
        connection_id: object,
        revoke_callback: Callable[[str, str, str | None], None],
    ) -> bool:
        """Delete one grant only after its authenticated tokens are revoked upstream."""
        team = _team_id(team_id)
        assistant = _component_id(assistant_id, "Assistant id")
        connection = _component_id(connection_id, "connection id")
        if not callable(revoke_callback):
            raise OAuthConnectionValidationError("OAuth revocation callback is invalid")
        with self._lock:
            state = self._read_state()
            teams = state["teams"]
            if not isinstance(teams, dict) or not isinstance(teams.get(team), dict):
                return False
            assistants = teams[team]
            records = self._records(state, team, assistant, create=False)
            raw_record = records.get(connection)
            if raw_record is None:
                return False
            record = _validate_record(raw_record)
            provider, _, _, _, _ = _record_metadata(record)
            grant = self._resolve_record(team, assistant, connection, record)
            revoke_callback(provider, grant.access_token, grant.refresh_token)
            records.pop(connection)
            if not records:
                assistants.pop(assistant, None)
            if not assistants:
                teams.pop(team, None)
            self._write_state(state)
            return True

    def delete_assistant(self, team_id: object, assistant_id: object) -> bool:
        team = _team_id(team_id)
        assistant = _component_id(assistant_id, "Assistant id")
        with self._lock:
            state = self._read_state()
            teams = state["teams"]
            if not isinstance(teams, dict) or not isinstance(teams.get(team), dict):
                return False
            assistants = teams[team]
            removed = assistants.pop(assistant, None) is not None
            if removed and not assistants:
                teams.pop(team, None)
            if removed:
                self._write_state(state)
            return removed

    def delete_team(self, team_id: object) -> bool:
        team = _team_id(team_id)
        with self._lock:
            state = self._read_state()
            teams = state["teams"]
            if not isinstance(teams, dict):
                raise OAuthConnectionStoreError("OAuth connection state is malformed")
            removed = teams.pop(team, None) is not None
            if removed:
                self._write_state(state)
            return removed

    def delete_all(self) -> bool:
        """Atomically purge all connection material during an owned Space reset."""
        with self._lock:
            state = self._read_state()
            if not _has_records(state):
                return False
            self._write_state(_empty_state())
            return True
