"""Encrypted, controller-owned OAuth tokens for Assistant accounts.

Tokens never belong to an Assistant manifest, Brain prompt, process environment,
command argument, public API response, or log.  Each encrypted record is bound to
the exact Team, Assistant, account, provider, scopes, expiry, status, and
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

STATE_PATH = Path("/var/lib/shimpz-local/assistant-accounts/state/accounts.json")
KEY_PATH = Path("/var/lib/shimpz-local/assistant-accounts/key/aes256.key")
MAX_STATE_BYTES = 4 * 1024 * 1024
MAX_TOKEN_BYTES = 16 * 1024
MAX_PLAINTEXT_BYTES = (MAX_TOKEN_BYTES * 3) + 2048
MAX_ACCOUNTS_PER_ASSISTANT = 16
MAX_TOTAL_RECORDS = 4096
MAX_ACCOUNT_ID_BYTES = 256
MAX_ACCOUNT_TEXT_BYTES = 512
REFRESH_WINDOW_SECONDS = 60
_TEAM_ID = re.compile(r"[a-z0-9_]{1,40}\Z")
_COMPONENT_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_STORED_STATUSES = frozenset({"connected", "reauthorization-required"})
StoredStatus = Literal["connected", "reauthorization-required"]
AccountStatus = Literal[
    "missing",
    "connected",
    "refresh-required",
    "reauthorization-required",
]


class OAuthAccountStoreError(RuntimeError):
    """OAuth account state is invalid, unavailable, or unauthentic."""


class OAuthAccountValidationError(OAuthAccountStoreError):
    """A caller supplied invalid OAuth account data."""


class OAuthAccountMissingError(OAuthAccountStoreError):
    """The requested OAuth account has not been configured."""


class OAuthAccountReauthorizationError(OAuthAccountStoreError):
    """A new provider authorization is required before this account can run."""


@dataclass(frozen=True, slots=True)
class OAuthAccountIdentity:
    id: str
    username: str | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class OAuthAccountMetadata:
    """Bounded public inventory that never contains either OAuth token."""

    id: str
    provider: str
    scopes: tuple[str, ...]
    status: AccountStatus
    account: OAuthAccountIdentity | None
    expires_at: int | None
    generation: int


@dataclass(frozen=True, slots=True, repr=False)
class _TokenGrant:
    access_token: str
    refresh_token: str | None
    broker_lease: str | None
    scopes: tuple[str, ...]
    expires_at: int
    account: OAuthAccountIdentity | None
    status: StoredStatus
    generation: int = 0


def _component_id(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) > 64 or _COMPONENT_ID.fullmatch(value) is None:
        raise OAuthAccountValidationError(f"{label} is invalid")
    return value


def _team_id(value: object) -> str:
    if not isinstance(value, str) or _TEAM_ID.fullmatch(value) is None:
        raise OAuthAccountValidationError("Team id is invalid")
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
        raise OAuthAccountValidationError(f"{label} is invalid")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        raise OAuthAccountValidationError(f"{label} is invalid") from exc
    if len(encoded) > maximum:
        raise OAuthAccountValidationError(f"{label} is invalid")
    return value


def _token(value: object, label: str, *, optional: bool = False) -> str | None:
    return _bounded_text(value, label, MAX_TOKEN_BYTES, optional=optional)


def _account(value: object) -> OAuthAccountIdentity | None:
    if value is None:
        return None
    if isinstance(value, OAuthAccountIdentity):
        raw_id, raw_username, raw_name = value.id, value.username, value.name
    elif isinstance(value, Mapping) and set(value) <= {"id", "username", "name"} and "id" in value:
        raw_id = value.get("id")
        raw_username = value.get("username")
        raw_name = value.get("name")
    else:
        raise OAuthAccountValidationError("OAuth account is invalid")
    return OAuthAccountIdentity(
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
        raise OAuthAccountValidationError("OAuth account status is invalid")
    return value  # type: ignore[return-value]


def _intent(provider_id: object, scopes: object) -> tuple[str, tuple[str, ...]]:
    try:
        intent = oauth_providers.account_intent(provider_id, scopes)
    except oauth_providers.OAuthProviderError as exc:
        raise OAuthAccountValidationError("OAuth account declaration is invalid") from exc
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
        broker_lease = getattr(value, "broker_lease", None)
        raw_scopes = value.scopes  # type: ignore[attr-defined]
        expires_in = value.expires_in  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as exc:
        raise OAuthAccountValidationError("OAuth token set is invalid") from exc
    if (
        not isinstance(raw_scopes, tuple)
        or raw_scopes != expected_scopes
        or type(expires_in) is not int
        or not 30 <= expires_in <= 31_536_000
    ):
        raise OAuthAccountValidationError("OAuth token set is invalid")
    expiry = now + expires_in
    if not 1 <= expiry <= (2**53 - 1):
        raise OAuthAccountValidationError("OAuth token expiry is invalid")
    return _TokenGrant(
        access_token=str(_token(access_token, "OAuth access token")),
        refresh_token=_token(refresh_token, "OAuth refresh token", optional=True),
        broker_lease=_token(broker_lease, "OAuth broker lease", optional=True),
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
                raise OAuthAccountStoreError("OAuth account state has duplicate fields")
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OAuthAccountStoreError("OAuth account state is not valid JSON") from exc


def _decode_part(
    value: object,
    *,
    expected: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> bytes:
    if not isinstance(value, str) or len(value) > (MAX_PLAINTEXT_BYTES * 2) + 128:
        raise OAuthAccountStoreError("OAuth account envelope is malformed")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise OAuthAccountStoreError("OAuth account envelope is malformed") from exc
    if (
        (expected is not None and len(decoded) != expected)
        or (minimum is not None and len(decoded) < minimum)
        or (maximum is not None and len(decoded) > maximum)
    ):
        raise OAuthAccountStoreError("OAuth account envelope is malformed")
    return decoded


def _read_private_file(path: Path, maximum: int, label: str) -> bytes | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise OAuthAccountStoreError(f"{label} is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size > maximum
        ):
            raise OAuthAccountStoreError(f"{label} failed its ownership contract")
        payload = bytearray()
        while len(payload) <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > maximum:
            raise OAuthAccountStoreError(f"{label} exceeds its fixed byte limit")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _require_private_parent(path: Path, label: str) -> None:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise OAuthAccountStoreError(f"{label} directory is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise OAuthAccountStoreError(f"{label} directory failed its ownership contract")


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
        raise OAuthAccountStoreError(f"{label} could not be persisted") from exc
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
    except OAuthAccountValidationError as exc:
        raise OAuthAccountStoreError("OAuth account state record is malformed") from exc
    if (
        not isinstance(raw_scopes, list)
        or tuple(raw_scopes) != scopes
        or type(expires_at) is not int
        or not 1 <= expires_at <= (2**53 - 1)
        or type(generation) is not int
        or generation < 1
    ):
        raise OAuthAccountStoreError("OAuth account state record is malformed")
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
        raise OAuthAccountStoreError("OAuth account state record is malformed")
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
        raise OAuthAccountStoreError("OAuth account state record is malformed")
    _decode_part(envelope.get("nonce"), expected=12)
    _decode_part(envelope.get("ciphertext"), minimum=17, maximum=MAX_PLAINTEXT_BYTES + 16)
    return value


def _validate_state(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"schema", "teams"} or value.get("schema") != 1:
        raise OAuthAccountStoreError("OAuth account state has an unsupported shape")
    teams = value.get("teams")
    if not isinstance(teams, dict):
        raise OAuthAccountStoreError("OAuth account state is malformed")
    total = 0
    for raw_team, raw_assistants in teams.items():
        try:
            _team_id(raw_team)
        except OAuthAccountValidationError as exc:
            raise OAuthAccountStoreError("OAuth account state is malformed") from exc
        if not isinstance(raw_assistants, dict):
            raise OAuthAccountStoreError("OAuth account state is malformed")
        for raw_assistant, raw_accounts in raw_assistants.items():
            try:
                _component_id(raw_assistant, "Assistant id")
            except OAuthAccountValidationError as exc:
                raise OAuthAccountStoreError("OAuth account state is malformed") from exc
            if not isinstance(raw_accounts, dict) or len(raw_accounts) > MAX_ACCOUNTS_PER_ASSISTANT:
                raise OAuthAccountStoreError("OAuth account state is malformed")
            for raw_account, raw_record in raw_accounts.items():
                try:
                    _component_id(raw_account, "account id")
                except OAuthAccountValidationError as exc:
                    raise OAuthAccountStoreError("OAuth account state is malformed") from exc
                _validate_record(raw_record)
                total += 1
                if total > MAX_TOTAL_RECORDS:
                    raise OAuthAccountStoreError("OAuth account state exceeds its record limit")
    return value


def _has_records(state: Mapping[str, object]) -> bool:
    teams = state.get("teams")
    if not isinstance(teams, dict):
        raise OAuthAccountStoreError("OAuth account state is malformed")
    return any(
        bool(accounts)
        for assistants in teams.values()
        if isinstance(assistants, dict)
        for accounts in assistants.values()
        if isinstance(accounts, dict)
    )


def _aad(
    team_id: str,
    assistant_id: str,
    account_id: str,
    record: Mapping[str, object],
) -> bytes:
    provider, scopes, expires_at, status, generation = _record_metadata(record)
    return json.dumps(
        [
            "shimpz-oauth-account-v1",
            team_id,
            assistant_id,
            account_id,
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
    if not isinstance(value, Mapping) or len(value) > MAX_ACCOUNTS_PER_ASSISTANT:
        raise OAuthAccountValidationError("OAuth account declarations are invalid")
    declared: dict[str, tuple[str, tuple[str, ...]]] = {}
    for raw_id, raw_spec in value.items():
        account_id = _component_id(raw_id, "account id")
        if isinstance(raw_spec, Mapping) and set(raw_spec) == {"provider", "scopes"}:
            provider = raw_spec.get("provider")
            scopes = raw_spec.get("scopes")
        else:
            try:
                provider = raw_spec.provider  # type: ignore[attr-defined]
                scopes = raw_spec.scopes  # type: ignore[attr-defined]
            except (AttributeError, TypeError) as exc:
                raise OAuthAccountValidationError("OAuth account declarations are invalid") from exc
        declared[account_id] = _intent(provider, scopes)
    return declared


def _declared_ids(value: object) -> tuple[str, ...]:
    values: Iterable[object]
    if isinstance(value, Mapping):
        values = value.keys()
    elif isinstance(value, Iterable) and not isinstance(value, str | bytes):
        values = value
    else:
        raise OAuthAccountValidationError("OAuth account ids are invalid")
    declared: list[str] = []
    seen: set[str] = set()
    for raw_id in values:
        if len(declared) == MAX_ACCOUNTS_PER_ASSISTANT:
            raise OAuthAccountValidationError("OAuth account ids are invalid")
        account_id = _component_id(raw_id, "account id")
        if account_id in seen:
            raise OAuthAccountValidationError("OAuth account ids are invalid")
        declared.append(account_id)
        seen.add(account_id)
    return tuple(declared)


class OAuthAccountStore:
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
            raise OAuthAccountStoreError("OAuth account state and key paths must be absolute")
        try:
            state_parent = self.state_path.parent.resolve()
            key_parent = self.key_path.parent.resolve()
        except OSError as exc:
            raise OAuthAccountStoreError("OAuth account storage paths are unavailable") from exc
        if state_parent == key_parent:
            raise OAuthAccountStoreError("OAuth account keyring must be separate from encrypted state")
        if not callable(clock):
            raise OAuthAccountStoreError("OAuth account clock is invalid")
        self._clock = clock
        self._lock = threading.RLock()

    def _now(self) -> int:
        now = self._clock()
        if not isinstance(now, int | float) or isinstance(now, bool) or not 0 <= now <= (2**53 - 1):
            raise OAuthAccountStoreError("OAuth account clock is invalid")
        return int(now)

    def _read_state(self) -> dict[str, object]:
        payload = _read_private_file(self.state_path, MAX_STATE_BYTES, "OAuth account state")
        return _empty_state() if payload is None else _validate_state(_strict_json(payload))

    def _write_state(self, state: Mapping[str, object]) -> None:
        validated = _validate_state(dict(state))
        payload = json.dumps(validated, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(payload) > MAX_STATE_BYTES:
            raise OAuthAccountStoreError("OAuth account state exceeds its fixed byte limit")
        _atomic_write(self.state_path, payload, "OAuth account state")

    def _key(self, *, allow_create: bool = False) -> bytes:
        payload = _read_private_file(self.key_path, 32, "OAuth account keyring")
        if payload is None:
            if not allow_create:
                raise OAuthAccountStoreError("OAuth account keyring is unavailable")
            payload = AESGCM.generate_key(bit_length=256)
            _atomic_write(self.key_path, payload, "OAuth account keyring")
        if len(payload) != 32:
            raise OAuthAccountStoreError("OAuth account keyring is invalid")
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
            raise OAuthAccountStoreError("OAuth account state is malformed")
        assistants = teams.get(team_id)
        if assistants is None:
            if not create:
                return {}
            assistants = {}
            teams[team_id] = assistants
        elif not isinstance(assistants, dict):
            raise OAuthAccountStoreError("OAuth account state is malformed")
        records = assistants.get(assistant_id)
        if records is None:
            if not create:
                return {}
            records = {}
            assistants[assistant_id] = records
        elif not isinstance(records, dict):
            raise OAuthAccountStoreError("OAuth account state is malformed")
        return records

    @staticmethod
    def _plaintext(grant: _TokenGrant) -> bytes:
        payload = json.dumps(
            {
                "access_token": grant.access_token,
                "refresh_token": grant.refresh_token,
                "broker_lease": grant.broker_lease,
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
            raise OAuthAccountValidationError("OAuth token set is too large")
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
            raise OAuthAccountStoreError("decrypted OAuth account is malformed")
        value = _strict_json(plaintext)
        if not isinstance(value, dict) or set(value) != {
            "access_token",
            "refresh_token",
            "broker_lease",
            "account",
        }:
            raise OAuthAccountStoreError("decrypted OAuth account is malformed")
        try:
            account = _account(value.get("account"))
            return _TokenGrant(
                access_token=str(_token(value.get("access_token"), "OAuth access token")),
                refresh_token=_token(value.get("refresh_token"), "OAuth refresh token", optional=True),
                broker_lease=_token(value.get("broker_lease"), "OAuth broker lease", optional=True),
                scopes=scopes,
                expires_at=expires_at,
                account=account,
                status=status,
                generation=generation,
            )
        except OAuthAccountValidationError as exc:
            raise OAuthAccountStoreError("decrypted OAuth account is malformed") from exc

    def _resolve_record(
        self,
        team: str,
        assistant: str,
        account: str,
        record: object,
    ) -> _TokenGrant:
        validated = _validate_record(record)
        provider, scopes, expires_at, status, generation = _record_metadata(validated)
        envelope = validated["envelope"]
        if not isinstance(envelope, dict):
            raise OAuthAccountStoreError("OAuth account envelope is malformed")
        try:
            plaintext = AESGCM(self._key()).decrypt(
                _decode_part(envelope.get("nonce"), expected=12),
                _decode_part(envelope.get("ciphertext")),
                _aad(team, assistant, account, validated),
            )
        except InvalidTag as exc:
            raise OAuthAccountStoreError("OAuth account envelope authentication failed") from exc
        return self._decrypted(plaintext, provider, scopes, expires_at, status, generation)

    def put(
        self,
        team_id: object,
        assistant_id: object,
        account_id: object,
        provider: object,
        scopes: object,
        token_set: object,
        identity: object = None,
    ) -> OAuthAccountMetadata:
        """Encrypt one exchanged token set and atomically advance its generation."""
        team = _team_id(team_id)
        assistant = _component_id(assistant_id, "Assistant id")
        account = _component_id(account_id, "account id")
        canonical_provider, canonical_scopes = _intent(provider, scopes)
        canonical = _token_set(token_set, canonical_scopes, self._now(), identity)
        plaintext = self._plaintext(canonical)
        with self._lock:
            state = self._read_state()
            key = self._key(allow_create=not _has_records(state))
            records = self._records(state, team, assistant, create=True)
            if account not in records and len(records) >= MAX_ACCOUNTS_PER_ASSISTANT:
                raise OAuthAccountStoreError("OAuth account capacity reached")
            previous = records.get(account)
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
                _aad(team, assistant, account, record),
            )
            record["envelope"] = {
                "algorithm": "AES-256-GCM",
                "nonce": base64.b64encode(nonce).decode("ascii"),
                "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            }
            records[account] = record
            self._write_state(state)
            return OAuthAccountMetadata(
                account,
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
        account_id: object,
        provider: object,
        scopes: object,
        refresh_callback: Callable[[str, str | None], object],
    ) -> str:
        """Return one bounded access token, refreshing once under a single-flight lock."""
        team = _team_id(team_id)
        assistant = _component_id(assistant_id, "Assistant id")
        account = _component_id(account_id, "account id")
        canonical_provider, expected_scopes = _intent(provider, scopes)
        if not callable(refresh_callback):
            raise OAuthAccountValidationError("OAuth refresh callback is invalid")
        with self._lock:
            state = self._read_state()
            records = self._records(state, team, assistant, create=False)
            if account not in records:
                raise OAuthAccountMissingError("OAuth account is not configured")
            grant = self._resolve_record(team, assistant, account, records[account])
            stored_provider, stored_scopes, _, _, _ = _record_metadata(_validate_record(records[account]))
            if stored_provider != canonical_provider or stored_scopes != expected_scopes:
                raise OAuthAccountReauthorizationError("OAuth account declaration changed; reauthorization is required")
            if grant.status == "reauthorization-required":
                raise OAuthAccountReauthorizationError("OAuth account requires reauthorization")
            if grant.expires_at > self._now() + REFRESH_WINDOW_SECONDS:
                return grant.access_token
            if grant.refresh_token is None:
                raise OAuthAccountReauthorizationError("OAuth account requires reauthorization")
            refreshed = refresh_callback(grant.refresh_token, grant.broker_lease)
            canonical = _token_set(refreshed, expected_scopes, self._now(), grant.account)
            self.put(
                team,
                assistant,
                account,
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
    ) -> tuple[OAuthAccountMetadata, ...]:
        """Return complete declared inventory, including missing account rows."""
        team = _team_id(team_id)
        assistant = _component_id(assistant_id, "Assistant id")
        declared = _declarations(declarations)
        with self._lock:
            state = self._read_state()
            records = self._records(state, team, assistant, create=False)
            now = self._now()
            result: list[OAuthAccountMetadata] = []
            for account, (provider, scopes) in declared.items():
                if account not in records:
                    result.append(
                        OAuthAccountMetadata(
                            account,
                            provider,
                            scopes,
                            "missing",
                            None,
                            None,
                            0,
                        )
                    )
                    continue
                record = _validate_record(records[account])
                stored_provider, stored_scopes, expires_at, status, generation = _record_metadata(record)
                grant = self._resolve_record(team, assistant, account, record)
                if stored_provider != provider or stored_scopes != scopes:
                    result.append(
                        OAuthAccountMetadata(
                            account,
                            provider,
                            scopes,
                            "reauthorization-required",
                            None,
                            expires_at,
                            generation,
                        )
                    )
                    continue
                projected: AccountStatus = status
                if status == "connected" and expires_at <= now:
                    projected = "refresh-required" if grant.refresh_token is not None else "reauthorization-required"
                result.append(
                    OAuthAccountMetadata(
                        account,
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
        """Atomically discard accounts removed from a new Assistant release."""
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
            for account in obsolete:
                records.pop(account)
            if not records:
                assistants.pop(assistant, None)
            if not assistants:
                teams.pop(team, None)
            self._write_state(state)
            return True

    def delete_account(
        self,
        team_id: object,
        assistant_id: object,
        account_id: object,
    ) -> bool:
        team = _team_id(team_id)
        assistant = _component_id(assistant_id, "Assistant id")
        account = _component_id(account_id, "account id")
        with self._lock:
            state = self._read_state()
            teams = state["teams"]
            if not isinstance(teams, dict) or not isinstance(teams.get(team), dict):
                return False
            assistants = teams[team]
            records = self._records(state, team, assistant, create=False)
            removed = records.pop(account, None) is not None
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
        account_id: object,
        revoke_callback: Callable[[str, str, str | None, str | None], None],
    ) -> bool:
        """Delete one grant only after its authenticated tokens are revoked upstream."""
        team = _team_id(team_id)
        assistant = _component_id(assistant_id, "Assistant id")
        account = _component_id(account_id, "account id")
        if not callable(revoke_callback):
            raise OAuthAccountValidationError("OAuth revocation callback is invalid")
        with self._lock:
            state = self._read_state()
            teams = state["teams"]
            if not isinstance(teams, dict) or not isinstance(teams.get(team), dict):
                return False
            assistants = teams[team]
            records = self._records(state, team, assistant, create=False)
            raw_record = records.get(account)
            if raw_record is None:
                return False
            record = _validate_record(raw_record)
            provider, _, _, _, _ = _record_metadata(record)
            grant = self._resolve_record(team, assistant, account, record)
            revoke_callback(provider, grant.access_token, grant.refresh_token, grant.broker_lease)
            records.pop(account)
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
                raise OAuthAccountStoreError("OAuth account state is malformed")
            removed = teams.pop(team, None) is not None
            if removed:
                self._write_state(state)
            return removed

    def delete_all(self) -> bool:
        """Atomically purge all account material during an owned Space reset."""
        with self._lock:
            state = self._read_state()
            if not _has_records(state):
                return False
            self._write_state(_empty_state())
            return True
