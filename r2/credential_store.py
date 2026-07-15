"""Encrypted Capsule-scoped credential sets for the R2 Driver Spec v1 form."""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import driver_manifest
from credential_bundle import (
    CREDENTIAL_SCHEMA,
    CredentialBundleValidationError,
    credential_profile,
)
from credential_bundle import (
    validate_bundle as validate_credential_bundle,
)
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

DRIVER = driver_manifest.load()
STATE_PATH = Path(
    os.environ.get(
        "SHIMPZ_R2DRIVER_CREDENTIAL_STATE_FILE",
        "/var/lib/shimpz-r2credentials/state.json",
    )
)
KEY_PATH = Path(
    os.environ.get(
        "SHIMPZ_R2DRIVER_CREDENTIAL_KEY_FILE",
        "/var/lib/shimpz-r2keyring/aes256.key",
    )
)
STATE_VERSION = 2
MAX_STATE_BYTES = 8 * 1024 * 1024
MAX_CREDENTIALS = 4096

_CAPSULE_ID_RE = re.compile(r"^[a-z0-9_]{1,40}$")
_CREDENTIAL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")
_IDEMPOTENCY_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_TIMESTAMP_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_ROOT_KEYS = {"version", "driver_id", "schema_version", "capsules"}
_RECORD_KEYS = {
    "profile_id",
    "label",
    "generation",
    "status",
    "created_at",
    "updated_at",
    "idempotency_hash",
    "create_fingerprint",
    "envelope",
}
_ENVELOPE_KEYS = {"algorithm", "nonce", "ciphertext"}


class CredentialStoreError(Exception):
    """Encrypted credential state could not be safely read, authenticated, or committed."""


class CredentialValidationError(CredentialStoreError):
    """A credential identity or bundle does not satisfy the closed R2 form."""


class CredentialNotFoundError(CredentialStoreError):
    """The exact Capsule-scoped credential does not exist."""


class CredentialConflictError(CredentialStoreError):
    """A create identity or compare-and-swap generation no longer matches."""


class CredentialRevokedError(CredentialStoreError):
    """The credential exists as metadata but its encrypted value has been destroyed."""


@dataclass(frozen=True)
class CredentialMetadata:
    capsule_id: str
    credential_id: str
    profile_id: str
    label: str
    generation: int
    status: str
    created_at: str
    updated_at: str

    def public(self) -> dict[str, object]:
        """Return metadata only; values, nonce, tag, and ciphertext are impossible here."""
        return {
            "capsule_id": self.capsule_id,
            "credential_id": self.credential_id,
            "profile_id": self.profile_id,
            "label": self.label,
            "generation": self.generation,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ResolvedCredential:
    """Private request-scoped values with an intentionally redacted representation."""

    __slots__ = ("_values", "metadata")

    def __init__(self, metadata: CredentialMetadata, values: Mapping[str, str]) -> None:
        self.metadata = metadata
        self._values = MappingProxyType(dict(values))

    def value(self, field_id: str) -> str:
        try:
            return self._values[field_id]
        except KeyError as exc:
            raise CredentialValidationError("resolved credential field is unavailable") from exc

    def values(self) -> dict[str, str]:
        """Make secret access explicit at the trusted driver boundary."""
        return dict(self._values)

    def __repr__(self) -> str:
        return f"ResolvedCredential(metadata={self.metadata!r}, values=<redacted>)"


def _validate_capsule_id(value: object) -> str:
    if not isinstance(value, str) or _CAPSULE_ID_RE.fullmatch(value) is None:
        raise CredentialValidationError("capsule_id must match [a-z0-9_]{1,40}")
    return value


def _validate_credential_id(value: object) -> str:
    if not isinstance(value, str) or _CREDENTIAL_ID_RE.fullmatch(value) is None:
        raise CredentialValidationError("credential_id must be a lowercase kebab-case identifier up to 80 characters")
    return value


def _validate_label(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 80
        or value.strip() != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise CredentialValidationError("label must be a trimmed printable string up to 80 characters")
    return value


def _validate_idempotency_key(value: object) -> str:
    if not isinstance(value, str) or _IDEMPOTENCY_KEY_RE.fullmatch(value) is None:
        raise CredentialValidationError("idempotency_key must be an opaque 16-128 character token")
    return value


def _profile(profile_id: object) -> driver_manifest.CredentialProfile:
    try:
        return credential_profile(profile_id)
    except CredentialBundleValidationError as exc:
        raise CredentialValidationError(str(exc)) from exc


def validate_bundle(profile_id: object, values: object) -> dict[str, str]:
    try:
        return validate_credential_bundle(profile_id, values)
    except CredentialBundleValidationError as exc:
        raise CredentialValidationError(str(exc)) from exc


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _canonical(values: Mapping[str, str]) -> bytes:
    return json.dumps(values, sort_keys=True, separators=(",", ":")).encode()


def _aad(capsule_id: str, credential_id: str, profile_id: str, generation: int) -> bytes:
    return json.dumps(
        [
            "shimpz-driver-credential",
            DRIVER.id,
            CREDENTIAL_SCHEMA.schema_version,
            capsule_id,
            credential_id,
            profile_id,
            generation,
        ],
        separators=(",", ":"),
    ).encode()


def _empty_state() -> dict[str, object]:
    return {
        "version": STATE_VERSION,
        "driver_id": DRIVER.id,
        "schema_version": CREDENTIAL_SCHEMA.schema_version,
        "capsules": {},
    }


def _ensure_private_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
        metadata = path.lstat()
    except OSError as exc:
        raise CredentialStoreError("credential storage directory is unavailable") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise CredentialStoreError("credential storage directory is not owner-only")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise CredentialStoreError("credential storage directory could not be committed") from exc


def _read_private_file(path: Path, maximum: int, description: str) -> bytes | None:
    _ensure_private_directory(path.parent)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CredentialStoreError(f"{description} could not be opened") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_size > maximum
        ):
            raise CredentialStoreError(f"{description} has unsafe filesystem metadata")
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
            raise CredentialStoreError(f"{description} changed while it was read")
    except OSError as exc:
        raise CredentialStoreError(f"{description} could not be read") from exc
    finally:
        os.close(descriptor)
    return payload


def _atomic_write(path: Path, payload: bytes, description: str) -> None:
    _ensure_private_directory(path.parent)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
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
                raise OSError("short private state write")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        temporary.replace(path)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise CredentialStoreError(f"{description} could not be committed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)


def _decode_part(value: object, expected_length: int | None = None) -> bytes:
    if not isinstance(value, str):
        raise CredentialStoreError("credential envelope is malformed")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise CredentialStoreError("credential envelope is malformed") from exc
    if expected_length is not None and len(decoded) != expected_length:
        raise CredentialStoreError("credential envelope is malformed")
    return decoded


def _validate_record(capsule_id: str, credential_id: str, value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _RECORD_KEYS:
        raise CredentialStoreError("credential state contains a malformed record")
    profile_id = value.get("profile_id")
    _profile(profile_id)
    _validate_label(value.get("label"))
    generation = value.get("generation")
    if type(generation) is not int or generation < 1:
        raise CredentialStoreError("credential state contains an invalid generation")
    status = value.get("status")
    if status not in {"active", "revoked"}:
        raise CredentialStoreError("credential state contains an invalid status")
    for field in ("created_at", "updated_at"):
        timestamp = value.get(field)
        if not isinstance(timestamp, str) or _TIMESTAMP_RE.fullmatch(timestamp) is None:
            raise CredentialStoreError("credential state contains an invalid timestamp")
    for field in ("idempotency_hash", "create_fingerprint"):
        digest = value.get(field)
        if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
            raise CredentialStoreError("credential state contains invalid idempotency metadata")

    envelope = value.get("envelope")
    if status == "revoked":
        if envelope is not None:
            raise CredentialStoreError("revoked credential unexpectedly retains an envelope")
    elif not isinstance(envelope, dict) or set(envelope) != _ENVELOPE_KEYS:
        raise CredentialStoreError("active credential has a malformed envelope")
    else:
        if envelope.get("algorithm") != "AES-256-GCM":
            raise CredentialStoreError("credential envelope algorithm is unsupported")
        _decode_part(envelope.get("nonce"), 12)
        if len(_decode_part(envelope.get("ciphertext"))) < 16:
            raise CredentialStoreError("credential envelope is malformed")
    _validate_capsule_id(capsule_id)
    _validate_credential_id(credential_id)
    return value


def _validate_state(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _ROOT_KEYS:
        raise CredentialStoreError("credential state has an invalid root")
    if (
        value.get("version") != STATE_VERSION
        or value.get("driver_id") != DRIVER.id
        or value.get("schema_version") != CREDENTIAL_SCHEMA.schema_version
    ):
        raise CredentialStoreError("credential state version or driver identity is invalid")
    capsules = value.get("capsules")
    if not isinstance(capsules, dict):
        raise CredentialStoreError("credential state capsules must be an object")
    count = 0
    for capsule_id, records in capsules.items():
        _validate_capsule_id(capsule_id)
        if not isinstance(records, dict) or not records:
            raise CredentialStoreError("credential state contains an empty or malformed Capsule set")
        for credential_id, record in records.items():
            _validate_record(capsule_id, credential_id, record)
            count += 1
    if count > MAX_CREDENTIALS:
        raise CredentialStoreError("credential state exceeds its fixed record limit")
    return value


def _metadata(capsule_id: str, credential_id: str, record: Mapping[str, object]) -> CredentialMetadata:
    return CredentialMetadata(
        capsule_id=capsule_id,
        credential_id=credential_id,
        profile_id=str(record["profile_id"]),
        label=str(record["label"]),
        generation=int(record["generation"]),
        status=str(record["status"]),
        created_at=str(record["created_at"]),
        updated_at=str(record["updated_at"]),
    )


class CredentialStore:
    """One in-process, thread-safe encrypted registry with durable compare-and-swap updates."""

    def __init__(self, state_path: Path = STATE_PATH, key_path: Path = KEY_PATH) -> None:
        self.state_path = Path(state_path)
        self.key_path = Path(key_path)
        if not self.state_path.is_absolute() or not self.key_path.is_absolute():
            raise CredentialStoreError("credential state and key paths must be absolute")
        if self.state_path.parent == self.key_path.parent:
            raise CredentialStoreError("credential keyring must be separate from encrypted state")
        self._lock = threading.RLock()

    def _read_state(self) -> dict[str, object]:
        payload = _read_private_file(self.state_path, MAX_STATE_BYTES, "credential state")
        if payload is None:
            return _empty_state()
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CredentialStoreError("credential state is not valid JSON") from exc
        return _validate_state(value)

    def _write_state(self, state: Mapping[str, object]) -> None:
        validated = _validate_state(dict(state))
        payload = json.dumps(validated, sort_keys=True, separators=(",", ":")).encode()
        if len(payload) > MAX_STATE_BYTES:
            raise CredentialStoreError("credential state exceeds its fixed byte limit")
        _atomic_write(self.state_path, payload, "credential state")

    def _key(self, *, allow_create: bool = False) -> bytes:
        payload = _read_private_file(self.key_path, 32, "credential keyring")
        if payload is None:
            if not allow_create:
                raise CredentialStoreError("credential keyring is unavailable")
            payload = AESGCM.generate_key(bit_length=256)
            _atomic_write(self.key_path, payload, "credential keyring")
        if len(payload) != 32:
            raise CredentialStoreError("credential keyring is not an AES-256 key")
        return payload

    def _encrypt(
        self,
        values: Mapping[str, str],
        capsule_id: str,
        credential_id: str,
        profile_id: str,
        generation: int,
        key: bytes | None = None,
    ) -> dict[str, str]:
        nonce = os.urandom(12)
        ciphertext = AESGCM(key if key is not None else self._key()).encrypt(
            nonce,
            _canonical(values),
            _aad(capsule_id, credential_id, profile_id, generation),
        )
        return {
            "algorithm": "AES-256-GCM",
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        }

    def _create_fingerprint(
        self,
        capsule_id: str,
        credential_id: str,
        profile_id: str,
        label: str,
        values: Mapping[str, str],
        key: bytes,
    ) -> str:
        request = json.dumps(
            [capsule_id, credential_id, profile_id, label, values],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        fingerprint_key = hmac.new(
            key,
            b"shimpz-r2-create-fingerprint-key-v1",
            hashlib.sha256,
        ).digest()
        return hmac.new(fingerprint_key, request, hashlib.sha256).hexdigest()

    @staticmethod
    def _idempotency_hash(capsule_id: str, idempotency_key: str, key: bytes) -> str:
        digest_key = hmac.new(
            key,
            b"shimpz-r2-idempotency-hash-key-v1",
            hashlib.sha256,
        ).digest()
        return hmac.new(
            digest_key,
            json.dumps([capsule_id, idempotency_key], separators=(",", ":")).encode(),
            hashlib.sha256,
        ).hexdigest()

    def credential_id(self, capsule_id: object, idempotency_key: object) -> str:
        """Derive a stable opaque identifier without exposing the caller's idempotency key."""
        capsule = _validate_capsule_id(capsule_id)
        idempotency = _validate_idempotency_key(idempotency_key)
        with self._lock:
            key = self._key(allow_create=not self.state_path.exists())
            identifier_key = hmac.new(
                key,
                b"shimpz-r2-credential-id-key-v1",
                hashlib.sha256,
            ).digest()
            digest = hmac.new(
                identifier_key,
                f"{capsule}\0{idempotency}".encode(),
                hashlib.sha256,
            ).hexdigest()
        return f"r2-{digest[:48]}"

    def preflight_create(
        self,
        capsule_id: object,
        credential_id: object,
        profile_id: object,
        label: object,
        values: object,
        idempotency_key: object,
    ) -> CredentialMetadata | None:
        """Return an exact prior create, reject conflicts, or admit a new provider probe."""
        capsule = _validate_capsule_id(capsule_id)
        credential = _validate_credential_id(credential_id)
        selected_profile = _profile(profile_id).id
        selected_label = _validate_label(label)
        bundle = validate_bundle(selected_profile, values)
        idempotency = _validate_idempotency_key(idempotency_key)
        with self._lock:
            state = self._read_state()
            key = self._key() if self.state_path.exists() else self._key(allow_create=True)
            idempotency_hash = self._idempotency_hash(capsule, idempotency, key)
            capsules = state["capsules"]
            if not isinstance(capsules, dict):
                raise CredentialStoreError("credential state capsules are malformed")
            records = capsules.get(capsule)
            if records is not None and not isinstance(records, dict):
                raise CredentialStoreError("credential state Capsule set is malformed")
            if isinstance(records, dict):
                for existing_id, existing_record in records.items():
                    if not isinstance(existing_record, dict):
                        raise CredentialStoreError("credential state record is malformed")
                    if existing_record.get("idempotency_hash") == idempotency_hash and existing_id != credential:
                        raise CredentialConflictError("idempotency key is already bound to another credential")
            if not isinstance(records, dict) or credential not in records:
                return None
            existing = self._record(state, capsule, credential)
            fingerprint = self._create_fingerprint(
                capsule,
                credential,
                selected_profile,
                selected_label,
                bundle,
                key,
            )
            if secrets.compare_digest(
                str(existing.get("idempotency_hash")), idempotency_hash
            ) and secrets.compare_digest(str(existing.get("create_fingerprint")), fingerprint):
                return _metadata(capsule, credential, existing)
            raise CredentialConflictError("credential identity or idempotency request conflicts")

    def preflight_rotate(
        self,
        capsule_id: object,
        credential_id: object,
        expected_generation: object,
        profile_id: object,
        label: object,
        values: object,
    ) -> None:
        """Validate identity, CAS, profile, label, and bundle before a provider network probe."""
        capsule = _validate_capsule_id(capsule_id)
        credential = _validate_credential_id(credential_id)
        selected_profile = _profile(profile_id).id
        _validate_label(label)
        validate_bundle(selected_profile, values)
        with self._lock:
            state = self._read_state()
            record = self._record(state, capsule, credential)
            if record.get("status") != "active":
                raise CredentialRevokedError("credential is revoked")
            self._expected_generation(record, expected_generation)
            if selected_profile != record.get("profile_id"):
                raise CredentialConflictError("credential profile cannot change during rotation")

    def _decrypt(
        self,
        record: Mapping[str, object],
        capsule_id: str,
        credential_id: str,
        key: bytes | None = None,
    ) -> dict[str, str]:
        envelope = record.get("envelope")
        if not isinstance(envelope, dict):
            raise CredentialRevokedError("credential is revoked")
        nonce = _decode_part(envelope.get("nonce"), 12)
        ciphertext = _decode_part(envelope.get("ciphertext"))
        try:
            plaintext = AESGCM(key if key is not None else self._key()).decrypt(
                nonce,
                ciphertext,
                _aad(
                    capsule_id,
                    credential_id,
                    str(record["profile_id"]),
                    int(record["generation"]),
                ),
            )
        except InvalidTag as exc:
            raise CredentialStoreError("credential envelope authentication failed") from exc
        try:
            values = json.loads(plaintext)
            return validate_bundle(record["profile_id"], values)
        except (UnicodeDecodeError, json.JSONDecodeError, CredentialValidationError) as exc:
            raise CredentialStoreError("decrypted credential bundle is malformed") from exc

    @staticmethod
    def _record(state: Mapping[str, object], capsule_id: str, credential_id: str) -> dict[str, object]:
        capsules = state["capsules"]
        if not isinstance(capsules, dict):
            raise CredentialStoreError("credential state capsules are malformed")
        records = capsules.get(capsule_id)
        if not isinstance(records, dict) or credential_id not in records:
            raise CredentialNotFoundError("credential does not exist in this Capsule")
        record = records[credential_id]
        if not isinstance(record, dict):
            raise CredentialStoreError("credential state record is malformed")
        return record

    @staticmethod
    def _expected_generation(record: Mapping[str, object], expected_generation: object) -> int:
        if type(expected_generation) is not int or expected_generation < 1:
            raise CredentialValidationError("expected_generation must be a positive integer")
        if record.get("generation") != expected_generation:
            raise CredentialConflictError("credential generation changed")
        return expected_generation

    def create(
        self,
        capsule_id: object,
        credential_id: object,
        profile_id: object,
        label: object,
        values: object,
        idempotency_key: object,
    ) -> CredentialMetadata:
        capsule = _validate_capsule_id(capsule_id)
        credential = _validate_credential_id(credential_id)
        selected_profile = _profile(profile_id).id
        selected_label = _validate_label(label)
        bundle = validate_bundle(selected_profile, values)
        idempotency = _validate_idempotency_key(idempotency_key)
        with self._lock:
            state = self._read_state()
            key = self._key(allow_create=not self.state_path.exists())
            idempotency_hash = self._idempotency_hash(capsule, idempotency, key)
            fingerprint = self._create_fingerprint(
                capsule,
                credential,
                selected_profile,
                selected_label,
                bundle,
                key,
            )
            capsules = state["capsules"]
            if not isinstance(capsules, dict):
                raise CredentialStoreError("credential state capsules are malformed")
            records = capsules.get(capsule)
            if records is not None and not isinstance(records, dict):
                raise CredentialStoreError("credential state Capsule set is malformed")
            if isinstance(records, dict):
                for existing_id, existing_record in records.items():
                    if not isinstance(existing_record, dict):
                        raise CredentialStoreError("credential state record is malformed")
                    if existing_record.get("idempotency_hash") == idempotency_hash and existing_id != credential:
                        raise CredentialConflictError("idempotency key is already bound to another credential")
            if isinstance(records, dict) and credential in records:
                existing = self._record(state, capsule, credential)
                same_idempotency_key = secrets.compare_digest(
                    str(existing.get("idempotency_hash")),
                    idempotency_hash,
                )
                same_request = secrets.compare_digest(
                    str(existing.get("create_fingerprint")),
                    fingerprint,
                )
                if same_idempotency_key and same_request:
                    return _metadata(capsule, credential, existing)
                raise CredentialConflictError("credential identity or idempotency request conflicts")
            count = sum(len(value) for value in capsules.values() if isinstance(value, dict))
            if count >= MAX_CREDENTIALS:
                raise CredentialConflictError("credential capacity is exhausted")
            now = _timestamp()
            record: dict[str, object] = {
                "profile_id": selected_profile,
                "label": selected_label,
                "generation": 1,
                "status": "active",
                "created_at": now,
                "updated_at": now,
                "idempotency_hash": idempotency_hash,
                "create_fingerprint": fingerprint,
                "envelope": self._encrypt(bundle, capsule, credential, selected_profile, 1, key),
            }
            if records is None:
                records = {}
                capsules[capsule] = records
            records[credential] = record
            self._write_state(state)
            return _metadata(capsule, credential, record)

    def list_metadata(self, capsule_id: object) -> tuple[CredentialMetadata, ...]:
        capsule = _validate_capsule_id(capsule_id)
        with self._lock:
            state = self._read_state()
            capsules = state["capsules"]
            if not isinstance(capsules, dict):
                raise CredentialStoreError("credential state capsules are malformed")
            records = capsules.get(capsule, {})
            if not isinstance(records, dict):
                raise CredentialStoreError("credential state Capsule set is malformed")
            return tuple(
                _metadata(capsule, credential_id, records[credential_id])
                for credential_id in sorted(records)
                if records[credential_id].get("status") == "active"
            )

    def capsule_record_count(self, capsule_id: object) -> int:
        """Count active records and tombstones toward the bounded per-Capsule inventory."""
        capsule = _validate_capsule_id(capsule_id)
        with self._lock:
            state = self._read_state()
            capsules = state["capsules"]
            if not isinstance(capsules, dict):
                raise CredentialStoreError("credential state capsules are malformed")
            records = capsules.get(capsule, {})
            if not isinstance(records, dict):
                raise CredentialStoreError("credential state Capsule set is malformed")
            return len(records)

    def check_health(self) -> None:
        """Authenticate all active envelopes and fail if non-empty state lost its keyring."""
        with self._lock:
            state = self._read_state()
            capsules = state["capsules"]
            if not isinstance(capsules, dict):
                raise CredentialStoreError("credential state capsules are malformed")
            records = [
                (capsule_id, credential_id, record)
                for capsule_id, capsule_records in capsules.items()
                for credential_id, record in capsule_records.items()
            ]
            if not records and not self.state_path.exists():
                return
            key = self._key()
            for capsule_id, credential_id, record in records:
                if record.get("status") == "active":
                    self._decrypt(record, capsule_id, credential_id, key)

    def resolve(self, capsule_id: object, credential_id: object) -> ResolvedCredential:
        capsule = _validate_capsule_id(capsule_id)
        credential = _validate_credential_id(credential_id)
        with self._lock:
            state = self._read_state()
            record = self._record(state, capsule, credential)
            if record.get("status") != "active":
                raise CredentialRevokedError("credential is revoked")
            return ResolvedCredential(
                _metadata(capsule, credential, record),
                self._decrypt(record, capsule, credential),
            )

    def rotate(
        self,
        capsule_id: object,
        credential_id: object,
        expected_generation: object,
        profile_id: object,
        label: object,
        values: object,
    ) -> CredentialMetadata:
        capsule = _validate_capsule_id(capsule_id)
        credential = _validate_credential_id(credential_id)
        with self._lock:
            state = self._read_state()
            record = self._record(state, capsule, credential)
            if record.get("status") != "active":
                raise CredentialRevokedError("credential is revoked")
            generation = self._expected_generation(record, expected_generation) + 1
            selected_profile = _profile(profile_id).id
            if selected_profile != record.get("profile_id"):
                raise CredentialConflictError("credential profile cannot change during rotation")
            selected_label = _validate_label(label)
            bundle = validate_bundle(selected_profile, values)
            record["generation"] = generation
            record["label"] = selected_label
            record["updated_at"] = _timestamp()
            record["envelope"] = self._encrypt(
                bundle,
                capsule,
                credential,
                str(record["profile_id"]),
                generation,
            )
            self._write_state(state)
            return _metadata(capsule, credential, record)

    def remove(
        self,
        capsule_id: object,
        credential_id: object,
        expected_generation: object,
    ) -> CredentialMetadata:
        """Destroy the envelope with CAS and retain an idempotent, non-public tombstone."""
        capsule = _validate_capsule_id(capsule_id)
        credential = _validate_credential_id(credential_id)
        with self._lock:
            state = self._read_state()
            record = self._record(state, capsule, credential)
            if type(expected_generation) is not int or expected_generation < 1:
                raise CredentialValidationError("expected_generation must be a positive integer")
            if record.get("status") == "revoked" and record.get("generation") in {
                expected_generation,
                expected_generation + 1,
            }:
                return _metadata(capsule, credential, record)
            self._expected_generation(record, expected_generation)
            record["generation"] = expected_generation + 1
            record["status"] = "revoked"
            record["updated_at"] = _timestamp()
            record["envelope"] = None
            self._write_state(state)
            return _metadata(capsule, credential, record)

    def revoke(
        self,
        capsule_id: object,
        credential_id: object,
        expected_generation: object,
    ) -> CredentialMetadata:
        return self.remove(capsule_id, credential_id, expected_generation)

    def revoke_capsule(self, capsule_id: object) -> None:
        """Atomically destroy every active envelope for one Capsule, retry-safe."""
        capsule = _validate_capsule_id(capsule_id)
        with self._lock:
            state = self._read_state()
            capsules = state["capsules"]
            if not isinstance(capsules, dict):
                raise CredentialStoreError("credential state Capsules are malformed")
            records = capsules.get(capsule)
            if records is None:
                return
            if not isinstance(records, dict):
                raise CredentialStoreError("credential state Capsule set is malformed")
            changed = False
            now = _timestamp()
            for record in records.values():
                if not isinstance(record, dict):
                    raise CredentialStoreError("credential state record is malformed")
                if record.get("status") == "active":
                    record["generation"] = int(record["generation"]) + 1
                    record["status"] = "revoked"
                    record["updated_at"] = now
                    record["envelope"] = None
                    changed = True
            if changed:
                self._write_state(state)

    def purge_revoked(self, capsule_id: object, credential_id: object, expected_generation: object) -> None:
        """Physically remove one exact tombstone during an authorized teardown."""
        capsule = _validate_capsule_id(capsule_id)
        credential = _validate_credential_id(credential_id)
        with self._lock:
            state = self._read_state()
            record = self._record(state, capsule, credential)
            self._expected_generation(record, expected_generation)
            if record.get("status") != "revoked" or record.get("envelope") is not None:
                raise CredentialConflictError("only a revoked credential can be purged")
            capsules = state["capsules"]
            if not isinstance(capsules, dict) or not isinstance(capsules.get(capsule), dict):
                raise CredentialStoreError("credential state Capsule set is malformed")
            del capsules[capsule][credential]
            if not capsules[capsule]:
                del capsules[capsule]
            self._write_state(state)

    def purge_capsule(self, capsule_id: object) -> None:
        """Remove all tombstones only after every Capsule credential has been revoked."""
        capsule = _validate_capsule_id(capsule_id)
        with self._lock:
            state = self._read_state()
            capsules = state["capsules"]
            if not isinstance(capsules, dict):
                raise CredentialStoreError("credential state Capsules are malformed")
            records = capsules.get(capsule)
            if records is None:
                return
            if not isinstance(records, dict) or any(
                record.get("status") != "revoked" or record.get("envelope") is not None for record in records.values()
            ):
                raise CredentialConflictError("Capsule still has active credentials")
            del capsules[capsule]
            self._write_state(state)


STORE = CredentialStore()
