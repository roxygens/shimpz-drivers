"""Immutable Assistant manifest admission for reviewed security intent."""

from __future__ import annotations

import io
import ipaddress
import re
import tarfile
import threading
import tomllib
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass

MANIFEST_PATH = "/opt/shimpz-assistant/shimpz.assistant.toml"
MAX_MANIFEST_BYTES = 256 * 1024
MAX_ARCHIVE_BYTES = MAX_MANIFEST_BYTES + (32 * 1024)
MAX_ALLOWED_HOSTS = 32
MAX_SECRETS = 32
MAX_POWER_SECRETS = 16
MAX_POWERS = 128
MAX_IDENTIFIER_LENGTH = 80
MAX_SECRET_ID_LENGTH = 64
DEFAULT_CACHE_ENTRIES = 256
_ID_RE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_SECRET_VALUE_RE = re.compile(
    r"(?i)(?:bearer\s+[a-z0-9._~-]{12,}|(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)"
    r"\s*[:=]\s*\S+|(?:sk|ghp|github_pat|glpat|xox[baprs])[-_][a-z0-9_-]{12,})"
)
_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\Z")
_PUBLIC_HOST_RE = re.compile(
    r"(?=.{1,253}\Z)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?\Z"
)
_NON_PUBLIC_HOST_SUFFIXES = (
    ".arpa",
    ".example",
    ".home",
    ".internal",
    ".invalid",
    ".lan",
    ".local",
    ".localdomain",
    ".localhost",
    ".onion",
    ".test",
)


class ManifestError(RuntimeError):
    """An immutable Assistant package did not expose its reviewed security intent."""


@dataclass(frozen=True, slots=True, order=True)
class SecretDeclaration:
    """Public metadata for one secret identifier; never a secret value."""

    id: str
    name: str
    summary: str


@dataclass(frozen=True, slots=True)
class ManifestContract:
    """Canonical security intent admitted from one immutable Assistant package."""

    allowed_hosts: tuple[str, ...]
    secrets: tuple[SecretDeclaration, ...]
    power_secrets: tuple[tuple[str, tuple[str, ...]], ...]


def canonical_allowed_hosts(value: object) -> tuple[str, ...]:
    """Return one deterministic list of exact public DNS host names."""
    if not isinstance(value, list | tuple) or len(value) > MAX_ALLOWED_HOSTS:
        raise ManifestError("Assistant allowed_hosts is invalid")
    hosts: list[str] = []
    for host in value:
        if not isinstance(host, str) or not 1 <= len(host) <= 253 or not host.isascii() or host != host.lower():
            raise ManifestError("Assistant allowed_hosts is invalid")
        if _PUBLIC_HOST_RE.fullmatch(host) is None or host.endswith(_NON_PUBLIC_HOST_SUFFIXES):
            raise ManifestError("Assistant allowed_hosts is invalid")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise ManifestError("Assistant allowed_hosts is invalid")
        hosts.append(host)
    if len(set(hosts)) != len(hosts):
        raise ManifestError("Assistant allowed_hosts is invalid")
    return tuple(sorted(hosts))


def _identifier(value: object, *, kind: str, maximum: int = MAX_IDENTIFIER_LENGTH) -> str:
    if not isinstance(value, str) or len(value) > maximum or _ID_RE.fullmatch(value) is None:
        raise ManifestError(f"Assistant {kind} identifier is invalid")
    return value


def _public_text(value: object, *, kind: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or "\n" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ManifestError(f"Assistant {kind} is invalid")
    if _SECRET_VALUE_RE.search(value) or _JWT_RE.fullmatch(value.strip()):
        raise ManifestError(f"Assistant {kind} resembles credential material")
    return value


def canonical_secret_declarations(value: object) -> tuple[SecretDeclaration, ...]:
    """Canonicalize public secret metadata keyed by stable secret id."""
    if not isinstance(value, Mapping) or len(value) > MAX_SECRETS:
        raise ManifestError("Assistant secret declarations are invalid")
    declarations: list[SecretDeclaration] = []
    for secret_id, metadata in value.items():
        identifier = _identifier(secret_id, kind="secret", maximum=MAX_SECRET_ID_LENGTH)
        if not isinstance(metadata, list | tuple) or len(metadata) != 2:
            raise ManifestError("Assistant secret declaration is invalid")
        declarations.append(
            SecretDeclaration(
                identifier,
                _public_text(metadata[0], kind="secret name", maximum=80),
                _public_text(metadata[1], kind="secret summary", maximum=160),
            )
        )
    return tuple(sorted(declarations))


def canonical_power_secret_refs(
    value: object,
    declared_secrets: tuple[SecretDeclaration, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Canonicalize every Power's exact, public secret-id dependency set."""
    if not isinstance(value, Mapping) or not value or len(value) > MAX_POWERS:
        raise ManifestError("Assistant Power secret references are invalid")
    declared_ids = {secret.id for secret in declared_secrets}
    used: set[str] = set()
    bindings: list[tuple[str, tuple[str, ...]]] = []
    for power_id, refs in value.items():
        identifier = _identifier(power_id, kind="Power")
        if not isinstance(refs, list | tuple) or len(refs) > MAX_POWER_SECRETS:
            raise ManifestError("Assistant Power secret references are invalid")
        normalized = tuple(_identifier(secret_id, kind="secret", maximum=MAX_SECRET_ID_LENGTH) for secret_id in refs)
        if len(normalized) != len(set(normalized)) or not set(normalized) <= declared_ids:
            raise ManifestError("Assistant Power secret references are invalid")
        used.update(normalized)
        bindings.append((identifier, tuple(sorted(normalized))))
    if used != declared_ids:
        raise ManifestError("Assistant secret declarations must each be used by a Power")
    return tuple(sorted(bindings))


def canonical_manifest_contract(
    *,
    allowed_hosts: object,
    secret_declarations: object,
    power_secret_refs: object,
) -> ManifestContract:
    """Build one deterministic contract for package and reviewed registry comparison."""
    secrets = canonical_secret_declarations(secret_declarations)
    return ManifestContract(
        allowed_hosts=canonical_allowed_hosts(allowed_hosts),
        secrets=secrets,
        power_secrets=canonical_power_secret_refs(power_secret_refs, secrets),
    )


def reviewed_manifest_contract(*, allowed_hosts: object, secrets: object, powers: object) -> ManifestContract:
    """Normalize the controller-owned registry dataclasses without trusting package input."""
    if not isinstance(secrets, Mapping) or not isinstance(powers, Mapping):
        raise ManifestError("Assistant reviewed manifest contract is invalid")
    try:
        secret_declarations = {secret_id: (metadata.name, metadata.summary) for secret_id, metadata in secrets.items()}
        power_secret_refs = {power_id: power.secrets for power_id, power in powers.items()}
    except AttributeError as exc:
        raise ManifestError("Assistant reviewed manifest contract is invalid") from exc
    return canonical_manifest_contract(
        allowed_hosts=allowed_hosts,
        secret_declarations=secret_declarations,
        power_secret_refs=power_secret_refs,
    )


def _reject_credential_material(value: object) -> None:
    pending: list[tuple[object, tuple[str, ...], int]] = [(value, (), 0)]
    while pending:
        current, path, depth = pending.pop()
        if depth > 64:
            raise ManifestError("Assistant manifest exceeds the safe nesting limit")
        if isinstance(current, dict):
            for key, child in current.items():
                if not isinstance(key, str):
                    raise ManifestError("Assistant manifest contains an invalid key")
                public_secret_key = (
                    (not path and key == "secrets")
                    or path == ("secrets",)
                    or (len(path) == 2 and path[0] == "powers" and key == "secrets")
                )
                lowered = key.lower()
                if not public_secret_key and any(
                    marker in lowered
                    for marker in ("secret", "password", "token", "api_key", "private_key", "access_key", "env")
                ):
                    raise ManifestError("Assistant manifest contains a forbidden credential field")
                pending.append((child, (*path, key), depth + 1))
        elif isinstance(current, list):
            pending.extend((child, path, depth + 1) for child in current)
        elif isinstance(current, str) and (_SECRET_VALUE_RE.search(current) or _JWT_RE.fullmatch(current.strip())):
            raise ManifestError("Assistant manifest contains credential material")


def _manifest_table(raw: bytes) -> dict[str, object]:
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= MAX_MANIFEST_BYTES:
        raise ManifestError("Assistant manifest has an invalid size")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError("Assistant manifest is not UTF-8") from exc
    if any(not character.isprintable() and character not in {"\n", "\r", "\t"} for character in text):
        raise ManifestError("Assistant manifest contains invalid text")
    try:
        manifest = tomllib.loads(text)
    except (RecursionError, tomllib.TOMLDecodeError) as exc:
        raise ManifestError("Assistant manifest is invalid TOML") from exc
    if not isinstance(manifest, dict):
        raise ManifestError("Assistant manifest is invalid")
    _reject_credential_material(manifest)
    return manifest


def parse_manifest_contract(raw: bytes) -> ManifestContract:
    """Parse the bounded public security contract from one UTF-8 TOML manifest."""
    manifest = _manifest_table(raw)
    if manifest.get("schema_version") != 2 or "allowed_hosts" not in manifest:
        raise ManifestError("Assistant manifest does not declare its security contract")

    raw_secrets = manifest.get("secrets", {})
    if not isinstance(raw_secrets, dict):
        raise ManifestError("Assistant secret declarations are invalid")
    declarations: dict[str, tuple[object, object]] = {}
    for secret_id, metadata in raw_secrets.items():
        if not isinstance(metadata, dict) or set(metadata) != {"name", "summary"}:
            raise ManifestError("Assistant secret declaration is invalid")
        declarations[secret_id] = (metadata["name"], metadata["summary"])

    raw_powers = manifest.get("powers")
    if not isinstance(raw_powers, dict):
        raise ManifestError("Assistant Power secret references are invalid")
    power_refs: dict[str, object] = {}
    for power_id, power in raw_powers.items():
        if not isinstance(power, dict) or set(power) - {"summary", "approval", "secrets"}:
            raise ManifestError("Assistant Power declaration is invalid")
        _public_text(power.get("summary"), kind="Power summary", maximum=160)
        approval = power.get("approval", "never")
        if approval not in {"never", "once", "always"}:
            raise ManifestError("Assistant Power approval is invalid")
        power_refs[power_id] = power.get("secrets", [])

    return canonical_manifest_contract(
        allowed_hosts=manifest["allowed_hosts"],
        secret_declarations=declarations,
        power_secret_refs=power_refs,
    )


def parse_allowed_hosts(raw: bytes) -> tuple[str, ...]:
    """Compatibility projection of the admitted complete manifest contract."""
    return parse_manifest_contract(raw).allowed_hosts


def _bounded_archive(chunks: Iterable[bytes]) -> bytes:
    archive = bytearray()
    try:
        with ExitStack() as cleanup:
            close = getattr(chunks, "close", None)
            if callable(close):
                cleanup.callback(close)
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise ManifestError("Assistant manifest archive is invalid")
                archive.extend(chunk)
                if len(archive) > MAX_ARCHIVE_BYTES:
                    raise ManifestError("Assistant manifest archive is too large")
    except ManifestError:
        raise
    except Exception as exc:
        raise ManifestError("Assistant manifest archive is unavailable") from exc
    return bytes(archive)


def read_container_manifest_contract(container) -> ManifestContract:
    """Read the fixed regular manifest contract from a digest-bound root."""
    try:
        chunks, metadata = container.get_archive(MANIFEST_PATH)
    except Exception as exc:
        raise ManifestError("Assistant manifest is unavailable") from exc
    if not isinstance(metadata, dict):
        raise ManifestError("Assistant manifest metadata is invalid")
    size = metadata.get("size")
    mode = metadata.get("mode")
    name = metadata.get("name")
    if (
        name != "shimpz.assistant.toml"
        or not isinstance(size, int)
        or isinstance(size, bool)
        or not 1 <= size <= MAX_MANIFEST_BYTES
        or not isinstance(mode, int)
        or isinstance(mode, bool)
        or mode != 0o444
    ):
        raise ManifestError("Assistant manifest metadata is invalid")

    archive = _bounded_archive(chunks)
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            members = bundle.getmembers()
            if (
                len(members) != 1
                or members[0].name not in {"shimpz.assistant.toml", "./shimpz.assistant.toml"}
                or not members[0].isreg()
                or members[0].size != size
                or members[0].mode & 0o777 != 0o444
            ):
                raise ManifestError("Assistant manifest archive is invalid")
            extracted = bundle.extractfile(members[0])
            if extracted is None:
                raise ManifestError("Assistant manifest archive is invalid")
            raw = extracted.read(MAX_MANIFEST_BYTES + 1)
    except ManifestError:
        raise
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ManifestError("Assistant manifest archive is invalid") from exc
    if len(raw) != size:
        raise ManifestError("Assistant manifest archive is invalid")
    return parse_manifest_contract(raw)


def read_container_allowed_hosts(container) -> tuple[str, ...]:
    """Compatibility projection of a container's admitted complete manifest contract."""
    return read_container_manifest_contract(container).allowed_hosts


class ManifestContractCache:
    """Admit reviewed security intent once per immutable container generation."""

    def __init__(self, max_entries: int = DEFAULT_CACHE_ENTRIES) -> None:
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries < 1:
            raise ValueError("Assistant manifest cache size must be positive")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, ManifestContract] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, container, reviewed: object) -> ManifestContract:
        """Return declared intent only when it exactly matches controller review."""
        container_id = getattr(container, "id", None)
        if (
            not isinstance(container_id, str)
            or not container_id
            or len(container_id) > 256
            or any(not character.isalnum() and character not in {"-", "_", "."} for character in container_id)
        ):
            raise ManifestError("Assistant container identity is invalid")
        if not isinstance(reviewed, ManifestContract):
            raise ManifestError("Assistant reviewed manifest contract is invalid")
        with self._lock:
            declared = self._entries.get(container_id)
            if declared is None:
                declared = read_container_manifest_contract(container)
                self._entries[container_id] = declared
                while len(self._entries) > self._max_entries:
                    self._entries.popitem(last=False)
            else:
                self._entries.move_to_end(container_id)
        if declared != reviewed:
            raise ManifestError("Assistant manifest does not match its reviewed contract")
        return declared

    def discard(self, container_id: object) -> None:
        if isinstance(container_id, str):
            with self._lock:
                self._entries.pop(container_id, None)
