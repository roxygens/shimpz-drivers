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

import oauth_providers

MANIFEST_PATH = "/opt/shimpz-assistant/shimpz.toml"
MAX_MANIFEST_BYTES = 256 * 1024
MAX_ARCHIVE_BYTES = MAX_MANIFEST_BYTES + (32 * 1024)
MAX_ALLOWED_HOSTS = 32
MAX_ACCOUNTS = 16
MAX_IDENTIFIER_LENGTH = 80
MAX_SECRET_ID_LENGTH = 64
DEFAULT_CACHE_ENTRIES = 256
_ID_RE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_CREATOR_RE = re.compile(r"@[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?\Z")
_GITHUB_RE = re.compile(r"https://github\.com/[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}[A-Za-z0-9])?/[A-Za-z0-9_.-]{1,100}\Z")
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
class AccountDeclaration:
    """Public provider intent for one controller-owned account."""

    id: str
    provider: str
    scopes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ManifestContract:
    """Canonical security intent admitted from one immutable Assistant package."""

    allowed_hosts: tuple[str, ...]
    accounts: tuple[AccountDeclaration, ...]


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


def canonical_account_declarations(value: object) -> tuple[AccountDeclaration, ...]:
    """Canonicalize accounts whose id is a controller-reviewed provider id."""
    if not isinstance(value, Mapping) or len(value) > MAX_ACCOUNTS:
        raise ManifestError("Assistant account declarations are invalid")
    declarations: list[AccountDeclaration] = []
    for account_id, scopes in value.items():
        identifier = _identifier(account_id, kind="account", maximum=MAX_SECRET_ID_LENGTH)
        try:
            intent = oauth_providers.account_intent(identifier, scopes)
        except oauth_providers.OAuthProviderError as exc:
            raise ManifestError("Assistant account declaration is invalid") from exc
        declarations.append(
            AccountDeclaration(
                identifier,
                intent.provider.id,
                intent.scopes,
            )
        )
    return tuple(sorted(declarations))


def canonical_manifest_contract(
    *,
    allowed_hosts: object,
    account_declarations: object | None = None,
) -> ManifestContract:
    """Build one deterministic contract for package and reviewed registry comparison."""
    accounts = canonical_account_declarations({} if account_declarations is None else account_declarations)
    return ManifestContract(
        allowed_hosts=canonical_allowed_hosts(allowed_hosts),
        accounts=accounts,
    )


def reviewed_manifest_contract(
    *,
    allowed_hosts: object,
    accounts: object | None = None,
) -> ManifestContract:
    """Normalize the controller-owned registry dataclasses without trusting package input."""
    if not isinstance(accounts, Mapping):
        raise ManifestError("Assistant reviewed manifest contract is invalid")
    try:
        account_declarations = {account_id: metadata.scopes for account_id, metadata in accounts.items()}
        if any(account_id != metadata.provider for account_id, metadata in accounts.items()):
            raise ManifestError("Assistant reviewed account provider does not match its id")
    except AttributeError as exc:
        raise ManifestError("Assistant reviewed manifest contract is invalid") from exc
    return canonical_manifest_contract(
        allowed_hosts=allowed_hosts,
        account_declarations=account_declarations,
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
                lowered = key.lower()
                if any(
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
    required = {"name", "summary", "creators", "github", "allowed_hosts"}
    if not required <= set(manifest) or set(manifest) - (required | {"accounts"}):
        raise ManifestError("Assistant manifest contains an unsupported top-level field")
    _reject_credential_material(manifest)
    return manifest


def parse_manifest_contract(raw: bytes) -> ManifestContract:
    """Parse the bounded public security contract from one UTF-8 TOML manifest."""
    manifest = _manifest_table(raw)
    _public_text(manifest["name"], kind="name", maximum=80)
    _public_text(manifest["summary"], kind="summary", maximum=160)
    creators = manifest["creators"]
    if (
        not isinstance(creators, list)
        or not 1 <= len(creators) <= 16
        or any(not isinstance(creator, str) or _CREATOR_RE.fullmatch(creator) is None for creator in creators)
        or len(creators) != len(set(creators))
    ):
        raise ManifestError("Assistant creators are invalid")
    github = manifest["github"]
    if not isinstance(github, str) or _GITHUB_RE.fullmatch(github) is None:
        raise ManifestError("Assistant github repository is invalid")

    raw_accounts = manifest.get("accounts", {})
    if not isinstance(raw_accounts, dict):
        raise ManifestError("Assistant account declarations are invalid")
    account_declarations: dict[str, object] = {}
    for account_id, metadata in raw_accounts.items():
        if not isinstance(metadata, dict) or set(metadata) != {"scopes"}:
            raise ManifestError("Assistant account declaration is invalid")
        account_declarations[account_id] = metadata["scopes"]

    return canonical_manifest_contract(
        allowed_hosts=manifest["allowed_hosts"],
        account_declarations=account_declarations,
    )


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


def _read_container_manifest_bytes(container) -> bytes:
    """Read the fixed regular manifest file from a digest-bound root."""
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
        name != "shimpz.toml"
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
                or members[0].name not in {"shimpz.toml", "./shimpz.toml"}
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
    return raw


def read_container_manifest_contract(container) -> ManifestContract:
    """Read the fixed regular manifest contract from a digest-bound root."""
    return parse_manifest_contract(_read_container_manifest_bytes(container))


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
