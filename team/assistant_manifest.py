"""Immutable Assistant manifest admission for reviewed security intent."""

from __future__ import annotations

import io
import ipaddress
import json
import re
import tarfile
import threading
import tomllib
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import oauth_providers
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

MANIFEST_PATH = "/opt/shimpz/shimpz.toml"
CONTRACT_PATH = "/opt/shimpz/shimpz.contract.json"
CATALOG_PATH = Path(__file__).with_name("assistant_catalog.json")
MAX_MANIFEST_BYTES = 256 * 1024
MAX_CONTRACT_BYTES = 512 * 1024
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


@dataclass(frozen=True, slots=True)
class ReviewedAssistant:
    """Controller-reviewed metadata and machine Power contract."""

    assistant_id: str
    name: str
    summary: str
    rpc_command: str
    health_path: str
    allowed_hosts: tuple[str, ...]
    accounts: tuple[AccountDeclaration, ...]
    powers: dict[str, dict[str, Any]]
    power_validators: dict[str, dict[str, Draft202012Validator]]
    machine_contract: dict[str, Any]


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


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _strict_json(raw: bytes, *, maximum: int, kind: str) -> object:
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= maximum:
        raise ManifestError(f"Assistant {kind} has an invalid size")
    try:
        return json.loads(raw, parse_constant=_reject_json_constant, object_pairs_hook=_unique_json_object)
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Assistant {kind} is invalid JSON") from exc


def _machine_schema(value: object, *, kind: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("type") != "object":
        raise ManifestError(f"Assistant Power {kind} schema must describe an object")
    try:
        Draft202012Validator.check_schema(value)
    except SchemaError as exc:
        raise ManifestError(f"Assistant Power {kind} schema is invalid") from exc
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(encoded) > 128 * 1024:
        raise ManifestError(f"Assistant Power {kind} schema is too large")
    pending: list[object] = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            schema_type = item.get("type")
            permits_object = schema_type == "object" or (isinstance(schema_type, list) and "object" in schema_type)
            if permits_object and item.get("additionalProperties") is not False:
                raise ManifestError(f"Assistant Power {kind} schema must close every object")
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return value


def canonical_machine_contract(value: object, declared_accounts: tuple[AccountDeclaration, ...]) -> dict[str, Any]:
    """Validate and canonicalize an untrusted SDK-generated Power contract."""
    if not isinstance(value, dict) or set(value) != {"version", "powers"} or value["version"] != 1:
        raise ManifestError("Assistant machine contract has an unsupported shape")
    raw_powers = value["powers"]
    if not isinstance(raw_powers, list) or not 1 <= len(raw_powers) <= 128:
        raise ManifestError("Assistant machine contract Powers are invalid")
    declared_ids = {account.id for account in declared_accounts}
    used_accounts: set[str] = set()
    powers: list[dict[str, Any]] = []
    ids: set[str] = set()
    for raw_power in raw_powers:
        if not isinstance(raw_power, dict) or set(raw_power) != {
            "id",
            "method",
            "path",
            "input_schema",
            "output_schema",
            "accounts",
        }:
            raise ManifestError("Assistant machine contract Power is invalid")
        power_id = _identifier(raw_power["id"], kind="Power")
        if power_id in ids or raw_power["method"] != "POST" or raw_power["path"] != f"/v1/powers/{power_id}":
            raise ManifestError("Assistant machine contract Power route is invalid")
        ids.add(power_id)
        accounts = raw_power["accounts"]
        if (
            not isinstance(accounts, list)
            or len(accounts) > 4
            or len(accounts) != len(set(accounts))
            or any(not isinstance(account_id, str) or account_id not in declared_ids for account_id in accounts)
        ):
            raise ManifestError("Assistant machine contract Power accounts are invalid")
        used_accounts.update(accounts)
        powers.append(
            {
                "id": power_id,
                "method": "POST",
                "path": raw_power["path"],
                "input_schema": _machine_schema(raw_power["input_schema"], kind="input"),
                "output_schema": _machine_schema(raw_power["output_schema"], kind="output"),
                "accounts": sorted(accounts),
            }
        )
    if used_accounts != declared_ids:
        raise ManifestError("Assistant machine contract must use every declared account")
    return {"version": 1, "powers": sorted(powers, key=lambda power: power["id"])}


def parse_machine_contract(raw: bytes, declared_accounts: tuple[AccountDeclaration, ...]) -> dict[str, Any]:
    """Parse a bounded SDK artifact without executing Assistant code."""
    return canonical_machine_contract(
        _strict_json(raw, maximum=MAX_CONTRACT_BYTES, kind="machine contract"),
        declared_accounts,
    )


def validate_schema_payload(validator: Draft202012Validator, payload: object) -> dict[str, object]:
    """Validate one untrusted Power input or output against its reviewed schema."""
    if not isinstance(payload, dict):
        raise ValueError("Power payload must be an object")
    try:
        validator.validate(payload)
    except ValidationError as exc:
        raise ValueError("Power payload does not match its reviewed schema") from exc
    return payload


def load_reviewed_catalog(path: Path = CATALOG_PATH) -> dict[str, ReviewedAssistant]:
    """Load the build-baked catalog of reviewed SDK machine contracts."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ManifestError("Assistant reviewed catalog is unavailable") from exc
    catalog = _strict_json(raw, maximum=MAX_CONTRACT_BYTES * 2, kind="reviewed catalog")
    if not isinstance(catalog, dict) or set(catalog) != {"version", "assistants"} or catalog["version"] != 1:
        raise ManifestError("Assistant reviewed catalog has an unsupported shape")
    assistants = catalog["assistants"]
    if not isinstance(assistants, dict) or not assistants or len(assistants) > 32:
        raise ManifestError("Assistant reviewed catalog is invalid")
    reviewed: dict[str, ReviewedAssistant] = {}
    for raw_id, metadata in assistants.items():
        assistant_id = _identifier(raw_id, kind="id")
        if not isinstance(metadata, dict) or set(metadata) != {
            "name",
            "summary",
            "rpc_command",
            "health_path",
            "allowed_hosts",
            "accounts",
            "contract",
        }:
            raise ManifestError("Assistant reviewed catalog entry is invalid")
        name = _public_text(metadata["name"], kind="name", maximum=80)
        summary = _public_text(metadata["summary"], kind="summary", maximum=160)
        rpc_command = metadata["rpc_command"]
        health_path = metadata["health_path"]
        if (
            not isinstance(rpc_command, str)
            or re.fullmatch(r"/usr/local/bin/[a-z0-9-]{1,80}", rpc_command) is None
            or not isinstance(health_path, str)
            or re.fullmatch(r"/[a-z0-9/-]{1,80}", health_path) is None
        ):
            raise ManifestError("Assistant reviewed catalog runtime is invalid")
        raw_accounts = metadata["accounts"]
        if not isinstance(raw_accounts, dict):
            raise ManifestError("Assistant reviewed catalog accounts are invalid")
        account_scopes: dict[str, object] = {}
        for account_id, account in raw_accounts.items():
            if not isinstance(account, dict) or set(account) != {"scopes"}:
                raise ManifestError("Assistant reviewed catalog account is invalid")
            account_scopes[account_id] = account["scopes"]
        accounts = canonical_account_declarations(account_scopes)
        machine_contract = canonical_machine_contract(metadata["contract"], accounts)
        reviewed[assistant_id] = ReviewedAssistant(
            assistant_id=assistant_id,
            name=name,
            summary=summary,
            rpc_command=rpc_command,
            health_path=health_path,
            allowed_hosts=canonical_allowed_hosts(metadata["allowed_hosts"]),
            accounts=accounts,
            powers={power["id"]: power for power in machine_contract["powers"]},
            power_validators={
                power["id"]: {
                    "input": Draft202012Validator(power["input_schema"]),
                    "output": Draft202012Validator(power["output_schema"]),
                }
                for power in machine_contract["powers"]
            },
            machine_contract=machine_contract,
        )
    return reviewed


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


def _bounded_archive(chunks: Iterable[bytes], maximum: int = MAX_ARCHIVE_BYTES) -> bytes:
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
                if len(archive) > maximum:
                    raise ManifestError("Assistant package archive is too large")
    except ManifestError:
        raise
    except Exception as exc:
        raise ManifestError("Assistant manifest archive is unavailable") from exc
    return bytes(archive)


def _read_container_file(container, *, path: str, name: str, maximum: int) -> bytes:
    """Read one fixed immutable regular file from a digest-bound root."""
    try:
        chunks, metadata = container.get_archive(path)
    except Exception as exc:
        raise ManifestError("Assistant package file is unavailable") from exc
    if not isinstance(metadata, dict):
        raise ManifestError("Assistant package metadata is invalid")
    size = metadata.get("size")
    mode = metadata.get("mode")
    metadata_name = metadata.get("name")
    if (
        metadata_name != name
        or not isinstance(size, int)
        or isinstance(size, bool)
        or not 1 <= size <= maximum
        or not isinstance(mode, int)
        or isinstance(mode, bool)
        or mode != 0o444
    ):
        raise ManifestError("Assistant package metadata is invalid")

    archive = _bounded_archive(chunks, maximum + (32 * 1024))
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            members = bundle.getmembers()
            if (
                len(members) != 1
                or members[0].name not in {name, f"./{name}"}
                or not members[0].isreg()
                or members[0].size != size
                or members[0].mode & 0o777 != 0o444
            ):
                raise ManifestError("Assistant package archive is invalid")
            extracted = bundle.extractfile(members[0])
            if extracted is None:
                raise ManifestError("Assistant package archive is invalid")
            raw = extracted.read(maximum + 1)
    except ManifestError:
        raise
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ManifestError("Assistant package archive is invalid") from exc
    if len(raw) != size:
        raise ManifestError("Assistant package archive is invalid")
    return raw


def _read_container_manifest_bytes(container) -> bytes:
    return _read_container_file(
        container,
        path=MANIFEST_PATH,
        name="shimpz.toml",
        maximum=MAX_MANIFEST_BYTES,
    )


def read_container_manifest_contract(container) -> ManifestContract:
    """Read the fixed regular manifest contract from a digest-bound root."""
    return parse_manifest_contract(_read_container_manifest_bytes(container))


def read_container_machine_contract(
    container,
    declared_accounts: tuple[AccountDeclaration, ...],
) -> dict[str, Any]:
    """Read and validate the fixed SDK contract artifact from an immutable image."""
    raw = _read_container_file(
        container,
        path=CONTRACT_PATH,
        name="shimpz.contract.json",
        maximum=MAX_CONTRACT_BYTES,
    )
    return parse_machine_contract(raw, declared_accounts)


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


class MachineContractCache:
    """Admit the image-baked SDK artifact only when it equals controller review."""

    def __init__(self, max_entries: int = DEFAULT_CACHE_ENTRIES) -> None:
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries < 1:
            raise ValueError("Assistant machine contract cache size must be positive")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(
        self,
        container,
        declared_accounts: tuple[AccountDeclaration, ...],
        reviewed: object,
    ) -> dict[str, Any]:
        """Return the machine contract only after exact semantic equality."""
        container_id = getattr(container, "id", None)
        if (
            not isinstance(container_id, str)
            or not container_id
            or len(container_id) > 256
            or any(not character.isalnum() and character not in {"-", "_", "."} for character in container_id)
        ):
            raise ManifestError("Assistant container identity is invalid")
        with self._lock:
            declared = self._entries.get(container_id)
            if declared is None:
                declared = read_container_machine_contract(container, declared_accounts)
                self._entries[container_id] = declared
                while len(self._entries) > self._max_entries:
                    self._entries.popitem(last=False)
            else:
                self._entries.move_to_end(container_id)
        if declared != reviewed:
            raise ManifestError("Assistant machine contract does not match controller review")
        return declared

    def discard(self, container_id: object) -> None:
        if isinstance(container_id, str):
            with self._lock:
                self._entries.pop(container_id, None)
