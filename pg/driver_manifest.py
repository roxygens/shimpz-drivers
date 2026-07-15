"""Strict, dependency-free parser for the Shimpz Driver Spec v1 manifest."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

MANIFEST_PATH = Path(__file__).with_name("shimpz.driver.toml")

_TOP_LEVEL_KEYS = {
    "schema_version",
    "id",
    "title",
    "version",
    "summary",
    "interface",
    "scope",
    "credential_policy",
    "data_plane",
    "port",
    "health_path",
    "metadata_path",
    "capabilities",
}
_CAPABILITY_KEYS = {"operations"}
_SCOPES = {"space"}
_CREDENTIAL_POLICIES = {"none", "managed", "managed-or-byok"}
_DATA_PLANES = {"direct", "brokered"}
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_VERSION_PATTERN = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
_INTERFACE_PATTERN = re.compile(r"^shimpz\.[a-z][a-z0-9-]*/v[1-9][0-9]*$")
_OPERATION_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:\.[a-z][a-z0-9]*)+$")
_PATH_PATTERN = re.compile(r"^/[a-z0-9][a-z0-9/-]*$")


class ManifestError(ValueError):
    """The driver manifest does not satisfy the closed Driver Spec v1 contract."""


@dataclass(frozen=True)
class DriverManifest:
    schema_version: int
    id: str
    title: str
    version: str
    summary: str
    interface: str
    scope: str
    credential_policy: str
    data_plane: str
    port: int
    health_path: str
    metadata_path: str
    operations: tuple[str, ...]

    def public(self) -> dict[str, object]:
        """Return only the non-secret, language-neutral discovery contract."""
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "title": self.title,
            "version": self.version,
            "summary": self.summary,
            "interface": self.interface,
            "scope": self.scope,
            "credential_policy": self.credential_policy,
            "data_plane": self.data_plane,
            "port": self.port,
            "health_path": self.health_path,
            "metadata_path": self.metadata_path,
            "capabilities": {"operations": list(self.operations)},
        }


def _closed_keys(value: object, allowed: set[str], context: str) -> dict:
    if not isinstance(value, dict):
        raise ManifestError(f"{context} must be a table")
    keys = set(value)
    missing = allowed - keys
    unknown = keys - allowed
    if missing:
        raise ManifestError(f"{context} is missing: {', '.join(sorted(missing))}")
    if unknown:
        raise ManifestError(f"{context} has unknown keys: {', '.join(sorted(unknown))}")
    return value


def _string(value: object, field: str, *, maximum: int = 80) -> str:
    if not isinstance(value, str) or not value or value.strip() != value or len(value) > maximum:
        raise ManifestError(f"{field} must be a non-empty trimmed string up to {maximum} characters")
    return value


def _choice(value: object, field: str, allowed: set[str]) -> str:
    selected = _string(value, field)
    if selected not in allowed:
        raise ManifestError(f"{field} must be one of: {', '.join(sorted(allowed))}")
    return selected


def load(path: Path = MANIFEST_PATH) -> DriverManifest:
    """Load a v1 manifest, rejecting missing fields, unknown fields, and invalid values."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ManifestError(f"cannot read driver manifest: {exc}") from exc

    values = _closed_keys(raw, _TOP_LEVEL_KEYS, "manifest")
    capabilities = _closed_keys(values["capabilities"], _CAPABILITY_KEYS, "capabilities")

    schema_version = values["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise ManifestError("schema_version must be the integer 1")

    driver_id = _string(values["id"], "id")
    if not _ID_PATTERN.fullmatch(driver_id):
        raise ManifestError("id must be a lowercase kebab-case identifier")

    title = _string(values["title"], "title")
    version = _string(values["version"], "version")
    if not _VERSION_PATTERN.fullmatch(version):
        raise ManifestError("version must be a stable semantic version such as 1.0.0")

    summary = _string(values["summary"], "summary", maximum=160)
    interface = _string(values["interface"], "interface")
    if not _INTERFACE_PATTERN.fullmatch(interface):
        raise ManifestError("interface must use the shimpz.<name>/v<number> form")

    port = values["port"]
    if type(port) is not int or not 1 <= port <= 65535:
        raise ManifestError("port must be an integer from 1 to 65535")

    paths = {}
    for field in ("health_path", "metadata_path"):
        paths[field] = _string(values[field], field)
        if not _PATH_PATTERN.fullmatch(paths[field]):
            raise ManifestError(f"{field} must be an absolute lowercase HTTP path")
    if paths["health_path"] == paths["metadata_path"]:
        raise ManifestError("health_path and metadata_path must be different")

    operations = capabilities["operations"]
    if not isinstance(operations, list) or not operations:
        raise ManifestError("capabilities.operations must be a non-empty array")
    if any(not isinstance(operation, str) or not _OPERATION_PATTERN.fullmatch(operation) for operation in operations):
        raise ManifestError("each capability operation must be a lowercase dotted identifier")
    if len(operations) != len(set(operations)):
        raise ManifestError("capability operations must be unique")

    return DriverManifest(
        schema_version=schema_version,
        id=driver_id,
        title=title,
        version=version,
        summary=summary,
        interface=interface,
        scope=_choice(values["scope"], "scope", _SCOPES),
        credential_policy=_choice(values["credential_policy"], "credential_policy", _CREDENTIAL_POLICIES),
        data_plane=_choice(values["data_plane"], "data_plane", _DATA_PLANES),
        port=port,
        health_path=paths["health_path"],
        metadata_path=paths["metadata_path"],
        operations=tuple(operations),
    )
