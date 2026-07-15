"""Strict, dependency-free parsers for the Shimpz Driver Spec v1 contract."""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

MANIFEST_PATH = Path(__file__).with_name("shimpz.driver.toml")
CREDENTIALS_PATH = Path(__file__).with_name("shimpz.credentials.json")

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
    "credential_schema_path",
    "capabilities",
}
_CAPABILITY_KEYS = {"operations"}
_CREDENTIAL_TOP_LEVEL_KEYS = {"schema_version", "owner_scope", "cardinality", "profiles"}
_PROFILE_REQUIRED_KEYS = {"id", "kind", "title", "fields"}
_PROFILE_OPTIONAL_KEYS = {"summary"}
_FIELD_BASE_KEYS = {"id", "label", "type", "format", "min_length", "max_length", "required"}
_SECRET_FIELD_KEYS = _FIELD_BASE_KEYS | {"write_only"}
_SCOPES = {"space"}
_CREDENTIAL_POLICIES = {"none", "managed", "managed-or-byok"}
_DATA_PLANES = {"direct", "brokered"}
_OWNER_SCOPES = {"capsule"}
_CARDINALITIES = {"one", "many"}
_PROFILE_KINDS = {"secret-fields"}
_FIELD_TYPES = {"text", "secret"}
_TEXT_FIELD_FORMATS = {
    "plain-text",
    "account-id",
    "bucket-name",
    "hostname",
    "region",
    "tenant-id",
    "username",
}
_SECRET_FIELD_FORMATS = {
    "access-key-id",
    "api-key",
    "password",
    "private-key",
    "secret-access-key",
    "secret-token",
}
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_FIELD_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_VERSION_PATTERN = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
_INTERFACE_PATTERN = re.compile(r"^shimpz\.[a-z][a-z0-9-]*/v[1-9][0-9]*$")
_OPERATION_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:\.[a-z][a-z0-9]*)+$")
_PATH_PATTERN = re.compile(r"^/[a-z0-9][a-z0-9/-]*$")


class ManifestError(ValueError):
    """The driver manifest does not satisfy the closed Driver Spec v1 contract."""


class CredentialSchemaError(ValueError):
    """The credential form does not satisfy the closed Driver Spec v1 contract."""


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
    credential_schema_path: str
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
            "credential_schema_path": self.credential_schema_path,
            "capabilities": {"operations": list(self.operations)},
        }


@dataclass(frozen=True)
class CredentialField:
    id: str
    label: str
    type: str
    format: str
    min_length: int
    max_length: int
    required: bool
    write_only: bool | None = None

    def public(self) -> dict[str, object]:
        result: dict[str, object] = {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "format": self.format,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "required": self.required,
        }
        if self.write_only is not None:
            result["write_only"] = self.write_only
        return result


@dataclass(frozen=True)
class CredentialProfile:
    id: str
    kind: str
    title: str
    fields: tuple[CredentialField, ...]
    summary: str | None = None

    def public(self) -> dict[str, object]:
        result: dict[str, object] = {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "fields": [field.public() for field in self.fields],
        }
        if self.summary is not None:
            result["summary"] = self.summary
        return result


@dataclass(frozen=True)
class CredentialSchema:
    schema_version: int
    owner_scope: str
    cardinality: str
    profiles: tuple[CredentialProfile, ...]

    def public(self) -> dict[str, object]:
        """Return definitions only; credential values and inventory cannot enter this model."""
        return {
            "schema_version": self.schema_version,
            "owner_scope": self.owner_scope,
            "cardinality": self.cardinality,
            "profiles": [profile.public() for profile in self.profiles],
        }


def _closed_keys(
    value: object,
    allowed: set[str],
    context: str,
    *,
    required: set[str] | None = None,
    error_type: type[ValueError] = ManifestError,
) -> dict:
    if not isinstance(value, dict):
        raise error_type(f"{context} must be an object")
    keys = set(value)
    missing = (required if required is not None else allowed) - keys
    unknown = keys - allowed
    if missing:
        raise error_type(f"{context} is missing: {', '.join(sorted(missing))}")
    if unknown:
        raise error_type(f"{context} has unknown keys: {', '.join(sorted(unknown))}")
    return value


def _string(
    value: object,
    field: str,
    *,
    maximum: int = 80,
    error_type: type[ValueError] = ManifestError,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\n" in value
        or "\r" in value
        or len(value) > maximum
    ):
        raise error_type(f"{field} must be a non-empty trimmed single-line string up to {maximum} characters")
    return value


def _choice(
    value: object,
    field: str,
    allowed: set[str],
    *,
    error_type: type[ValueError] = ManifestError,
) -> str:
    selected = _string(value, field, error_type=error_type)
    if selected not in allowed:
        raise error_type(f"{field} must be one of: {', '.join(sorted(allowed))}")
    return selected


def _positive_length(value: object, field: str) -> int:
    if type(value) is not int or not 1 <= value <= 4096:
        raise CredentialSchemaError(f"{field} must be an integer from 1 to 4096")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CredentialSchemaError(f"credential schema repeats key: {key}")
        result[key] = value
    return result


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
    for field in ("health_path", "metadata_path", "credential_schema_path"):
        paths[field] = _string(values[field], field)
        if not _PATH_PATTERN.fullmatch(paths[field]):
            raise ManifestError(f"{field} must be an absolute lowercase HTTP path")
    if len(set(paths.values())) != len(paths):
        raise ManifestError("health and discovery paths must be different")

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
        credential_schema_path=paths["credential_schema_path"],
        operations=tuple(operations),
    )


def _parse_credential_field(raw_field: object, context: str) -> CredentialField:
    if not isinstance(raw_field, dict):
        raise CredentialSchemaError(f"{context} must be an object")
    field_type = raw_field.get("type")
    allowed_keys = _SECRET_FIELD_KEYS if field_type == "secret" else _FIELD_BASE_KEYS
    field = _closed_keys(raw_field, allowed_keys, context, error_type=CredentialSchemaError)

    field_id = _string(field["id"], f"{context}.id", error_type=CredentialSchemaError)
    if not _FIELD_ID_PATTERN.fullmatch(field_id):
        raise CredentialSchemaError(f"{context}.id must be a lowercase snake_case identifier")
    selected_type = _choice(
        field["type"],
        f"{context}.type",
        _FIELD_TYPES,
        error_type=CredentialSchemaError,
    )
    selected_format = _choice(
        field["format"],
        f"{context}.format",
        _SECRET_FIELD_FORMATS if selected_type == "secret" else _TEXT_FIELD_FORMATS,
        error_type=CredentialSchemaError,
    )
    minimum = _positive_length(field["min_length"], f"{context}.min_length")
    maximum = _positive_length(field["max_length"], f"{context}.max_length")
    if minimum > maximum:
        raise CredentialSchemaError(f"{context}.min_length cannot exceed max_length")
    required = field["required"]
    if type(required) is not bool:
        raise CredentialSchemaError(f"{context}.required must be a boolean")
    write_only = field.get("write_only")
    if selected_type == "secret" and write_only is not True:
        raise CredentialSchemaError(f"{context}.write_only must be true for secret fields")

    return CredentialField(
        id=field_id,
        label=_string(field["label"], f"{context}.label", error_type=CredentialSchemaError),
        type=selected_type,
        format=selected_format,
        min_length=minimum,
        max_length=maximum,
        required=required,
        write_only=write_only,
    )


def _parse_credential_profile(raw_profile: object, index: int) -> CredentialProfile:
    context = f"profiles[{index}]"
    profile = _closed_keys(
        raw_profile,
        _PROFILE_REQUIRED_KEYS | _PROFILE_OPTIONAL_KEYS,
        context,
        required=_PROFILE_REQUIRED_KEYS,
        error_type=CredentialSchemaError,
    )
    profile_id = _string(profile["id"], f"{context}.id", error_type=CredentialSchemaError)
    if not _ID_PATTERN.fullmatch(profile_id):
        raise CredentialSchemaError(f"{context}.id must be a lowercase kebab-case identifier")

    raw_fields = profile["fields"]
    if not isinstance(raw_fields, list) or not 1 <= len(raw_fields) <= 32:
        raise CredentialSchemaError(f"{context}.fields must contain from 1 to 32 definitions")
    fields = tuple(
        _parse_credential_field(raw_field, f"{context}.fields[{field_index}]")
        for field_index, raw_field in enumerate(raw_fields)
    )
    if len(fields) != len({field.id for field in fields}):
        raise CredentialSchemaError(f"{context}.field ids must be unique")

    summary = profile.get("summary")
    return CredentialProfile(
        id=profile_id,
        kind=_choice(
            profile["kind"],
            f"{context}.kind",
            _PROFILE_KINDS,
            error_type=CredentialSchemaError,
        ),
        title=_string(profile["title"], f"{context}.title", error_type=CredentialSchemaError),
        fields=fields,
        summary=(
            _string(summary, f"{context}.summary", maximum=240, error_type=CredentialSchemaError)
            if summary is not None
            else None
        ),
    )


def load_credentials(path: Path = CREDENTIALS_PATH) -> CredentialSchema:
    """Load credential form definitions without accepting credential values or inventory."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except CredentialSchemaError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CredentialSchemaError(f"cannot read credential schema: {exc}") from exc

    values = _closed_keys(
        raw,
        _CREDENTIAL_TOP_LEVEL_KEYS,
        "credential schema",
        error_type=CredentialSchemaError,
    )
    schema_version = values["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        raise CredentialSchemaError("schema_version must be the integer 1")

    raw_profiles = values["profiles"]
    if not isinstance(raw_profiles, list) or not 1 <= len(raw_profiles) <= 16:
        raise CredentialSchemaError("profiles must contain from 1 to 16 definitions")

    profiles = tuple(_parse_credential_profile(profile, index) for index, profile in enumerate(raw_profiles))
    if len(profiles) != len({profile.id for profile in profiles}):
        raise CredentialSchemaError("profile ids must be unique")

    return CredentialSchema(
        schema_version=schema_version,
        owner_scope=_choice(
            values["owner_scope"],
            "owner_scope",
            _OWNER_SCOPES,
            error_type=CredentialSchemaError,
        ),
        cardinality=_choice(
            values["cardinality"],
            "cardinality",
            _CARDINALITIES,
            error_type=CredentialSchemaError,
        ),
        profiles=profiles,
    )
