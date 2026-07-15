"""Fixed, bounded capsule-driver client for Capsule-scoped R2 credentials.

Only this controller persists a cleartext Capsule principal.  R2 stores its hash and all credential
values stay inside r2-driver.  Every response crossing this boundary is projected onto a closed,
non-secret shape before the Admin can observe it.
"""

from __future__ import annotations

import contextlib
import errno
import http.client
import ipaddress
import json
import os
import re
import secrets
import stat
import threading
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from urllib.parse import urlsplit

R2DRIVER_URL = os.environ.get("SHIMPZ_R2DRIVER_URL", "http://r2-driver:7075")
PROVISIONER_TOKEN_FILE = Path(
    os.environ.get("SHIMPZ_R2DRIVER_PROVISIONER_TOKEN_FILE", "/run/shimpz-r2provisioner/token")
)
PRINCIPAL_DIR = Path(os.environ.get("SHIMPZ_R2_PRINCIPAL_DIR", "/var/lib/capsule-driver/r2-principals"))
PROVISIONER_TOKEN_UID = int(os.environ.get("SHIMPZ_R2DRIVER_UID", "10007"))
PROVISIONER_TOKEN_GID = int(os.environ.get("SHIMPZ_R2PROVISIONER_TOKEN_GID", "10015"))

TIMEOUT_SECONDS = 30
MAX_JSON_REQUEST_BYTES = 64 * 1024
MAX_JSON_RESPONSE_BYTES = 256 * 1024
MAX_CREDENTIALS = 256

_CAPSULE_ID_RE = re.compile(r"^[a-z0-9_]{1,40}$")
_CREDENTIAL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")
_DNS_NAME_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*$"
)
_principal_guard = threading.RLock()


class R2DriverError(Exception):
    """A safe, typed boundary failure; it never contains an upstream body or capability."""

    def __init__(self, status: HTTPStatus, message: str, *, category: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.category = category


@dataclass(frozen=True)
class _Endpoint:
    host: str
    port: int


def _parse_endpoint(value: str) -> _Endpoint:
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port or 7075
    except (TypeError, ValueError) as exc:
        raise RuntimeError("SHIMPZ_R2DRIVER_URL is invalid") from exc
    if (
        parsed.scheme != "http"
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or not 1 <= port <= 65535
    ):
        raise RuntimeError("SHIMPZ_R2DRIVER_URL must be one fixed HTTP origin")
    try:
        ipaddress.ip_address(host)
    except ValueError as exc:
        if host != host.lower() or _DNS_NAME_RE.fullmatch(host) is None:
            raise RuntimeError("SHIMPZ_R2DRIVER_URL has an invalid host") from exc
    return _Endpoint(host, port)


_ENDPOINT = _parse_endpoint(R2DRIVER_URL)


def _capsule_id(value: object) -> str:
    if not isinstance(value, str) or _CAPSULE_ID_RE.fullmatch(value) is None:
        raise R2DriverError(HTTPStatus.BAD_REQUEST, "Capsule id is invalid", category="request")
    return value


def _credential_id(value: object) -> str:
    if not isinstance(value, str) or _CREDENTIAL_ID_RE.fullmatch(value) is None:
        raise R2DriverError(HTTPStatus.BAD_REQUEST, "credential id is invalid", category="request")
    return value


def _encode_json(payload: object | None) -> bytes | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise R2DriverError(HTTPStatus.BAD_REQUEST, "request body must be a JSON object", category="request")
    try:
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise R2DriverError(HTTPStatus.BAD_REQUEST, "request body is invalid", category="request") from exc
    if not raw or len(raw) > MAX_JSON_REQUEST_BYTES:
        raise R2DriverError(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            "request body exceeds its fixed limit",
            category="request",
        )
    return raw


def _read_json_response(response: http.client.HTTPResponse) -> dict[str, object]:
    content_type = (response.getheader("Content-Type") or "").partition(";")[0].strip().lower()
    if content_type != "application/json":
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver returned an invalid response", category="protocol")
    declared = response.getheader("Content-Length")
    if declared is not None:
        try:
            length = int(declared)
        except ValueError as exc:
            raise R2DriverError(
                HTTPStatus.BAD_GATEWAY,
                "R2 Driver returned an invalid response",
                category="protocol",
            ) from exc
        if length < 0 or length > MAX_JSON_RESPONSE_BYTES:
            raise R2DriverError(
                HTTPStatus.BAD_GATEWAY,
                "R2 Driver returned an invalid response",
                category="protocol",
            )
    raw = response.read(MAX_JSON_RESPONSE_BYTES + 1)
    if not raw or len(raw) > MAX_JSON_RESPONSE_BYTES:
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver returned an invalid response", category="protocol")
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise R2DriverError(
            HTTPStatus.BAD_GATEWAY,
            "R2 Driver returned an invalid response",
            category="protocol",
        ) from exc
    if not isinstance(payload, dict):
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver returned an invalid response", category="protocol")
    return payload


_SAFE_STATUS = {
    HTTPStatus.BAD_REQUEST: ("R2 Driver rejected the request", "request"),
    HTTPStatus.NOT_FOUND: ("R2 Driver resource was not found", "not-found"),
    HTTPStatus.CONFLICT: ("R2 Driver resource changed or conflicts", "conflict"),
    HTTPStatus.REQUEST_ENTITY_TOO_LARGE: ("R2 Driver request is too large", "request"),
    HTTPStatus.UNPROCESSABLE_ENTITY: ("R2 rejected the credential bundle", "verification"),
    HTTPStatus.TOO_MANY_REQUESTS: ("R2 Driver is busy; retry later", "rate-limit"),
}


def _call(method: str, path: str, payload: object | None = None, *, bearer: str | None = None) -> dict[str, object]:
    if method not in {"GET", "POST", "PUT", "DELETE"} or not path.startswith("/") or "?" in path or "#" in path:
        raise R2DriverError(HTTPStatus.INTERNAL_SERVER_ERROR, "invalid R2 Driver operation", category="local")
    raw = _encode_json(payload)
    headers = {"Accept": "application/json"}
    if raw is not None:
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(raw))
    if bearer is not None:
        if _TOKEN_RE.fullmatch(bearer) is None:
            raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver capability is unavailable", category="local")
        headers["Authorization"] = f"Bearer {bearer}"
    connection: http.client.HTTPConnection | None = None
    try:
        connection = http.client.HTTPConnection(_ENDPOINT.host, _ENDPOINT.port, timeout=TIMEOUT_SECONDS)
        connection.request(method, path, body=raw, headers=headers)
        response = connection.getresponse()
        result = _read_json_response(response)
    except R2DriverError:
        raise
    except (OSError, TimeoutError, UnicodeError, http.client.HTTPException) as exc:
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver is unavailable", category="transport") from exc
    finally:
        if connection is not None:
            with contextlib.suppress(OSError, http.client.HTTPException):
                connection.close()
    if response.status == HTTPStatus.OK:
        return result
    try:
        upstream = HTTPStatus(response.status)
    except ValueError:
        upstream = None
    if upstream in _SAFE_STATUS:
        message, category = _SAFE_STATUS[upstream]
        raise R2DriverError(upstream, message, category=category)
    # Authentication failures, redirects and every upstream 5xx collapse to the same boundary error.
    raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver is unavailable", category="upstream")


def _ensure_principal_directory() -> None:
    if not PRINCIPAL_DIR.is_absolute():
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver capability is unavailable", category="local")
    try:
        PRINCIPAL_DIR.mkdir(parents=True, mode=0o700, exist_ok=True)
        metadata = PRINCIPAL_DIR.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise OSError("unsafe principal directory")
    except OSError as exc:
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver capability is unavailable", category="local") from exc


def _fsync_principal_directory() -> None:
    try:
        descriptor = os.open(PRINCIPAL_DIR, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver capability is unavailable", category="local") from exc


def _principal_path(capsule_id: object) -> Path:
    return PRINCIPAL_DIR / f"{_capsule_id(capsule_id)}.token"


def _read_principal(path: Path) -> str:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver capability is unavailable", category="local") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_size != 64
        ):
            raise OSError("unsafe principal token")
        raw = os.read(descriptor, 65)
    except OSError as exc:
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver capability is unavailable", category="local") from exc
    finally:
        os.close(descriptor)
    try:
        token = raw.decode("ascii")
    except UnicodeError as exc:
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver capability is unavailable", category="local") from exc
    if _TOKEN_RE.fullmatch(token) is None:
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver capability is unavailable", category="local")
    return token


def _write_new_principal(path: Path, token: str) -> None:
    temporary = PRINCIPAL_DIR / f".{path.name}.{secrets.token_hex(12)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(temporary, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        raw = token.encode("ascii")
        while raw:
            written = os.write(descriptor, raw)
            if written < 1:
                raise OSError("short principal write")
            raw = raw[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        # Hard-link publication is atomic and cannot replace a pre-existing file or symlink.
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        _fsync_principal_directory()
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            return
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver capability is unavailable", category="local") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)


def _principal(capsule_id: object, *, create: bool) -> str:
    path = _principal_path(capsule_id)
    with _principal_guard:
        _ensure_principal_directory()
        try:
            return _read_principal(path)
        except R2DriverError as exc:
            if path.exists() or not create:
                raise
            if exc.category != "local":
                raise
        _write_new_principal(path, secrets.token_hex(32))
        return _read_principal(path)


def _read_provisioner() -> str:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(PROVISIONER_TOKEN_FILE, flags)
    except OSError as exc:
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver provisioner is unavailable", category="local") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != PROVISIONER_TOKEN_UID
            or metadata.st_gid != PROVISIONER_TOKEN_GID
            or stat.S_IMODE(metadata.st_mode) != 0o440
            or metadata.st_nlink != 1
            or metadata.st_size != 64
        ):
            raise OSError("unsafe provisioner token")
        raw = os.read(descriptor, 65)
    except OSError as exc:
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver provisioner is unavailable", category="local") from exc
    finally:
        os.close(descriptor)
    try:
        token = raw.decode("ascii")
    except UnicodeError as exc:
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver provisioner is unavailable", category="local") from exc
    if _TOKEN_RE.fullmatch(token) is None:
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver provisioner is unavailable", category="local")
    return token


def _remove_principal(capsule_id: object) -> None:
    path = _principal_path(capsule_id)
    with _principal_guard:
        _ensure_principal_directory()
        try:
            # Validate before deletion so a hostile filesystem object is never silently accepted.
            _read_principal(path)
        except R2DriverError:
            if not path.exists():
                return
            raise
        try:
            path.unlink()
            _fsync_principal_directory()
        except OSError as exc:
            raise R2DriverError(
                HTTPStatus.BAD_GATEWAY, "R2 Driver capability is unavailable", category="local"
            ) from exc


def provision_capsule(capsule_id: object) -> dict[str, object]:
    cid = _capsule_id(capsule_id)
    principal = _principal(cid, create=True)
    return _call(
        "POST",
        "/v1/capsules/provision",
        {"capsule_id": cid, "principal_token": principal},
        bearer=_read_provisioner(),
    )


def ensure_provisioned(capsule_id: object) -> None:
    # Replaying provision is intentional: it resolves an ambiguous earlier HTTP result without local state.
    provision_capsule(capsule_id)


def retire_capsule(capsule_id: object) -> dict[str, object]:
    cid = _capsule_id(capsule_id)
    return _call(
        "POST",
        "/v1/capsules/retire",
        {"capsule_id": cid},
        bearer=_read_provisioner(),
    )


def finalize_capsule_drop(capsule_id: object) -> dict[str, object]:
    cid = _capsule_id(capsule_id)
    result = _call(
        "POST",
        "/v1/capsules/finalize",
        {"capsule_id": cid},
        bearer=_read_provisioner(),
    )
    # Never remove the only cleartext principal until the finalizer returned an authenticated 200.
    _remove_principal(cid)
    return result


def _project_driver(value: dict[str, object]) -> dict[str, object]:
    allowed = {
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
    if value.get("id") != "r2" or not allowed.issuperset(value):
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver returned an invalid definition", category="protocol")
    return {key: value[key] for key in allowed if key in value}


def _project_form(value: dict[str, object]) -> dict[str, object]:
    if set(value) != {"schema_version", "owner_scope", "cardinality", "profiles"}:
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver returned an invalid form", category="protocol")
    profiles = value.get("profiles")
    if not isinstance(profiles, list) or len(profiles) > 32:
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver returned an invalid form", category="protocol")
    projected_profiles: list[dict[str, object]] = []
    for profile in profiles:
        if not isinstance(profile, dict) or not set(profile) <= {"id", "kind", "title", "summary", "fields"}:
            raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver returned an invalid form", category="protocol")
        fields = profile.get("fields")
        if not isinstance(fields, list) or len(fields) > 64:
            raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver returned an invalid form", category="protocol")
        projected_fields: list[dict[str, object]] = []
        field_keys = {
            "id",
            "label",
            "type",
            "format",
            "min_length",
            "max_length",
            "required",
            "write_only",
            "help",
            "options",
        }
        for field in fields:
            if not isinstance(field, dict) or not set(field) <= field_keys:
                raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver returned an invalid form", category="protocol")
            projected_fields.append({key: field[key] for key in field_keys if key in field})
        projected = {key: profile[key] for key in ("id", "kind", "title", "summary") if key in profile}
        projected["fields"] = projected_fields
        projected_profiles.append(projected)
    return {
        "schema_version": value["schema_version"],
        "owner_scope": value["owner_scope"],
        "cardinality": value["cardinality"],
        "profiles": projected_profiles,
    }


def _project_credential(value: object) -> dict[str, object]:
    allowed = {"id", "profile_id", "label", "generation", "status", "created_at", "updated_at"}
    required = {"id", "profile_id", "label", "generation", "status", "created_at", "updated_at"}
    if not isinstance(value, dict) or not required <= set(value) or not set(value) <= allowed:
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver returned invalid metadata", category="protocol")
    _credential_id(value.get("id"))
    return {key: value[key] for key in allowed}


def driver_document(capsule_id: object) -> dict[str, object]:
    cid = _capsule_id(capsule_id)
    driver = _project_driver(_call("GET", "/v1/driver"))
    credential_form = _project_form(_call("GET", "/v1/driver/credentials"))
    inventory = _call("GET", f"/v1/capsules/{cid}/credentials", bearer=_principal(cid, create=False))
    if set(inventory) != {"credentials"} or not isinstance(inventory.get("credentials"), list):
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver returned invalid inventory", category="protocol")
    credentials = inventory["credentials"]
    if len(credentials) > MAX_CREDENTIALS:
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver returned invalid inventory", category="protocol")
    return {
        "driver": driver,
        "credential_form": credential_form,
        "credentials": [_project_credential(item) for item in credentials],
    }


def create_credential(capsule_id: object, payload: dict[str, object]) -> dict[str, object]:
    cid = _capsule_id(capsule_id)
    result = _call(
        "POST",
        f"/v1/capsules/{cid}/credentials",
        payload,
        bearer=_principal(cid, create=False),
    )
    return _project_credential(result)


def rotate_credential(capsule_id: object, credential_id: object, payload: dict[str, object]) -> dict[str, object]:
    cid, crid = _capsule_id(capsule_id), _credential_id(credential_id)
    result = _call(
        "PUT",
        f"/v1/capsules/{cid}/credentials/{crid}",
        payload,
        bearer=_principal(cid, create=False),
    )
    return _project_credential(result)


def remove_credential(capsule_id: object, credential_id: object, payload: dict[str, object]) -> dict[str, object]:
    cid, crid = _capsule_id(capsule_id), _credential_id(credential_id)
    result = _call(
        "DELETE",
        f"/v1/capsules/{cid}/credentials/{crid}",
        payload,
        bearer=_principal(cid, create=False),
    )
    return _project_credential(result)


def verify_credential(capsule_id: object, credential_id: object) -> dict[str, object]:
    cid, crid = _capsule_id(capsule_id), _credential_id(credential_id)
    result = _call(
        "POST",
        f"/v1/capsules/{cid}/credentials/{crid}/verify",
        {},
        bearer=_principal(cid, create=False),
    )
    allowed = {"id", "generation", "verdict", "trace_id"}
    if set(result) != allowed or result.get("id") != crid or result.get("verdict") != "valid":
        raise R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver returned an invalid verdict", category="protocol")
    return {key: result[key] for key in allowed}
