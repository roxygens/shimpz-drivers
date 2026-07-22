"""Allowlist validation for pg-driver — runs BEFORE any psql/createdb/dropdb call.

Nothing here touches Postgres; it only decides yes/no and returns a validated project name the
caller (app.py) turns into pg_client.py calls. This validator is the actual security boundary,
not the client that acts on its output.
"""

from __future__ import annotations

import hashlib
import re
import secrets

# Postgres identifier limit is 63 bytes; dbname/role are "proj_" + this, so leave room for the prefix.
PROJECT_NAME_RE = re.compile(r"^[a-z0-9_]{1,58}$")
TEAM_ID_RE = re.compile(r"^[a-z0-9_]{1,40}$")
APP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
PRINCIPAL_TOKEN_RE = re.compile(r"^[a-f0-9]{64}$")


class ValidationError(Exception):
    """A pg-driver request failed the allowlist — nothing was touched."""


def sanitize_proj(name: str) -> str:
    """Port of shimpzdetect.sh's _sanitize_proj / drivers/apps/validate.py's sanitize_proj.

    MUST match both exactly — shimpz-app and the server-side drivers independently derive the same
    proj_<name> identity from a raw project name.
    """
    lowered = re.sub(r"[^a-z0-9_]+", "_", str(name).lower())
    return lowered.strip("_")


def validate_project(name: object) -> str:
    if not isinstance(name, str) or not name:
        raise ValidationError(f"project name must be a non-empty string: {name!r}")
    sanitized = sanitize_proj(name)
    if not sanitized or not PROJECT_NAME_RE.match(sanitized):
        raise ValidationError(f"project name sanitizes to empty or invalid: {name!r} -> {sanitized!r}")
    return sanitized


def validate_team_id(value: object) -> str:
    if not isinstance(value, str) or not TEAM_ID_RE.fullmatch(value):
        raise ValidationError("team_id must match [a-z0-9_]{1,40}")
    return value


def validate_app_id(value: object) -> str:
    if not isinstance(value, str) or not APP_ID_RE.fullmatch(value):
        raise ValidationError("app_id must match [a-z0-9][a-z0-9-]{0,39}")
    return value


def validate_principal_token(value: object) -> str:
    if not isinstance(value, str) or not PRINCIPAL_TOKEN_RE.fullmatch(value):
        raise ValidationError("principal_token must be a 256-bit lowercase hex token")
    return value


def team_project(team_id: str) -> str:
    return f"team_{validate_team_id(team_id)}"


def team_app_project(team_id: str, app_id: str) -> str:
    digest = hashlib.sha256(validate_team_id(team_id).encode()).hexdigest()[:10]
    app = validate_app_id(app_id).replace("-", "_")
    return f"team_{digest}_{app}"


def tokens_equal(left: str, right: str) -> bool:
    """Constant-time comparison for the control-plane bearer."""
    return secrets.compare_digest(left, right)
