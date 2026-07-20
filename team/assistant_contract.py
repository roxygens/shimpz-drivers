"""Closed first-party contract for the Shimpz Assistant reference artifact.

The hosted and single-owner controllers deliberately share this module so the
Brain never sees a Power that one runtime validates differently from the other.
Secret declarations are public metadata; values remain Controller-owned.
"""

from __future__ import annotations

import re
from typing import Any

ASSISTANT_ID = "shimpz-assistant"
ASSISTANT_NAME = "Shimpz Assistant"
ASSISTANT_SUMMARY = "Read public X profiles and manage approved Posts for one connected X account."
ASSISTANT_RPC_COMMAND = "/usr/local/bin/shimpz-assistant-rpc"
ASSISTANT_ALLOWED_HOSTS = ("api.x.com",)
MAX_HELP_BYTES = 32 * 1024
HELP_LOCALES = frozenset({"en", "pt", "es", "zh", "fr", "de", "ja", "ar"})
_USERNAME = re.compile(r"[A-Za-z0-9_]{1,15}")
_SNOWFLAKE = re.compile(r"[0-9]{1,19}")


def secret_contracts() -> dict[str, dict[str, str]]:
    """Return fresh public metadata; no secret value or transport hint lives here."""
    return {
        "x-bearer-token": {
            "name": "X Bearer Token",
            "summary": "App-only token used exclusively for public X profile reads.",
        },
        "x-api-key": {
            "name": "X API Key",
            "summary": "OAuth 1.0a consumer key identifying the connected X application.",
        },
        "x-api-key-secret": {
            "name": "X API Key Secret",
            "summary": "OAuth 1.0a consumer secret used to sign account requests.",
        },
        "x-access-token": {
            "name": "X Access Token",
            "summary": "OAuth 1.0a token identifying the connected X account.",
        },
        "x-access-token-secret": {
            "name": "X Access Token Secret",
            "summary": "OAuth 1.0a token secret used to sign account requests.",
        },
    }


def _user_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string", "pattern": "^[0-9]{1,19}$"},
            "name": {"type": "string", "minLength": 1, "maxLength": 80},
            "username": {"type": "string", "pattern": "^[A-Za-z0-9_]{1,15}$"},
        },
        "required": ["id", "name", "username"],
        "additionalProperties": False,
    }


def power_contracts() -> dict[str, dict[str, Any]]:
    """Return fresh closed schemas so callers cannot mutate another registry."""
    oauth = ("x-api-key", "x-api-key-secret", "x-access-token", "x-access-token-secret")
    return {
        "public-user-lookup": {
            "method": "POST",
            "path": "/v1/powers/public-user-lookup",
            "summary": "Read one public X profile by username.",
            "input_schema": {
                "type": "object",
                "properties": {"username": {"type": "string", "pattern": "^[A-Za-z0-9_]{1,15}$"}},
                "required": ["username"],
                "additionalProperties": False,
            },
            "output_schema": _user_schema(),
            "approval": "none",
            "secrets": ("x-bearer-token",),
        },
        "identity-me": {
            "method": "POST",
            "path": "/v1/powers/identity-me",
            "summary": "Read the identity of the connected X account.",
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
            "output_schema": _user_schema(),
            "approval": "none",
            "secrets": oauth,
        },
        "create-post": {
            "method": "POST",
            "path": "/v1/powers/create-post",
            "summary": "Publish one Post from the connected X account after explicit approval.",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string", "minLength": 1, "maxLength": 280}},
                "required": ["text"],
                "additionalProperties": False,
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "pattern": "^[0-9]{1,19}$"},
                    "text": {"type": "string", "minLength": 1, "maxLength": 280},
                },
                "required": ["id", "text"],
                "additionalProperties": False,
            },
            "approval": "each-run",
            "secrets": oauth,
        },
        "delete-post": {
            "method": "POST",
            "path": "/v1/powers/delete-post",
            "summary": "Delete one Post owned by the connected X account after explicit approval.",
            "input_schema": {
                "type": "object",
                "properties": {"id": {"type": "string", "pattern": "^[0-9]{1,19}$"}},
                "required": ["id"],
                "additionalProperties": False,
            },
            "output_schema": {
                "type": "object",
                "properties": {"deleted": {"const": True}},
                "required": ["deleted"],
                "additionalProperties": False,
            },
            "approval": "each-run",
            "secrets": oauth,
        },
    }


def _closed_object(payload: object, allowed: set[str], *, required: set[str]) -> dict[str, object]:
    if not isinstance(payload, dict) or not required <= set(payload) <= allowed:
        raise ValueError("Power payload does not match its declared fields")
    return payload


def _bounded_text(value: object, *, minimum: int, maximum: int, field: str) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum or value.strip() != value:
        raise ValueError(f"{field} is invalid")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field} is invalid")
    return value


def _username(value: object) -> str:
    if not isinstance(value, str) or _USERNAME.fullmatch(value) is None:
        raise ValueError("username is invalid")
    return value


def _snowflake(value: object) -> str:
    if not isinstance(value, str) or _SNOWFLAKE.fullmatch(value) is None:
        raise ValueError("id is invalid")
    return value


def validate_power_input(assistant_id: str, power: str, payload: object) -> dict[str, object]:
    if assistant_id != ASSISTANT_ID:
        raise ValueError("the Power has no declared input contract")
    if power == "public-user-lookup":
        safe = _closed_object(payload, {"username"}, required={"username"})
        return {"username": _username(safe["username"])}
    if power == "identity-me":
        _closed_object(payload, set(), required=set())
        return {}
    if power == "create-post":
        safe = _closed_object(payload, {"text"}, required={"text"})
        return {"text": _bounded_text(safe["text"], minimum=1, maximum=280, field="text")}
    if power == "delete-post":
        safe = _closed_object(payload, {"id"}, required={"id"})
        return {"id": _snowflake(safe["id"])}
    raise ValueError("the Power has no declared input contract")


def _user(payload: object) -> dict[str, object]:
    safe = _closed_object(payload, {"id", "name", "username"}, required={"id", "name", "username"})
    return {
        "id": _snowflake(safe["id"]),
        "name": _bounded_text(safe["name"], minimum=1, maximum=80, field="name"),
        "username": _username(safe["username"]),
    }


def validate_power_output(assistant_id: str, power: str, payload: object) -> dict[str, object]:
    if assistant_id != ASSISTANT_ID:
        raise ValueError("the Power has no declared output contract")
    if power in {"public-user-lookup", "identity-me"}:
        return _user(payload)
    if power == "create-post":
        safe = _closed_object(payload, {"id", "text"}, required={"id", "text"})
        return {
            "id": _snowflake(safe["id"]),
            "text": _bounded_text(safe["text"], minimum=1, maximum=280, field="text"),
        }
    if power == "delete-post":
        safe = _closed_object(payload, {"deleted"}, required={"deleted"})
        if safe["deleted"] is not True:
            raise ValueError("deleted is invalid")
        return {"deleted": True}
    raise ValueError("the Power has no declared output contract")


def validate_help_payload(payload: object) -> dict[str, str]:
    """Accept only one bounded UTF-8 Markdown document from the fixed RPC."""
    if not isinstance(payload, dict) or set(payload) != {"markdown"}:
        raise ValueError("Assistant Help returned an invalid result")
    markdown = payload["markdown"]
    if not isinstance(markdown, str) or not markdown:
        raise ValueError("Assistant Help returned an invalid result")
    try:
        encoded = markdown.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("Assistant Help returned an invalid result") from exc
    if len(encoded) > MAX_HELP_BYTES or any(
        (ord(character) < 32 and character not in "\n\t") or 127 <= ord(character) <= 159 for character in markdown
    ):
        raise ValueError("Assistant Help returned an invalid result")
    return {"markdown": markdown}


def validate_help_locale(locale: object) -> str:
    """Accept only the fixed locale identifiers implemented by the Assistant Help RPC."""
    if not isinstance(locale, str) or locale not in HELP_LOCALES:
        raise ValueError("Assistant Help locale is not supported")
    return locale
