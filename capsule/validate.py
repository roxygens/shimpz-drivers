"""Allowlist validation for capsule-driver — runs BEFORE any Docker or pg-driver call.

Nothing here touches Docker; it only decides yes/no and returns a validated capsule id the caller
(app.py) turns into container/network/volume/DB names. Same shape as the other drivers' validate.py
modules — the actual security boundary, not the client that acts on its output.
"""

from __future__ import annotations

import re

# The id becomes the DB project "capsule_<id>"; Postgres identifiers are 63 bytes and dbname/role are
# "proj_capsule_" + this, so cap it well under the limit. It also names the container/network/volumes,
# so keep it to the Docker-safe [a-z0-9_] set.
CAPSULE_ID_RE = re.compile(r"^[a-z0-9_]{1,40}$")


class ValidationError(Exception):
    """A capsule-driver request failed the allowlist — nothing was touched."""


def sanitize(name: str) -> str:
    lowered = re.sub(r"[^a-z0-9_]+", "_", str(name).lower())
    return lowered.strip("_")


def validate_capsule_id(name: object) -> str:
    if not isinstance(name, str) or not name:
        raise ValidationError(f"capsule id must be a non-empty string: {name!r}")
    sanitized = sanitize(name)
    if not sanitized or not CAPSULE_ID_RE.match(sanitized):
        raise ValidationError(f"capsule id sanitizes to empty or invalid: {name!r} -> {sanitized!r}")
    return sanitized


MAX_CHAT_MESSAGE = 16000


def validate_chat_message(message: object) -> str:
    """A Captain-to-Assistant chat message: non-empty text, size-bounded."""
    if not isinstance(message, str):
        raise ValidationError("message must be a string")
    text = message.strip()
    if not text:
        raise ValidationError("message must be non-empty")
    if len(text) > MAX_CHAT_MESSAGE:
        raise ValidationError(f"message too long (> {MAX_CHAT_MESSAGE} chars)")
    return text
