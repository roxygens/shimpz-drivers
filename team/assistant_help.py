"""Closed, provider-neutral validation for Assistant Help responses."""

from __future__ import annotations

MAX_HELP_BYTES = 32 * 1024
HELP_LOCALES = frozenset({"en", "pt", "es", "zh", "fr", "de", "ja", "ar"})


def validate_payload(payload: object) -> dict[str, str]:
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


def validate_locale(locale: object) -> str:
    """Accept only the fixed locale identifiers implemented by Assistant Help RPCs."""
    if not isinstance(locale, str) or locale not in HELP_LOCALES:
        raise ValueError("Assistant Help locale is not supported")
    return locale
