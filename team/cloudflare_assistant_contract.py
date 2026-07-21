"""Closed controller contract for the read-only Cloudflare Assistant."""

from __future__ import annotations

import re
from typing import Any

ASSISTANT_ID = "shimpz-cloudflare"
ASSISTANT_NAME = "Shimpz Cloudflare"
ASSISTANT_SUMMARY = "List Cloudflare zones and inspect their DNS records through OAuth."
ASSISTANT_RPC_COMMAND = "/usr/local/bin/shimpz-assistant-rpc"
ASSISTANT_ALLOWED_HOSTS = ("api.cloudflare.com",)
ASSISTANT_HEALTH_PATH = "/healthz"

_HEX_ID = re.compile(r"[0-9a-f]{32}\Z")
_STATUS = re.compile(r"[a-z][a-z0-9_-]{0,31}\Z")
_ZONE_TYPES = frozenset({"full", "partial", "secondary", "internal"})
_DNS_TYPES = frozenset(
    {
        "A",
        "AAAA",
        "CAA",
        "CERT",
        "CNAME",
        "DNSKEY",
        "DS",
        "HTTPS",
        "LOC",
        "MX",
        "NAPTR",
        "NS",
        "OPENPGPKEY",
        "PTR",
        "SMIMEA",
        "SRV",
        "SSHFP",
        "SVCB",
        "TLSA",
        "TXT",
        "URI",
    }
)


def secret_contracts() -> dict[str, dict[str, str]]:
    return {}


def account_contracts() -> dict[str, dict[str, object]]:
    return {
        "cloudflare": {
            "provider": "cloudflare",
            "scopes": ("dns.read", "offline_access", "zone.read"),
        }
    }


def _pagination_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "page": {"type": "integer", "minimum": 1},
            "per_page": {"type": "integer", "minimum": 1, "maximum": 100},
            "count": {"type": "integer", "minimum": 0, "maximum": 100},
            "total_count": {"type": "integer", "minimum": 0},
            "total_pages": {"type": "integer", "minimum": 0},
        },
        "required": ["page", "per_page", "count", "total_count", "total_pages"],
        "additionalProperties": False,
    }


def power_contracts() -> dict[str, dict[str, Any]]:
    pagination_input = {
        "page": {"type": "integer", "minimum": 1, "maximum": 100000},
        "per_page": {"type": "integer", "minimum": 1, "maximum": 100},
    }
    zone = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "pattern": "^[0-9a-f]{32}$"},
            "name": {"type": "string", "minLength": 1, "maxLength": 255},
            "status": {"type": "string", "pattern": "^[a-z][a-z0-9_-]{0,31}$"},
            "type": {"enum": sorted(_ZONE_TYPES)},
            "paused": {"type": "boolean"},
            "account": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "pattern": "^[0-9a-f]{32}$"},
                    "name": {"type": "string", "minLength": 1, "maxLength": 160},
                },
                "required": ["id", "name"],
                "additionalProperties": False,
            },
        },
        "required": ["id", "name", "status", "type", "paused", "account"],
        "additionalProperties": False,
    }
    record = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "pattern": "^[0-9a-f]{32}$"},
            "type": {"enum": sorted(_DNS_TYPES)},
            "name": {"type": "string", "minLength": 1, "maxLength": 255},
            "content": {"type": "string", "minLength": 1, "maxLength": 65535},
            "ttl": {"type": "integer", "minimum": 1, "maximum": 2147483647},
            "proxied": {"type": "boolean"},
            "proxiable": {"type": "boolean"},
        },
        "required": ["id", "type", "name", "content", "ttl", "proxied", "proxiable"],
        "additionalProperties": False,
    }
    return {
        "list-zones": {
            "method": "POST",
            "path": "/v1/powers/list-zones",
            "summary": "List a bounded page of Cloudflare zones and domains.",
            "input_schema": {
                "type": "object",
                "properties": pagination_input,
                "required": ["page", "per_page"],
                "additionalProperties": False,
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "zones": {"type": "array", "maxItems": 100, "items": zone},
                    "pagination": _pagination_schema(),
                },
                "required": ["zones", "pagination"],
                "additionalProperties": False,
            },
            "approval": "none",
            "secrets": (),
            "accounts": ("cloudflare",),
        },
        "list-dns-records": {
            "method": "POST",
            "path": "/v1/powers/list-dns-records",
            "summary": "List a bounded page of DNS records from one Cloudflare zone.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "zone_id": {"type": "string", "pattern": "^[0-9a-f]{32}$"},
                    **pagination_input,
                },
                "required": ["zone_id", "page", "per_page"],
                "additionalProperties": False,
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "records": {"type": "array", "maxItems": 100, "items": record},
                    "pagination": _pagination_schema(),
                },
                "required": ["records", "pagination"],
                "additionalProperties": False,
            },
            "approval": "none",
            "secrets": (),
            "accounts": ("cloudflare",),
        },
    }


def _closed(payload: object, fields: set[str]) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("Power payload does not match its declared fields")
    return payload


def _integer(value: object, minimum: int, maximum: int, field: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{field} is invalid")
    return value


def _identifier(value: object) -> str:
    if not isinstance(value, str) or _HEX_ID.fullmatch(value) is None:
        raise ValueError("Cloudflare identifier is invalid")
    return value


def _text(value: object, maximum: int, field: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        raise ValueError(f"{field} is invalid")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field} is invalid")
    return value


def validate_power_input(assistant_id: str, power: str, payload: object) -> dict[str, object]:
    if assistant_id != ASSISTANT_ID or power not in {"list-zones", "list-dns-records"}:
        raise ValueError("the Power has no declared input contract")
    fields = {"page", "per_page"} | ({"zone_id"} if power == "list-dns-records" else set())
    safe = _closed(payload, fields)
    result: dict[str, object] = {
        "page": _integer(safe["page"], 1, 100000, "page"),
        "per_page": _integer(safe["per_page"], 1, 100, "per_page"),
    }
    if power == "list-dns-records":
        result = {"zone_id": _identifier(safe["zone_id"]), **result}
    return result


def _pagination(payload: object) -> dict[str, object]:
    safe = _closed(payload, {"page", "per_page", "count", "total_count", "total_pages"})
    result = {
        "page": _integer(safe["page"], 1, 100000, "page"),
        "per_page": _integer(safe["per_page"], 1, 100, "per_page"),
        "count": _integer(safe["count"], 0, 100, "count"),
        "total_count": _integer(safe["total_count"], 0, (1 << 63) - 1, "total_count"),
        "total_pages": _integer(safe["total_pages"], 0, (1 << 63) - 1, "total_pages"),
    }
    if result["count"] > result["total_count"]:
        raise ValueError("pagination is invalid")
    return result


def _zone(payload: object) -> dict[str, object]:
    safe = _closed(payload, {"id", "name", "status", "type", "paused", "account"})
    account = _closed(safe["account"], {"id", "name"})
    status = safe["status"]
    zone_type = safe["type"]
    if not isinstance(status, str) or _STATUS.fullmatch(status) is None:
        raise ValueError("status is invalid")
    if not isinstance(zone_type, str) or zone_type not in _ZONE_TYPES or type(safe["paused"]) is not bool:
        raise ValueError("zone is invalid")
    return {
        "id": _identifier(safe["id"]),
        "name": _text(safe["name"], 255, "name"),
        "status": status,
        "type": zone_type,
        "paused": safe["paused"],
        "account": {
            "id": _identifier(account["id"]),
            "name": _text(account["name"], 160, "account name"),
        },
    }


def _record(payload: object) -> dict[str, object]:
    safe = _closed(payload, {"id", "type", "name", "content", "ttl", "proxied", "proxiable"})
    record_type = safe["type"]
    if not isinstance(record_type, str) or record_type not in _DNS_TYPES:
        raise ValueError("DNS record type is invalid")
    if type(safe["proxied"]) is not bool or type(safe["proxiable"]) is not bool:
        raise ValueError("DNS proxy state is invalid")
    return {
        "id": _identifier(safe["id"]),
        "type": record_type,
        "name": _text(safe["name"], 255, "name"),
        "content": _text(safe["content"], 65535, "content"),
        "ttl": _integer(safe["ttl"], 1, 2147483647, "ttl"),
        "proxied": safe["proxied"],
        "proxiable": safe["proxiable"],
    }


def validate_power_output(assistant_id: str, power: str, payload: object) -> dict[str, object]:
    if assistant_id != ASSISTANT_ID or power not in {"list-zones", "list-dns-records"}:
        raise ValueError("the Power has no declared output contract")
    item_field = "zones" if power == "list-zones" else "records"
    safe = _closed(payload, {item_field, "pagination"})
    items = safe[item_field]
    if not isinstance(items, list) or len(items) > 100:
        raise ValueError(f"{item_field} is invalid")
    validator = _zone if power == "list-zones" else _record
    return {item_field: [validator(item) for item in items], "pagination": _pagination(safe["pagination"])}
