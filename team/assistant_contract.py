"""Closed first-party contract for the Shimpz Assistant reference artifact.

The hosted and single-owner controllers deliberately share this module so the
Brain never sees a Power that one runtime validates differently from the other.
Artifact bytes and Docker policy remain controller-owned release data.
"""

from __future__ import annotations

import math
from typing import Any

ASSISTANT_ID = "shimpz-assistant"
ASSISTANT_NAME = "Shimpz Assistant"
ASSISTANT_SUMMARY = "Search places and inspect current and forecast weather through Open-Meteo."
ASSISTANT_RPC_COMMAND = "/usr/local/bin/shimpz-assistant-rpc"
ASSISTANT_EGRESS = (
    "api.open-meteo.com",
    "geocoding-api.open-meteo.com",
)
MAX_HELP_BYTES = 32 * 1024
HELP_LOCALES = frozenset({"en", "pt", "es", "zh", "fr", "de", "ja", "ar"})


def power_contracts() -> dict[str, dict[str, Any]]:
    """Return fresh closed schemas so callers cannot mutate another registry."""
    coordinates = {
        "latitude": {"type": "number", "minimum": -90, "maximum": 90},
        "longitude": {"type": "number", "minimum": -180, "maximum": 180},
    }
    return {
        "search-location": {
            "method": "POST",
            "path": "/v1/powers/search-location",
            "summary": "Find geographic coordinates for a city or postal code.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 2, "maxLength": 100},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "locations": {
                        "type": "array",
                        "maxItems": 10,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "minLength": 1, "maxLength": 160},
                                "country": {"type": "string", "maxLength": 120},
                                "latitude": coordinates["latitude"],
                                "longitude": coordinates["longitude"],
                                "timezone": {"type": "string", "minLength": 1, "maxLength": 100},
                            },
                            "required": ["name", "country", "latitude", "longitude", "timezone"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["locations"],
                "additionalProperties": False,
            },
            "approval": "none",
        },
        "current-weather": {
            "method": "POST",
            "path": "/v1/powers/current-weather",
            "summary": "Read the current weather for one coordinate.",
            "input_schema": {
                "type": "object",
                "properties": coordinates,
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "observed_at": {"type": "string", "minLength": 1, "maxLength": 64},
                    "temperature_c": {"type": "number"},
                    "apparent_temperature_c": {"type": "number"},
                    "wind_speed_kmh": {"type": "number", "minimum": 0},
                    "weather_code": {"type": "integer", "minimum": 0, "maximum": 999},
                    "timezone": {"type": "string", "minLength": 1, "maxLength": 100},
                },
                "required": [
                    "observed_at",
                    "temperature_c",
                    "apparent_temperature_c",
                    "wind_speed_kmh",
                    "weather_code",
                    "timezone",
                ],
                "additionalProperties": False,
            },
            "approval": "none",
        },
        "daily-forecast": {
            "method": "POST",
            "path": "/v1/powers/daily-forecast",
            "summary": "Read a daily weather forecast for one coordinate.",
            "input_schema": {
                "type": "object",
                "properties": {
                    **coordinates,
                    "days": {"type": "integer", "minimum": 1, "maximum": 16},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "minLength": 1, "maxLength": 100},
                    "days": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 16,
                        "items": {
                            "type": "object",
                            "properties": {
                                "date": {"type": "string", "minLength": 1, "maxLength": 32},
                                "temperature_min_c": {"type": "number"},
                                "temperature_max_c": {"type": "number"},
                                "precipitation_probability_max": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 100,
                                },
                                "weather_code": {"type": "integer", "minimum": 0, "maximum": 999},
                            },
                            "required": [
                                "date",
                                "temperature_min_c",
                                "temperature_max_c",
                                "precipitation_probability_max",
                                "weather_code",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["timezone", "days"],
                "additionalProperties": False,
            },
            "approval": "none",
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


def _number(value: object, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("coordinate is invalid")
    result = float(value)
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise ValueError("coordinate is invalid")
    return result


def _integer(value: object, *, minimum: int, maximum: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{field} is invalid")
    return value


def _coordinates(payload: dict[str, object]) -> tuple[float, float]:
    return (
        _number(payload["latitude"], minimum=-90, maximum=90),
        _number(payload["longitude"], minimum=-180, maximum=180),
    )


def validate_power_input(assistant_id: str, power: str, payload: object) -> dict[str, object]:
    if assistant_id != ASSISTANT_ID:
        raise ValueError("the Power has no declared input contract")
    if power == "search-location":
        safe = _closed_object(payload, {"query", "limit"}, required={"query"})
        raw_query = safe["query"]
        if not isinstance(raw_query, str):
            raise ValueError("query is invalid")
        query = _bounded_text(raw_query.strip(), minimum=2, maximum=100, field="query")
        limit = _integer(safe.get("limit", 5), minimum=1, maximum=10, field="limit")
        return {"query": query, "limit": limit}
    if power == "current-weather":
        safe = _closed_object(payload, {"latitude", "longitude"}, required={"latitude", "longitude"})
        latitude, longitude = _coordinates(safe)
        return {"latitude": latitude, "longitude": longitude}
    if power == "daily-forecast":
        safe = _closed_object(
            payload,
            {"latitude", "longitude", "days"},
            required={"latitude", "longitude"},
        )
        latitude, longitude = _coordinates(safe)
        days = _integer(safe.get("days", 7), minimum=1, maximum=16, field="days")
        return {"latitude": latitude, "longitude": longitude, "days": days}
    raise ValueError("the Power has no declared input contract")


def _location(value: object) -> dict[str, object]:
    safe = _closed_object(
        value,
        {"name", "country", "latitude", "longitude", "timezone"},
        required={"name", "country", "latitude", "longitude", "timezone"},
    )
    latitude, longitude = _coordinates(safe)
    country = _bounded_text(safe["country"], minimum=0, maximum=120, field="country")
    return {
        "name": _bounded_text(safe["name"], minimum=1, maximum=160, field="name"),
        "country": country,
        "latitude": latitude,
        "longitude": longitude,
        "timezone": _bounded_text(safe["timezone"], minimum=1, maximum=100, field="timezone"),
    }


def _finite_measurement(value: object, *, minimum: float | None = None) -> float:
    return _number(value, minimum=minimum)


def validate_power_output(assistant_id: str, power: str, payload: object) -> dict[str, object]:
    if assistant_id != ASSISTANT_ID:
        raise ValueError("the Power has no declared output contract")
    if power == "search-location":
        safe = _closed_object(payload, {"locations"}, required={"locations"})
        locations = safe["locations"]
        if not isinstance(locations, list) or len(locations) > 10:
            raise ValueError("locations are invalid")
        return {"locations": [_location(location) for location in locations]}
    if power == "current-weather":
        keys = {
            "observed_at",
            "temperature_c",
            "apparent_temperature_c",
            "wind_speed_kmh",
            "weather_code",
            "timezone",
        }
        safe = _closed_object(payload, keys, required=keys)
        return {
            "observed_at": _bounded_text(safe["observed_at"], minimum=1, maximum=64, field="observed_at"),
            "temperature_c": _finite_measurement(safe["temperature_c"]),
            "apparent_temperature_c": _finite_measurement(safe["apparent_temperature_c"]),
            "wind_speed_kmh": _finite_measurement(safe["wind_speed_kmh"], minimum=0),
            "weather_code": _integer(safe["weather_code"], minimum=0, maximum=999, field="weather_code"),
            "timezone": _bounded_text(safe["timezone"], minimum=1, maximum=100, field="timezone"),
        }
    if power == "daily-forecast":
        safe = _closed_object(payload, {"timezone", "days"}, required={"timezone", "days"})
        days = safe["days"]
        if not isinstance(days, list) or not 1 <= len(days) <= 16:
            raise ValueError("forecast days are invalid")
        normalized_days: list[dict[str, object]] = []
        keys = {
            "date",
            "temperature_min_c",
            "temperature_max_c",
            "precipitation_probability_max",
            "weather_code",
        }
        for day in days:
            item = _closed_object(day, keys, required=keys)
            minimum = _finite_measurement(item["temperature_min_c"])
            maximum = _finite_measurement(item["temperature_max_c"])
            if minimum > maximum:
                raise ValueError("forecast temperatures are invalid")
            normalized_days.append(
                {
                    "date": _bounded_text(item["date"], minimum=1, maximum=32, field="date"),
                    "temperature_min_c": minimum,
                    "temperature_max_c": maximum,
                    "precipitation_probability_max": _integer(
                        item["precipitation_probability_max"],
                        minimum=0,
                        maximum=100,
                        field="precipitation_probability_max",
                    ),
                    "weather_code": _integer(item["weather_code"], minimum=0, maximum=999, field="weather_code"),
                }
            )
        return {
            "timezone": _bounded_text(safe["timezone"], minimum=1, maximum=100, field="timezone"),
            "days": normalized_days,
        }
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
