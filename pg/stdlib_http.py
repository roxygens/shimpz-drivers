"""Small fail-closed primitives for stdlib HTTP control-plane services."""

from __future__ import annotations

import hmac
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from http import HTTPStatus
from typing import BinaryIO
from urllib.parse import parse_qs, urlsplit


class HttpError(ValueError):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class Route:
    method: str
    pattern: re.Pattern[str]
    operation: str


@dataclass(frozen=True)
class RouteMatch:
    operation: str
    params: dict[str, str]
    query: dict[str, list[str]]


@dataclass(frozen=True)
class HttpFailure:
    status: HTTPStatus
    public_message: str
    audit_reason: str
    result: str


def bearer_token(headers: object) -> str:
    """Return one exactly framed bearer value, otherwise the empty string."""
    get_all = getattr(headers, "get_all", None)
    values = get_all("Authorization", failobj=[]) if get_all is not None else []
    if len(values) != 1:
        return ""
    scheme, separator, value = values[0].partition(" ")
    return value if separator and scheme == "Bearer" else ""


def bearer_authorized(headers: object, token: str) -> bool:
    """Accept one exact bearer header and compare it in constant time."""
    supplied = bearer_token(headers)
    return bool(supplied) and hmac.compare_digest(supplied, token)


def send_json(handler: object, status: HTTPStatus, payload: object) -> None:
    body = json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json_body(headers: object, stream: BinaryIO, *, max_bytes: int) -> dict[str, object]:
    raw_length = headers.get("Content-Length", "0") or "0"
    try:
        length = int(raw_length)
    except ValueError as exc:
        raise HttpError(HTTPStatus.BAD_REQUEST, "invalid Content-Length") from exc
    if length < 0 or length > max_bytes:
        raise HttpError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body is too large")
    if length == 0:
        return {}
    try:
        body = json.loads(stream.read(length))
    except json.JSONDecodeError as exc:
        raise HttpError(HTTPStatus.BAD_REQUEST, "invalid JSON body") from exc
    if not isinstance(body, dict):
        raise HttpError(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
    return body


def resolve_route(routes: Iterable[Route], method: str, raw_target: str) -> RouteMatch:
    target = urlsplit(raw_target)
    for route in routes:
        if route.method != method or (matched := route.pattern.fullmatch(target.path)) is None:
            continue
        return RouteMatch(route.operation, matched.groupdict(), parse_qs(target.query))
    raise HttpError(HTTPStatus.NOT_FOUND, f"no route for {method} {target.path}")


def dispatch(
    action: Callable[[], None],
    *,
    classify: Callable[[Exception], HttpFailure | None],
    emit: Callable[[HttpFailure], None],
    unexpected_message: str,
) -> None:
    """Run one HTTP action and redact every unclassified ordinary exception."""
    try:
        action()
    except Exception as exc:
        failure = classify(exc)
        if failure is None:
            failure = HttpFailure(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                unexpected_message,
                type(exc).__name__,
                "error",
            )
        emit(failure)
