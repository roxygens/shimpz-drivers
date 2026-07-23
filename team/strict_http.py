"""Fail-closed HTTP parsing primitives shared by both Team Controllers."""

from __future__ import annotations

import hmac
import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import BinaryIO
from urllib.parse import parse_qsl, urlsplit

MAX_REQUEST_TARGET_BYTES = 512


class HttpContractError(ValueError):
    def __init__(self, status: HTTPStatus, message: str, *, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code


@dataclass(frozen=True)
class RequestTarget:
    path: str
    parts: tuple[str, ...]
    query: dict[str, str]


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def bearer_matches(headers: object, token: str) -> bool:
    """Accept exactly one bearer header and compare it in constant time."""
    values = headers.get_all("Authorization", failobj=[])
    return len(values) == 1 and hmac.compare_digest(values[0], f"Bearer {token}")


def read_json_object(
    headers: object,
    stream: BinaryIO,
    *,
    max_bytes: int,
) -> dict[str, object]:
    """Read one length-delimited, finite, duplicate-free JSON object."""
    if headers.get_all("Transfer-Encoding", failobj=[]):
        raise HttpContractError(
            HTTPStatus.BAD_REQUEST,
            "chunked requests are not accepted",
            code="chunked-request",
        )
    lengths = headers.get_all("Content-Length", failobj=[])
    if len(lengths) != 1:
        raise HttpContractError(
            HTTPStatus.LENGTH_REQUIRED,
            "one Content-Length is required",
            code="content-length",
        )
    try:
        length = int(lengths[0])
    except ValueError as exc:
        raise HttpContractError(
            HTTPStatus.BAD_REQUEST,
            "invalid Content-Length",
            code="content-length",
        ) from exc
    if length < 2 or length > max_bytes:
        raise HttpContractError(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            f"request body is too large (max {max_bytes} bytes)",
            code="body-too-large",
        )
    content_types = headers.get_all("Content-Type", failobj=[])
    if len(content_types) != 1 or content_types[0].partition(";")[0].strip().lower() != "application/json":
        raise HttpContractError(
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            "Content-Type must be application/json",
            code="content-type",
        )
    try:
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError("short request body")
        body = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise HttpContractError(
            HTTPStatus.BAD_REQUEST,
            "invalid JSON body",
            code="invalid-json",
        ) from exc
    if not isinstance(body, dict):
        raise HttpContractError(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            "a JSON object is required",
            code="invalid-body",
        )
    return body


def reject_body(headers: object) -> None:
    """Reject transfer framing or a nonzero body on a bodyless route."""
    if headers.get_all("Transfer-Encoding", failobj=[]):
        raise HttpContractError(
            HTTPStatus.BAD_REQUEST,
            "this request cannot have a body",
            code="unexpected-body",
        )
    lengths = headers.get_all("Content-Length", failobj=[])
    if len(lengths) > 1:
        raise HttpContractError(
            HTTPStatus.BAD_REQUEST,
            "invalid Content-Length",
            code="content-length",
        )
    if not lengths:
        return
    try:
        length = int(lengths[0])
    except ValueError as exc:
        raise HttpContractError(
            HTTPStatus.BAD_REQUEST,
            "invalid Content-Length",
            code="content-length",
        ) from exc
    if length != 0:
        raise HttpContractError(
            HTTPStatus.BAD_REQUEST,
            "this request cannot have a body",
            code="unexpected-body",
        )


def parse_request_target(
    raw_target: str,
    *,
    allow_query: bool,
    max_bytes: int = MAX_REQUEST_TARGET_BYTES,
) -> RequestTarget:
    """Parse one bounded origin-form target without encoded or ambiguous routing."""
    if len(raw_target.encode("utf-8", "replace")) > max_bytes:
        raise HttpContractError(
            HTTPStatus.URI_TOO_LONG,
            "request path is too long",
            code="path-too-long",
        )
    parsed = urlsplit(raw_target)
    if parsed.fragment or "%" in parsed.path or (parsed.query and not allow_query):
        raise HttpContractError(
            HTTPStatus.BAD_REQUEST,
            "query and encoded paths are not accepted",
            code="invalid-path",
        )
    query: dict[str, str] = {}
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if not key or key in query:
            raise HttpContractError(
                HTTPStatus.BAD_REQUEST,
                "request query is ambiguous",
                code="invalid-path",
            )
        query[key] = value
    return RequestTarget(
        path=parsed.path,
        parts=tuple(part for part in parsed.path.split("/") if part),
        query=query,
    )


def parse_routed_request(
    headers: object,
    raw_target: str,
    method: str,
    *,
    body_methods: frozenset[str],
    allow_query: bool,
    max_bytes: int = MAX_REQUEST_TARGET_BYTES,
) -> RequestTarget:
    """Parse a request target and enforce the route table's body-capable methods."""
    target = parse_request_target(raw_target, allow_query=allow_query, max_bytes=max_bytes)
    if method not in body_methods:
        reject_body(headers)
    return target


# Canonical route matching lives beside strict target parsing so adding a Team endpoint cannot make
# the hosted and local Controllers disagree about method/path semantics.
HOSTED_CONTROLLER = "hosted"
LOCAL_CONTROLLER = "local"
_BOTH_CONTROLLERS = frozenset({HOSTED_CONTROLLER, LOCAL_CONTROLLER})


@dataclass(frozen=True, slots=True)
class ControllerRoute:
    method: str
    pattern: tuple[str, ...]
    operation: str
    profiles: frozenset[str] = _BOTH_CONTROLLERS


@dataclass(frozen=True, slots=True)
class ControllerRouteMatch:
    operation: str
    params: dict[str, str]

    @property
    def group(self) -> str | None:
        fixed = {"health", "registry-list", "team-list", "space-reset", "assistant-account-complete"}
        if self.operation in fixed:
            return "fixed"
        if self.operation in {"team-create", "team-destroy"}:
            return "team"
        for prefix, group in (
            ("file-", "file"),
            ("inference-", "inference"),
            ("chat-", "chat"),
            ("assistant-secret-", "assistant-secret"),
            ("assistant-approval-", "assistant-approval"),
            ("assistant-account-", "assistant-account"),
        ):
            if self.operation.startswith(prefix):
                return group
        return "chat" if self.operation == "chat" else None


def _controller_route(
    method: str,
    path: str,
    operation: str,
    profiles: frozenset[str] = _BOTH_CONTROLLERS,
) -> ControllerRoute:
    return ControllerRoute(method, tuple(part for part in path.split("/") if part), operation, profiles)


_HOSTED_CONTROLLER_ONLY = frozenset({HOSTED_CONTROLLER})
_LOCAL_CONTROLLER_ONLY = frozenset({LOCAL_CONTROLLER})
CONTROLLER_ROUTES = (
    _controller_route("GET", "/v1/teams", "team-list"),
    _controller_route("POST", "/v1/oauth/cloudflare/callback", "assistant-account-complete"),
    _controller_route("POST", "/v1/teams/:team_id/create", "team-create"),
    _controller_route("DELETE", "/v1/teams/:team_id", "team-destroy"),
    _controller_route("GET", "/v1/teams/:team_id/files", "file-list"),
    _controller_route("POST", "/v1/teams/:team_id/files", "file-upload"),
    _controller_route("DELETE", "/v1/teams/:team_id/files/:file_id", "file-delete"),
    _controller_route("GET", "/v1/teams/:team_id/inference", "inference-status"),
    _controller_route("PUT", "/v1/teams/:team_id/inference", "inference-configure"),
    _controller_route("POST", "/v1/teams/:team_id/chat", "chat"),
    _controller_route("GET", "/v1/teams/:team_id/chat/accounts", "chat-account-pending"),
    _controller_route("POST", "/v1/teams/:team_id/chat/accounts", "chat-account-submit"),
    _controller_route("GET", "/v1/teams/:team_id/chat/secrets", "chat-secret-pending"),
    _controller_route("POST", "/v1/teams/:team_id/chat/secrets", "chat-secret-submit"),
    _controller_route("POST", "/v1/teams/:team_id/chat/stop", "chat-stop"),
    _controller_route("GET", "/v1/teams/:team_id/assistant-secrets", "assistant-secret-list"),
    _controller_route("PUT", "/v1/teams/:team_id/assistant-secrets", "assistant-secret-replace"),
    _controller_route("GET", "/v1/teams/:team_id/assistant-accounts", "assistant-account-list"),
    _controller_route(
        "POST",
        "/v1/teams/:team_id/assistant-accounts/challenges/:challenge_id/authorize",
        "assistant-account-authorize",
    ),
    _controller_route(
        "DELETE",
        "/v1/teams/:team_id/assistant-accounts/:assistant_id/:account_id",
        "assistant-account-disconnect",
    ),
    _controller_route("GET", "/v1/teams/:team_id/assistants/:assistant_id/help", "assistant-help"),
    _controller_route("GET", "/v1/teams/:team_id/assistants/:assistant_id/help/:locale", "assistant-help"),
    _controller_route("POST", "/v1/teams/:team_id/chat/stream", "chat-stream", _HOSTED_CONTROLLER_ONLY),
    _controller_route("GET", "/v1/teams/:team_id/apps", "app-list", _HOSTED_CONTROLLER_ONLY),
    _controller_route("POST", "/v1/teams/:team_id/apps", "app-install", _HOSTED_CONTROLLER_ONLY),
    _controller_route("DELETE", "/v1/teams/:team_id/apps/:app_id", "app-uninstall", _HOSTED_CONTROLLER_ONLY),
    _controller_route("GET", "/v1/teams/:team_id/status", "team-status", _HOSTED_CONTROLLER_ONLY),
    _controller_route("GET", "/v1/teams/:team_id/logs", "team-logs", _HOSTED_CONTROLLER_ONLY),
    _controller_route("POST", "/v1/teams/:team_id/stop", "team-stop", _HOSTED_CONTROLLER_ONLY),
    _controller_route("POST", "/v1/teams/:team_id/start", "team-start", _HOSTED_CONTROLLER_ONLY),
    _controller_route("POST", "/v1/teams/:team_id/restart", "team-restart", _HOSTED_CONTROLLER_ONLY),
    _controller_route("GET", "/healthz", "health", _LOCAL_CONTROLLER_ONLY),
    _controller_route("GET", "/v1/assistants", "registry-list", _LOCAL_CONTROLLER_ONLY),
    _controller_route("DELETE", "/v1/space", "space-reset", _LOCAL_CONTROLLER_ONLY),
    _controller_route("GET", "/v1/teams/:team_id/assistants", "assistant-list", _LOCAL_CONTROLLER_ONLY),
    _controller_route("POST", "/v1/teams/:team_id/assistants", "assistant-install", _LOCAL_CONTROLLER_ONLY),
    _controller_route(
        "DELETE",
        "/v1/teams/:team_id/assistants/:assistant_id",
        "assistant-uninstall",
        _LOCAL_CONTROLLER_ONLY,
    ),
    _controller_route(
        "POST",
        "/v1/teams/:team_id/assistants/:assistant_id/powers/:power_id",
        "assistant-invoke",
        _LOCAL_CONTROLLER_ONLY,
    ),
    _controller_route(
        "GET",
        "/v1/teams/:team_id/assistant-approvals",
        "assistant-approval-list",
        _LOCAL_CONTROLLER_ONLY,
    ),
    _controller_route(
        "DELETE",
        "/v1/teams/:team_id/assistant-approvals",
        "assistant-approval-revoke",
        _LOCAL_CONTROLLER_ONLY,
    ),
    _controller_route("GET", "/v1/teams/:team_id/chat/approval", "chat-approval-pending"),
    _controller_route("POST", "/v1/teams/:team_id/chat/approval", "chat-approval-submit"),
    _controller_route("GET", "/v1/teams/:team_id/chat/input", "chat-input-pending"),
    _controller_route("POST", "/v1/teams/:team_id/chat/input", "chat-input-submit"),
)


def resolve_controller_route(profile: str, method: str, parts: tuple[str, ...]) -> ControllerRouteMatch | None:
    """Resolve one exact origin-form path without wildcard suffixes or method fallthrough."""
    if profile not in {HOSTED_CONTROLLER, LOCAL_CONTROLLER}:
        raise ValueError("unknown Controller routing profile")
    for route in CONTROLLER_ROUTES:
        if profile not in route.profiles or method != route.method or len(parts) != len(route.pattern):
            continue
        params: dict[str, str] = {}
        for expected, actual in zip(route.pattern, parts, strict=True):
            if expected.startswith(":"):
                params[expected[1:]] = actual
            elif expected != actual:
                break
        else:
            return ControllerRouteMatch(route.operation, params)
    return None
