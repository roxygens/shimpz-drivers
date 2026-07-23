"""Bounded HTTP adapter for the local Team controller."""

import base64
import binascii
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from docker.errors import DockerException
from http_boundary import local
from http_boundary import strict as strict_http

from local_support import audit as local_audit
from local_support.errors import ApiProblemError as ApiProblem
from local_support.validation import (
    validate_model_credential_headers,
    validate_team_id,
    validate_team_name,
)

MAX_BODY_BYTES = 16 * 1024
MAX_CHAT_BODY_BYTES = 24 * 1024
MAX_SECRET_BODY_BYTES = 512 * 1024
MAX_API_RESPONSE_BYTES = 128 * 1024
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_FILE_BODY_BYTES = 4 * ((MAX_UPLOAD_BYTES + 2) // 3) + 8192
MAX_PATH_BYTES = 512
REQUEST_TIMEOUT_SECONDS = 10
CHAT_PAUSED_STATUSES = frozenset({"accounts-required", "secrets-required", "input-required", "approval-required"})
_FILE_UPLOAD_SLOTS = threading.BoundedSemaphore(1)


class BoundedServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 32

    def __init__(self, address, handler, controller: object, token: str) -> None:
        super().__init__(address, handler)
        self.controller = controller
        self.token = token
        self._slots = threading.BoundedSemaphore(16)

    def process_request(self, request, client_address) -> None:
        if not self._slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


class Handler(BaseHTTPRequestHandler):
    server: BoundedServer
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args) -> None:
        return

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(REQUEST_TIMEOUT_SECONDS)

    def _authorized(self) -> bool:
        return strict_http.bearer_matches(self.headers, self.server.token)

    def _send(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
        if len(encoded) > MAX_API_RESPONSE_BYTES:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            encoded = b'{"error":"response exceeded its limit"}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        if status == HTTPStatus.UNAUTHORIZED:
            self.send_header("WWW-Authenticate", 'Bearer realm="shimpz-local"')
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(encoded)

    def _body(self, *, max_bytes: int = MAX_BODY_BYTES) -> dict[str, object]:
        try:
            return strict_http.read_json_object(
                self.headers,
                self.rfile,
                max_bytes=max_bytes,
            )
        except strict_http.HttpContractError as exc:
            raise ApiProblem(exc.status, exc.message, code=exc.code) from exc

    def _file_body(self) -> tuple[object, bytes, object]:
        body = self._body(max_bytes=MAX_FILE_BODY_BYTES)
        if set(body) not in ({"filename", "content_b64"}, {"filename", "content_b64", "media_type"}):
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "file upload requires filename, content_b64, and optional media_type",
                code="invalid-body",
            )
        encoded = body["content_b64"]
        if not isinstance(encoded, str):
            raise ApiProblem(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid file content", code="invalid-file")
        try:
            content = base64.b64decode(encoded, validate=True)
        except (binascii.Error, UnicodeError, ValueError) as exc:
            raise ApiProblem(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid file content", code="invalid-file") from exc
        if not content or len(content) > MAX_UPLOAD_BYTES:
            raise ApiProblem(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"file must contain 1 to {MAX_UPLOAD_BYTES} bytes",
                code="file-too-large",
            )
        return body["filename"], content, body.get("media_type")

    def _team_create_body(self) -> str:
        body = self._body()
        if set(body) != {"team_name"}:
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "Team creation requires only team_name",
                code="invalid-body",
            )
        return validate_team_name(body["team_name"])

    def _install_body(self) -> str:
        body = self._body()
        if set(body) != {"assistant"} or not isinstance(body["assistant"], str):
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "assistant must identify one allowlisted Assistant",
                code="invalid-body",
            )
        return body["assistant"]

    def _model_credential_headers(self) -> tuple[str, str]:
        return validate_model_credential_headers(
            self.headers.get_all("X-Shimpz-Model-Provider", failobj=[]),
            self.headers.get_all("X-Shimpz-Model-Api-Key", failobj=[]),
        )

    def _reject_body(self) -> None:
        try:
            strict_http.reject_body(self.headers)
        except strict_http.HttpContractError as exc:
            raise ApiProblem(exc.status, exc.message, code=exc.code) from exc

    def _path_parts(self) -> list[str]:
        try:
            return list(
                strict_http.parse_request_target(
                    self.path,
                    allow_query=False,
                    max_bytes=MAX_PATH_BYTES,
                ).parts
            )
        except strict_http.HttpContractError as exc:
            raise ApiProblem(exc.status, exc.message, code=exc.code) from exc

    def _fixed_route(
        self, parts: list[str]
    ) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        controller = self.server.controller
        if self.command == "GET" and parts == ["healthz"]:
            return HTTPStatus.OK, controller.health(), "health", None, None
        if self.command == "GET" and parts == ["v1", "assistants"]:
            return HTTPStatus.OK, controller.list_registry(), "registry-list", None, None
        if self.command == "GET" and parts == ["v1", "teams"]:
            return HTTPStatus.OK, controller.list_teams(), "team-list", None, None
        if self.command == "DELETE" and parts == ["v1", "space"]:
            return HTTPStatus.OK, controller.reset_space(), "space-reset", None, None
        if self.command == "POST" and parts == ["v1", "oauth", "cloudflare", "callback"]:
            body = self._body()
            if set(body) != {"state", "claim", "session_binding"}:
                raise ApiProblem(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "OAuth callback is invalid",
                    code="invalid-body",
                )
            result = controller.complete_cloudflare_oauth_callback(
                state=body["state"],
                claim=body["claim"],
                session_binding=body["session_binding"],
            )
            return HTTPStatus.OK, result, "assistant-account-complete", None, None
        return None

    def _file_route(self, parts: list[str]) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        if len(parts) not in {4, 5} or parts[:2] != ["v1", "teams"] or parts[3] != "files":
            return None
        controller = self.server.controller
        team_id = validate_team_id(parts[2])
        if len(parts) == 4 and self.command == "GET":
            return HTTPStatus.OK, controller.list_files(team_id), "file-list", team_id, None
        if len(parts) == 4 and self.command == "POST":
            if not _FILE_UPLOAD_SLOTS.acquire(blocking=False):
                raise ApiProblem(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    "another Team file upload is in progress",
                    code="file-upload-busy",
                )
            try:
                filename, content, media_type = self._file_body()
                return (
                    HTTPStatus.OK,
                    controller.put_file(team_id, filename, content, media_type),
                    "file-upload",
                    team_id,
                    None,
                )
            finally:
                _FILE_UPLOAD_SLOTS.release()
        if len(parts) == 5 and self.command == "DELETE":
            return (
                HTTPStatus.OK,
                controller.delete_file(team_id, parts[4]),
                "file-delete",
                team_id,
                None,
            )
        return None

    def _inference_route(
        self, parts: list[str]
    ) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        if len(parts) != 4 or parts[:2] != ["v1", "teams"] or parts[3] != "inference":
            return None
        team_id = validate_team_id(parts[2])
        if self.command == "GET":
            return (
                HTTPStatus.OK,
                self.server.controller.inference_status(team_id),
                "inference-status",
                team_id,
                None,
            )
        if self.command == "PUT":
            return (
                HTTPStatus.OK,
                self.server.controller.configure_inference(team_id, self._body()),
                "inference-configure",
                team_id,
                None,
            )
        return None

    @staticmethod
    def _chat_status(payload: dict[str, object]) -> HTTPStatus:
        return HTTPStatus.PRECONDITION_REQUIRED if payload.get("status") in CHAT_PAUSED_STATUSES else HTTPStatus.OK

    def _chat_start(
        self,
        team_id: str,
    ) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None]:
        provider, api_key = self._model_credential_headers()
        body = self._body(max_bytes=MAX_CHAT_BODY_BYTES)
        payload = self.server.controller.chat(team_id, body, provider, api_key)
        return self._chat_status(payload), payload, "chat", team_id, None

    def _chat_pending(
        self,
        team_id: str,
        segment: str,
    ) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        pending = {
            "accounts": ("pending_chat_accounts", "chat-account-pending"),
            "secrets": ("pending_chat_secrets", "chat-secret-pending"),
            "approval": ("pending_chat_approval", "chat-approval-pending"),
            "input": ("pending_chat_input", "chat-input-pending"),
        }.get(segment)
        if pending is None:
            return None
        method_name, operation_name = pending
        operation = getattr(self.server.controller, method_name)
        return HTTPStatus.OK, operation(team_id), operation_name, team_id, None

    def _chat_submit(
        self,
        team_id: str,
        segment: str,
    ) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        submission = {
            "accounts": ("resume_chat_accounts", "chat-account-submit", MAX_BODY_BYTES),
            "secrets": ("submit_chat_secrets", "chat-secret-submit", MAX_SECRET_BODY_BYTES),
            "input": ("submit_chat_input", "chat-input-submit", MAX_SECRET_BODY_BYTES),
            "approval": ("submit_chat_approval", "chat-approval-submit", MAX_SECRET_BODY_BYTES),
        }.get(segment)
        if submission is None:
            return None
        method_name, operation_name, max_bytes = submission
        operation = getattr(self.server.controller, method_name)
        provider, api_key = self._model_credential_headers()
        payload = operation(team_id, self._body(max_bytes=max_bytes), provider, api_key)
        return self._chat_status(payload), payload, operation_name, team_id, None

    def _chat_stop(
        self,
        team_id: str,
    ) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None]:
        if self._body() != {}:
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "chat stop requires an empty object",
                code="invalid-body",
            )
        return HTTPStatus.OK, self.server.controller.stop_chat(team_id), "chat-stop", team_id, None

    def _chat_route(
        self,
        parts: list[str],
    ) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        if len(parts) not in {4, 5} or parts[:2] != ["v1", "teams"] or parts[3] != "chat":
            return None
        team_id = validate_team_id(parts[2])
        if len(parts) == 4:
            return self._chat_start(team_id) if self.command == "POST" else None
        segment = parts[4]
        if self.command == "GET":
            return self._chat_pending(team_id, segment)
        if self.command == "POST" and segment == "stop":
            return self._chat_stop(team_id)
        return self._chat_submit(team_id, segment) if self.command == "POST" else None

    def _assistant_secret_route(
        self,
        parts: list[str],
    ) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        if len(parts) != 4 or parts[:2] != ["v1", "teams"] or parts[3] != "assistant-secrets":
            return None
        team_id = validate_team_id(parts[2])
        if self.command == "GET":
            return (
                HTTPStatus.OK,
                self.server.controller.list_assistant_secrets(team_id),
                "assistant-secret-list",
                team_id,
                None,
            )
        if self.command == "PUT":
            return (
                HTTPStatus.OK,
                self.server.controller.replace_assistant_secrets(
                    team_id,
                    self._body(max_bytes=MAX_SECRET_BODY_BYTES),
                ),
                "assistant-secret-replace",
                team_id,
                None,
            )
        return None

    def _assistant_approval_route(
        self,
        parts: list[str],
    ) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        if len(parts) != 4 or parts[:2] != ["v1", "teams"] or parts[3] != "assistant-approvals":
            return None
        team_id = validate_team_id(parts[2])
        if self.command == "GET":
            return (
                HTTPStatus.OK,
                self.server.controller.list_assistant_approval_grants(team_id),
                "assistant-approval-list",
                team_id,
                None,
            )
        if self.command == "DELETE":
            return (
                HTTPStatus.OK,
                self.server.controller.revoke_assistant_approval_grants(team_id),
                "assistant-approval-revoke",
                team_id,
                None,
            )
        return None

    def _assistant_account_route(
        self,
        parts: list[str],
    ) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        if len(parts) < 4 or parts[:2] != ["v1", "teams"] or parts[3] != "assistant-accounts":
            return None
        team_id = validate_team_id(parts[2])
        if len(parts) == 4 and self.command == "GET":
            return (
                HTTPStatus.OK,
                self.server.controller.list_assistant_accounts(team_id),
                "assistant-account-list",
                team_id,
                None,
            )
        if len(parts) == 7 and parts[4] == "challenges" and parts[6] == "authorize" and self.command == "POST":
            body = self._body()
            if set(body) != {"session_binding"}:
                raise ApiProblem(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "OAuth authorization is invalid",
                    code="invalid-body",
                )
            return (
                HTTPStatus.OK,
                self.server.controller.start_assistant_account_authorization(
                    team_id,
                    parts[5],
                    body["session_binding"],
                ),
                "assistant-account-authorize",
                team_id,
                None,
            )
        if len(parts) == 6 and self.command == "DELETE":
            return (
                HTTPStatus.OK,
                self.server.controller.disconnect_assistant_account(
                    team_id,
                    parts[4],
                    parts[5],
                ),
                "assistant-account-disconnect",
                team_id,
                parts[4],
            )
        return None

    def _team_route(self, parts: list[str]) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        if len(parts) == 4 and parts[:2] == ["v1", "teams"] and parts[3] == "create":
            team_id = validate_team_id(parts[2])
            if self.command == "POST":
                return (
                    HTTPStatus.OK,
                    self.server.controller.create_team(team_id, self._team_create_body()),
                    "team-create",
                    team_id,
                    None,
                )
        if len(parts) == 3 and parts[:2] == ["v1", "teams"] and self.command == "DELETE":
            team_id = validate_team_id(parts[2])
            return (
                HTTPStatus.OK,
                self.server.controller.destroy_team(team_id),
                "team-destroy",
                team_id,
                None,
            )
        return None

    def _route(self) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None]:
        parts = self._path_parts()
        controller = self.server.controller
        if self.command not in {"POST", "PUT"}:
            self._reject_body()
        route = strict_http.resolve_controller_route(strict_http.LOCAL_CONTROLLER, self.command, tuple(parts))
        if route is None:
            raise ApiProblem(HTTPStatus.NOT_FOUND, "route not found", code="route-not-found")

        operation = route.operation
        grouped_resolver = {
            "fixed": self._fixed_route,
            "file": self._file_route,
            "inference": self._inference_route,
            "chat": self._chat_route,
            "assistant-secret": self._assistant_secret_route,
            "assistant-approval": self._assistant_approval_route,
            "assistant-account": self._assistant_account_route,
            "team": self._team_route,
        }.get(route.group)
        if grouped_resolver is not None:
            result = grouped_resolver(parts)
            if result is None:
                raise AssertionError("canonical local route group was not dispatched")
            return result

        team_id = validate_team_id(route.params["team_id"])
        if operation == "assistant-list":
            return HTTPStatus.OK, controller.list_assistants(team_id), operation, team_id, None
        if operation == "assistant-install":
            assistant_id = self._install_body()
            return (
                HTTPStatus.OK,
                controller.install_assistant(team_id, assistant_id),
                operation,
                team_id,
                assistant_id,
            )
        assistant_id = route.params["assistant_id"]
        if operation == "assistant-uninstall":
            return (
                HTTPStatus.OK,
                controller.uninstall_assistant(team_id, assistant_id),
                operation,
                team_id,
                assistant_id,
            )
        if operation == "assistant-help":
            return (
                HTTPStatus.OK,
                controller.assistant_help(team_id, assistant_id, route.params.get("locale", "en")),
                operation,
                team_id,
                assistant_id,
            )
        if operation == "assistant-invoke":
            return (
                HTTPStatus.OK,
                controller.invoke(team_id, assistant_id, route.params["power_id"], self._body()),
                operation,
                team_id,
                assistant_id,
            )
        raise AssertionError("canonical local route was not dispatched")

    def _handle(self) -> None:
        self.close_connection = True
        if not self._authorized():
            trace_id = local_audit.record("authentication", result="denied", detail="invalid-bearer")
            self._send(HTTPStatus.UNAUTHORIZED, {"error": "authentication required", "trace_id": trace_id})
            return

        local.dispatch_route(
            self._route,
            local_audit.record,
            self._send,
            ApiProblem,
            DockerException,
        )

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_DELETE(self) -> None:
        self._handle()

    def do_HEAD(self) -> None:
        self._handle()

    def do_OPTIONS(self) -> None:
        self._handle()

    def do_PATCH(self) -> None:
        self._handle()

    def do_PUT(self) -> None:
        self._handle()
