"""Bounded HTTP transport and route dispatch for the hosted Team controller."""

from __future__ import annotations

import contextlib
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType

import accounts_client
import audit
import brain_runtime_token_store
import docker
import marketplace
import validate
from assistant_human import approval_flow as assistant_approval_flow
from assistant_human import input_flow as assistant_input_flow

from http_boundary import hosted, stdlib
from http_boundary import strict as strict_http

_controller: ModuleType


def bind_controller(controller: ModuleType) -> None:
    global _controller
    _controller = controller


class _BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Thread-per-request server with hard admission and slow-client expiry."""

    daemon_threads = True

    def __init__(self, *args, max_concurrency: int | None = None, **kwargs) -> None:
        concurrency = _controller.MAX_HTTP_CONCURRENCY if max_concurrency is None else max_concurrency
        self._request_slots = threading.BoundedSemaphore(concurrency)
        super().__init__(*args, **kwargs)

    def get_request(self):
        request, client_address = super().get_request()
        request.settimeout(_controller.HTTP_CONNECTION_TIMEOUT_SECONDS)
        return request, client_address

    def process_request(self, request, client_address) -> None:
        # Backpressure happens before a thread exists. At the ceiling, at most the kernel's bounded
        # listen backlog plus this accepted socket waits; Python thread count cannot grow unbounded.
        self._request_slots.acquire()
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._request_slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()


class Handler(BaseHTTPRequestHandler):
    server_version = "team-driver/1.0"

    def log_message(self, *_args) -> None:  # audit.log is the ONLY log source
        pass

    def _principal(self) -> tuple[str, str | None] | None:
        """('operator', None) for the admin bearer; ('account', <id>) for a valid account token; else None.

        The operator token (the admin panel) has full access. A store-forwarded account token is verified
        against the accounts service and scopes every op to that account's OWN teams — the store holds
        no privileged secret, this driver is the enforcer.
        """
        if strict_http.bearer_matches(self.headers, _controller._token):
            return ("operator", None)
        account_token = self.headers.get("X-Shimpz-Account", "")
        if account_token:
            account_id = accounts_client.verify(account_token)
            if account_id:
                return ("account", account_id)
        return None

    def _send_json(self, status: HTTPStatus, payload: dict, *, no_store: bool = False) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if no_store:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream_chat(
        self,
        team_id: str,
        message: str,
        file_ids: object,
        assistant_ids: tuple[str, ...],
        lease: _controller._AuthorizationLease,
    ) -> None:
        """Preserve the NDJSON transport while exposing only the validated terminal reply."""
        terminal: dict[str, object]
        stream_error = None
        with _controller._exclusive_chat_turn(team_id, lease) as (token, container):
            pending = _controller._pending_hosted_chat(team_id)
            if pending is not None:
                self._send_json(
                    HTTPStatus.PRECONDITION_REQUIRED,
                    pending,
                    no_store=True,
                )
                return
                # The durable token is claimed before a 200 or any response byte reaches the client.
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            def emit(obj: dict) -> None:
                line = (json.dumps(obj, ensure_ascii=False) + "\n").encode()
                self.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n")
                self.wfile.flush()

            try:
                result = _controller._chat_in_turn(
                    team_id,
                    message,
                    file_ids,
                    assistant_ids,
                    token,
                    container,
                    lease.owner,
                )
                paused = result.get("status") in _controller.CHAT_PAUSED_STATUSES
                terminal = (
                    {"type": str(result["status"]), **result}
                    if paused
                    else {
                        "type": "done",
                        "reply": result["reply"],
                        "team_id": result["team_id"],
                        "team_name": result["team_name"],
                    }
                )
                emit(terminal)
            except _controller.ApiError as exc:
                terminal = (
                    {"type": "stopped"}
                    if exc.status == HTTPStatus.CONFLICT and exc.message == "brain turn stopped"
                    else {"type": "error", "status": int(exc.status), "detail": exc.message}
                )
                with contextlib.suppress(OSError):
                    emit(terminal)
            except (docker.errors.DockerException, OSError) as exc:
                stream_error = type(exc).__name__
                terminal = {"type": "error", "status": 500, "detail": "brain stream failed"}
                with contextlib.suppress(OSError):
                    emit(terminal)
            finally:
                with contextlib.suppress(OSError):
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
        audit.log(
            "chat",
            team_id,
            result="ok" if terminal["type"] in {"done", "accounts-required", "secrets-required"} else "error",
            streamed=True,
            status=terminal.get("status"),
            reason=stream_error,
        )

    def _read_body(self, *, max_bytes: int | None = None) -> dict:
        try:
            return strict_http.read_json_object(
                self.headers,
                self.rfile,
                max_bytes=_controller.MAX_JSON_BODY_BYTES if max_bytes is None else max_bytes,
            )
        except strict_http.HttpContractError as exc:
            raise _controller.ApiError(exc.status, exc.message) from exc

    def _read_driver_body(self, keys: set[str]) -> dict[str, object]:
        """Read one closed Driver mutation document; arbitrary scripts/shapes never cross the bridge."""
        body = self._read_body(max_bytes=_controller.MAX_DRIVER_JSON_BODY_BYTES)
        if not isinstance(body, dict) or set(body) != keys:
            raise _controller.ApiError(HTTPStatus.BAD_REQUEST, "request body does not match the Driver operation")
        return body

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        principal = self._principal()
        if principal is None:
            if self.client_address[0] == "127.0.0.1":
                audit.log("auth", self.path, result="denied", level="info", source="loopback-probe")
            else:
                audit.log("auth", self.path, result="denied")
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "invalid or missing credentials"})
            return
        stdlib.dispatch(
            lambda: self._route(method, principal),
            classify=lambda exc: hosted.classify_failure(
                exc,
                _controller.ApiError,
                validate.ValidationError,
                marketplace.MarketplaceError,
            ),
            emit=lambda failure: self._emit_failure(method, failure),
            unexpected_message="internal driver error",
        )

    def _emit_failure(self, method: str, failure: stdlib.HttpFailure) -> None:
        audit.log(method.lower(), self.path, result=failure.result, reason=failure.audit_reason)
        self._send_json(failure.status, {"error": failure.public_message})

    def _route(self, method: str, principal: tuple[str, str | None]) -> None:
        target, route = hosted.route_target(self.headers, self.path, method, _controller.ApiError)
        query = target.query
        parts = list(target.parts)
        kind, account_id = principal
        operation = route.operation

        if Handler._route_global_operation(self, operation, principal, kind, account_id):
            return

        team_id = validate.validate_team_id(route.params["team_id"])
        if operation == "team-create":
            _controller._enforce_rate("create", principal)
            body = self._read_body()
            # an account owns what it creates; an operator may create-on-behalf via an explicit owner
            owner = account_id or str(body.get("owner", "")).strip()
            result = _controller._create(team_id, body, owner)
            trace = audit.log("create", team_id, result="ok", created=result.get("created"), owner=owner)
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        if operation == "team-destroy":
            # A completed Brain removal may leave a bounded durable cleanup record while volume
            # deletion is retried. Only Destroy may authorize against that non-runnable successor.
            lease = _controller._authorize_destroy(team_id, principal)
            result = _controller._destroy(team_id, lease)
            trace = audit.log("destroy", team_id, result="ok", db_dropped=result["db_dropped"])
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
            # Every other operation acts on an existing Team and therefore gates on ownership first.
        lease = _controller._authorize(team_id, principal)
        if operation.startswith("app-"):
            self._route_apps(method, parts, team_id, principal, lease)
        elif operation.startswith("assistant-secret-"):
            self._route_assistant_secrets(method, parts, team_id, lease, principal)
        elif operation.startswith("assistant-account-"):
            self._route_assistant_accounts(method, parts, team_id, lease)
        elif operation == "assistant-help":
            if target.query:
                raise _controller.ApiError(HTTPStatus.BAD_REQUEST, "query and encoded paths are not accepted")
            self._route_assistants(method, parts, team_id, lease)
        elif operation.startswith("inference-"):
            self._route_inference(method, parts, team_id, lease)
        elif operation == "chat" or operation.startswith("chat-"):
            self._route_chat(method, parts, team_id, principal, lease)
        elif operation.startswith("file-"):
            self._route_files(method, parts, team_id, lease, principal)
        else:
            self._route_team_runtime(method, operation.removeprefix("team-"), team_id, lease, query)

    def _route_global_operation(
        self,
        operation: str,
        principal: tuple[str, str | None],
        kind: str,
        account_id: str | None,
    ) -> bool:
        if operation == "team-list":
            self._send_json(HTTPStatus.OK, _controller._list(owner=account_id if kind == "account" else None))
            return True

        if operation == "assistant-account-complete":
            result = _controller._complete_oauth_account(self._read_body(), principal)
            audit.log(
                "assistant_account_complete",
                result["team_id"],
                result="ok",
                assistant=result["assistant_id"],
                account=result["account_id"],
                provider=result["provider"],
            )
            self._send_json(HTTPStatus.OK, result, no_store=True)
            return True
        return False

    def _route_assistant_accounts(
        self,
        method: str,
        parts: list[str],
        team_id: str,
        lease: _controller._AuthorizationLease,
    ) -> None:
        if method == "GET" and len(parts) == 4:
            self._send_json(
                HTTPStatus.OK,
                _controller._assistant_account_inventory(team_id, lease),
                no_store=True,
            )
            return
        if method == "POST" and len(parts) == 7 and parts[4] == "challenges" and parts[6] == "authorize":
            body = self._read_body()
            if not isinstance(body, dict) or set(body) != {"session_binding"}:
                raise _controller.ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "OAuth authorization request is invalid")
            result = _controller._start_oauth_account(
                team_id,
                parts[5],
                body["session_binding"],
                lease,
            )
            audit.log("assistant_account_start", team_id, result="ok")
            self._send_json(HTTPStatus.OK, result, no_store=True)
            return
        if method == "DELETE" and len(parts) == 6:
            result = _controller._disconnect_oauth_account(team_id, parts[4], parts[5], lease)
            audit.log(
                "assistant_account_disconnect",
                team_id,
                result="ok",
                assistant=parts[4],
                account=parts[5],
                disconnected=result["disconnected"],
            )
            self._send_json(HTTPStatus.OK, result, no_store=True)
            return
        raise _controller.ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} /{'/'.join(parts)}")

    def _route_assistant_secrets(
        self,
        method: str,
        parts: list[str],
        team_id: str,
        lease: _controller._AuthorizationLease,
        principal: tuple[str, str | None],
    ) -> None:
        if method == "GET" and len(parts) == 4:
            self._send_json(
                HTTPStatus.OK,
                _controller._assistant_secret_inventory(team_id, lease),
                no_store=True,
            )
            return
        if method == "PUT" and len(parts) == 4:
            _controller._enforce_rate("secret", principal)
            body = self._read_body(max_bytes=_controller.MAX_ASSISTANT_SECRET_BODY_BYTES)
            result = _controller._replace_assistant_secrets(team_id, body, lease)
            audit.log(
                "assistant_secret_replace",
                team_id,
                result="ok",
                assistant=body.get("assistant_id") if isinstance(body, dict) else None,
            )
            self._send_json(HTTPStatus.OK, result, no_store=True)
            return
        raise _controller.ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} /{'/'.join(parts)}")

    def _route_team_runtime(
        self,
        method: str,
        operation: str,
        team_id: str,
        lease: _controller._AuthorizationLease,
        query: dict[str, str],
    ) -> None:
        if method == "GET" and operation == "status":
            self._send_json(HTTPStatus.OK, _controller._status(team_id, lease))
            return
        if method == "GET" and operation == "logs":
            self._send_json(HTTPStatus.OK, _controller._logs(team_id, int(query.get("lines", "200")), lease))
            return
        if method == "POST" and operation in ("stop", "start", "restart"):
            result = _controller._lifecycle(team_id, operation, lease)
            audit.log(operation, team_id, result="ok")
            self._send_json(HTTPStatus.OK, result)
            return
        raise _controller.ApiError(HTTPStatus.NOT_FOUND, f"no such Team operation: {method} {operation}")

    def _route_files(
        self,
        method: str,
        parts: list[str],
        team_id: str,
        lease: _controller._AuthorizationLease,
        principal: tuple[str, str | None],
    ) -> None:
        if method == "GET" and len(parts) == 4:
            self._send_json(HTTPStatus.OK, _controller._list_team_files(team_id, lease))
            return
        if method == "POST" and len(parts) == 4:
            _controller._enforce_rate("file_upload", principal)
            if not _controller._file_upload_slots.acquire(blocking=False):
                raise _controller.ApiError(HTTPStatus.TOO_MANY_REQUESTS, "another Team file upload is in progress")
            try:
                body = self._read_body(max_bytes=_controller.MAX_FILE_BODY_BYTES)
                if not isinstance(body, dict) or set(body) not in (
                    {"filename", "content_b64"},
                    {"filename", "content_b64", "media_type"},
                ):
                    raise _controller.ApiError(
                        HTTPStatus.UNPROCESSABLE_ENTITY,
                        "file upload requires filename, content_b64, and optional media_type",
                    )
                result = _controller._put_inbox_file(
                    team_id,
                    body["filename"],
                    body["content_b64"],
                    body.get("media_type"),
                    lease,
                )
            finally:
                _controller._file_upload_slots.release()
            trace = audit.log(
                "team_file_upload",
                team_id,
                result="ok",
                file_id=result["file"]["id"],
                bytes=result["file"]["size"],
            )
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        if method == "DELETE" and len(parts) == 5:
            result = _controller._delete_team_file(team_id, parts[4], lease)
            trace = audit.log(
                "team_file_delete",
                team_id,
                result="ok",
                file_id=result["id"],
                deleted=result["deleted"],
            )
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        raise _controller.ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} /{'/'.join(parts)}")

    def _route_inference(
        self,
        method: str,
        parts: list[str],
        team_id: str,
        lease: _controller._AuthorizationLease,
    ) -> None:
        if len(parts) == 4 and method == "GET":
            self._send_json(HTTPStatus.OK, _controller._inference_status(team_id, lease))
            return
        if len(parts) == 4 and method == "PUT":
            self._send_json(HTTPStatus.OK, _controller._configure_inference(team_id, self._read_body(), lease))
            return
        raise _controller.ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} /{'/'.join(parts)}")

    def _route_human_chat(
        self,
        method: str,
        kind: str,
        team_id: str,
        principal: tuple[str, str | None],
        lease: _controller._AuthorizationLease,
    ) -> None:
        if kind == "input":
            challenge = _controller._assistant_input_challenges.current(team_id)
            payload = assistant_input_flow.challenge_payload
            submit = _controller._submit_chat_input
        else:
            challenge = _controller._assistant_approval_challenges.current(team_id)
            payload = assistant_approval_flow.challenge_payload
            submit = _controller._submit_chat_approval
        if method == "GET":
            self._send_json(
                HTTPStatus.OK,
                payload(challenge) if challenge is not None else {"team_id": team_id, "status": "none"},
                no_store=True,
            )
            return
        if method == "POST":
            _controller._enforce_rate("chat", principal)
            result = submit(team_id, self._read_body(), lease)
            paused = result.get("status") in _controller.CHAT_PAUSED_STATUSES
            self._send_json(
                HTTPStatus.PRECONDITION_REQUIRED if paused else HTTPStatus.OK,
                result,
                no_store=True,
            )
            return
        raise _controller.ApiError(HTTPStatus.NOT_FOUND, f"no such chat {kind} operation")

    def _route_chat(
        self,
        method: str,
        parts: list[str],
        team_id: str,
        principal: tuple[str, str | None],
        lease: _controller._AuthorizationLease,
    ) -> None:
        """/v1/teams/{team_id}/chat[/stream|/stop|/asks|/answer] — the Captain's brain conversation.

        Ownership was already enforced by _authorize. `chat` (bare) is the non-streaming fallback;
        `chat/stream` is the live NDJSON turn; the rest are the shimpz-ask surface + the Stop control.
        """
        sub2 = parts[4] if len(parts) > 4 else ""
        if sub2 in {"", "stream"}:
            Handler._route_chat_turn(self, method, sub2, team_id, principal, lease)
            return
        if sub2 == "accounts" and len(parts) == 5:
            Handler._route_chat_accounts(self, method, team_id, principal, lease)
            return
        if sub2 == "secrets" and len(parts) == 5:
            Handler._route_chat_secrets(self, method, team_id, principal, lease)
            return
        if sub2 in {"input", "approval"} and len(parts) == 5:
            self._route_human_chat(method, sub2, team_id, principal, lease)
            return
        if sub2 == "stop":
            Handler._route_chat_stop(self, method, team_id, principal, lease)
            return
        raise _controller.ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} /{'/'.join(parts)}")

    def _route_chat_turn(
        self,
        method: str,
        mode: str,
        team_id: str,
        principal: tuple[str, str | None],
        lease: _controller._AuthorizationLease,
    ) -> None:
        if method != "POST":
            raise _controller.ApiError(HTTPStatus.NOT_FOUND, f"no such chat operation: {method}")
        body = self._read_body()
        if not isinstance(body, dict) or set(body) != {"message", "files", "assistant_ids"}:
            raise _controller.ApiError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "Team chat requires message, files, and assistant_ids",
            )
        message = validate.validate_chat_message(body["message"])
        file_ids = body["files"]
        assistant_ids = _controller._chat_assistant_ids(body["assistant_ids"])
        if mode == "stream":
            _controller._enforce_rate("stream", principal)
            pending = _controller._pending_hosted_chat(team_id)
            if pending is not None:
                self._send_json(
                    HTTPStatus.PRECONDITION_REQUIRED,
                    pending,
                    no_store=True,
                )
                return
            self._stream_chat(team_id, message, file_ids, assistant_ids, lease)
            return
        _controller._enforce_rate("chat", principal)
        result = _controller._chat(team_id, message, file_ids, assistant_ids, lease)
        audit.log(
            "chat",
            team_id,
            result="ok",
            chars_in=len(message),
            chars_out=len(str(result.get("reply", ""))),
            paused=result.get("status") in _controller.CHAT_PAUSED_STATUSES,
        )
        paused = result.get("status") in _controller.CHAT_PAUSED_STATUSES
        self._send_json(
            HTTPStatus.PRECONDITION_REQUIRED if paused else HTTPStatus.OK,
            result,
            no_store=paused,
        )

    def _route_chat_accounts(
        self,
        method: str,
        team_id: str,
        principal: tuple[str, str | None],
        lease: _controller._AuthorizationLease,
    ) -> None:
        if method == "GET":
            pending = _controller._assistant_account_challenges.current(team_id)
            self._send_json(
                HTTPStatus.OK,
                (
                    _controller._hosted_account_challenge_payload(pending)
                    if pending is not None
                    else {"team_id": team_id, "status": "none"}
                ),
                no_store=True,
            )
            return
        if method == "POST":
            _controller._enforce_rate("chat", principal)
            body = self._read_body()
            if not isinstance(body, dict) or set(body) != {"challenge_id"}:
                raise _controller.ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "account continuation is invalid")
            result = _controller._resume_chat_accounts(team_id, body["challenge_id"], lease)
            paused = result.get("status") in _controller.CHAT_PAUSED_STATUSES
            self._send_json(
                HTTPStatus.PRECONDITION_REQUIRED if paused else HTTPStatus.OK,
                result,
                no_store=True,
            )
            return
        raise _controller.ApiError(HTTPStatus.NOT_FOUND, f"no such chat account operation: {method}")

    def _route_chat_secrets(
        self,
        method: str,
        team_id: str,
        principal: tuple[str, str | None],
        lease: _controller._AuthorizationLease,
    ) -> None:
        if method == "GET":
            self._send_json(
                HTTPStatus.OK,
                _controller._pending_chat_secrets(team_id, lease),
                no_store=True,
            )
            return
        if method == "POST":
            _controller._enforce_rate("chat", principal)
            result = _controller._submit_chat_secrets(
                team_id,
                self._read_body(max_bytes=_controller.MAX_ASSISTANT_SECRET_BODY_BYTES),
                lease,
            )
            paused = result.get("status") in _controller.CHAT_PAUSED_STATUSES
            self._send_json(
                HTTPStatus.PRECONDITION_REQUIRED if paused else HTTPStatus.OK,
                result,
                no_store=True,
            )
            return
        raise _controller.ApiError(HTTPStatus.NOT_FOUND, f"no such chat secret operation: {method}")

    def _route_chat_stop(
        self,
        method: str,
        team_id: str,
        principal: tuple[str, str | None],
        lease: _controller._AuthorizationLease,
    ) -> None:
        if method != "POST":
            raise _controller.ApiError(HTTPStatus.NOT_FOUND, f"no such chat stop operation: {method}")
        _controller._enforce_rate("stop", principal)
        self._send_json(HTTPStatus.OK, _controller._stop_chat(team_id, lease))

    def _route_apps(
        self,
        method: str,
        parts: list[str],
        team_id: str,
        principal: tuple[str, str | None],
        lease: _controller._AuthorizationLease,
    ) -> None:
        """/v1/teams/{team_id}/apps[/{app}] — the P4 deploy arm. Ownership was already enforced."""
        kind, account_id = principal
        if method == "POST" and len(parts) == 4:
            _controller._enforce_rate("install", principal)
            app_id, spec = marketplace.resolve(self._read_body().get("app"))
            # The marketplace gate, enforced where the socket lives: a NON-first-party app needs a
            # VERIFIED Shimpz account — on a self-hosted Space the verify call IS the phone-home
            # (SHIMPZ_ACCOUNTS_URL → shimpz.com), so not even the Space operator bypasses it.
            if not spec.first_party and kind != "account":
                raise _controller.ApiError(
                    HTTPStatus.UNAUTHORIZED, f"installing {app_id!r} requires a valid Shimpz account"
                )
            owner = account_id or lease.owner
            result = _controller._install_app(team_id, app_id, spec, owner, lease)
            trace = audit.log("install", team_id, result="ok", app=app_id, installed=result["installed"])
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        if method == "GET" and len(parts) == 4:
            self._send_json(HTTPStatus.OK, _controller._list_apps(team_id, lease))
            return
        if method == "DELETE" and len(parts) == 5:
            # Shape-validated only — NOT resolved: an app later pulled from the registry must still
            # be uninstallable from every team that has it.
            app_id = marketplace.validate_app_id(parts[4])
            result = _controller._uninstall_app(team_id, app_id, lease)
            trace = audit.log("uninstall", team_id, result="ok", app=app_id, db_dropped=result["db_dropped"])
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        raise _controller.ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} /{'/'.join(parts)}")

    def _route_assistants(
        self,
        method: str,
        parts: list[str],
        team_id: str,
        lease: _controller._AuthorizationLease,
    ) -> None:
        """Expose only fixed read contracts; install lifecycle remains on the canonical Apps route."""
        if method == "GET" and len(parts) in {6, 7} and parts[5] == "help":
            assistant_id = marketplace.validate_app_id(parts[4])
            locale = parts[6] if len(parts) == 7 else "en"
            help_payload = _controller._assistant_help(team_id, assistant_id, lease, locale)
            trace = audit.log(
                "assistant_help",
                team_id,
                result="ok",
                assistant=help_payload["assistant"],
            )
            self._send_json(
                HTTPStatus.OK,
                {**help_payload, "trace_id": trace},
                no_store=True,
            )
            return
        raise _controller.ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} /{'/'.join(parts)}")


def main() -> None:
    # The Controller owns this bearer. The runtime receives the same named volume read-only and
    # cannot rotate or replace its authority.
    brain_runtime_token_store.ensure()
    _BoundedThreadingHTTPServer((_controller.ALL_INTERFACES, _controller.LISTEN_PORT), Handler).serve_forever()
