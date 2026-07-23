"""Local Assistant RPC transport and fail-stop readiness."""

import socket
import time
from contextlib import suppress
from http import HTTPStatus
from pathlib import Path

import assistant_help
import assistant_secret_flow
import power_execution
from container_policy import local as local_container_policy
from docker.errors import DockerException, NotFound
from local_registry import AssistantSpec

from local_support.errors import ApiProblemError as ApiProblem

MAX_RESPONSE_BYTES = assistant_help.MAX_HELP_BYTES * 6 + 1024
RPC_TIMEOUT_SECONDS = 8
HEALTH_TIMEOUT_SECONDS = 15
ASSISTANT_UID = local_container_policy.ASSISTANT_UID
ASSISTANT_WORKDIR = str(Path("/") / "tmp")


class UnsupportedAssistantRpcPathError(RuntimeError):
    """The fixed Assistant RPC adapter rejected a path it does not implement."""


class LocalAssistantRpcMixin:
    @staticmethod
    def _close_exec_stream(stream) -> None:
        power_execution.close_exec_stream(stream)

    def _fail_stop_power(self, container) -> None:
        """Stop, then kill if needed, and prove an ambiguous local Power cannot keep running."""
        try:
            container.stop(timeout=3)
        except NotFound:
            return
        except DockerException:
            pass
        if self._power_not_running(container):
            return
        try:
            container.kill()
        except NotFound:
            return
        except DockerException:
            pass
        if self._power_not_running(container):
            return
        self._blocked_power_workloads.add(container.id)
        raise ApiProblem(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Assistant Power termination could not be proved; reinstall the Assistant",
            code="assistant-power-blocked",
        )

    @staticmethod
    def _power_not_running(container) -> bool:
        try:
            container.reload()
        except NotFound:
            return True
        except DockerException:
            return False
        state = container.attrs.get("State")
        return isinstance(state, dict) and state.get("Running") is False

    def _read_rpc_frames(self, raw_socket: socket.socket, deadline: float) -> tuple[bytes, bytes]:
        return power_execution.read_rpc_frames(raw_socket, deadline, MAX_RESPONSE_BYTES)

    def _rpc(
        self,
        container,
        spec: AssistantSpec,
        method: str,
        path: str,
        payload: dict,
        *,
        detect_unsupported_path: bool = False,
    ) -> object:
        try:
            encoded = assistant_secret_flow.encode_private_rpc_envelope(payload)
        except assistant_secret_flow.SecretFlowError as exc:
            raise ApiProblem(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request is too large",
                code="body-too-large",
            ) from exc

        def close_stream(stream: object) -> None:
            with suppress(Exception):
                self._close_exec_stream(stream)

        try:
            return power_execution.rpc_exchange(
                container.id,
                [spec.rpc_command, method, path],
                encoded,
                power_execution.RpcExchangeStrategy(
                    api=self.client.api,
                    user=ASSISTANT_UID,
                    workdir=ASSISTANT_WORKDIR,
                    timeout=RPC_TIMEOUT_SECONDS,
                    maximum=MAX_RESPONSE_BYTES,
                    transport_errors=(DockerException,),
                    fail_stop=lambda: self._fail_stop_power(container),
                    cancelled=lambda _exc: None,
                    close_stream=close_stream,
                ),
                detect_unsupported_path=detect_unsupported_path,
            )
        except power_execution.RpcExchangeError as exc:
            if exc.kind == "unsupported-path":
                raise UnsupportedAssistantRpcPathError(path) from None
            message, code = {
                "timeout": ("Assistant Power timed out", "assistant-timeout"),
                "ambiguous": ("Assistant Power status is ambiguous", "assistant-rpc-failed"),
                "failed": ("Assistant Power failed", "assistant-rpc-failed"),
                "invalid-result": ("Assistant Power failed", "assistant-rpc-failed"),
            }.get(exc.kind, (None, None))
            status = power_execution.rpc_failure_status(exc.kind)
            raise ApiProblem(status, message, code=code) from exc

    def _wait_ready(self, container, spec: AssistantSpec) -> None:
        deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            container.reload()
            if container.status not in {"created", "running"}:
                break
            if container.status == "running":
                try:
                    result = self._rpc(container, spec, "GET", spec.health_path, {})
                except ApiProblem:
                    pass
                else:
                    if result == {"status": "ok"}:
                        return
            time.sleep(0.2)
        raise ApiProblem(HTTPStatus.BAD_GATEWAY, "Assistant did not become ready", code="assistant-not-ready")
