"""Shared real-Docker and controller HTTP harness for live controller suites."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

_T = TypeVar("_T")


def _docker_argv(executable: str, arguments: tuple[str, ...]) -> list[str]:
    command = [executable, *arguments]
    socket = Path("/var/run/docker.sock")
    sg = shutil.which("sg")
    if socket.exists() and not os.access(socket, os.R_OK | os.W_OK) and sg is not None:
        return [sg, "docker", "-c", shlex.join(command)]
    return command


class DockerHarnessMixin:
    docker_command = shutil.which("docker") or "/usr/bin/docker"
    docker_cwd: Path
    credential_header = "Authorization"
    credential_prefix = "Bearer "
    api_timeout = 30
    api_read_limit = 32 * 1024 + 1
    controller_kind = "controller"

    def _run(
        self,
        *arguments: str,
        check: bool = True,
        timeout: int = 600,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            _docker_argv(self.docker_command, arguments),
            cwd=self.docker_cwd,
            env={**os.environ, "DOCKER_BUILDKIT": "1"},
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if check and result.returncode != 0:
            self.fail(f"docker {arguments[0]} failed (rc={result.returncode}): {result.stderr[-2000:]}")
        return result

    def _remove(self, *arguments: str) -> None:
        self._run(*arguments, check=False, timeout=120)

    def _api(
        self,
        port: int,
        credential: str | None,
        method: str,
        path: str,
        body: dict[str, object] | bytes | None = None,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, object]]:
        encoded = (
            body
            if isinstance(body, bytes)
            else None
            if body is None
            else json.dumps(body, separators=(",", ":")).encode()
        )
        headers = {"Connection": "close"}
        if credential is not None:
            headers[self.credential_header] = f"{self.credential_prefix}{credential}"
        if body is not None and not isinstance(body, bytes):
            headers["Content-Type"] = "application/json"
        headers.update(extra_headers or {})
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=encoded,
            headers=headers,
            method=method,
        )
        try:
            response = urllib.request.urlopen(request, timeout=self.api_timeout)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            payload = json.loads(response.read(self.api_read_limit))
            self.assertIsInstance(payload, dict)
            return response.status, payload

    def _wait_controller(
        self,
        container: str,
        probe: Callable[[], _T | None],
        *,
        interval: float = 0.2,
    ) -> _T:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            ready = probe()
            if ready is not None:
                return ready
            time.sleep(interval)
        log_result = self._run("logs", container, check=False)
        logs = (log_result.stdout + log_result.stderr)[-2000:]
        self.fail(f"the {self.controller_kind} did not become ready: {logs}")
        raise AssertionError("unreachable")
