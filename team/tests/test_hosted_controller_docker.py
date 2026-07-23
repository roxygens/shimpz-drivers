"""Live cross-tenant contract for the hosted Controller and real Docker inventory."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import unittest
import urllib.error
import urllib.request
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

TEAM = Path(__file__).resolve().parents[1]
DOCKER = shutil.which("docker") or "/usr/bin/docker"


class _AccountsHandler(BaseHTTPRequestHandler):
    sessions: ClassVar[dict[str, str]] = {"session-a": "account_a", "session-b": "account_b"}

    def log_message(self, *_args) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            token = json.loads(self.rfile.read(length))["token"]
            account_id = self.sessions[token]
        except KeyError, TypeError, json.JSONDecodeError:
            status = HTTPStatus.FORBIDDEN
            payload = {"error": "invalid session"}
        else:
            status = HTTPStatus.OK
            payload = {"account_id": account_id}
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class HostedControllerDockerTests(unittest.TestCase):
    maxDiff = None

    def _run(self, *arguments: str, check: bool = True, timeout: int = 600) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(  # noqa: S603 - fixed Docker executable with test-owned arguments
            [DOCKER, *arguments],
            cwd=TEAM,
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
        session: str,
        method: str,
        path: str,
        body: dict[str, object] | None = None,
    ) -> tuple[int, dict[str, object]]:
        encoded = None if body is None else json.dumps(body, separators=(",", ":")).encode()
        headers = {"Connection": "close", "X-Shimpz-Account": session}
        if encoded is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=encoded,
            headers=headers,
            method=method,
        )
        try:
            response = urllib.request.urlopen(request, timeout=15)  # noqa: S310 - fixed loopback URL
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            payload = json.loads(response.read(64 * 1024 + 1))
            self.assertIsInstance(payload, dict)
            return response.status, payload

    def _wait_controller(self, container: str) -> int:
        mapping = self._run("port", container, "7077/tcp").stdout.strip()
        port = int(mapping.rsplit(":", 1)[1])
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                status, _ = self._api(port, "session-b", "GET", "/v1/teams")
            except OSError, urllib.error.URLError:
                time.sleep(0.2)
            else:
                if status == HTTPStatus.OK:
                    return port
        logs = self._run("logs", container, check=False).stdout[-2000:]
        self.fail(f"the hosted Controller did not become ready: {logs}")
        raise AssertionError("unreachable")

    @unittest.skipUnless(os.environ.get("SHIMPZ_RUN_DOCKER_TESTS") == "1", "real Docker test is opt-in")
    def test_account_b_cannot_reach_any_account_a_team_route(self) -> None:
        unique = uuid.uuid4().hex[:12]
        image = f"shimpz-team-driver-hosted-test:{unique}"
        controller = f"shimpz-hosted-controller-{unique}"
        team_id = f"live_{unique}"
        anchor = f"team_{team_id}"
        socket_gid = str(Path("/var/run/docker.sock").stat().st_gid)
        bridge_gateway = self._run(
            "network",
            "inspect",
            "bridge",
            "--format",
            "{{(index .IPAM.Config 0).Gateway}}",
        ).stdout.strip()
        accounts = ThreadingHTTPServer((bridge_gateway, 0), _AccountsHandler)
        accounts_thread = threading.Thread(
            target=accounts.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        accounts_thread.start()

        try:
            self._run(
                "build",
                "--build-arg",
                f"DOCKER_GID={socket_gid}",
                "--tag",
                image,
                ".",
            )
            self._run(
                "run",
                "--detach",
                "--name",
                anchor,
                "--cpus",
                "0.25",
                "--memory",
                "64m",
                "--memory-swap",
                "64m",
                "--pids-limit",
                "32",
                "--label",
                "team.driver=1",
                "--label",
                f"team.id={team_id}",
                "--label",
                "team.name=Account A Team",
                "--label",
                "team.owner=account_a",
                "--label",
                "team.brain=runtime",
                "--label",
                "team.model=gpt-5-nano",
                "--entrypoint",
                "/bin/sh",
                image,
                "-c",
                "sleep 600",
            )
            self._run(
                "run",
                "--detach",
                "--name",
                controller,
                "--cpus",
                "1",
                "--memory",
                "512m",
                "--memory-swap",
                "512m",
                "--pids-limit",
                "128",
                "--group-add",
                socket_gid,
                "--volume",
                "/var/run/docker.sock:/var/run/docker.sock",
                "--env",
                f"SHIMPZ_ACCOUNTS_URL=http://{bridge_gateway}:{accounts.server_port}",
                "--publish",
                "127.0.0.1::7077",
                image,
            )
            port = self._wait_controller(controller)

            owner_status, owner_team = self._api(port, "session-a", "GET", f"/v1/teams/{team_id}/status")
            self.assertEqual(owner_status, HTTPStatus.OK, owner_team)
            self.assertEqual(owner_team["owner"], "account_a")

            other_status, other_teams = self._api(port, "session-b", "GET", "/v1/teams")
            self.assertEqual(other_status, HTTPStatus.OK)
            self.assertEqual(other_teams, {"teams": []})

            base = f"/v1/teams/{team_id}"
            routes = (
                ("DELETE", base),
                ("GET", f"{base}/status"),
                ("GET", f"{base}/logs?lines=1"),
                ("POST", f"{base}/stop"),
                ("POST", f"{base}/start"),
                ("POST", f"{base}/restart"),
                ("GET", f"{base}/apps"),
                ("POST", f"{base}/apps"),
                ("DELETE", f"{base}/apps/notification-center"),
                ("GET", f"{base}/assistant-secrets"),
                ("PUT", f"{base}/assistant-secrets"),
                ("GET", f"{base}/assistant-accounts"),
                ("POST", f"{base}/assistant-accounts/challenges/{'a' * 32}/authorize"),
                ("DELETE", f"{base}/assistant-accounts/shimpz-cloudflare/cloudflare"),
                ("GET", f"{base}/assistants/shimpz-cloudflare/help/en"),
                ("GET", f"{base}/inference"),
                ("PUT", f"{base}/inference"),
                ("POST", f"{base}/chat"),
                ("POST", f"{base}/chat/stream"),
                ("POST", f"{base}/chat/stop"),
                ("GET", f"{base}/chat/accounts"),
                ("POST", f"{base}/chat/accounts"),
                ("GET", f"{base}/chat/secrets"),
                ("POST", f"{base}/chat/secrets"),
                ("GET", f"{base}/files"),
                ("POST", f"{base}/files"),
                ("DELETE", f"{base}/files/{'b' * 32}"),
            )
            expected = {"error": f"team {team_id!r} not found"}
            for method, path in routes:
                body = {} if method in {"POST", "PUT"} else None
                with self.subTest(method=method, path=path):
                    status, payload = self._api(port, "session-b", method, path, body)
                    self.assertEqual(status, HTTPStatus.NOT_FOUND)
                    self.assertEqual(payload, expected)
        finally:
            accounts.shutdown()
            accounts.server_close()
            accounts_thread.join(timeout=2)
            self._remove("rm", "--force", controller, anchor)
            self._remove("image", "rm", "--force", image)


if __name__ == "__main__":
    unittest.main()
