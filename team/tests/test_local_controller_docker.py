"""No-mock end-to-end contract against the real local Docker daemon."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import os
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]
FIXTURE = TEAM / "tests" / "fixtures" / "shimpz-assistant"
REGISTRY_IMAGE = "registry:2.8.3@sha256:a3d8aaa63ed8681a604f1dea0aa03f100d5895b6a58ace528858a7b332415373"
BUILDKIT_IMAGE = "moby/buildkit:v0.31.1@sha256:6b59b7df63a8cb9902736f9ddf7fcff8261613d3e7449b8ea8b7537fc399c03a"
APP_EGRESS_IMAGE = (
    "ghcr.io/roxygens/shimpz-space@sha256:8e5e2d97a9332304db265a9b2ccf052d50cef74eb11553adc40ec7c85e69a13f"
)
MANAGED_LABEL = "com.shimpz.local.managed"
PROFILE_LABEL = "com.shimpz.local.profile"
SPACE_LABEL = "com.shimpz.local.space-id"
KIND_LABEL = "com.shimpz.local.kind"
LOCAL_PROFILE = "single-owner-local-v1"

sys.path.insert(0, str(TEAM))
from local_app import half_cpu_set


class _BrainLifecycleHandler(BaseHTTPRequestHandler):
    """Minimal real HTTP peer for the controller's closed thread-deletion contract."""

    def log_message(self, *_args) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            document = json.loads(body)
        except UnicodeError, json.JSONDecodeError:
            document = None
        valid = (
            self.path == "/v1/threads/delete"
            and isinstance(document, dict)
            and set(document) == {"thread_id"}
            and isinstance(document["thread_id"], str)
        )
        response = json.dumps({"status": "deleted"} if valid else {"error": "invalid request"}).encode()
        self.send_response(200 if valid else 400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(response)


class DockerFlowTests(unittest.TestCase):
    maxDiff = None

    def _run(self, *arguments: str, check: bool = True, timeout: int = 600) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["docker", *arguments],
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

    @staticmethod
    def _ownership(space_id: str, kind: str) -> dict[str, str]:
        return {
            MANAGED_LABEL: "1",
            PROFILE_LABEL: LOCAL_PROFILE,
            SPACE_LABEL: space_id,
            KIND_LABEL: kind,
        }

    def _owned_ids(self, resource: str, space_id: str, kind: str) -> list[str]:
        expected = self._ownership(space_id, kind)
        filters: list[str] = []
        for key, value in expected.items():
            filters.extend(("--filter", f"label={key}={value}"))
        if resource == "container":
            result = self._run("container", "ls", "--all", "--quiet", *filters, check=False)
        else:
            result = self._run("network", "ls", "--quiet", *filters, check=False)
        if result.returncode != 0:
            return []

        verified: list[str] = []
        for resource_id in result.stdout.splitlines():
            inspected = self._run("inspect", resource_id, check=False)
            if inspected.returncode != 0:
                continue
            try:
                metadata = json.loads(inspected.stdout)[0]
            except IndexError, TypeError, json.JSONDecodeError:
                continue
            labels = metadata.get("Config", {}).get("Labels") if resource == "container" else metadata.get("Labels")
            if isinstance(labels, dict) and all(labels.get(key) == value for key, value in expected.items()):
                verified.append(resource_id)
        return verified

    def _cleanup_owned_space(self, space_id: str) -> None:
        # Workloads must leave their networks before Docker can remove those networks.
        for container_id in self._owned_ids("container", space_id, "assistant"):
            self._remove("container", "rm", "--force", container_id)
        for network_id in self._owned_ids("network", space_id, "team"):
            self._remove("network", "rm", network_id)

    def _wait_registry(self, port: int) -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v2/", timeout=1) as response:
                    if response.status == 200:
                        return
            except OSError, urllib.error.URLError:
                time.sleep(0.2)
        self.fail("the test OCI registry did not become ready")

    def _api(
        self,
        port: int,
        token: str | None,
        method: str,
        path: str,
        body: dict[str, object] | None = None,
    ) -> tuple[int, dict[str, object]]:
        encoded = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers = {"Connection": "close"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        if encoded is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=encoded,
            headers=headers,
            method=method,
        )
        try:
            response = urllib.request.urlopen(request, timeout=30)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            payload = json.loads(response.read(32 * 1024 + 1))
            self.assertIsInstance(payload, dict)
            return response.status, payload

    def _wait_controller(self, container: str) -> tuple[int, str]:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            state = self._run("inspect", "--format", "{{.State.Status}}", container, check=False)
            if state.returncode == 0 and state.stdout.strip() == "running":
                token_result = self._run(
                    "exec",
                    container,
                    "/opt/venv/bin/python",
                    "-c",
                    "from pathlib import Path; print(Path('/run/shimpz-local/token').read_text())",
                    check=False,
                )
                if token_result.returncode == 0 and len(token_result.stdout.strip()) == 64:
                    mapping = self._run("port", container, "7077/tcp").stdout.strip()
                    port = int(mapping.rsplit(":", 1)[1])
                    try:
                        status, _ = self._api(port, token_result.stdout.strip(), "GET", "/healthz")
                    except OSError, urllib.error.URLError:
                        pass
                    else:
                        if status == 200:
                            return port, token_result.stdout.strip()
            time.sleep(0.25)
        log_result = self._run("logs", container, check=False)
        logs = (log_result.stdout + log_result.stderr)[-2000:]
        self.fail(f"the local controller did not become ready: {logs}")
        raise AssertionError("unreachable")

    @unittest.skipUnless(os.environ.get("SHIMPZ_RUN_DOCKER_TESTS") == "1", "real Docker test is opt-in")
    def test_real_pull_isolation_lifecycle_and_space_reset(self) -> None:
        unique = uuid.uuid4().hex[:12]
        builder = f"shimpz-local-test-{unique}"
        registry = f"shimpz-registry-{unique}"
        controller = f"shimpz-controller-{unique}"
        egress_proxy = f"shimpz-egress-proxy-{unique}"
        fixture_tag = f"shimpz-assistant-test:{unique}"
        controller_tag = f"shimpz-team-driver-local-test:{unique}"
        token_volume = f"shimpz-local-token-{unique}"
        runtime_token_volume = f"shimpz-local-runtime-token-{unique}"
        audit_volume = f"shimpz-local-audit-{unique}"
        storage_volume = f"shimpz-local-storage-{unique}"
        inference_volume = f"shimpz-local-inference-{unique}"
        power_journal_volume = f"shimpz-local-power-journal-{unique}"
        approval_state_volume = f"shimpz-local-approval-state-{unique}"
        egress_policy_volume = f"shimpz-local-egress-policy-{unique}"
        egress_audit_volume = f"shimpz-local-egress-audit-{unique}"
        space_id = f"test-space-{unique}"
        foreign_network = f"shimpz-foreign-{unique}"
        outbound_network = f"shimpz-egress-outbound-{unique}"
        trusted_ref = ""
        daemon_processors = int(self._run("info", "--format", "{{.NCPU}}").stdout.strip())
        test_cpuset = half_cpu_set(daemon_processors)
        bridge_gateway = ipaddress.IPv4Address(
            self._run(
                "network",
                "inspect",
                "bridge",
                "--format",
                "{{(index .IPAM.Config 0).Gateway}}",
            ).stdout.strip()
        )
        brain_server = ThreadingHTTPServer((str(bridge_gateway), 0), _BrainLifecycleHandler)
        brain_thread = threading.Thread(target=brain_server.serve_forever, daemon=True)
        brain_thread.start()

        try:
            self._run(
                "buildx",
                "create",
                "--name",
                builder,
                "--driver",
                "docker-container",
                "--driver-opt",
                "network=host",
                "--driver-opt",
                f"image={BUILDKIT_IMAGE}",
                "--driver-opt",
                f"cpuset-cpus={test_cpuset}",
                "--driver-opt",
                "memory=4g",
                "--driver-opt",
                "memory-swap=4g",
                "--bootstrap",
            )
            self._run(
                "buildx",
                "build",
                "--builder",
                builder,
                "--load",
                "--tag",
                fixture_tag,
                str(FIXTURE),
            )
            fixture_id = self._run("image", "inspect", "--format", "{{.Id}}", fixture_tag).stdout.strip()

            self._run(
                "run",
                "--detach",
                "--name",
                registry,
                "--cpuset-cpus",
                test_cpuset,
                "--cpus",
                "1",
                "--memory",
                "256m",
                "--memory-swap",
                "256m",
                "--pids-limit",
                "128",
                "--publish",
                "127.0.0.1::5000",
                REGISTRY_IMAGE,
            )
            registry_port = int(self._run("port", registry, "5000/tcp").stdout.strip().rsplit(":", 1)[1])
            self._wait_registry(registry_port)
            repository_tag = f"127.0.0.1:{registry_port}/shimpz/shimpz-assistant:test"
            self._run("tag", fixture_tag, repository_tag)
            self._run("push", repository_tag)
            repo_digests = json.loads(
                self._run("image", "inspect", "--format", "{{json .RepoDigests}}", repository_tag).stdout
            )
            trusted_ref = next(item for item in repo_digests if item.startswith(repository_tag.rsplit(":", 1)[0] + "@"))
            self.assertRegex(trusted_ref, r"@sha256:[0-9a-f]{64}$")

            # Remove every local fixture reference so installation must perform a real digest pull.
            self._remove("image", "rm", "--force", repository_tag, fixture_tag, fixture_id)
            self.assertNotEqual(self._run("image", "inspect", trusted_ref, check=False).returncode, 0)

            self._run(
                "buildx",
                "build",
                "--builder",
                builder,
                "--load",
                "--file",
                str(TEAM / "Dockerfile.local"),
                "--build-arg",
                f"SHIMPZ_ASSISTANT_IMAGE={trusted_ref}",
                "--tag",
                controller_tag,
                str(TEAM),
            )
            self._run("volume", "create", token_volume)
            self._run("volume", "create", runtime_token_volume)
            self._run("volume", "create", audit_volume)
            self._run("volume", "create", storage_volume)
            self._run("volume", "create", inference_volume)
            self._run("volume", "create", power_journal_volume)
            self._run("volume", "create", approval_state_volume)
            self._run("volume", "create", egress_policy_volume)
            self._run("volume", "create", egress_audit_volume)
            self._run("network", "create", outbound_network)
            self._run("pull", APP_EGRESS_IMAGE)
            self._run(
                "run",
                "--detach",
                "--name",
                egress_proxy,
                "--network",
                outbound_network,
                "--cpuset-cpus",
                test_cpuset,
                "--cpus",
                "1",
                "--user",
                "10005:10005",
                "--group-add",
                "10017",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--memory",
                "128m",
                "--memory-swap",
                "128m",
                "--pids-limit",
                "64",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=16m",
                "--volume",
                f"{egress_policy_volume}:/policy:ro",
                "--volume",
                f"{egress_audit_volume}:/var/log/app-egress-proxy",
                "--label",
                "com.shimpz.local.managed=1",
                "--label",
                "com.shimpz.local.profile=single-owner-local-v1",
                "--label",
                f"com.shimpz.local.space-id={space_id}",
                "--label",
                "com.shimpz.local.kind=app-egress-proxy",
                APP_EGRESS_IMAGE,
            )
            socket_gid = str(Path("/var/run/docker.sock").stat().st_gid)
            self._run(
                "run",
                "--detach",
                "--name",
                controller,
                "--cpuset-cpus",
                test_cpuset,
                "--cpus",
                "2",
                "--memory",
                "512m",
                "--memory-swap",
                "512m",
                "--pids-limit",
                "128",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=32m",
                "--group-add",
                socket_gid,
                "--group-add",
                "10016",
                "--group-add",
                "10017",
                "--volume",
                "/var/run/docker.sock:/var/run/docker.sock",
                "--volume",
                f"{token_volume}:/run/shimpz-local",
                "--volume",
                f"{runtime_token_volume}:/run/shimpz-brain-runtime",
                "--volume",
                f"{audit_volume}:/var/log/shimpz-local",
                "--volume",
                f"{storage_volume}:/var/lib/shimpz-local/storage",
                "--volume",
                f"{inference_volume}:/var/lib/shimpz-local/inference",
                "--volume",
                f"{power_journal_volume}:/var/lib/shimpz-local/power-journal",
                "--volume",
                f"{approval_state_volume}:/var/lib/shimpz-local/assistant-approvals",
                "--volume",
                f"{egress_policy_volume}:/var/lib/shimpz-local/app-egress",
                "--env",
                f"SHIMPZ_SPACE_ID={space_id}",
                "--env",
                f"SHIMPZ_APP_EGRESS_PROXY_CONTAINER={egress_proxy}",
                "--env",
                "SHIMPZ_APP_EGRESS_POLICY_DIR=/var/lib/shimpz-local/app-egress",
                "--env",
                f"SHIMPZ_BRAIN_RUNTIME_URL=http://{bridge_gateway}:{brain_server.server_port}",
                "--publish",
                "127.0.0.1::7077",
                controller_tag,
            )
            port, token = self._wait_controller(controller)
            journal_mode = self._run(
                "exec",
                controller,
                "/opt/venv/bin/python",
                "-c",
                "import os,stat; s=os.stat('/var/lib/shimpz-local/power-journal/journal.sqlite3'); "
                "print(oct(stat.S_IMODE(s.st_mode)),s.st_uid,s.st_gid,s.st_nlink)",
            ).stdout.strip()
            self.assertEqual(journal_mode, "0o600 10001 10001 1")
            approval_mode = self._run(
                "exec",
                controller,
                "/opt/venv/bin/python",
                "-c",
                "import os,stat; s=os.stat('/var/lib/shimpz-local/assistant-approvals/grants.sqlite3'); "
                "print(oct(stat.S_IMODE(s.st_mode)),s.st_uid,s.st_gid,s.st_nlink)",
            ).stdout.strip()
            self.assertEqual(approval_mode, "0o600 10001 10001 1")

            unauthenticated, _ = self._api(port, None, "GET", "/v1/assistants")
            self.assertEqual(unauthenticated, 401)
            status, catalog = self._api(port, token, "GET", "/v1/assistants")
            self.assertEqual(status, 200)
            self.assertEqual(catalog["assistants"][0]["id"], "shimpz-assistant")
            self.assertEqual(
                catalog["assistants"][0]["powers"],
                [
                    "cancel-direct-upload",
                    "create-post",
                    "create-test-direct-upload",
                    "delete-post",
                    "identity-me",
                    "list-direct-uploads",
                    "public-user-lookup",
                    "verify-mux-webhook",
                ],
            )

            status, created = self._api(
                port,
                token,
                "POST",
                "/v1/teams/demo_team/create",
                {"team_name": "Demo Team"},
            )
            self.assertEqual(status, 200, created)
            self.assertTrue(created.get("created"), created)
            _, created_again = self._api(
                port,
                token,
                "POST",
                "/v1/teams/demo_team/create",
                {"team_name": "Demo Team"},
            )
            self.assertFalse(created_again["created"])
            _, teams = self._api(port, token, "GET", "/v1/teams")
            self.assertEqual(
                teams["teams"],
                [{"team_id": "demo_team", "team_name": "Demo Team", "status": "running"}],
            )

            file_status, uploaded = self._api(
                port,
                token,
                "POST",
                "/v1/teams/demo_team/files",
                {
                    "filename": "brief.txt",
                    "media_type": "text/plain",
                    "content_b64": base64.b64encode(b"Team private data").decode("ascii"),
                },
            )
            self.assertEqual(file_status, 200)
            file_id = uploaded["file"]["id"]
            self.assertRegex(file_id, r"^[0-9a-f]{32}$")
            self.assertEqual(uploaded["file"]["limit_bytes"], 100 * 1024 * 1024)
            _, files = self._api(port, token, "GET", "/v1/teams/demo_team/files")
            self.assertEqual(files["files"][0]["id"], file_id)
            self.assertEqual(files["used_bytes"], len(b"Team private data"))
            self.assertEqual(
                self._run(
                    "exec",
                    controller,
                    "test",
                    "-f",
                    "/var/lib/shimpz-local/storage/demo_team/files.sqlite3",
                    check=False,
                ).returncode,
                0,
            )

            self._run("restart", controller)
            port, token = self._wait_controller(controller)
            _, files_after_restart = self._api(port, token, "GET", "/v1/teams/demo_team/files")
            self.assertEqual(files_after_restart["files"][0]["id"], file_id)

            # A daemon-side network loss must not let a new lifecycle inherit the old opaque data.
            self._api(
                port,
                token,
                "POST",
                "/v1/teams/orphan_team/create",
                {"team_name": "Orphan Team"},
            )
            self._api(
                port,
                token,
                "POST",
                "/v1/teams/orphan_team/files",
                {
                    "filename": "stale.txt",
                    "content_b64": base64.b64encode(b"must not survive").decode("ascii"),
                },
            )
            prefix = hashlib.sha256(space_id.encode("ascii")).hexdigest()[:12]
            self._run("network", "rm", f"shimpz-local-{prefix}-team-orphan_team")
            _, recreated = self._api(
                port,
                token,
                "POST",
                "/v1/teams/orphan_team/create",
                {"team_name": "Orphan Team"},
            )
            self.assertTrue(recreated["created"])
            _, orphan_files = self._api(port, token, "GET", "/v1/teams/orphan_team/files")
            self.assertEqual(orphan_files["files"], [])
            self._api(port, token, "DELETE", "/v1/teams/orphan_team")

            # An unknown ID is rejected while the trusted image is still absent from the daemon.
            unknown_status, _ = self._api(
                port,
                token,
                "POST",
                "/v1/teams/demo_team/assistants",
                {"assistant": "unknown-assistant"},
            )
            self.assertEqual(unknown_status, 404)
            self.assertNotEqual(self._run("image", "inspect", trusted_ref, check=False).returncode, 0)

            installed_status, installed = self._api(
                port,
                token,
                "POST",
                "/v1/teams/demo_team/assistants",
                {"assistant": "shimpz-assistant"},
            )
            self.assertEqual(installed_status, 200, installed)
            self.assertTrue(installed["installed"], installed)
            self.assertEqual(self._run("image", "inspect", trusted_ref, check=False).returncode, 0)

            assistant_name = self._run(
                "ps",
                "--all",
                "--filter",
                f"label=com.shimpz.local.space-id={space_id}",
                "--filter",
                "label=com.shimpz.local.assistant-id=shimpz-assistant",
                "--format",
                "{{.Names}}",
            ).stdout.strip()
            self.assertTrue(assistant_name)
            original_assistant_id = self._run("inspect", "--format", "{{.Id}}", assistant_name).stdout.strip()
            metadata = json.loads(self._run("inspect", assistant_name).stdout)[0]
            host = metadata["HostConfig"]
            self.assertEqual(metadata["Config"]["User"], "10001:10001")
            self.assertTrue(host["ReadonlyRootfs"])
            self.assertIn("ALL", host["CapDrop"])
            self.assertTrue(any(item.startswith("no-new-privileges") for item in host["SecurityOpt"]))
            self.assertNotIn("seccomp=unconfined", host["SecurityOpt"])
            self.assertEqual(host["Memory"], 128 * 1024 * 1024)
            self.assertEqual(host["MemorySwap"], 128 * 1024 * 1024)
            self.assertEqual(host["NanoCpus"], 250_000_000)
            self.assertEqual(host["PidsLimit"], 64)
            self.assertEqual(host["CpusetCpus"], test_cpuset)
            self.assertIn(host.get("Tmpfs"), (None, {}))
            self.assertEqual(metadata["Mounts"], [])
            self.assertIn(host["PortBindings"], (None, {}))
            networks = metadata["NetworkSettings"]["Networks"]
            self.assertEqual(len(networks), 1)
            network_name = next(iter(networks))
            network_metadata = json.loads(self._run("network", "inspect", network_name).stdout)[0]
            self.assertTrue(network_metadata["Internal"])
            self.assertEqual(network_metadata["Labels"]["com.shimpz.local.space-id"], space_id)
            self.assertEqual(network_metadata["Labels"]["com.shimpz.local.team-name"], "Demo Team")

            # Docker still reports "running" when PID 1 is stopped. A retry must replace this explicitly
            # stateless runtime; relying on an external SIGCONT would leave the documented recovery false.
            self._run("kill", "--signal", "STOP", assistant_name)
            stopped_state = self._run("inspect", "--format", "{{.State.Status}}", assistant_name).stdout.strip()
            self.assertEqual(stopped_state, "running")
            recovered_status, recovered = self._api(
                port,
                token,
                "POST",
                "/v1/teams/demo_team/assistants",
                {"assistant": "shimpz-assistant"},
            )
            self.assertEqual((recovered_status, recovered["installed"]), (200, False))
            replacement_assistant_id = self._run("inspect", "--format", "{{.Id}}", assistant_name).stdout.strip()
            self.assertNotEqual(replacement_assistant_id, original_assistant_id)
            self.assertNotEqual(self._run("inspect", original_assistant_id, check=False).returncode, 0)

            _, installed_again = self._api(
                port,
                token,
                "POST",
                "/v1/teams/demo_team/assistants",
                {"assistant": "shimpz-assistant"},
            )
            self.assertFalse(installed_again["installed"])
            self.assertEqual(
                self._run("inspect", "--format", "{{.Id}}", assistant_name).stdout.strip(),
                replacement_assistant_id,
            )

            _, listed = self._api(port, token, "GET", "/v1/teams/demo_team/assistants")
            self.assertEqual(listed["assistants"], [{"assistant": "shimpz-assistant", "status": "running"}])
            help_status, assistant_help = self._api(
                port,
                token,
                "GET",
                "/v1/teams/demo_team/assistants/shimpz-assistant/help/pt",
            )
            self.assertEqual(help_status, 200)
            self.assertEqual(assistant_help["assistant"], "shimpz-assistant")
            self.assertIn("# Shimpz Assistant", assistant_help["markdown"])
            self.assertRegex(assistant_help["trace_id"], r"^[0-9a-f]{32}$")
            secret_status, secret_inventory = self._api(
                port,
                token,
                "GET",
                "/v1/teams/demo_team/assistant-secrets",
            )
            self.assertEqual(secret_status, 200)
            secret_items = secret_inventory["assistants"][0]["secrets"]
            self.assertEqual(
                [(item["id"], item["configured"], item["mask"]) for item in secret_items],
                [
                    ("mux-token-id", False, None),
                    ("mux-token-secret", False, None),
                    ("mux-webhook-signing-secret", False, None),
                ],
            )
            account_status, account_inventory = self._api(
                port,
                token,
                "GET",
                "/v1/teams/demo_team/assistant-accounts",
            )
            self.assertEqual(account_status, 200)
            self.assertEqual(len(account_inventory["accounts"]), 1)
            account = account_inventory["accounts"][0]
            self.assertEqual(
                {
                    "assistant_id": account["assistant_id"],
                    "id": account["id"],
                    "provider": account["provider"],
                    "scopes": account["scopes"],
                    "status": account["status"],
                    "account": account["account"],
                    "expires_at": account["expires_at"],
                },
                {
                    "assistant_id": "shimpz-assistant",
                    "id": "x",
                    "provider": "x",
                    "scopes": ["offline.access", "tweet.read", "tweet.write", "users.read"],
                    "status": "missing",
                    "account": None,
                    "expires_at": None,
                },
            )
            account_required, missing_account = self._api(
                port,
                token,
                "POST",
                "/v1/teams/demo_team/assistants/shimpz-assistant/powers/public-user-lookup",
                {"username": "OpenAI"},
            )
            self.assertEqual(account_required, 409)
            self.assertEqual(missing_account["code"], "assistant-account-unavailable")
            unknown_power, _ = self._api(
                port,
                token,
                "POST",
                "/v1/teams/demo_team/assistants/shimpz-assistant/powers/shell",
                {},
            )
            self.assertEqual(unknown_power, 404)

            proxy_metadata = json.loads(self._run("inspect", egress_proxy).stdout)[0]
            proxy_networks = proxy_metadata["NetworkSettings"]["Networks"]
            self.assertEqual(set(proxy_networks), {outbound_network, network_name})
            self.assertIn("app-egress-proxy", proxy_networks[network_name]["Aliases"])
            policy_contract = self._run(
                "exec",
                controller,
                "/opt/venv/bin/python",
                "-c",
                "import json,os,stat; from pathlib import Path; "
                "p=next(Path('/var/lib/shimpz-local/app-egress').glob('*.json')); s=p.stat(); "
                "print(json.dumps(json.loads(p.read_text())),oct(stat.S_IMODE(s.st_mode)),s.st_uid,s.st_gid)",
            ).stdout.strip()
            self.assertEqual(
                policy_contract,
                '["api.mux.com", "api.x.com"] 0o640 10001 10017',
            )

            _, removed = self._api(
                port,
                token,
                "DELETE",
                "/v1/teams/demo_team/assistants/shimpz-assistant",
            )
            self.assertTrue(removed["uninstalled"])
            _, removed_again = self._api(
                port,
                token,
                "DELETE",
                "/v1/teams/demo_team/assistants/shimpz-assistant",
            )
            self.assertFalse(removed_again["uninstalled"])
            proxy_networks_after_uninstall = json.loads(self._run("inspect", egress_proxy).stdout)[0][
                "NetworkSettings"
            ]["Networks"]
            self.assertEqual(set(proxy_networks_after_uninstall), {outbound_network})
            remaining_policy_files = self._run(
                "exec",
                controller,
                "/opt/venv/bin/python",
                "-c",
                "from pathlib import Path; p=Path('/var/lib/shimpz-local/app-egress'); "
                "print(len(list(p.glob('*.json'))),len(list((p/'.tokens').glob('*.token'))))",
            ).stdout.strip()
            self.assertEqual(remaining_policy_files, "0 0")
            _, deleted_file = self._api(
                port,
                token,
                "DELETE",
                f"/v1/teams/demo_team/files/{file_id}",
            )
            self.assertTrue(deleted_file["deleted"])
            _, destroyed = self._api(port, token, "DELETE", "/v1/teams/demo_team")
            self.assertTrue(destroyed["destroyed"])
            self.assertTrue(destroyed["storage_removed"])
            self.assertNotEqual(
                self._run(
                    "exec",
                    controller,
                    "test",
                    "-e",
                    "/var/lib/shimpz-local/storage/demo_team",
                    check=False,
                ).returncode,
                0,
            )
            _, destroyed_again = self._api(port, token, "DELETE", "/v1/teams/demo_team")
            self.assertFalse(destroyed_again["destroyed"])

            # Reset owns no identifiers and ignores a similarly labeled resource missing the exact kind label.
            self._run(
                "network",
                "create",
                "--internal",
                "--label",
                "com.shimpz.local.managed=1",
                "--label",
                "com.shimpz.local.profile=single-owner-local-v1",
                "--label",
                f"com.shimpz.local.space-id={space_id}",
                foreign_network,
            )
            self._api(
                port,
                token,
                "POST",
                "/v1/teams/reset_team/create",
                {"team_name": "Reset Team"},
            )
            self._api(
                port,
                token,
                "POST",
                "/v1/teams/reset_team/assistants",
                {"assistant": "shimpz-assistant"},
            )
            self._api(
                port,
                token,
                "POST",
                "/v1/teams/reset_team/files",
                {
                    "filename": "reset.txt",
                    "content_b64": base64.b64encode(b"remove me").decode("ascii"),
                },
            )
            reset_status, reset = self._api(port, token, "DELETE", "/v1/space")
            self.assertEqual(reset_status, 200)
            self.assertEqual((reset["assistants_removed"], reset["teams_removed"]), (1, 1))
            _, reset_again = self._api(port, token, "DELETE", "/v1/space")
            self.assertEqual((reset_again["assistants_removed"], reset_again["teams_removed"]), (0, 0))
            self.assertNotEqual(
                self._run(
                    "exec",
                    controller,
                    "test",
                    "-e",
                    "/var/lib/shimpz-local/storage/reset_team",
                    check=False,
                ).returncode,
                0,
            )
            self.assertEqual(self._run("network", "inspect", foreign_network, check=False).returncode, 0)

            audit = self._run(
                "exec",
                controller,
                "/opt/venv/bin/python",
                "-c",
                "from pathlib import Path; print(Path('/var/log/shimpz-local/audit.jsonl').read_text())",
            ).stdout
            self.assertIn('"operation":"space-reset"', audit)
            self.assertIn('"detail":"assistant-account-unavailable"', audit)
            self.assertNotIn("Captain", audit)
            self.assertNotIn(token, audit)

            token_mode = self._run(
                "exec",
                controller,
                "/opt/venv/bin/python",
                "-c",
                "import os,stat; s=os.stat('/run/shimpz-local/token'); "
                "print(oct(stat.S_IMODE(s.st_mode)),s.st_uid,s.st_gid,s.st_nlink)",
            ).stdout.strip()
            self.assertEqual(token_mode, "0o440 10001 10010 1")
            runtime_token_mode = self._run(
                "exec",
                controller,
                "/opt/venv/bin/python",
                "-c",
                "import os,stat; s=os.stat('/run/shimpz-brain-runtime/token'); "
                "print(oct(stat.S_IMODE(s.st_mode)),s.st_uid,s.st_gid,s.st_nlink,s.st_size)",
            ).stdout.strip()
            self.assertEqual(runtime_token_mode, "0o440 10001 10016 1 64")

            # Leave one exact-owned pair for the outer finally. This proves cleanup does not depend
            # on reaching the controller reset route and therefore also runs after an earlier failure.
            self._api(
                port,
                token,
                "POST",
                "/v1/teams/cleanup_team/create",
                {"team_name": "Cleanup Team"},
            )
            self._api(
                port,
                token,
                "POST",
                "/v1/teams/cleanup_team/assistants",
                {"assistant": "shimpz-assistant"},
            )
            self.assertEqual(len(self._owned_ids("container", space_id, "assistant")), 1)
            self.assertEqual(len(self._owned_ids("network", space_id, "team")), 1)
        finally:
            brain_server.shutdown()
            brain_server.server_close()
            brain_thread.join(timeout=2)
            # Cleanup remains strictly scoped to this test's unique names/labels.
            self._remove("rm", "--force", egress_proxy)
            self._cleanup_owned_space(space_id)
            owned_containers = self._owned_ids("container", space_id, "assistant")
            owned_networks = self._owned_ids("network", space_id, "team")
            self._remove("rm", "--force", controller)
            self._remove("rm", "--force", registry)
            self._remove("network", "rm", foreign_network)
            self._remove("network", "rm", outbound_network)
            self._remove(
                "volume",
                "rm",
                "--force",
                token_volume,
                runtime_token_volume,
                audit_volume,
                storage_volume,
                inference_volume,
                power_journal_volume,
                approval_state_volume,
                egress_policy_volume,
                egress_audit_volume,
            )
            if trusted_ref:
                self._remove("image", "rm", "--force", trusted_ref)
            self._remove("image", "rm", "--force", fixture_tag, controller_tag)
            self._remove("buildx", "rm", "--force", builder)
            self.assertEqual(owned_containers, [])
            self.assertEqual(owned_networks, [])


if __name__ == "__main__":
    unittest.main()
