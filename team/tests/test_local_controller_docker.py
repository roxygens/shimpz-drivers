"""No-mock end-to-end contract against the real local Docker daemon."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]
FIXTURE = TEAM / "tests" / "fixtures" / "reference-assistant"
REGISTRY_IMAGE = "registry:2.8.3@sha256:a3d8aaa63ed8681a604f1dea0aa03f100d5895b6a58ace528858a7b332415373"
BUILDKIT_IMAGE = "moby/buildkit:v0.31.1@sha256:6b59b7df63a8cb9902736f9ddf7fcff8261613d3e7449b8ea8b7537fc399c03a"
MANAGED_LABEL = "com.shimpz.local.managed"
PROFILE_LABEL = "com.shimpz.local.profile"
SPACE_LABEL = "com.shimpz.local.space-id"
KIND_LABEL = "com.shimpz.local.kind"
LOCAL_PROFILE = "single-owner-local-v1"

sys.path.insert(0, str(TEAM))
import power_execution
from docker_harness import DockerHarnessMixin
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


@dataclass(slots=True)
class _DockerFlow:
    builder: str
    registry: str
    controller: str
    egress_proxy: str
    fixture_tag: str
    controller_tag: str
    egress_proxy_tag: str
    token_volume: str
    runtime_token_volume: str
    audit_volume: str
    storage_volume: str
    inference_volume: str
    power_journal_volume: str
    approval_state_volume: str
    continuation_state_volume: str
    continuation_key_volume: str
    egress_policy_volume: str
    egress_audit_volume: str
    space_id: str
    foreign_network: str
    outbound_network: str
    test_cpuset: str
    bridge_gateway: ipaddress.IPv4Address
    brain_server: ThreadingHTTPServer
    brain_thread: threading.Thread
    trusted_ref: str = ""
    port: int = 0
    token: str = ""
    file_id: str = ""
    network_name: str = ""
    assistant_name: str = ""
    original_assistant_id: str = ""


class DockerFlowTests(DockerHarnessMixin, unittest.TestCase):
    maxDiff = None
    docker_command = "docker"
    docker_cwd = TEAM
    controller_kind = "local controller"

    def _new_flow(self) -> _DockerFlow:
        unique = uuid.uuid4().hex[:12]
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
        brain_thread = threading.Thread(
            target=brain_server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        brain_thread.start()
        return _DockerFlow(
            builder=f"shimpz-local-test-{unique}",
            registry=f"shimpz-registry-{unique}",
            controller=f"shimpz-controller-{unique}",
            egress_proxy=f"shimpz-egress-proxy-{unique}",
            fixture_tag=f"shimpz-cloudflare-test:{unique}",
            controller_tag=f"shimpz-team-driver-local-test:{unique}",
            egress_proxy_tag=f"shimpz-app-egress-test:{unique}",
            token_volume=f"shimpz-local-token-{unique}",
            runtime_token_volume=f"shimpz-local-runtime-token-{unique}",
            audit_volume=f"shimpz-local-audit-{unique}",
            storage_volume=f"shimpz-local-storage-{unique}",
            inference_volume=f"shimpz-local-inference-{unique}",
            power_journal_volume=f"shimpz-local-power-journal-{unique}",
            approval_state_volume=f"shimpz-local-approval-state-{unique}",
            continuation_state_volume=f"shimpz-local-continuation-state-{unique}",
            continuation_key_volume=f"shimpz-local-continuation-key-{unique}",
            egress_policy_volume=f"shimpz-local-egress-policy-{unique}",
            egress_audit_volume=f"shimpz-local-egress-audit-{unique}",
            space_id=f"test-space-{unique}",
            foreign_network=f"shimpz-foreign-{unique}",
            outbound_network=f"shimpz-egress-outbound-{unique}",
            test_cpuset=test_cpuset,
            bridge_gateway=bridge_gateway,
            brain_server=brain_server,
            brain_thread=brain_thread,
        )

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

    def _wait_local_controller(self, container: str) -> tuple[int, str]:
        def probe() -> tuple[int, str] | None:
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
            return None

        return self._wait_controller(container, probe, interval=0.25)

    def _prepare_images(self, flow: _DockerFlow) -> None:
        self._run(
            "buildx",
            "create",
            "--name",
            flow.builder,
            "--driver",
            "docker-container",
            "--driver-opt",
            "network=host",
            "--driver-opt",
            f"image={BUILDKIT_IMAGE}",
            "--driver-opt",
            f"cpuset-cpus={flow.test_cpuset}",
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
            flow.builder,
            "--load",
            "--tag",
            flow.fixture_tag,
            "--file",
            str(FIXTURE / "Dockerfile"),
            str(TEAM),
        )
        fixture_id = self._run("image", "inspect", "--format", "{{.Id}}", flow.fixture_tag).stdout.strip()

        self._run(
            "run",
            "--detach",
            "--name",
            flow.registry,
            "--cpuset-cpus",
            flow.test_cpuset,
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
        registry_port = int(self._run("port", flow.registry, "5000/tcp").stdout.strip().rsplit(":", 1)[1])
        self._wait_registry(registry_port)
        repository_tag = f"127.0.0.1:{registry_port}/shimpz/shimpz-cloudflare:test"
        self._run("tag", flow.fixture_tag, repository_tag)
        self._run("push", repository_tag)
        repo_digests = json.loads(
            self._run("image", "inspect", "--format", "{{json .RepoDigests}}", repository_tag).stdout
        )
        flow.trusted_ref = next(
            item for item in repo_digests if item.startswith(repository_tag.rsplit(":", 1)[0] + "@")
        )
        self.assertRegex(flow.trusted_ref, r"@sha256:[0-9a-f]{64}$")

        # Remove every local fixture reference so installation must perform a real digest pull.
        self._remove("image", "rm", "--force", repository_tag, flow.fixture_tag, fixture_id)
        self.assertNotEqual(self._run("image", "inspect", flow.trusted_ref, check=False).returncode, 0)

        self._run(
            "buildx",
            "build",
            "--builder",
            flow.builder,
            "--load",
            "--file",
            str(TEAM / "Dockerfile.local"),
            "--build-arg",
            f"SHIMPZ_ASSISTANT_IMAGE={flow.trusted_ref}",
            "--build-arg",
            f"SHIMPZ_CLOUDFLARE_ASSISTANT_IMAGE={flow.trusted_ref}",
            "--tag",
            flow.controller_tag,
            str(TEAM),
        )

    def _start_controller(self, flow: _DockerFlow) -> None:
        self._run("volume", "create", flow.token_volume)
        self._run("volume", "create", flow.runtime_token_volume)
        self._run("volume", "create", flow.audit_volume)
        self._run("volume", "create", flow.storage_volume)
        self._run("volume", "create", flow.inference_volume)
        self._run("volume", "create", flow.power_journal_volume)
        self._run("volume", "create", flow.approval_state_volume)
        self._run("volume", "create", flow.continuation_state_volume)
        self._run("volume", "create", flow.continuation_key_volume)
        self._run("volume", "create", flow.egress_policy_volume)
        self._run("volume", "create", flow.egress_audit_volume)
        self._run("network", "create", flow.outbound_network)
        self._run("build", "--tag", flow.egress_proxy_tag, str(TEAM.parent / "app-egress"))
        self._run(
            "run",
            "--detach",
            "--name",
            flow.egress_proxy,
            "--network",
            flow.outbound_network,
            "--cpuset-cpus",
            flow.test_cpuset,
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
            f"{flow.egress_policy_volume}:/policy:ro",
            "--volume",
            f"{flow.egress_audit_volume}:/var/log/app-egress-proxy",
            "--label",
            "com.shimpz.local.managed=1",
            "--label",
            "com.shimpz.local.profile=single-owner-local-v1",
            "--label",
            f"com.shimpz.local.space-id={flow.space_id}",
            "--label",
            "com.shimpz.local.kind=app-egress-proxy",
            flow.egress_proxy_tag,
        )
        socket_gid = str(Path("/var/run/docker.sock").stat().st_gid)
        self._run(
            "run",
            "--detach",
            "--name",
            flow.controller,
            "--cpuset-cpus",
            flow.test_cpuset,
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
            f"{flow.token_volume}:/run/shimpz-local",
            "--volume",
            f"{flow.runtime_token_volume}:/run/shimpz-brain-runtime",
            "--volume",
            f"{flow.audit_volume}:/var/log/shimpz-local",
            "--volume",
            f"{flow.storage_volume}:/var/lib/shimpz-local/storage",
            "--volume",
            f"{flow.inference_volume}:/var/lib/shimpz-local/inference",
            "--volume",
            f"{flow.power_journal_volume}:/var/lib/shimpz-local/power-journal",
            "--volume",
            f"{flow.approval_state_volume}:/var/lib/shimpz-local/assistant-approvals",
            "--volume",
            f"{flow.continuation_state_volume}:/var/lib/shimpz-local/chat-continuations/state",
            "--volume",
            f"{flow.continuation_key_volume}:/var/lib/shimpz-local/chat-continuations/key",
            "--volume",
            f"{flow.egress_policy_volume}:/var/lib/shimpz-local/app-egress",
            "--env",
            f"SHIMPZ_SPACE_ID={flow.space_id}",
            "--env",
            f"SHIMPZ_APP_EGRESS_PROXY_CONTAINER={flow.egress_proxy}",
            "--env",
            "SHIMPZ_APP_EGRESS_POLICY_DIR=/var/lib/shimpz-local/app-egress",
            "--env",
            f"SHIMPZ_BRAIN_RUNTIME_URL=http://{flow.bridge_gateway}:{flow.brain_server.server_port}",
            "--publish",
            "127.0.0.1::7077",
            flow.controller_tag,
        )
        flow.port, flow.token = self._wait_local_controller(flow.controller)
        journal_mode = self._run(
            "exec",
            flow.controller,
            "/opt/venv/bin/python",
            "-c",
            "import os,stat; s=os.stat('/var/lib/shimpz-local/power-journal/journal.sqlite3'); "
            "print(oct(stat.S_IMODE(s.st_mode)),s.st_uid,s.st_gid,s.st_nlink)",
        ).stdout.strip()
        self.assertEqual(journal_mode, "0o600 10001 10001 1")
        approval_mode = self._run(
            "exec",
            flow.controller,
            "/opt/venv/bin/python",
            "-c",
            "import os,stat; s=os.stat('/var/lib/shimpz-local/assistant-approvals/grants.sqlite3'); "
            "print(oct(stat.S_IMODE(s.st_mode)),s.st_uid,s.st_gid,s.st_nlink)",
        ).stdout.strip()
        self.assertEqual(approval_mode, "0o600 10001 10001 1")
        continuation_files = self._run(
            "exec",
            flow.controller,
            "/opt/venv/bin/python",
            "-c",
            "import os,stat,time; from local_chat_continuation_store import EncryptedContinuationStore; "
            "s=EncryptedContinuationStore(); "
            "s.put('demo_team','input','0'*32,int(time.time())+60,['thread:test'],b'opaque'); "
            "assert s.delete('demo_team'); "
            "paths=(s.state_path,s.key_path); "
            "print(' '.join(f'{oct(stat.S_IMODE(p.stat().st_mode))}:{p.stat().st_uid}:{p.stat().st_gid}' "
            "for p in paths))",
        ).stdout.strip()
        self.assertEqual(continuation_files, "0o600:10001:10001 0o600:10001:10001")

        unauthenticated, _ = self._api(flow.port, None, "GET", "/v1/assistants")
        self.assertEqual(unauthenticated, 401)
        status, catalog = self._api(flow.port, flow.token, "GET", "/v1/assistants")
        self.assertEqual(status, 200)
        self.assertEqual(catalog["assistants"][0]["id"], "shimpz-cloudflare")
        self.assertEqual(
            catalog["assistants"][0]["powers"],
            ["list-dns-records", "list-zones"],
        )

    def _exercise_team_storage(self, flow: _DockerFlow) -> None:
        status, created = self._api(
            flow.port,
            flow.token,
            "POST",
            "/v1/teams/demo_team/create",
            {"team_name": "Demo Team"},
        )
        self.assertEqual(status, 200, created)
        self.assertTrue(created.get("created"), created)
        _, created_again = self._api(
            flow.port,
            flow.token,
            "POST",
            "/v1/teams/demo_team/create",
            {"team_name": "Demo Team"},
        )
        self.assertFalse(created_again["created"])
        _, teams = self._api(flow.port, flow.token, "GET", "/v1/teams")
        self.assertEqual(
            teams["teams"],
            [{"team_id": "demo_team", "team_name": "Demo Team", "status": "running"}],
        )

        file_status, uploaded = self._api(
            flow.port,
            flow.token,
            "POST",
            "/v1/teams/demo_team/files",
            b"Team private data",
            extra_headers={
                "Content-Type": "text/plain",
                "X-Shimpz-Filename": "brief.txt",
            },
        )
        self.assertEqual(file_status, 200)
        flow.file_id = uploaded["file"]["id"]
        self.assertRegex(flow.file_id, r"^[0-9a-f]{32}$")
        self.assertEqual(uploaded["file"]["limit_bytes"], 100 * 1024 * 1024)
        _, files = self._api(flow.port, flow.token, "GET", "/v1/teams/demo_team/files")
        self.assertEqual(files["files"][0]["id"], flow.file_id)
        self.assertEqual(files["used_bytes"], len(b"Team private data"))
        self.assertEqual(
            self._run(
                "exec",
                flow.controller,
                "test",
                "-f",
                "/var/lib/shimpz-local/storage/demo_team/files.sqlite3",
                check=False,
            ).returncode,
            0,
        )

        self._run("restart", flow.controller)
        flow.port, flow.token = self._wait_local_controller(flow.controller)
        _, files_after_restart = self._api(flow.port, flow.token, "GET", "/v1/teams/demo_team/files")
        self.assertEqual(files_after_restart["files"][0]["id"], flow.file_id)

        # A daemon-side network loss must not let a new lifecycle inherit the old opaque data.
        self._api(
            flow.port,
            flow.token,
            "POST",
            "/v1/teams/orphan_team/create",
            {"team_name": "Orphan Team"},
        )
        self._api(
            flow.port,
            flow.token,
            "POST",
            "/v1/teams/orphan_team/files",
            b"must not survive",
            extra_headers={
                "Content-Type": "application/octet-stream",
                "X-Shimpz-Filename": "stale.txt",
            },
        )
        prefix = hashlib.sha256(flow.space_id.encode("ascii")).hexdigest()[:12]
        self._run("network", "rm", f"shimpz-local-{prefix}-team-orphan_team")
        _, recreated = self._api(
            flow.port,
            flow.token,
            "POST",
            "/v1/teams/orphan_team/create",
            {"team_name": "Orphan Team"},
        )
        self.assertTrue(recreated["created"])
        _, orphan_files = self._api(flow.port, flow.token, "GET", "/v1/teams/orphan_team/files")
        self.assertEqual(orphan_files["files"], [])
        self._api(flow.port, flow.token, "DELETE", "/v1/teams/orphan_team")

    def _exercise_assistant(self, flow: _DockerFlow) -> None:
        # An unknown ID is rejected while the trusted image is still absent from the daemon.
        unknown_status, _ = self._api(
            flow.port,
            flow.token,
            "POST",
            "/v1/teams/demo_team/assistants",
            {"assistant": "unknown-assistant"},
        )
        self.assertEqual(unknown_status, 404)
        self.assertNotEqual(self._run("image", "inspect", flow.trusted_ref, check=False).returncode, 0)

        installed_status, installed = self._api(
            flow.port,
            flow.token,
            "POST",
            "/v1/teams/demo_team/assistants",
            {"assistant": "shimpz-cloudflare"},
        )
        controller_logs = ""
        if installed_status != 200:
            log_result = self._run("logs", flow.controller, check=False)
            controller_logs = (log_result.stdout + log_result.stderr)[-2000:]
        self.assertEqual(installed_status, 200, f"{installed}\n{controller_logs}")
        self.assertTrue(installed["installed"], installed)
        self.assertEqual(self._run("image", "inspect", flow.trusted_ref, check=False).returncode, 0)

        flow.assistant_name = self._run(
            "ps",
            "--all",
            "--filter",
            f"label=com.shimpz.local.space-id={flow.space_id}",
            "--filter",
            "label=com.shimpz.local.assistant-id=shimpz-cloudflare",
            "--format",
            "{{.Names}}",
        ).stdout.strip()
        self.assertTrue(flow.assistant_name)
        flow.original_assistant_id = self._run("inspect", "--format", "{{.Id}}", flow.assistant_name).stdout.strip()
        metadata = json.loads(self._run("inspect", flow.assistant_name).stdout)[0]
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
        self.assertEqual(host["CpusetCpus"], flow.test_cpuset)
        self.assertIn(host.get("Tmpfs"), (None, {}))
        self.assertEqual(metadata["Mounts"], [])
        self.assertIn(host["PortBindings"], (None, {}))
        networks = metadata["NetworkSettings"]["Networks"]
        self.assertEqual(len(networks), 1)
        flow.network_name = next(iter(networks))
        network_metadata = json.loads(self._run("network", "inspect", flow.network_name).stdout)[0]
        self.assertTrue(network_metadata["Internal"])
        self.assertEqual(network_metadata["Labels"]["com.shimpz.local.space-id"], flow.space_id)
        self.assertEqual(network_metadata["Labels"]["com.shimpz.local.team-name"], "Demo Team")

    def _exercise_assistant_recovery(self, flow: _DockerFlow) -> None:
        # Docker still reports "running" when PID 1 is stopped. A retry must replace this explicitly
        # stateless runtime; relying on an external SIGCONT would leave the documented recovery false.
        self._run("kill", "--signal", "STOP", flow.assistant_name)
        stopped_state = self._run("inspect", "--format", "{{.State.Status}}", flow.assistant_name).stdout.strip()
        self.assertEqual(stopped_state, "running")
        recovered_status, recovered = self._api(
            flow.port,
            flow.token,
            "POST",
            "/v1/teams/demo_team/assistants",
            {"assistant": "shimpz-cloudflare"},
        )
        self.assertEqual((recovered_status, recovered["installed"]), (200, False))
        replacement_assistant_id = self._run("inspect", "--format", "{{.Id}}", flow.assistant_name).stdout.strip()
        self.assertNotEqual(replacement_assistant_id, flow.original_assistant_id)
        self.assertNotEqual(self._run("inspect", flow.original_assistant_id, check=False).returncode, 0)

        _, installed_again = self._api(
            flow.port,
            flow.token,
            "POST",
            "/v1/teams/demo_team/assistants",
            {"assistant": "shimpz-cloudflare"},
        )
        self.assertFalse(installed_again["installed"])
        self.assertEqual(
            self._run("inspect", "--format", "{{.Id}}", flow.assistant_name).stdout.strip(),
            replacement_assistant_id,
        )

        _, listed = self._api(flow.port, flow.token, "GET", "/v1/teams/demo_team/assistants")
        self.assertEqual(listed["assistants"], [{"assistant": "shimpz-cloudflare", "status": "running"}])
        help_status, assistant_help = self._api(
            flow.port,
            flow.token,
            "GET",
            "/v1/teams/demo_team/assistants/shimpz-cloudflare/help/pt",
        )
        self.assertEqual(help_status, 200)
        self.assertEqual(assistant_help["assistant"], "shimpz-cloudflare")
        self.assertIn("# Shimpz Cloudflare", assistant_help["markdown"])
        self.assertRegex(assistant_help["trace_id"], r"^[0-9a-f]{32}$")
        secret_status, secret_inventory = self._api(
            flow.port,
            flow.token,
            "GET",
            "/v1/teams/demo_team/assistant-secrets",
        )
        self.assertEqual(secret_status, 200)
        secret_items = secret_inventory["assistants"][0]["secrets"]
        self.assertEqual(secret_items, [])
        account_status, account_inventory = self._api(
            flow.port,
            flow.token,
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
                "assistant_id": "shimpz-cloudflare",
                "id": "cloudflare",
                "provider": "cloudflare",
                "scopes": ["dns.read", "offline_access", "zone.read"],
                "status": "missing",
                "account": None,
                "expires_at": None,
            },
        )
        account_required, missing_account = self._api(
            flow.port,
            flow.token,
            "POST",
            "/v1/teams/demo_team/assistants/shimpz-cloudflare/powers/list-zones",
            {"page": 1, "per_page": 25},
        )
        self.assertEqual(account_required, power_execution.ACCOUNT_PRECONDITION_STATUS)
        self.assertEqual(missing_account["code"], "assistant-account-unavailable")
        unknown_power, _ = self._api(
            flow.port,
            flow.token,
            "POST",
            "/v1/teams/demo_team/assistants/shimpz-cloudflare/powers/shell",
            {},
        )
        self.assertEqual(unknown_power, 404)

    def _exercise_teardown(self, flow: _DockerFlow) -> None:
        proxy_metadata = json.loads(self._run("inspect", flow.egress_proxy).stdout)[0]
        proxy_networks = proxy_metadata["NetworkSettings"]["Networks"]
        self.assertEqual(set(proxy_networks), {flow.outbound_network, flow.network_name})
        self.assertIn("app-egress-proxy", proxy_networks[flow.network_name]["Aliases"])
        policy_contract = self._run(
            "exec",
            flow.controller,
            "/opt/venv/bin/python",
            "-c",
            "import json,os,stat; from pathlib import Path; "
            "p=next(Path('/var/lib/shimpz-local/app-egress').glob('*.json')); s=p.stat(); "
            "print(json.dumps(json.loads(p.read_text())),oct(stat.S_IMODE(s.st_mode)),s.st_uid,s.st_gid)",
        ).stdout.strip()
        self.assertEqual(
            policy_contract,
            '["api.cloudflare.com"] 0o640 10001 10017',
        )

        _, removed = self._api(
            flow.port,
            flow.token,
            "DELETE",
            "/v1/teams/demo_team/assistants/shimpz-cloudflare",
        )
        self.assertTrue(removed["uninstalled"])
        _, removed_again = self._api(
            flow.port,
            flow.token,
            "DELETE",
            "/v1/teams/demo_team/assistants/shimpz-cloudflare",
        )
        self.assertFalse(removed_again["uninstalled"])
        proxy_networks_after_uninstall = json.loads(self._run("inspect", flow.egress_proxy).stdout)[0][
            "NetworkSettings"
        ]["Networks"]
        self.assertEqual(set(proxy_networks_after_uninstall), {flow.outbound_network})
        remaining_policy_files = self._run(
            "exec",
            flow.controller,
            "/opt/venv/bin/python",
            "-c",
            "from pathlib import Path; p=Path('/var/lib/shimpz-local/app-egress'); "
            "print(len(list(p.glob('*.json'))),len(list((p/'.tokens').glob('*.token'))))",
        ).stdout.strip()
        self.assertEqual(remaining_policy_files, "0 0")
        _, deleted_file = self._api(
            flow.port,
            flow.token,
            "DELETE",
            f"/v1/teams/demo_team/files/{flow.file_id}",
        )
        self.assertTrue(deleted_file["deleted"])
        _, destroyed = self._api(flow.port, flow.token, "DELETE", "/v1/teams/demo_team")
        self.assertTrue(destroyed["destroyed"])
        self.assertTrue(destroyed["storage_removed"])
        self.assertNotEqual(
            self._run(
                "exec",
                flow.controller,
                "test",
                "-e",
                "/var/lib/shimpz-local/storage/demo_team",
                check=False,
            ).returncode,
            0,
        )
        _, destroyed_again = self._api(flow.port, flow.token, "DELETE", "/v1/teams/demo_team")
        self.assertFalse(destroyed_again["destroyed"])

    def _exercise_reset(self, flow: _DockerFlow) -> None:
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
            f"com.shimpz.local.space-id={flow.space_id}",
            flow.foreign_network,
        )
        self._api(
            flow.port,
            flow.token,
            "POST",
            "/v1/teams/reset_team/create",
            {"team_name": "Reset Team"},
        )
        self._api(
            flow.port,
            flow.token,
            "POST",
            "/v1/teams/reset_team/assistants",
            {"assistant": "shimpz-cloudflare"},
        )
        self._api(
            flow.port,
            flow.token,
            "POST",
            "/v1/teams/reset_team/files",
            b"remove me",
            extra_headers={
                "Content-Type": "application/octet-stream",
                "X-Shimpz-Filename": "reset.txt",
            },
        )
        reset_status, reset = self._api(flow.port, flow.token, "DELETE", "/v1/space")
        self.assertEqual(reset_status, 200)
        self.assertEqual((reset["assistants_removed"], reset["teams_removed"]), (1, 1))
        _, reset_again = self._api(flow.port, flow.token, "DELETE", "/v1/space")
        self.assertEqual((reset_again["assistants_removed"], reset_again["teams_removed"]), (0, 0))
        self.assertNotEqual(
            self._run(
                "exec",
                flow.controller,
                "test",
                "-e",
                "/var/lib/shimpz-local/storage/reset_team",
                check=False,
            ).returncode,
            0,
        )
        self.assertEqual(self._run("network", "inspect", flow.foreign_network, check=False).returncode, 0)

        audit = self._run(
            "exec",
            flow.controller,
            "/opt/venv/bin/python",
            "-c",
            "from pathlib import Path; print(Path('/var/log/shimpz-local/audit.jsonl').read_text())",
        ).stdout
        self.assertIn('"operation":"space-reset"', audit)
        self.assertIn('"detail":"assistant-account-unavailable"', audit)
        self.assertNotIn("Captain", audit)
        self.assertNotIn(flow.token, audit)

        token_mode = self._run(
            "exec",
            flow.controller,
            "/opt/venv/bin/python",
            "-c",
            "import os,stat; s=os.stat('/run/shimpz-local/token'); "
            "print(oct(stat.S_IMODE(s.st_mode)),s.st_uid,s.st_gid,s.st_nlink)",
        ).stdout.strip()
        self.assertEqual(token_mode, "0o440 10001 10010 1")
        runtime_token_mode = self._run(
            "exec",
            flow.controller,
            "/opt/venv/bin/python",
            "-c",
            "import os,stat; s=os.stat('/run/shimpz-brain-runtime/token'); "
            "print(oct(stat.S_IMODE(s.st_mode)),s.st_uid,s.st_gid,s.st_nlink,s.st_size)",
        ).stdout.strip()
        self.assertEqual(runtime_token_mode, "0o440 10001 10016 1 64")

        # Leave one exact-owned pair for the outer finally. This proves cleanup does not depend
        # on reaching the controller reset route and therefore also runs after an earlier failure.
        self._api(
            flow.port,
            flow.token,
            "POST",
            "/v1/teams/cleanup_team/create",
            {"team_name": "Cleanup Team"},
        )
        self._api(
            flow.port,
            flow.token,
            "POST",
            "/v1/teams/cleanup_team/assistants",
            {"assistant": "shimpz-cloudflare"},
        )
        self.assertEqual(len(self._owned_ids("container", flow.space_id, "assistant")), 1)
        self.assertEqual(len(self._owned_ids("network", flow.space_id, "team")), 1)

    def _cleanup(self, flow: _DockerFlow) -> None:
        flow.brain_server.shutdown()
        flow.brain_server.server_close()
        flow.brain_thread.join(timeout=2)
        # Cleanup remains strictly scoped to this test's unique names/labels.
        self._remove("rm", "--force", flow.egress_proxy)
        self._cleanup_owned_space(flow.space_id)
        owned_containers = self._owned_ids("container", flow.space_id, "assistant")
        owned_networks = self._owned_ids("network", flow.space_id, "team")
        self._remove("rm", "--force", flow.controller)
        self._remove("rm", "--force", flow.registry)
        self._remove("network", "rm", flow.foreign_network)
        self._remove("network", "rm", flow.outbound_network)
        self._remove(
            "volume",
            "rm",
            "--force",
            flow.token_volume,
            flow.runtime_token_volume,
            flow.audit_volume,
            flow.storage_volume,
            flow.inference_volume,
            flow.power_journal_volume,
            flow.approval_state_volume,
            flow.continuation_state_volume,
            flow.continuation_key_volume,
            flow.egress_policy_volume,
            flow.egress_audit_volume,
        )
        if flow.trusted_ref:
            self._remove("image", "rm", "--force", flow.trusted_ref)
        self._remove("image", "rm", "--force", flow.fixture_tag, flow.controller_tag, flow.egress_proxy_tag)
        self._remove("buildx", "rm", "--force", flow.builder)
        self.assertEqual(owned_containers, [])
        self.assertEqual(owned_networks, [])

    @unittest.skipUnless(os.environ.get("SHIMPZ_RUN_DOCKER_TESTS") == "1", "real Docker test is opt-in")
    def test_real_pull_isolation_lifecycle_and_space_reset(self) -> None:
        flow = self._new_flow()
        try:
            self._prepare_images(flow)
            self._start_controller(flow)
            self._exercise_team_storage(flow)
            self._exercise_assistant(flow)
            self._exercise_assistant_recovery(flow)
            self._exercise_teardown(flow)
            self._exercise_reset(flow)
        finally:
            self._cleanup(flow)


if __name__ == "__main__":
    unittest.main()
