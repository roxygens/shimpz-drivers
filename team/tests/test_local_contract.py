from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import threading
import unittest
from contextlib import closing
from dataclasses import replace
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM))
import assistant_secret_challenges
import brain_runtime_client
import inference_config
import local_app
import local_audit
import local_chat_continuation_store
import local_healthcheck
import local_registry
import local_token_store
from assistant_human import approval_challenges as assistant_approval_challenges
from assistant_human import approval_grants as assistant_approval_grants
from assistant_human import input_challenges as assistant_input_challenges
from local_controller_harness import LocalContractCase

LOOKUP_INPUT = {"page": 1, "per_page": 25}
LOOKUP_RESULT = {
    "zones": [],
    "pagination": {"page": 1, "per_page": 25, "count": 0, "total_count": 0, "total_pages": 0},
}
DNS_INPUT = {"zone_id": "a" * 32, "page": 1, "per_page": 25}
DNS_RESULT = {
    "records": [],
    "pagination": {"page": 1, "per_page": 25, "count": 0, "total_count": 0, "total_pages": 0},
}
TEST_SECRET_VALUES = {
    "service-token": "service-test-credential-123456789",
    "client-key": "client-key-test-credential-123456789",
    "client-secret": "client-secret-test-credential-123456789",
    "session-token": "session-token-test-credential-123456789",
    "session-secret": "session-secret-test-credential-123456789",
}
TEST_ACCOUNT_ACCESS_TOKEN = "-".join(("oauth", "access", "test", "token", "123456789"))
TEST_ACCOUNT_REFRESH_TOKEN = "-".join(("oauth", "refresh", "test", "token", "123456789"))
CURRENT_ASSISTANT_IMAGE = "ghcr.io/theshimpz/shimpz-space@sha256:" + "b" * 64
OUTDATED_ASSISTANT_IMAGE = "ghcr.io/theshimpz/shimpz-space@sha256:" + "a" * 64


class LocalContractTests(LocalContractCase):
    def test_local_state_defaults_match_the_installer_mount_contract(self) -> None:
        self.assertEqual(local_token_store.TOKEN_PATH, Path("/run/shimpz-local/token"))
        self.assertEqual(local_healthcheck.TOKEN_PATH, local_token_store.TOKEN_PATH)
        self.assertEqual(local_audit.AUDIT_PATH, Path("/var/log/shimpz-local/audit.jsonl"))
        self.assertEqual(local_app.STORAGE_ROOT, Path("/var/lib/shimpz-local/storage"))
        self.assertEqual(local_app.INFERENCE_ROOT, Path("/var/lib/shimpz-local/inference"))
        self.assertEqual(
            local_app.LOCAL_POWER_JOURNAL_PATH,
            Path("/var/lib/shimpz-local/power-journal/journal.sqlite3"),
        )
        self.assertEqual(
            local_app.LOCAL_CHAT_CONTINUATIONS_STATE_PATH,
            local_chat_continuation_store.STATE_PATH,
        )
        self.assertEqual(
            local_app.LOCAL_CHAT_CONTINUATIONS_KEY_PATH,
            local_chat_continuation_store.KEY_PATH,
        )

    def test_registry_accepts_only_a_non_placeholder_digest(self) -> None:
        digest = "127.0.0.1:5000/shimpz/shimpz-cloudflare@sha256:" + "a" * 64
        registry = self._registry(digest)
        self.assertEqual(registry["shimpz-cloudflare"].image, digest)
        self.assertEqual(registry["shimpz-cloudflare"].name, "Shimpz Cloudflare")
        self.assertEqual(
            set(registry["shimpz-cloudflare"].powers),
            {"list-zones", "list-dns-records"},
        )
        self.assertEqual(
            registry["shimpz-cloudflare"].powers["list-zones"].path,
            "/v1/powers/list-zones",
        )
        self.assertEqual(
            registry["shimpz-cloudflare"].allowed_hosts,
            ("api.cloudflare.com",),
        )
        invalid = (
            "ghcr.io/theshimpz/shimpz-space:latest",
            "ghcr.io/theshimpz/shimpz-space@sha256:" + "0" * 64,
            "https://ghcr.io/theshimpz/hello@sha256:" + "a" * 64,
        )
        for image in invalid:
            with self.subTest(image=image), self.assertRaises(local_registry.RegistryError):
                self._registry(image)

    def test_registry_shape_is_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": 2,
                        "images": {"shimpz-cloudflare": "x"},
                        "command": ["/bin/sh"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(local_registry.RegistryError):
                local_registry.load_registry(path)

    def test_cloudflare_assistant_contract_is_read_only_closed_and_bounded(self) -> None:
        registry = self._registry(CURRENT_ASSISTANT_IMAGE)
        spec = registry["shimpz-cloudflare"]
        self.assertEqual(spec.allowed_hosts, ("api.cloudflare.com",))
        self.assertEqual(spec.health_path, "/healthz")
        self.assertEqual(set(spec.powers), {"list-zones", "list-dns-records"})
        self.assertTrue(all(not hasattr(power, "approval") for power in spec.powers.values()))
        self.assertTrue(all(power.accounts == ("cloudflare",) for power in spec.powers.values()))
        self.assertEqual(
            local_registry.validate_power_input(
                "shimpz-cloudflare",
                "list-dns-records",
                {"zone_id": "a" * 32, "page": 1, "per_page": 100},
            ),
            {"zone_id": "a" * 32, "page": 1, "per_page": 100},
        )
        zones = {
            "zones": [
                {
                    "id": "a" * 32,
                    "name": "example.com",
                    "status": "active",
                    "type": "full",
                    "paused": False,
                    "account": {"id": "b" * 32, "name": "Shimpz"},
                }
            ],
            "pagination": {"page": 1, "per_page": 100, "count": 1, "total_count": 1, "total_pages": 1},
        }
        self.assertEqual(
            local_registry.validate_power_output("shimpz-cloudflare", "list-zones", zones),
            zones,
        )
        with self.assertRaises(ValueError):
            local_registry.validate_power_output(
                "shimpz-cloudflare",
                "list-zones",
                zones | {"access_token": "must-not-cross"},
            )
        for invalid in (
            {"page": 0, "per_page": 100},
            {"page": 1, "per_page": 101},
            {"zone_id": "../zone", "page": 1, "per_page": 10},
            {"page": 1, "per_page": 10, "access_token": "must-not-cross"},
        ):
            power = "list-dns-records" if "zone_id" in invalid else "list-zones"
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                local_registry.validate_power_input("shimpz-cloudflare", power, invalid)

    def test_identifiers_are_strict_and_bounded(self) -> None:
        self.assertEqual(local_app.validate_team_id("demo_team"), "demo_team")
        for invalid in ("Demo", "-demo", "demo-1", "a" * 41, "demo space", ""):
            with self.subTest(invalid=invalid), self.assertRaises(local_app.ApiProblem) as caught:
                local_app.validate_team_id(invalid)
            self.assertEqual(caught.exception.status, HTTPStatus.UNPROCESSABLE_ENTITY)

    def test_local_thread_identity_is_scoped_to_the_network_generation(self) -> None:
        first = local_app._brain_thread_id("local-space", "team_1", "a" * 64)
        second = local_app._brain_thread_id("local-space", "team_1", "b" * 64)

        self.assertEqual(first, f"local:local-space:team_1:{'a' * 64}:default")
        self.assertNotEqual(first, second)
        for space_id, team_id, network_id in (
            ("bad space", "team_1", "a" * 64),
            ("local-space", "bad-team", "a" * 64),
            ("local-space", "team_1", "not-a-docker-id"),
        ):
            with (
                self.subTest(
                    space_id=space_id,
                    team_id=team_id,
                    network_id=network_id,
                ),
                self.assertRaises(local_app.ApiProblem) as caught,
            ):
                local_app._brain_thread_id(space_id, team_id, network_id)
            self.assertEqual(caught.exception.status, HTTPStatus.CONFLICT)

    def test_team_name_matches_the_admin_contract(self) -> None:
        self.assertEqual(local_app.validate_team_name("My Team"), "My Team")
        for invalid in ("", " padded", "padded ", "x\n", "x" * 81, None):
            with self.subTest(invalid=invalid), self.assertRaises(local_app.ApiProblem):
                local_app.validate_team_name(invalid)

    def test_container_limits_and_stateless_recovery_are_intentionally_narrow(self) -> None:
        self.assertEqual(local_app.ASSISTANT_NANO_CPUS, 250_000_000)
        self.assertEqual(local_app.ASSISTANT_MEMORY, 128 * 1024 * 1024)
        self.assertEqual(local_app.ASSISTANT_PIDS, 64)
        self.assertEqual(local_app.half_cpu_set(96), "0-47")
        self.assertEqual(local_app.half_cpu_set(8), "0-3")
        self.assertEqual(local_app.half_cpu_set(1), "0")
        readiness = local_app.ApiProblem(HTTPStatus.BAD_GATEWAY, "not ready", code="assistant-not-ready")
        ownership = local_app.ApiProblem(HTTPStatus.CONFLICT, "drift", code="ownership-conflict")
        self.assertTrue(local_app._is_replaceable_readiness_failure("shimpz-cloudflare", readiness))
        self.assertFalse(local_app._is_replaceable_readiness_failure("future-stateful-assistant", readiness))
        self.assertFalse(local_app._is_replaceable_readiness_failure("shimpz-cloudflare", ownership))

    def test_local_controller_bootstraps_tokens_before_serving(self) -> None:
        events: list[str] = []
        client = SimpleNamespace(close=lambda: events.append("client-close"))
        server = SimpleNamespace(
            serve_forever=lambda **_kwargs: events.append("serve"),
            server_close=lambda: events.append("server-close"),
        )

        with (
            mock.patch.dict(os.environ, {"SHIMPZ_SPACE_ID": "local-space"}),
            mock.patch.object(local_app, "load_registry", side_effect=lambda: events.append("registry") or {}),
            mock.patch.object(
                local_app.local_token_store,
                "ensure_token",
                side_effect=lambda: events.append("controller-token") or "a" * 64,
            ),
            mock.patch.object(
                local_app.brain_runtime_token_store,
                "ensure",
                side_effect=lambda: events.append("runtime-token") or "b" * 64,
            ),
            mock.patch.object(
                local_app.docker,
                "from_env",
                side_effect=lambda **_kwargs: events.append("docker") or client,
            ),
            mock.patch.object(
                local_app.team_storage,
                "TeamStorage",
                side_effect=lambda _path: events.append("storage") or SimpleNamespace(),
            ),
            mock.patch.object(
                local_app,
                "LocalController",
                side_effect=lambda *_args: events.append("controller") or SimpleNamespace(),
            ),
            mock.patch.object(
                local_app,
                "BoundedServer",
                side_effect=lambda *_args: events.append("server") or server,
            ),
            mock.patch.object(local_app.local_audit, "record", side_effect=lambda *_args, **_kwargs: "trace"),
        ):
            result = local_app.main()

        self.assertEqual(result, 0)
        self.assertEqual(
            events,
            [
                "registry",
                "controller-token",
                "runtime-token",
                "docker",
                "storage",
                "controller",
                "server",
                "serve",
                "server-close",
                "client-close",
            ],
        )

    def test_local_controller_accepts_an_injected_power_journal(self) -> None:
        image = "127.0.0.1:5000/shimpz/shimpz-cloudflare@sha256:" + "a" * 64
        injected = SimpleNamespace()
        approval_grants = SimpleNamespace()
        client = SimpleNamespace(
            info=lambda: {"SecurityOptions": ["name=seccomp"], "NCPU": 2},
        )

        controller = local_app.LocalController(
            client,
            "local-space",
            self._registry(image),
            SimpleNamespace(),
            brain_runtime=SimpleNamespace(),
            power_state=injected,
            approval_grants=approval_grants,
        )

        self.assertIs(controller.power_state, injected)
        self.assertIs(controller.approval_grants, approval_grants)
        self.assertEqual(
            local_app.LOCAL_POWER_JOURNAL_PATH,
            Path("/var/lib/shimpz-local/power-journal/journal.sqlite3"),
        )

    def test_ambiguous_power_rpc_is_fail_stopped_or_permanently_blocked(self) -> None:
        controller = object.__new__(local_app.LocalController)
        controller._blocked_power_workloads = set()

        class Stoppable:
            id = "stoppable"
            status = "running"

            def __init__(self) -> None:
                self.attrs = {"State": {"Running": True}}

            def stop(self, *, timeout: int) -> None:
                self.status = "exited"
                self.attrs["State"]["Running"] = False

            def reload(self) -> None:
                return None

            def kill(self) -> None:
                raise AssertionError("a proved stop must not be killed")

        stopped = Stoppable()
        controller._fail_stop_power(stopped)
        self.assertEqual(stopped.status, "exited")
        self.assertNotIn(stopped.id, controller._blocked_power_workloads)

        class Paused:
            id = "paused"
            status = "paused"
            killed = False

            def __init__(self) -> None:
                self.attrs = {"State": {"Running": True}}

            def stop(self, *, timeout: int) -> None:
                return None

            def reload(self) -> None:
                return None

            def kill(self) -> None:
                self.killed = True
                self.attrs["State"]["Running"] = False

        paused = Paused()
        controller._fail_stop_power(paused)
        self.assertTrue(paused.killed)
        self.assertNotIn(paused.id, controller._blocked_power_workloads)

    def test_unprovable_power_stop_is_permanently_blocked(self) -> None:
        controller = object.__new__(local_app.LocalController)
        controller._blocked_power_workloads = set()

        class Ambiguous:
            id = "ambiguous"

            def stop(self, *, timeout: int) -> None:
                raise local_app.DockerException("ambiguous stop")

            def reload(self) -> None:
                raise local_app.DockerException("ambiguous inspect")

            def kill(self) -> None:
                raise local_app.DockerException("ambiguous kill")

        ambiguous = Ambiguous()
        with self.assertRaises(local_app.ApiProblem) as caught:
            controller._fail_stop_power(ambiguous)
        self.assertEqual(caught.exception.code, "assistant-power-blocked")
        self.assertIn(ambiguous.id, controller._blocked_power_workloads)

        class Malformed:
            id = "malformed"

            def __init__(self) -> None:
                self.attrs = {"State": {}}

            def stop(self, *, timeout: int) -> None:
                return None

            def reload(self) -> None:
                return None

            def kill(self) -> None:
                return None

        malformed = Malformed()
        with self.assertRaises(local_app.ApiProblem):
            controller._fail_stop_power(malformed)
        self.assertIn(malformed.id, controller._blocked_power_workloads)

    def test_large_upload_admission_is_single_slot(self) -> None:
        self.assertTrue(local_app._FILE_UPLOAD_SLOTS.acquire(blocking=False))
        try:
            self.assertFalse(local_app._FILE_UPLOAD_SLOTS.acquire(blocking=False))
        finally:
            local_app._FILE_UPLOAD_SLOTS.release()

    def test_inference_configuration_persists_only_provider_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = object.__new__(local_app.LocalController)
            controller._locks = tuple(__import__("threading").RLock() for _ in range(64))
            controller.inference_store = inference_config.InferenceConfigStore(Path(directory) / "inference")
            controller._network = lambda _team_id: object()

            configured = controller.configure_inference(
                "team_1",
                {"provider": "openai", "model": "gpt-5.5"},
            )

            self.assertEqual(
                configured,
                {"team_id": "team_1", "provider": "openai", "model": "gpt-5.5"},
            )
            self.assertEqual(controller.inference_status("team_1"), configured)
            stored = next((Path(directory) / "inference").iterdir()).read_text(encoding="utf-8")
            self.assertNotIn("api_key", stored)
            with self.assertRaises(local_app.ApiProblem):
                controller.configure_inference(
                    "team_1",
                    {"provider": "openai", "model": "gpt-5.5", "api_key": "never"},
                )

    def test_private_model_headers_are_closed_and_never_echoed(self) -> None:
        key = "sk-test-0123456789"
        self.assertEqual(
            local_app.validate_model_credential_headers(["openai"], [key]),
            ("openai", key),
        )
        invalid = (
            ([], [key]),
            (["openai", "openai"], [key]),
            (["unsupported"], [key]),
            (["openai"], []),
            (["openai"], [key, key]),
            (["openai"], ["short"]),
            (["openai"], ["x" * 16 + "\n"]),
        )
        for providers, keys in invalid:
            with self.subTest(providers=providers, keys=len(keys)), self.assertRaises(local_app.ApiProblem) as caught:
                local_app.validate_model_credential_headers(providers, keys)
            self.assertNotIn(key, str(caught.exception))

    def test_private_chat_route_reads_key_from_header_not_json(self) -> None:
        key = "sk-test-0123456789"
        body = json.dumps({"message": "Hello", "files": [], "assistant_ids": ["shimpz-cloudflare"]}).encode()
        captured: dict[str, object] = {}

        class Controller:
            @staticmethod
            def chat(team_id, payload, provider, api_key):
                captured.update(
                    team_id=team_id,
                    payload=payload,
                    provider=provider,
                    api_key=api_key,
                )
                return {"team_name": "Marketing", "reply": "Hello!"}

        token_value = "a" * 32
        handler = object.__new__(local_app.Handler)
        handler.command = "POST"
        handler.server = SimpleNamespace(controller=Controller(), token=token_value)
        handler.headers = Message()
        handler.headers["Authorization"] = f"Bearer {token_value}"
        handler.headers["Content-Type"] = "application/json"
        handler.headers["Content-Length"] = str(len(body))
        handler.headers["X-Shimpz-Model-Provider"] = "openai"
        handler.headers["X-Shimpz-Model-Api-Key"] = key
        handler.rfile = BytesIO(body)

        self.assertTrue(handler._authorized())
        status, response, *_audit = handler._chat_route(["v1", "teams", "team_1", "chat"])

        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(
            captured["payload"],
            {"message": "Hello", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
        )
        self.assertEqual(captured["provider"], "openai")
        self.assertEqual(captured["api_key"], key)
        self.assertNotIn(key, json.dumps(response))

    def test_power_rpc_receives_only_the_validated_input_and_private_envelopes(self) -> None:
        captured: list[object] = []
        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, object())

            def rpc(_container, spec, method, path, payload):
                captured.append((spec.assistant_id, method, path, payload))
                return LOOKUP_RESULT

            controller._rpc = rpc
            audit = mock.patch.object(local_app.local_audit, "record", return_value="trace")
            audit.start()
            self.addCleanup(audit.stop)
            with mock.patch.object(local_app.local_audit, "record", return_value="trace"):
                response = controller.invoke(
                    "team_1",
                    "shimpz-cloudflare",
                    "list-zones",
                    LOOKUP_INPUT,
                )

        self.assertEqual(
            captured,
            [
                (
                    "shimpz-cloudflare",
                    "POST",
                    "/v1/powers/list-zones",
                    {
                        "input": LOOKUP_INPUT,
                        "secrets": {},
                        "accounts": {
                            "cloudflare": {
                                "type": "oauth2-bearer",
                                "access_token": TEST_ACCOUNT_ACCESS_TOKEN,
                            }
                        },
                        "answers": [],
                    },
                )
            ],
        )
        self.assertEqual(response["result"], LOOKUP_RESULT)

    def test_power_rpc_surfaces_a_human_suspension_before_output_validation(self) -> None:
        suspension = local_app.power_execution.RpcSuspension(
            {
                "ordinal": 0,
                "kind": "request",
                "request_type": "str",
                "title": "Name",
                "summary": "Choose a name.",
                "docs": None,
                "options": [],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, object())
            controller._rpc = lambda *_args: suspension
            with mock.patch.object(local_app.local_audit, "record", return_value="trace"):
                response = controller.invoke(
                    "team_1",
                    "shimpz-cloudflare",
                    "list-zones",
                    LOOKUP_INPUT,
                )

        self.assertEqual(response["suspend"], suspension.payload)
        self.assertNotIn("result", response)

    def test_power_output_containing_a_secret_is_blocked_and_redacted(self) -> None:
        raw_secret = TEST_ACCOUNT_ACCESS_TOKEN
        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, object())
            controller._rpc = lambda *_args: {
                "id": "123456789",
                "name": f"unsafe {raw_secret}",
                "username": "OpenAI",
            }
            with (
                mock.patch.object(local_app.local_audit, "record", return_value="trace"),
                self.assertRaises(local_app.ApiProblem) as leaked,
            ):
                controller.invoke(
                    "team_1",
                    "shimpz-cloudflare",
                    "list-zones",
                    LOOKUP_INPUT,
                )

        self.assertEqual(leaked.exception.code, "assistant-secret-exposure")
        self.assertNotIn(raw_secret, str(leaked.exception))

    def test_power_output_containing_a_human_answer_is_blocked_and_redacted(self) -> None:
        answer = "human-submitted-private-value"
        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, object())
            controller._rpc = lambda *_args: {"echo": answer}
            with (
                mock.patch.object(local_app.local_audit, "record", return_value="trace"),
                self.assertRaises(local_app.ApiProblem) as leaked,
            ):
                controller.invoke(
                    "team_1",
                    "shimpz-cloudflare",
                    "list-zones",
                    LOOKUP_INPUT,
                    answers=(answer,),
                )

        self.assertEqual(leaked.exception.code, "assistant-secret-exposure")
        self.assertNotIn(answer, str(leaked.exception))

    def test_chat_exposes_every_active_assistant_to_the_team_brain(self) -> None:
        class Runtime:
            context = None

            def start(self, context, _message):
                self.context = context
                return brain_runtime_client.RuntimeTurn(status="completed", reply="Integrated.", powers=())

        runtime = Runtime()
        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, runtime)
            hello = controller.registry["shimpz-cloudflare"]
            account_helper = replace(
                hello,
                assistant_id="account-helper",
                image=hello.image.replace("a" * 64, "b" * 64),
                powers={"lookup": replace(hello.powers["list-zones"], path="/v1/powers/lookup")},
            )
            controller.registry[account_helper.assistant_id] = account_helper
            controller._active_chat_assistants = lambda _team_id, _network: (
                local_app._ActiveAssistant(hello, "hello-container"),
                local_app._ActiveAssistant(account_helper, "account-helper-container"),
            )

            response = controller.chat(
                "team_1",
                {
                    "message": "Check the accounts",
                    "files": [],
                    "assistant_ids": ["account-helper", "shimpz-cloudflare"],
                },
                "openai",
                "sk-test-0123456789",
            )

        self.assertEqual(
            [assistant.id for assistant in runtime.context.assistants], ["account-helper", "shimpz-cloudflare"]
        )
        self.assertEqual(
            [assistant.genesis for assistant in runtime.context.assistants],
            ["Use only the declared Cloudflare Powers.", "Use only the declared Cloudflare Powers."],
        )
        self.assertEqual(
            runtime.context.thread_id,
            f"local:local-space:team_1:{'a' * 64}:default",
        )
        self.assertEqual(response["team_name"], "Marketing")

    def test_chat_empty_scope_is_brain_only_but_still_scans_installed_workloads(self) -> None:
        class Runtime:
            context = None

            def start(self, context, _message):
                self.context = context
                return brain_runtime_client.RuntimeTurn(status="completed", reply="Brain only.", powers=())

        runtime = Runtime()
        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, runtime)
            scanner = controller._active_chat_assistants
            calls: list[str] = []
            controller._active_chat_assistants = lambda team_id, network: (
                calls.append(f"{team_id}:{network}") or scanner(team_id, network)
            )

            response = controller.chat(
                "team_1",
                {"message": "Hello", "files": [], "assistant_ids": []},
                "openai",
                "sk-test-0123456789",
            )

        self.assertEqual(runtime.context.assistants, ())
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(response["reply"], "Brain only.")

    def test_chat_rejects_invalid_or_unavailable_assistant_scope_before_runtime(self) -> None:
        class Runtime:
            def start(self, _context, _message):
                raise AssertionError("an invalid Assistant scope must not reach the Brain")

        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, Runtime())
            invalid = (
                ["shimpz-cloudflare", "shimpz-cloudflare"],
                ["bad_assistant"],
                [f"helper-{index}" for index in range(local_app.MAX_CHAT_ASSISTANTS + 1)],
            )
            for assistant_ids in invalid:
                with self.subTest(assistant_ids=assistant_ids), self.assertRaises(local_app.ApiProblem) as caught:
                    controller.chat(
                        "team_1",
                        {"message": "Hello", "files": [], "assistant_ids": assistant_ids},
                        "openai",
                        "sk-test-0123456789",
                    )
                self.assertEqual(caught.exception.code, "invalid-assistants")

            with self.assertRaises(local_app.ApiProblem) as unavailable:
                controller.chat(
                    "team_1",
                    {"message": "Hello", "files": [], "assistant_ids": ["account-helper"]},
                    "openai",
                    "sk-test-0123456789",
                )

        self.assertEqual(unavailable.exception.status, HTTPStatus.CONFLICT)
        self.assertEqual(unavailable.exception.code, "assistant-unavailable")
        self.assertEqual(unavailable.exception.message, "a selected Assistant is unavailable")

    def test_chat_revalidates_the_selected_assistant_generation_before_provider_use(self) -> None:
        class Runtime:
            def start(self, _context, _message):
                raise AssertionError("Assistant generation drift must not reach the Brain")

        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, Runtime())
            spec = controller.registry["shimpz-cloudflare"]
            generations = iter(("assistant-v1", "assistant-v2"))
            controller._active_chat_assistants = lambda _team_id, _network: (
                local_app._ActiveAssistant(spec, next(generations)),
            )

            with self.assertRaises(local_app.ApiProblem) as caught:
                controller.chat(
                    "team_1",
                    {"message": "Hello", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                    "openai",
                    "sk-test-0123456789",
                )

        self.assertEqual(caught.exception.code, "team-context-changed")

    def test_chat_power_rejects_a_container_replaced_between_selection_and_rpc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, object())
            frozen = SimpleNamespace(id="assistant-v1", status="running", reload=lambda: None)
            replacement = SimpleNamespace(id="assistant-v2", status="running", reload=lambda: None)
            discovered = iter((frozen, replacement))
            lookups: list[str] = []

            def assistant_container(_team_id: str, _assistant_id: str):
                container = next(discovered)
                lookups.append(container.id)
                return container

            controller._assistant_container = assistant_container
            controller._rpc = lambda *_args: self.fail("a replacement Assistant container executed the Power")
            controller._active_chat_tokens["team_1"] = "turn-token"

            with self.assertRaises(local_app.ApiProblem) as caught:
                controller._invoke_chat_power(
                    "team_1",
                    "turn-token",
                    "shimpz-cloudflare",
                    frozen.id,
                    "list-zones",
                    LOOKUP_INPUT,
                )

        self.assertEqual(lookups, [frozen.id, replacement.id])
        self.assertEqual(caught.exception.status, HTTPStatus.CONFLICT)
        self.assertEqual(caught.exception.code, "team-context-changed")
        self.assertEqual(controller._active_power_containers, {})

    def test_chat_never_exposes_or_executes_an_unselected_assistant(self) -> None:
        class Runtime:
            def start(self, context, _message):
                self.context = context
                return brain_runtime_client.RuntimeTurn(
                    status="power-required",
                    reply="",
                    powers=(
                        brain_runtime_client.PowerRequest(
                            interrupt_id="power-1",
                            assistant_id="account-helper",
                            power="lookup",
                            input=LOOKUP_INPUT,
                        ),
                    ),
                )

        runtime = Runtime()
        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, runtime)
            hello = controller.registry["shimpz-cloudflare"]
            account_helper = replace(
                hello,
                assistant_id="account-helper",
                image=hello.image.replace("a" * 64, "b" * 64),
                powers={"lookup": replace(hello.powers["list-zones"], path="/v1/powers/lookup")},
            )
            controller.registry[account_helper.assistant_id] = account_helper
            controller._active_chat_assistants = lambda _team_id, _network: (
                local_app._ActiveAssistant(hello, "hello-container"),
                local_app._ActiveAssistant(account_helper, "account-helper-container"),
            )
            controller.invoke = lambda *_args: self.fail("an unselected Assistant Power executed")

            with self.assertRaises(local_app.ApiProblem) as caught:
                controller.chat(
                    "team_1",
                    {"message": "Accounts", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                    "openai",
                    "sk-test-0123456789",
                )

        self.assertEqual([assistant.id for assistant in runtime.context.assistants], ["shimpz-cloudflare"])
        self.assertEqual(caught.exception.code, "brain-runtime-failed")

    def test_destroy_drains_chat_and_deletes_generation_before_teardown(self) -> None:
        events: list[object] = []
        controller = object.__new__(local_app.LocalController)
        controller.space_id = "local-space"
        controller.secret_challenges = assistant_secret_challenges.SecretChallengeStore()
        controller.approval_challenges = assistant_approval_challenges.ApprovalChallengeStore()
        controller.input_challenges = assistant_input_challenges.InputChallengeStore()
        controller.approval_grants = SimpleNamespace(revoke_team=lambda _team_id: 0)
        controller.assistant_secrets = SimpleNamespace(delete_team=lambda _team_id: False)
        controller.assistant_accounts = SimpleNamespace(
            delete_team=lambda team_id: events.append(("accounts-delete", team_id))
        )
        controller._active_chat_guard = threading.Lock()
        controller._active_chat_tokens = {"team_1": "turn-token"}
        controller._cancelled_chat_tokens = set()
        controller._active_power_containers = {"team_1": ("turn-token", object())}
        controller._blocked_power_workloads = set()

        class ChatLock:
            def acquire(self, *, timeout: int) -> bool:
                events.append(("chat-lock", timeout))
                return True

            def release(self) -> None:
                events.append("chat-release")

        class LifecycleLock:
            def __enter__(self):
                events.append("lifecycle-lock")

            def __exit__(self, *_args) -> None:
                events.append("lifecycle-release")

        network = SimpleNamespace(
            id="a" * 64,
            name="team-network",
            attrs={"Containers": {}},
            reload=lambda: None,
            remove=lambda: events.append("network-remove"),
        )
        container = SimpleNamespace(
            id="assistant-container",
            labels={local_app.ASSISTANT_LABEL: "shimpz-cloudflare"},
            remove=lambda *, force: events.append(("container-remove", force)),
        )

        def list_containers(**_filters):
            events.append("containers-read")
            return [container]

        controller._fail_stop_power = lambda _container: events.append("power-stopped")
        controller._chat_lock = lambda _team_id: ChatLock()
        controller._lock = lambda _team_id: LifecycleLock()
        controller._network = lambda _team_id, *, required=False: events.append("network-read") or network
        controller._assistant_filters = lambda _team_id: {}
        controller._validate_container_security = lambda *_args: events.append("container-validated")
        controller.registry = {"shimpz-cloudflare": SimpleNamespace(allowed_hosts=())}
        controller.client = SimpleNamespace(containers=SimpleNamespace(list=list_containers))
        controller.brain_runtime = SimpleNamespace(
            delete_thread=lambda thread_id: events.append(("thread-delete", thread_id))
        )
        controller.power_state = SimpleNamespace(purge=lambda generation: events.append(("power-purge", generation)))
        controller.storage = SimpleNamespace(destroy=lambda _team_id: events.append("storage-destroy") or True)
        controller.inference_store = SimpleNamespace(delete=lambda _team_id: events.append("inference-delete"))

        result = controller.destroy_team("team_1")

        expected_thread = local_app._brain_thread_id("local-space", "team_1", "a" * 64)
        self.assertEqual(
            events,
            [
                "power-stopped",
                ("chat-lock", 30),
                "lifecycle-lock",
                "network-read",
                "containers-read",
                "container-validated",
                ("thread-delete", expected_thread),
                ("power-purge", "a" * 64),
                ("container-remove", True),
                "storage-destroy",
                "inference-delete",
                "network-remove",
                ("accounts-delete", "team_1"),
                "lifecycle-release",
                "chat-release",
            ],
        )
        self.assertEqual(
            result,
            {
                "team_id": "team_1",
                "destroyed": True,
                "assistants_removed": 1,
                "storage_removed": True,
            },
        )

    def test_reset_removes_orphan_egress_authority_for_owned_teams(self) -> None:
        events: list[object] = []
        controller = object.__new__(local_app.LocalController)
        controller.space_id = "local-space"
        controller.secret_challenges = SimpleNamespace(cancel_all=lambda: events.append("cancel-secrets"))
        controller.approval_challenges = SimpleNamespace(cancel_all=lambda: events.append("cancel-approvals"))
        controller.input_challenges = SimpleNamespace(cancel_all=lambda: events.append("cancel-inputs"))
        controller._locks = (threading.RLock(),)
        controller._blocked_power_workloads = set()
        controller.registry = {"shimpz-cloudflare": SimpleNamespace()}
        network = SimpleNamespace(
            attrs={"Labels": {local_app.TEAM_LABEL: "team_1"}},
            remove=lambda: events.append("network-remove"),
        )
        controller.client = SimpleNamespace(
            containers=SimpleNamespace(list=lambda **_kwargs: []),
            networks=SimpleNamespace(list=lambda **_kwargs: [network]),
        )
        controller._validate_network = lambda _network, team_id: events.append(("validate-network", team_id))
        controller._delete_all_secret_state = lambda: events.append("delete-secrets")
        controller._delete_all_account_state = lambda: events.append("delete-accounts")
        controller._revoke_all_approval_grants = lambda: events.append("revoke-approvals")
        controller._remove_egress_policy = lambda team_id, assistant_id: events.append(
            ("remove-policy", team_id, assistant_id)
        )
        controller._disconnect_egress_proxy_if_attached = lambda _network: events.append("disconnect-proxy")
        controller.storage = SimpleNamespace(destroy_all=lambda: events.append("destroy-storage") or True)
        controller.inference_store = SimpleNamespace(
            delete=lambda team_id: events.append(("delete-inference", team_id))
        )

        result = controller.reset_space()

        self.assertEqual(result["assistants_removed"], 0)
        self.assertEqual(result["teams_removed"], 1)
        self.assertIn(("remove-policy", "team_1", "shimpz-cloudflare"), events)
        self.assertLess(events.index("delete-accounts"), events.index("network-remove"))

    def test_destroy_brain_failure_is_redacted_and_mutates_nothing(self) -> None:
        events: list[str] = []
        controller = object.__new__(local_app.LocalController)
        controller.space_id = "local-space"
        controller.secret_challenges = assistant_secret_challenges.SecretChallengeStore()
        controller.approval_challenges = assistant_approval_challenges.ApprovalChallengeStore()
        controller.input_challenges = assistant_input_challenges.InputChallengeStore()
        controller.approval_grants = SimpleNamespace(revoke_team=lambda _team_id: 0)
        controller.assistant_secrets = SimpleNamespace(delete_team=lambda _team_id: False)
        controller._active_chat_guard = threading.Lock()
        controller._active_chat_tokens = {}
        controller._cancelled_chat_tokens = set()
        controller._active_power_containers = {}
        controller._blocked_power_workloads = set()
        lock = threading.Lock()
        network = SimpleNamespace(
            id="a" * 64,
            name="team-network",
            remove=lambda: events.append("network-remove"),
        )
        container = SimpleNamespace(
            id="assistant-container",
            labels={local_app.ASSISTANT_LABEL: "shimpz-cloudflare"},
            remove=lambda *, force: events.append("container-remove"),
        )
        controller._chat_lock = lambda _team_id: lock
        controller._lock = lambda _team_id: threading.RLock()
        controller._network = lambda _team_id, *, required=False: network
        controller._assistant_filters = lambda _team_id: {}
        controller._validate_container_security = lambda *_args: None
        controller.registry = {"shimpz-cloudflare": SimpleNamespace(allowed_hosts=())}
        controller.client = SimpleNamespace(containers=SimpleNamespace(list=lambda **_filters: [container]))

        def fail_delete(_thread_id: str) -> None:
            raise brain_runtime_client.BrainRuntimeError("private-checkpoint-data")

        controller.brain_runtime = SimpleNamespace(delete_thread=fail_delete)
        controller.power_state = SimpleNamespace(
            purge=lambda _generation: self.fail("journal purge ran after Brain deletion failed")
        )
        controller.storage = SimpleNamespace(destroy=lambda _team_id: events.append("storage-destroy"))
        controller.inference_store = SimpleNamespace(delete=lambda _team_id: events.append("inference-delete"))

        with self.assertRaises(local_app.ApiProblem) as caught:
            controller.destroy_team("team_1")

        self.assertEqual(caught.exception.status, HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertEqual(caught.exception.message, "Team conversation state could not be deleted")
        self.assertNotIn("private-checkpoint-data", str(caught.exception))
        self.assertEqual(events, [])
        self.assertFalse(lock.locked())

    def test_destroy_journal_failure_is_redacted_before_teardown(self) -> None:
        events: list[object] = []
        controller = object.__new__(local_app.LocalController)
        controller.space_id = "local-space"
        controller.secret_challenges = assistant_secret_challenges.SecretChallengeStore()
        controller.approval_challenges = assistant_approval_challenges.ApprovalChallengeStore()
        controller.input_challenges = assistant_input_challenges.InputChallengeStore()
        controller.approval_grants = SimpleNamespace(revoke_team=lambda _team_id: 0)
        controller.assistant_secrets = SimpleNamespace(delete_team=lambda _team_id: False)
        controller._active_chat_guard = threading.Lock()
        controller._active_chat_tokens = {}
        controller._cancelled_chat_tokens = set()
        controller._active_power_containers = {}
        controller._blocked_power_workloads = set()
        lock = threading.Lock()
        network = SimpleNamespace(
            id="a" * 64,
            name="team-network",
            remove=lambda: events.append("network-remove"),
        )
        container = SimpleNamespace(
            id="assistant-container",
            labels={local_app.ASSISTANT_LABEL: "shimpz-cloudflare"},
            remove=lambda *, force: events.append(("container-remove", force)),
        )
        controller._chat_lock = lambda _team_id: lock
        controller._lock = lambda _team_id: threading.RLock()
        controller._network = lambda _team_id, *, required=False: network
        controller._assistant_filters = lambda _team_id: {}
        controller._validate_container_security = lambda *_args: None
        controller.registry = {"shimpz-cloudflare": SimpleNamespace(allowed_hosts=())}
        controller.client = SimpleNamespace(containers=SimpleNamespace(list=lambda **_filters: [container]))
        controller.brain_runtime = SimpleNamespace(
            delete_thread=lambda thread_id: events.append(("thread-delete", thread_id))
        )

        def fail_purge(generation: str) -> None:
            events.append(("power-purge", generation))
            raise local_app.power_journal.PowerJournalError("private-journal-path")

        controller.power_state = SimpleNamespace(purge=fail_purge)
        controller.storage = SimpleNamespace(destroy=lambda _team_id: events.append("storage-destroy"))
        controller.inference_store = SimpleNamespace(delete=lambda _team_id: events.append("inference-delete"))

        with self.assertRaises(local_app.ApiProblem) as caught:
            controller.destroy_team("team_1")

        expected_thread = local_app._brain_thread_id("local-space", "team_1", "a" * 64)
        self.assertEqual(caught.exception.status, HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertEqual(caught.exception.code, "power-state-unavailable")
        self.assertEqual(caught.exception.message, "Team Power execution state could not be deleted")
        self.assertNotIn("private-journal-path", str(caught.exception))
        self.assertEqual(
            events,
            [("thread-delete", expected_thread), ("power-purge", "a" * 64)],
        )
        self.assertFalse(lock.locked())

    def test_team_identity_drift_stops_before_the_provider_call(self) -> None:
        class Runtime:
            def start(self, _context, _message):
                raise AssertionError("a changed Team must not reach the provider")

        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, Runtime())
            names = iter(("Marketing", "Renamed"))
            controller._validate_network = lambda _network, _team_id: next(names)

            with self.assertRaises(local_app.ApiProblem) as caught:
                controller.chat(
                    "team_1",
                    {"message": "Hello", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                    "openai",
                    "sk-test-0123456789",
                )

        self.assertEqual(caught.exception.code, "team-context-changed")

    def test_chat_executes_only_a_controller_owned_declared_power(self) -> None:
        class Runtime:
            def start(self, _context, _message):
                return brain_runtime_client.RuntimeTurn(
                    status="power-required",
                    reply="",
                    powers=(
                        brain_runtime_client.PowerRequest(
                            interrupt_id="power-1",
                            assistant_id="shimpz-cloudflare",
                            power="list-zones",
                            input=LOOKUP_INPUT,
                        ),
                    ),
                )

            def resume(self, _context, results):
                if results != {"power-1": LOOKUP_RESULT}:
                    raise AssertionError("Power result did not return through the Controller")
                return brain_runtime_client.RuntimeTurn(status="completed", reply="Done", powers=())

        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, Runtime())
            invoked: list[tuple[str, str, object]] = []
            controller.invoke = lambda team_id, assistant, power, payload: (
                invoked.append((team_id, assistant, payload))
                or {"assistant": assistant, "power": power, "result": LOOKUP_RESULT}
            )
            response = controller.chat(
                "team_1",
                {"message": "Greet me", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                "openai",
                "sk-test-0123456789",
            )

        self.assertEqual(invoked, [("team_1", "shimpz-cloudflare", LOOKUP_INPUT)])
        self.assertEqual(response, {"team_id": "team_1", "team_name": "Marketing", "reply": "Done"})

    def test_chat_reuses_a_completed_power_after_resume_failure_then_delivers(self) -> None:
        request = brain_runtime_client.PowerRequest(
            interrupt_id="power-1",
            assistant_id="shimpz-cloudflare",
            power="list-zones",
            input=LOOKUP_INPUT,
        )

        class Runtime:
            resumes = 0

            def start(self, _context, _message):
                return brain_runtime_client.RuntimeTurn(status="power-required", reply="", powers=(request,))

            def resume(self, _context, results):
                self.resumes += 1
                if results != {"power-1": LOOKUP_RESULT}:
                    raise AssertionError("cached Power result changed")
                if self.resumes == 1:
                    raise brain_runtime_client.BrainRuntimeError("private-resume-failure")
                return brain_runtime_client.RuntimeTurn(status="completed", reply="Done", powers=())

        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, Runtime())
            invocations: list[object] = []
            controller.invoke = lambda _team_id, assistant, power, payload: (
                invocations.append(payload) or {"assistant": assistant, "power": power, "result": LOOKUP_RESULT}
            )
            with self.assertRaises(local_app.ApiProblem) as first:
                controller.chat(
                    "team_1",
                    {"message": "Greet me", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                    "openai",
                    "sk-test-0123456789",
                )

            response = controller.chat(
                "team_1",
                {"message": "Greet me", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                "openai",
                "sk-test-0123456789",
            )
            with closing(sqlite3.connect(controller.power_state.path)) as connection:
                pending = connection.execute("SELECT COUNT(*) FROM batches").fetchone()

        self.assertEqual(first.exception.code, "brain-runtime-failed")
        self.assertNotIn("private-resume-failure", str(first.exception))
        self.assertEqual(invocations, [LOOKUP_INPUT])
        self.assertEqual(response["reply"], "Done")
        self.assertEqual(pending, (0,))

    def test_chat_refuses_to_repeat_an_uncertain_power_execution(self) -> None:
        request = brain_runtime_client.PowerRequest(
            interrupt_id="power-1",
            assistant_id="shimpz-cloudflare",
            power="list-zones",
            input=LOOKUP_INPUT,
        )

        class Runtime:
            def start(self, _context, _message):
                return brain_runtime_client.RuntimeTurn(status="power-required", reply="", powers=(request,))

            def resume(self, _context, _results):
                raise AssertionError("an uncertain Power must never reach resume")

        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, Runtime())
            invocations: list[object] = []

            def fail_rpc(*_args):
                invocations.append("rpc")
                raise local_app.ApiProblem(
                    HTTPStatus.BAD_GATEWAY,
                    "private Assistant failure",
                    code="assistant-rpc-failed",
                )

            controller.invoke = fail_rpc
            with self.assertRaises(local_app.ApiProblem) as first:
                controller.chat(
                    "team_1",
                    {"message": "Greet me", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                    "openai",
                    "sk-test-0123456789",
                )
            with self.assertRaises(local_app.ApiProblem) as retry:
                controller.chat(
                    "team_1",
                    {"message": "Greet me", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                    "openai",
                    "sk-test-0123456789",
                )

        self.assertEqual(first.exception.code, "assistant-rpc-failed")
        self.assertEqual(retry.exception.status, HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertEqual(retry.exception.code, "power-state-unavailable")
        self.assertEqual(retry.exception.message, "Team Power execution state is unavailable")
        self.assertNotIn("private Assistant failure", str(retry.exception))
        self.assertEqual(invocations, ["rpc"])

    def test_chat_approval_is_explicit_team_bound_single_use_and_continues_exact_power(self) -> None:
        request = brain_runtime_client.PowerRequest(
            interrupt_id="power-1",
            assistant_id="shimpz-cloudflare",
            power="list-zones",
            input=LOOKUP_INPUT,
        )

        class Runtime:
            def __init__(self) -> None:
                self.resumes: list[dict[str, object]] = []

            def start(self, _context, _message):
                return brain_runtime_client.RuntimeTurn(
                    status="power-required",
                    reply="",
                    powers=(request,),
                )

            def resume(self, _context, results):
                self.resumes.append(dict(results))
                return brain_runtime_client.RuntimeTurn(status="completed", reply="Published.", powers=())

        runtime = Runtime()
        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, runtime)
            answers: list[list[object]] = []

            def rpc(_container, _spec, _method, _path, envelope):
                answers.append(envelope["answers"])
                if not envelope["answers"]:
                    return local_app.power_execution.RpcSuspension(
                        {
                            "ordinal": 0,
                            "kind": "approval",
                            "request_type": "bool",
                            "title": "Publish zones",
                            "summary": "Publish the current zones?",
                            "docs": "https://docs.shimpz.com/",
                            "options": [],
                            "runs": "always",
                        }
                    )
                return LOOKUP_RESULT

            controller._rpc = rpc
            audit = mock.patch.object(local_app.local_audit, "record", return_value="trace")
            audit.start()
            self.addCleanup(audit.stop)
            challenge = controller.chat(
                "team_1",
                {"message": "Publish it", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                "openai",
                "sk-test-0123456789",
            )
            self.assertEqual(challenge["status"], "approval-required")
            self.assertEqual(challenge["requirements"][0]["title"], "Publish zones")
            self.assertEqual(challenge["requirements"][0]["summary"], "Publish the current zones?")
            self.assertEqual(challenge["requirements"][0]["docs"], "https://docs.shimpz.com/")
            self.assertNotIn("power-1", repr(challenge))

            submission = {"challenge_id": challenge["challenge_id"], "approved": True}
            for invalid in (
                {**submission, "approved": False},
                {**submission, "unexpected": True},
            ):
                with self.subTest(invalid=invalid), self.assertRaises(local_app.ApiProblem) as rejected:
                    controller.submit_chat_approval("team_1", invalid, "openai", "sk-test-0123456789")
                self.assertEqual(rejected.exception.code, "invalid-assistant-approval")

            with self.assertRaises(local_app.ApiProblem) as isolated:
                controller.submit_chat_approval("team_2", submission, "openai", "sk-test-0123456789")
            self.assertEqual(isolated.exception.code, "assistant-approval-challenge-expired")

            response = controller.submit_chat_approval("team_1", submission, "openai", "sk-test-0123456789")
            with self.assertRaises(local_app.ApiProblem) as replay:
                controller.submit_chat_approval(
                    "team_1",
                    submission,
                    "openai",
                    "sk-test-0123456789",
                )

        self.assertEqual(response["reply"], "Published.")
        self.assertEqual(answers, [[], [True]])
        self.assertEqual(runtime.resumes, [{"power-1": LOOKUP_RESULT}])
        self.assertEqual(replay.exception.code, "assistant-approval-challenge-expired")

    def test_chat_human_input_replays_typed_answer_into_the_exact_power(self) -> None:
        request = brain_runtime_client.PowerRequest(
            interrupt_id="power-1",
            assistant_id="shimpz-cloudflare",
            power="list-zones",
            input=LOOKUP_INPUT,
        )

        class Runtime:
            def __init__(self) -> None:
                self.resumes: list[dict[str, object]] = []

            def start(self, _context, _message):
                return brain_runtime_client.RuntimeTurn(status="power-required", reply="", powers=(request,))

            def resume(self, _context, results):
                self.resumes.append(dict(results))
                return brain_runtime_client.RuntimeTurn(status="completed", reply="Answered.", powers=())

        runtime = Runtime()
        envelopes: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, runtime)

            def rpc(_container, _spec, _method, _path, envelope):
                envelopes.append(envelope)
                if not envelope["answers"]:
                    return local_app.power_execution.RpcSuspension(
                        {
                            "ordinal": 0,
                            "kind": "request",
                            "request_type": "str",
                            "title": "Name",
                            "summary": "Provide a name.",
                            "docs": None,
                            "options": [],
                        }
                    )
                if len(envelope["answers"]) == 1:
                    return local_app.power_execution.RpcSuspension(
                        {
                            "ordinal": 1,
                            "kind": "request",
                            "request_type": "int",
                            "title": "Count",
                            "summary": "Provide a count.",
                            "docs": None,
                            "options": [],
                        }
                    )
                return LOOKUP_RESULT

            controller._rpc = rpc
            audit = mock.patch.object(local_app.local_audit, "record", return_value="trace")
            audit.start()
            self.addCleanup(audit.stop)
            with mock.patch.object(local_app.local_audit, "record", return_value="trace"):
                challenge = controller.chat(
                    "team_1",
                    {"message": "Ask", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                    "openai",
                    "sk-test-0123456789",
                )
                self.assertEqual(
                    challenge,
                    {
                        "team_id": "team_1",
                        "status": "input-required",
                        "turn_id": challenge["challenge_id"],
                        "challenge_id": challenge["challenge_id"],
                        "request": {
                            "type": "str",
                            "title": "Name",
                            "summary": "Provide a name.",
                            "docs": None,
                            "options": [],
                        },
                    },
                )
                second = controller.submit_chat_input(
                    "team_1",
                    {"challenge_id": challenge["challenge_id"], "answer": "Ada"},
                    "openai",
                    "sk-test-0123456789",
                )
                self.assertEqual(second["status"], "input-required")
                self.assertEqual(second["request"]["type"], "int")
                response = controller.submit_chat_input(
                    "team_1",
                    {"challenge_id": second["challenge_id"], "answer": 3},
                    "openai",
                    "sk-test-0123456789",
                )
                with self.assertRaises(local_app.ApiProblem) as replay:
                    controller.submit_chat_input(
                        "team_1",
                        {"challenge_id": challenge["challenge_id"], "answer": "Ada"},
                        "openai",
                        "sk-test-0123456789",
                    )

        self.assertEqual(response["reply"], "Answered.")
        self.assertEqual([envelope["answers"] for envelope in envelopes], [[], ["Ada"], ["Ada", 3]])
        self.assertEqual(runtime.resumes, [{"power-1": LOOKUP_RESULT}])
        self.assertEqual(replay.exception.code, "assistant-input-challenge-expired")

    def test_chat_human_input_resumes_from_encrypted_disk_after_controller_restart(self) -> None:
        request = brain_runtime_client.PowerRequest(
            interrupt_id="power-1",
            assistant_id="shimpz-cloudflare",
            power="list-zones",
            input=LOOKUP_INPUT,
        )

        class Runtime:
            def __init__(self) -> None:
                self.starts = 0
                self.resumes: list[dict[str, object]] = []

            def start(self, _context, _message):
                self.starts += 1
                return brain_runtime_client.RuntimeTurn(status="power-required", reply="", powers=(request,))

            def resume(self, _context, results):
                self.resumes.append(dict(results))
                return brain_runtime_client.RuntimeTurn(status="completed", reply="Restarted.", powers=())

        runtime = Runtime()
        envelopes: list[dict[str, object]] = []

        def rpc(_container, _spec, _method, _path, envelope):
            envelopes.append(envelope)
            if not envelope["answers"]:
                return local_app.power_execution.RpcSuspension(
                    {
                        "ordinal": 0,
                        "kind": "request",
                        "request_type": "str",
                        "title": "Restart name",
                        "summary": "Provide a restart name.",
                        "docs": None,
                        "options": [],
                    }
                )
            if len(envelope["answers"]) == 1:
                return local_app.power_execution.RpcSuspension(
                    {
                        "ordinal": 1,
                        "kind": "request",
                        "request_type": "int",
                        "title": "Restart count",
                        "summary": "Provide a restart count.",
                        "docs": None,
                        "options": [],
                    }
                )
            return LOOKUP_RESULT

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(local_app.local_audit, "record", return_value="trace"):
                before_restart = self._chat_controller(directory, runtime)
                before_restart._rpc = rpc
                first = before_restart.chat(
                    "team_1",
                    {"message": "Ask across restart", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                    "openai",
                    "sk-test-0123456789",
                )
                second = before_restart.submit_chat_input(
                    "team_1",
                    {"challenge_id": first["challenge_id"], "answer": "Ada"},
                    "openai",
                    "sk-test-0123456789",
                )

                encrypted_state = json.loads(
                    (Path(directory) / "chat-continuations" / "state" / "continuations.json").read_bytes()
                )

                def cleartext_strings(value: object) -> list[str]:
                    if isinstance(value, dict):
                        exposed: list[str] = []
                        for key, item in value.items():
                            exposed.append(str(key))
                            if key not in {"ciphertext", "nonce"}:
                                exposed.extend(cleartext_strings(item))
                        return exposed
                    if isinstance(value, list):
                        return [item for member in value for item in cleartext_strings(member)]
                    return [value] if isinstance(value, str) else []

                exposed_strings = cleartext_strings(encrypted_state)
                self.assertNotIn("Ada", exposed_strings)
                self.assertNotIn("Restart count", exposed_strings)

                after_restart = self._chat_controller(directory, runtime)
                after_restart._rpc = rpc
                restored = after_restart._pending_chat_continuation("team_1")
                self.assertIsNotNone(restored)
                self.assertEqual(restored["challenge_id"], second["challenge_id"])
                response = after_restart.submit_chat_input(
                    "team_1",
                    {"challenge_id": second["challenge_id"], "answer": 3},
                    "openai",
                    "sk-test-0123456789",
                )

            self.assertIsNone(after_restart.chat_continuations.current("team_1"))

        self.assertEqual(response["reply"], "Restarted.")
        self.assertEqual(runtime.starts, 1)
        self.assertEqual(runtime.resumes, [{"power-1": LOOKUP_RESULT}])
        self.assertEqual([envelope["answers"] for envelope in envelopes], [[], ["Ada"], ["Ada", 3]])

    def test_once_approval_is_remembered_for_one_team_assistant_power_release_and_can_be_revoked(self) -> None:
        class Runtime:
            def __init__(self) -> None:
                self.turn = 0

            def start(self, _context, _message):
                self.turn += 1
                request = brain_runtime_client.PowerRequest(
                    interrupt_id=f"power-{self.turn}",
                    assistant_id="shimpz-cloudflare",
                    power="list-zones",
                    input=LOOKUP_INPUT,
                )
                return brain_runtime_client.RuntimeTurn(status="power-required", reply="", powers=(request,))

            def resume(self, _context, _results):
                return brain_runtime_client.RuntimeTurn(status="completed", reply="Published.", powers=())

        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, Runtime())
            answers: list[list[object]] = []

            def rpc(_container, _spec, _method, _path, envelope):
                answers.append(envelope["answers"])
                if not envelope["answers"]:
                    return local_app.power_execution.RpcSuspension(
                        {
                            "ordinal": 0,
                            "kind": "approval",
                            "request_type": "bool",
                            "title": "Publish zones",
                            "summary": "Publish the current zones?",
                            "docs": None,
                            "options": [],
                            "runs": "once",
                        }
                    )
                return LOOKUP_RESULT

            controller._rpc = rpc
            audit = mock.patch.object(local_app.local_audit, "record", return_value="trace")
            audit.start()
            self.addCleanup(audit.stop)

            first = controller.chat(
                "team_1",
                {"message": "First", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                "openai",
                "sk-test-0123456789",
            )
            self.assertEqual(first["requirements"][0]["approval"], "once")
            controller.submit_chat_approval(
                "team_1",
                {"challenge_id": first["challenge_id"], "approved": True},
                "openai",
                "sk-test-0123456789",
            )
            second = controller.chat(
                "team_1",
                {"message": "Second", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                "openai",
                "sk-test-0123456789",
            )
            controller.approval_grants.grant_many(
                (
                    assistant_approval_grants.Grant(
                        "team_1",
                        "shimpz-cloudflare",
                        "list-zones",
                        CURRENT_ASSISTANT_IMAGE,
                        1,
                    ),
                )
            )
            inventory = controller.list_assistant_approval_grants("team_1")
            revoked = controller.revoke_assistant_approval_grants("team_1")
            third = controller.chat(
                "team_1",
                {"message": "Third", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                "openai",
                "sk-test-0123456789",
            )

        self.assertEqual(second["reply"], "Published.")
        self.assertEqual(answers, [[], [True], [], [True], []])
        self.assertEqual(
            inventory,
            {
                "team_id": "team_1",
                "grants": [{"assistant_id": "shimpz-cloudflare", "power_id": "list-zones"}],
            },
        )
        self.assertEqual(revoked, {"team_id": "team_1", "revoked": 2})
        self.assertEqual(third["status"], "approval-required")

    def test_secret_continuation_can_pause_for_approval_before_any_power_runs(self) -> None:
        request = brain_runtime_client.PowerRequest(
            interrupt_id="create-1",
            assistant_id="shimpz-cloudflare",
            power="list-zones",
            input=LOOKUP_INPUT,
        )

        class Runtime:
            def start(self, _context, _message):
                return brain_runtime_client.RuntimeTurn(status="power-required", reply="", powers=(request,))

            def resume(self, _context, results):
                self.results = dict(results)
                return brain_runtime_client.RuntimeTurn(status="completed", reply="Published.", powers=())

        runtime = Runtime()
        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, runtime, configure_secrets=False)
            answers: list[list[object]] = []

            def rpc(_container, _spec, _method, _path, envelope):
                answers.append(envelope["answers"])
                if not envelope["answers"]:
                    return local_app.power_execution.RpcSuspension(
                        {
                            "ordinal": 0,
                            "kind": "approval",
                            "request_type": "bool",
                            "title": "Publish zones",
                            "summary": "Publish the current zones?",
                            "docs": None,
                            "options": [],
                            "runs": "always",
                        }
                    )
                return LOOKUP_RESULT

            controller._rpc = rpc
            audit = mock.patch.object(local_app.local_audit, "record", return_value="trace")
            audit.start()
            self.addCleanup(audit.stop)
            secret_challenge = controller.chat(
                "team_1",
                {"message": "Publish", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                "openai",
                "sk-test-0123456789",
            )
            approval_challenge = controller.submit_chat_secrets(
                "team_1",
                self._secret_submission(secret_challenge),
                "openai",
                "sk-test-0123456789",
            )
            self.assertEqual(approval_challenge["status"], "approval-required")
            self.assertEqual(approval_challenge["requirements"][0]["title"], "Publish zones")
            response = controller.submit_chat_approval(
                "team_1",
                {"challenge_id": approval_challenge["challenge_id"], "approved": True},
                "openai",
                "sk-test-0123456789",
            )

        self.assertEqual(response["reply"], "Published.")
        self.assertEqual(answers, [[], [True]])
        self.assertEqual(runtime.results, {"create-1": LOOKUP_RESULT})

    def test_approval_challenge_transfers_to_a_cancellable_active_turn_without_a_gap(self) -> None:
        request = brain_runtime_client.PowerRequest(
            interrupt_id="power-1",
            assistant_id="shimpz-cloudflare",
            power="list-zones",
            input=LOOKUP_INPUT,
        )

        class Runtime:
            def start(self, _context, _message):
                return brain_runtime_client.RuntimeTurn(status="power-required", reply="", powers=(request,))

            def resume(self, _context, _results):
                raise AssertionError("a cancelled approval must never resume")

        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, Runtime())
            controller._rpc = lambda *_args: local_app.power_execution.RpcSuspension(
                {
                    "ordinal": 0,
                    "kind": "approval",
                    "request_type": "bool",
                    "title": "Publish zones",
                    "summary": "Publish the current zones?",
                    "docs": None,
                    "options": [],
                    "runs": "always",
                }
            )
            audit = mock.patch.object(local_app.local_audit, "record", return_value="trace")
            audit.start()
            self.addCleanup(audit.stop)
            challenge = controller.chat(
                "team_1",
                {"message": "Publish", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                "openai",
                "sk-test-0123456789",
            )
            claiming = threading.Event()
            release = threading.Event()
            original_claim = controller.approval_challenges.claim

            def blocked_claim(team_id, challenge_id):
                claiming.set()
                if not release.wait(timeout=2):
                    raise AssertionError("test did not release approval claim")
                return original_claim(team_id, challenge_id)

            failures: list[BaseException] = []

            def submit() -> None:
                try:
                    controller.submit_chat_approval(
                        "team_1",
                        {"challenge_id": challenge["challenge_id"], "approved": True},
                        "openai",
                        "sk-test-0123456789",
                    )
                except local_app.ApiProblem as exc:
                    failures.append(exc)

            with mock.patch.object(controller.approval_challenges, "claim", side_effect=blocked_claim):
                thread = threading.Thread(target=submit)
                thread.start()
                self.assertTrue(claiming.wait(timeout=2))
                repeated = controller.chat(
                    "team_1",
                    {"message": "Different turn", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                    "openai",
                    "sk-test-0123456789",
                )
                self.assertEqual(repeated["challenge_id"], challenge["challenge_id"])
                stopped = controller.stop_chat("team_1")
                release.set()
                thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertTrue(stopped["accepted"])
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], local_app.ApiProblem)
        self.assertIn(failures[0].code, {"assistant-approval-challenge-expired", "chat-stopped"})

    def test_stop_discards_a_runtime_reply_that_finishes_late(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class Runtime:
            def start(self, _context, _message):
                started.set()
                if not release.wait(timeout=2):
                    raise AssertionError("test did not release runtime")
                return brain_runtime_client.RuntimeTurn(status="completed", reply="must be discarded", powers=())

        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, Runtime())
            failures: list[BaseException] = []

            def turn() -> None:
                try:
                    controller.chat(
                        "team_1",
                        {"message": "Wait", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                        "openai",
                        "sk-test-0123456789",
                    )
                except local_app.ApiProblem as exc:
                    failures.append(exc)

            worker = threading.Thread(target=turn)
            worker.start()
            self.assertTrue(started.wait(timeout=1))
            stopped = controller.stop_chat("team_1")
            release.set()
            worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertTrue(stopped["accepted"])
        self.assertFalse(stopped["confirmed"])
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], local_app.ApiProblem)
        self.assertEqual(failures[0].code, "chat-stopped")

    def test_assistant_lifecycle_is_rejected_before_mutation_during_an_active_chat(self) -> None:
        controller, _container, events = self._lifecycle_controller()
        chat_lock = controller._chat_lock("team_1")
        self.assertTrue(chat_lock.acquire(blocking=False))
        try:
            operations = (controller.install_assistant, controller.uninstall_assistant)
            for operation in operations:
                with self.subTest(operation=operation.__name__), self.assertRaises(local_app.ApiProblem) as caught:
                    operation("team_1", "shimpz-cloudflare")
                self.assertEqual((caught.exception.status, caught.exception.code), (HTTPStatus.CONFLICT, "chat-active"))
        finally:
            chat_lock.release()

        self.assertEqual(events, [])

    def test_install_replaces_an_outdated_release_after_current_contract_admission(self) -> None:
        controller, container, events = self._lifecycle_controller()
        controller.assistant_accounts.put(
            "team_1",
            "shimpz-cloudflare",
            "obsolete-account",
            "cloudflare",
            ("zone.read",),
            SimpleNamespace(
                access_token=TEST_ACCOUNT_ACCESS_TOKEN,
                refresh_token=TEST_ACCOUNT_REFRESH_TOKEN,
                scopes=("zone.read",),
                expires_in=3600,
            ),
        )
        trusted_image = object()
        controller._trusted_image = lambda _spec: events.append("trusted") or trusted_image
        controller._create_assistant_container = lambda _team_id, _spec, _network, image: events.append(
            ("create", image)
        )

        result = controller.install_assistant("team_1", "shimpz-cloudflare")

        self.assertEqual(result, {"assistant": "shimpz-cloudflare", "installed": False})
        self.assertEqual(events, ["reload", "trusted", "reload", ("remove", True), ("create", trusted_image)])
        self.assertEqual(container.attrs["Config"]["Image"], OUTDATED_ASSISTANT_IMAGE)
        self.assertFalse(controller.assistant_accounts.delete_assistant("team_1", "shimpz-cloudflare"))

    def test_release_update_is_generic_for_future_assistants(self) -> None:
        controller, container, events = self._lifecycle_controller()
        spec = controller.registry.pop("shimpz-cloudflare")
        spec.assistant_id = "future-assistant"
        controller.registry[spec.assistant_id] = spec
        labels = container.attrs["Config"]["Labels"]
        labels[local_app.ASSISTANT_LABEL] = spec.assistant_id
        container.name = controller._container_name("team_1", spec.assistant_id)
        controller._trusted_image = lambda _spec: events.append("trusted") or object()
        controller._create_assistant_container = lambda *_args: events.append("create")

        self.assertEqual(
            controller.list_assistants("team_1"),
            {"assistants": [{"assistant": "future-assistant", "status": "outdated"}]},
        )
        self.assertEqual(
            controller.install_assistant("team_1", "future-assistant"),
            {"assistant": "future-assistant", "installed": False},
        )
        self.assertEqual(events, ["reload", "reload", "trusted", "reload", ("remove", True), "create"])

    def test_release_update_rejects_a_previous_security_contract(self) -> None:
        controller, _container, events = self._lifecycle_controller()
        controller.registry["shimpz-cloudflare"].allowed_hosts = ("api.example.com",)
        controller._trusted_image = lambda _spec: self.fail("contract drift reached image resolution")

        with self.assertRaises(local_app.ApiProblem) as caught:
            controller.install_assistant("team_1", "shimpz-cloudflare")

        self.assertEqual(caught.exception.code, "egress-policy-drift")
        self.assertEqual(events, ["reload"])

    def test_container_profile_rejects_duplicate_or_malformed_environment_entries(self) -> None:
        invalid_environments = (
            ["SHIMPZ_TEAM_ID=team_1", "SHIMPZ_TEAM_ID=other"],
            ["HTTPS_PROXY=http://safe", "HTTPS_PROXY=http://evil"],
            ["missing-separator"],
        )
        for environment in invalid_environments:
            with self.subTest(environment=environment):
                controller, container, events = self._lifecycle_controller()
                container.attrs["Config"]["Env"] = environment

                with self.assertRaises(local_app.ApiProblem) as caught:
                    controller.list_assistants("team_1")

                self.assertEqual(caught.exception.code, "assistant-isolation-drift")
                self.assertEqual(events, ["reload"])

    def test_unready_same_release_recovery_preserves_once_approval(self) -> None:
        controller, container, _events = self._lifecycle_controller()
        spec = controller.registry["shimpz-cloudflare"]
        controller.approval_grants.grant_many(
            (
                assistant_approval_grants.Grant(
                    "team_1",
                    "shimpz-cloudflare",
                    "list-zones",
                    CURRENT_ASSISTANT_IMAGE,
                    0,
                ),
            )
        )
        controller._trusted_image = lambda _spec: object()
        controller._validate_container = lambda *_args: None
        controller._create_assistant_container = lambda *_args: None

        controller._replace_unready_assistant("team_1", spec, SimpleNamespace(name="team-network"), container)

        self.assertTrue(
            controller.approval_grants.is_granted(
                "team_1",
                "shimpz-cloudflare",
                "list-zones",
                CURRENT_ASSISTANT_IMAGE,
                0,
            )
        )

    def test_new_assistant_is_admitted_before_egress_and_start(self) -> None:
        events: list[object] = []
        controller = object.__new__(local_app.LocalController)
        controller.space_id = "local-space"
        controller.cpuset_cpus = "0"
        controller._assistant_genesis_cache = local_app.assistant_genesis.GenesisCache()
        controller._assistant_allowed_hosts_cache = local_app.assistant_manifest.ManifestContractCache()
        controller._assistant_machine_contract_cache = local_app.assistant_manifest.MachineContractCache()
        controller._blocked_power_workloads = set()
        spec = SimpleNamespace(
            assistant_id="shimpz-cloudflare",
            image=CURRENT_ASSISTANT_IMAGE,
            allowed_hosts=("api.open-meteo.com", "geocoding-api.open-meteo.com"),
        )
        network = SimpleNamespace(name=controller._network_name("team_1"))
        image = SimpleNamespace(id="sha256:" + "d" * 64)
        container = SimpleNamespace(
            id="assistant-generation",
            attrs={"Image": image.id},
            reload=lambda: events.append("reload"),
            start=lambda: events.append("start"),
            remove=lambda *, force: events.append(("remove", force)),
        )
        controller.client = SimpleNamespace(
            containers=SimpleNamespace(
                create=lambda **_kwargs: events.append("create") or container,
            )
        )
        controller._egress_token = lambda *_args, **_kwargs: events.append("token") or "a" * 32
        controller._admit_assistant_allowed_hosts = lambda _container, _spec: (
            events.append("admit") or tuple(sorted(_spec.allowed_hosts))
        )
        controller._activate_assistant_egress = lambda *_args: events.append("activate-egress")
        controller._validate_container = lambda *_args: events.append("validate")
        controller._wait_ready = lambda *_args: events.append("ready")
        controller._active_assistant_genesis = lambda *_args: events.append("genesis") or "Genesis"

        controller._create_assistant_container("team_1", spec, network, image)

        self.assertLess(events.index("admit"), events.index("activate-egress"))
        self.assertLess(events.index("admit"), events.index("start"))
        self.assertEqual(events[-4:], ["start", "validate", "ready", "genesis"])

    def test_local_admission_reviews_hosts_and_accounts(self) -> None:
        controller = object.__new__(local_app.LocalController)
        reviewed_contracts: list[local_app.assistant_manifest.ManifestContract] = []

        def admit(_container, reviewed):
            reviewed_contracts.append(reviewed)
            return reviewed

        controller._assistant_allowed_hosts_cache = SimpleNamespace(get=admit)
        controller._assistant_machine_contract_cache = SimpleNamespace(
            get=lambda _container, _accounts, reviewed: reviewed
        )
        spec = self._registry(CURRENT_ASSISTANT_IMAGE)["shimpz-cloudflare"]

        allowed_hosts = controller._admit_assistant_allowed_hosts(SimpleNamespace(id="generation"), spec)

        self.assertEqual(allowed_hosts, tuple(sorted(spec.allowed_hosts)))
        self.assertEqual(len(reviewed_contracts), 1)
        self.assertEqual(
            {account.id: (account.provider, account.scopes) for account in reviewed_contracts[0].accounts},
            {
                account_id: (account.provider, tuple(sorted(account.scopes)))
                for account_id, account in spec.accounts.items()
            },
        )
        exact = reviewed_contracts[0]
        account = exact.accounts[0]
        drifted = (
            replace(exact, accounts=(replace(account, provider="other"),)),
            replace(exact, accounts=(replace(account, scopes=("tweet.read",)),)),
        )
        controller._assistant_allowed_hosts_cache = local_app.assistant_manifest.ManifestContractCache()
        with mock.patch.object(
            local_app.assistant_manifest,
            "read_container_manifest_contract",
            return_value=exact,
        ):
            self.assertEqual(
                controller._admit_assistant_allowed_hosts(SimpleNamespace(id="exact-generation"), spec),
                exact.allowed_hosts,
            )
        for index, declared in enumerate(drifted):
            controller._assistant_allowed_hosts_cache = local_app.assistant_manifest.ManifestContractCache()
            with (
                self.subTest(declared=declared),
                mock.patch.object(
                    local_app.assistant_manifest,
                    "read_container_manifest_contract",
                    return_value=declared,
                ),
                self.assertRaises(local_app.ApiProblem) as drift,
            ):
                controller._admit_assistant_allowed_hosts(SimpleNamespace(id=f"drift-generation-{index}"), spec)
            self.assertEqual(drift.exception.code, "assistant-manifest-invalid")

    def test_manifest_mismatch_removes_stopped_container_without_activating_egress(self) -> None:
        events: list[object] = []
        controller = object.__new__(local_app.LocalController)
        controller.space_id = "local-space"
        controller.cpuset_cpus = "0"
        controller._assistant_genesis_cache = local_app.assistant_genesis.GenesisCache()
        controller._assistant_allowed_hosts_cache = local_app.assistant_manifest.ManifestContractCache()
        controller._assistant_machine_contract_cache = local_app.assistant_manifest.MachineContractCache()
        spec = SimpleNamespace(
            assistant_id="shimpz-cloudflare",
            image=CURRENT_ASSISTANT_IMAGE,
            allowed_hosts=("api.open-meteo.com",),
        )
        network = SimpleNamespace(name=controller._network_name("team_1"))
        image = SimpleNamespace(id="sha256:" + "d" * 64)
        container = SimpleNamespace(
            id="assistant-generation",
            attrs={"Image": image.id},
            reload=lambda: events.append("reload"),
            start=lambda: events.append("start"),
            remove=lambda *, force: events.append(("remove", force)),
        )
        controller.client = SimpleNamespace(containers=SimpleNamespace(create=lambda **_kwargs: container))
        controller._egress_token = lambda *_args, **_kwargs: "a" * 32
        controller._admit_assistant_allowed_hosts = lambda *_args: (_ for _ in ()).throw(
            local_app.ApiProblem(
                HTTPStatus.CONFLICT,
                "installed Assistant manifest failed its reviewed contract",
                code="assistant-manifest-invalid",
            )
        )
        controller._activate_assistant_egress = lambda *_args: events.append("activate-egress")
        controller._release_assistant_egress = lambda *_args: events.append("release-egress")

        with self.assertRaises(local_app.ApiProblem) as caught:
            controller._create_assistant_container("team_1", spec, network, image)

        self.assertEqual(caught.exception.code, "assistant-manifest-invalid")
        self.assertNotIn("start", events)
        self.assertNotIn("activate-egress", events)
        self.assertEqual(events, ["reload", ("remove", True), "release-egress"])

    def test_failed_install_removal_still_revokes_egress_and_reports_incomplete_rollback(self) -> None:
        events: list[object] = []
        controller = object.__new__(local_app.LocalController)
        controller.space_id = "local-space"
        controller.cpuset_cpus = "0"
        controller._assistant_genesis_cache = local_app.assistant_genesis.GenesisCache()
        controller._assistant_allowed_hosts_cache = local_app.assistant_manifest.ManifestContractCache()
        controller._assistant_machine_contract_cache = local_app.assistant_manifest.MachineContractCache()
        controller._blocked_power_workloads = set()
        spec = SimpleNamespace(
            assistant_id="shimpz-cloudflare",
            image=CURRENT_ASSISTANT_IMAGE,
            allowed_hosts=("api.open-meteo.com",),
        )
        network = SimpleNamespace(name=controller._network_name("team_1"))
        image = SimpleNamespace(id="sha256:" + "d" * 64)

        class Container:
            id = "assistant-generation"

            def __init__(self) -> None:
                self.attrs = {"Image": image.id, "State": {"Running": False}}

            def reload(self) -> None:
                events.append("reload")

            def remove(self, *, force: bool) -> None:
                events.append(("remove", force))
                raise local_app.DockerException("ambiguous removal")

            def stop(self, *, timeout: int) -> None:
                events.append(("stop", timeout))

            def kill(self) -> None:
                self.fail("a proved stopped container must not be killed")

        container = Container()
        controller.client = SimpleNamespace(containers=SimpleNamespace(create=lambda **_kwargs: container))
        controller._egress_token = lambda *_args, **_kwargs: "a" * 32
        controller._admit_assistant_allowed_hosts = lambda *_args: (_ for _ in ()).throw(
            local_app.ApiProblem(
                HTTPStatus.CONFLICT,
                "installed Assistant manifest failed its reviewed contract",
                code="assistant-manifest-invalid",
            )
        )
        controller._activate_assistant_egress = lambda *_args: events.append("activate-egress")
        controller._release_assistant_egress = lambda *_args: events.append("release-egress")

        with self.assertRaises(local_app.ApiProblem) as caught:
            controller._create_assistant_container("team_1", spec, network, image)

        self.assertEqual(caught.exception.code, "assistant-install-rollback-incomplete")
        self.assertNotIn("activate-egress", events)
        self.assertEqual(
            events,
            ["reload", ("remove", True), ("stop", 3), "reload", "release-egress"],
        )

    def test_uninstall_removes_an_outdated_release_after_current_contract_admission(self) -> None:
        controller, _container, events = self._lifecycle_controller()
        controller.assistant_accounts.put(
            "team_1",
            "shimpz-cloudflare",
            "cloudflare",
            "cloudflare",
            ("zone.read",),
            SimpleNamespace(
                access_token=TEST_ACCOUNT_ACCESS_TOKEN,
                refresh_token=TEST_ACCOUNT_REFRESH_TOKEN,
                scopes=("zone.read",),
                expires_in=3600,
            ),
        )

        result = controller.uninstall_assistant("team_1", "shimpz-cloudflare")

        self.assertEqual(result, {"assistant": "shimpz-cloudflare", "uninstalled": True})
        self.assertEqual(events, ["reload", ("remove", True)])
        self.assertFalse(controller.assistant_accounts.delete_assistant("team_1", "shimpz-cloudflare"))

    def test_install_rejects_security_drift_without_resolving_or_removing(self) -> None:
        controller, container, events = self._lifecycle_controller()
        container.attrs["HostConfig"]["Privileged"] = True
        controller._trusted_image = lambda _spec: self.fail("security drift reached image resolution")

        with self.assertRaises(local_app.ApiProblem) as caught:
            controller.install_assistant("team_1", "shimpz-cloudflare")

        self.assertEqual(
            (caught.exception.status, caught.exception.code),
            (HTTPStatus.CONFLICT, "assistant-isolation-drift"),
        )
        self.assertEqual(events, ["reload"])

    def test_uninstall_never_removes_a_container_with_wrong_ownership(self) -> None:
        controller, container, events = self._lifecycle_controller()
        container.labels[local_app.SPACE_LABEL] = "other-space"

        with self.assertRaises(local_app.ApiProblem) as caught:
            controller.uninstall_assistant("team_1", "shimpz-cloudflare")

        self.assertEqual(caught.exception.code, "assistant-isolation-drift")
        self.assertEqual(events, ["reload"])

    def test_list_marks_only_artifact_drift_outdated_and_rejects_security_drift(self) -> None:
        controller, container, events = self._lifecycle_controller()

        self.assertEqual(
            controller.list_assistants("team_1"),
            {"assistants": [{"assistant": "shimpz-cloudflare", "status": "outdated"}]},
        )
        with self.assertRaises(local_app.ApiProblem) as update_required:
            controller._validate_container(
                container,
                "team_1",
                controller.registry["shimpz-cloudflare"],
                controller._network_name("team_1"),
            )
        self.assertEqual(update_required.exception.code, "assistant-update-required")
        self.assertEqual(update_required.exception.message, "the installed Assistant must be updated")
        container.attrs["HostConfig"]["ReadonlyRootfs"] = False
        with self.assertRaises(local_app.ApiProblem) as caught:
            controller.list_assistants("team_1")

        self.assertEqual(caught.exception.code, "assistant-isolation-drift")
        self.assertEqual(events, ["reload", "reload", "reload"])

    def test_list_keeps_the_new_manifest_contract_strict(self) -> None:
        controller, container, _events = self._lifecycle_controller()
        container.labels[local_app.IMAGE_LABEL] = CURRENT_ASSISTANT_IMAGE
        container.attrs["Config"]["Image"] = CURRENT_ASSISTANT_IMAGE

        def reject(*_args):
            raise local_app.ApiProblem(
                HTTPStatus.CONFLICT,
                "installed Assistant manifest failed its reviewed contract",
                code="assistant-manifest-invalid",
            )

        controller._admit_assistant_allowed_hosts = reject
        with self.assertRaises(local_app.ApiProblem) as caught:
            controller.list_assistants("team_1")

        self.assertEqual(caught.exception.code, "assistant-manifest-invalid")

    def test_outdated_release_lineage_is_closed_before_lifecycle_actions(self) -> None:
        self.assertTrue(local_registry.is_digest_ref(OUTDATED_ASSISTANT_IMAGE))
        self.assertFalse(local_registry.is_digest_ref("ghcr.io/theshimpz/shimpz-space@sha256:" + "0" * 64))
        self.assertFalse(local_registry.is_digest_ref("ghcr.io/theshimpz/shimpz-space:latest"))

        for drift in ("missing-label", "image-label-mismatch", "foreign-repository", "wrong-name"):
            with self.subTest(drift=drift):
                controller, container, events = self._lifecycle_controller()
                if drift == "missing-label":
                    container.labels.pop(local_app.IMAGE_LABEL)
                elif drift == "image-label-mismatch":
                    container.attrs["Config"]["Image"] = CURRENT_ASSISTANT_IMAGE
                elif drift == "foreign-repository":
                    foreign = "evil.example/shimpz-space@sha256:" + "c" * 64
                    container.labels[local_app.IMAGE_LABEL] = foreign
                    container.attrs["Config"]["Image"] = foreign
                else:
                    container.name = "foreign-container"

                with self.assertRaises(local_app.ApiProblem) as caught:
                    controller.list_assistants("team_1")

                self.assertEqual(caught.exception.code, "assistant-isolation-drift")
                self.assertEqual(events, ["reload"])


if __name__ == "__main__":
    unittest.main()
