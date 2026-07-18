from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

CAPSULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAPSULE))

import brain_runtime_client
import inference_config
import local_app
import local_registry


class LocalContractTests(unittest.TestCase):
    def _registry(self, image: str) -> dict[str, local_registry.AssistantSpec]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps({"schema": 1, "hello_pulse_image": image}), encoding="utf-8")
            return local_registry.load_registry(path)

    def _chat_controller(self, directory: str, runtime) -> local_app.LocalController:
        image = "127.0.0.1:5000/shimpz/hello-pulse@sha256:" + "a" * 64
        controller = object.__new__(local_app.LocalController)
        controller.space_id = "local-space"
        controller.registry = self._registry(image)
        controller.storage = SimpleNamespace(metadata=lambda _cid, _files: [])
        controller.inference_store = inference_config.InferenceConfigStore(Path(directory) / "inference")
        controller.inference_store.save(
            "capsule_1",
            inference_config.normalize("openai", "gpt-5.5"),
        )
        controller.brain_runtime = runtime
        controller._blocked_power_workloads = set()
        controller._locks = tuple(threading.RLock() for _ in range(64))
        controller._active_chat_guard = threading.Lock()
        controller._chat_locks = {}
        controller._active_chat_tokens = {}
        controller._active_power_containers = {}
        controller._cancelled_chat_tokens = set()
        container = SimpleNamespace(id="assistant-container", status="running", reload=lambda: None)
        network = SimpleNamespace(id="team-network-id", name="capsule-network")
        controller._network = lambda _cid: network
        controller._validate_network = lambda _network, _cid: "Marketing"
        controller._assistant_container = lambda _cid, _assistant: container
        controller._validate_container = lambda *_args: None
        controller._active_chat_assistants = lambda _cid, _network: (
            local_app._ActiveAssistant(controller.registry["hello-pulse"], container.id),
        )
        return controller

    def test_registry_accepts_only_a_non_placeholder_digest(self) -> None:
        digest = "127.0.0.1:5000/shimpz/hello-pulse@sha256:" + "a" * 64
        registry = self._registry(digest)
        self.assertEqual(registry["hello-pulse"].image, digest)
        self.assertEqual(set(registry["hello-pulse"].powers), {"hello"})
        self.assertEqual(registry["hello-pulse"].powers["hello"].path, "/v1/powers/hello")
        self.assertIn("Respond naturally to questions and conversation", registry["hello-pulse"].rules)
        self.assertIn("only when the Captain explicitly asks", registry["hello-pulse"].rules)

        invalid = (
            "ghcr.io/roxygens/shimpz-space:latest",
            "ghcr.io/roxygens/shimpz-space@sha256:" + "0" * 64,
            "https://ghcr.io/roxygens/hello@sha256:" + "a" * 64,
        )
        for image in invalid:
            with self.subTest(image=image), self.assertRaises(local_registry.RegistryError):
                self._registry(image)

    def test_registry_shape_is_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(
                json.dumps({"schema": 1, "hello_pulse_image": "x", "command": ["/bin/sh"]}),
                encoding="utf-8",
            )
            with self.assertRaises(local_registry.RegistryError):
                local_registry.load_registry(path)

    def test_hello_contract_is_closed_and_bounded(self) -> None:
        self.assertEqual(local_registry.validate_hello_input({}), {"name": "Shimpz"})
        self.assertEqual(local_registry.validate_hello_input({"name": "Captain"}), {"name": "Captain"})
        for invalid in ({"name": ""}, {"name": " x"}, {"name": "x\n"}, {"extra": True}, []):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                local_registry.validate_hello_input(invalid)

    def test_identifiers_are_strict_and_bounded(self) -> None:
        self.assertEqual(local_app.validate_capsule_id("demo_capsule"), "demo_capsule")
        for invalid in ("Demo", "-demo", "demo-1", "a" * 41, "demo space", ""):
            with self.subTest(invalid=invalid), self.assertRaises(local_app.ApiProblem) as caught:
                local_app.validate_capsule_id(invalid)
            self.assertEqual(caught.exception.status, HTTPStatus.UNPROCESSABLE_ENTITY)

    def test_capsule_name_matches_the_admin_contract(self) -> None:
        self.assertEqual(local_app.validate_capsule_name("My Capsule"), "My Capsule")
        for invalid in ("", " padded", "padded ", "x\n", "x" * 81, None):
            with self.subTest(invalid=invalid), self.assertRaises(local_app.ApiProblem):
                local_app.validate_capsule_name(invalid)

    def test_container_limits_and_stateless_recovery_are_intentionally_narrow(self) -> None:
        self.assertEqual(local_app.ASSISTANT_NANO_CPUS, 250_000_000)
        self.assertEqual(local_app.ASSISTANT_MEMORY, 128 * 1024 * 1024)
        self.assertEqual(local_app.ASSISTANT_PIDS, 64)
        self.assertEqual(local_app.half_cpu_set(96), "0-47")
        self.assertEqual(local_app.half_cpu_set(8), "0-3")
        self.assertEqual(local_app.half_cpu_set(1), "0")
        readiness = local_app.ApiProblem(HTTPStatus.BAD_GATEWAY, "not ready", code="assistant-not-ready")
        ownership = local_app.ApiProblem(HTTPStatus.CONFLICT, "drift", code="ownership-conflict")
        self.assertTrue(local_app._is_replaceable_readiness_failure("hello-pulse", readiness))
        self.assertFalse(local_app._is_replaceable_readiness_failure("future-stateful-assistant", readiness))
        self.assertFalse(local_app._is_replaceable_readiness_failure("hello-pulse", ownership))

    def test_local_controller_owns_private_runtime_token_bootstrap(self) -> None:
        source = (CAPSULE / "local_app.py").read_text(encoding="utf-8")
        dockerfile = (CAPSULE / "Dockerfile.local").read_text(encoding="utf-8")
        self.assertIn("brain_runtime_token_store.ensure()", source)
        for marker in (
            "brain_runtime_token_store.py",
            "groupadd --gid 10016 shimpzbrain-runtime-token",
            "--groups 10010,10016",
            "/run/shimpz-brain-runtime",
            "chmod 0750 /run/shimpz-brain-runtime",
        ):
            self.assertIn(marker, dockerfile)

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
            controller._network = lambda _capsule_id: object()

            configured = controller.configure_inference(
                "capsule_1",
                {"provider": "openai", "model": "gpt-5.5"},
            )

            self.assertEqual(
                configured,
                {"capsule": "capsule_1", "provider": "openai", "model": "gpt-5.5"},
            )
            self.assertEqual(controller.inference_status("capsule_1"), configured)
            stored = next((Path(directory) / "inference").iterdir()).read_text(encoding="utf-8")
            self.assertNotIn("api_key", stored)
            with self.assertRaises(local_app.ApiProblem):
                controller.configure_inference(
                    "capsule_1",
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
        body = json.dumps({"message": "Hello", "files": []}).encode()
        captured: dict[str, object] = {}

        class Controller:
            @staticmethod
            def chat(capsule_id, payload, provider, api_key):
                captured.update(
                    capsule=capsule_id,
                    payload=payload,
                    provider=provider,
                    api_key=api_key,
                )
                return {"team": "Marketing", "reply": "Hello!"}

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
        status, response, *_audit = handler._chat_route(["v1", "capsules", "capsule_1", "chat"])

        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(captured["payload"], {"message": "Hello", "files": []})
        self.assertEqual(captured["provider"], "openai")
        self.assertEqual(captured["api_key"], key)
        self.assertNotIn(key, json.dumps(response))

    def test_chat_sends_key_only_to_runtime_and_returns_no_secret(self) -> None:
        class Runtime:
            context = None

            def start(self, context, _message):
                self.context = context
                return brain_runtime_client.RuntimeTurn(status="completed", reply="Hello!", powers=())

            def resume(self, _context, _results):
                raise AssertionError("a direct reply must not resume")

        runtime = Runtime()
        key = "sk-test-0123456789"
        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, runtime)
            response = controller.chat(
                "capsule_1",
                {"message": "Hello", "files": []},
                "openai",
                key,
            )
            with self.assertRaises(local_app.ApiProblem):
                controller.chat(
                    "capsule_1",
                    {
                        "message": "Hello",
                        "files": [],
                        "api_key": key,
                    },
                    "openai",
                    key,
                )
            persisted = "".join(path.read_text(encoding="utf-8") for path in (Path(directory) / "inference").iterdir())

        self.assertEqual(response["reply"], "Hello!")
        self.assertNotIn(key, json.dumps(response))
        self.assertNotIn(key, persisted)
        self.assertNotIn(key, repr(runtime.context))
        self.assertEqual(runtime.context.api_key, key)
        self.assertEqual(runtime.context.team_name, "Marketing")
        self.assertEqual([assistant.id for assistant in runtime.context.assistants], ["hello-pulse"])

    def test_chat_exposes_every_active_assistant_to_the_team_brain(self) -> None:
        class Runtime:
            context = None

            def start(self, context, _message):
                self.context = context
                return brain_runtime_client.RuntimeTurn(status="completed", reply="Integrated.", powers=())

        runtime = Runtime()
        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, runtime)
            hello = controller.registry["hello-pulse"]
            weather = replace(
                hello,
                assistant_id="weather-pulse",
                image=hello.image.replace("a" * 64, "b" * 64),
                rules="Use weather Powers only for weather data.",
                powers={"current": replace(hello.powers["hello"], path="/v1/powers/current")},
            )
            controller.registry[weather.assistant_id] = weather
            controller._active_chat_assistants = lambda _cid, _network: (
                local_app._ActiveAssistant(hello, "hello-container"),
                local_app._ActiveAssistant(weather, "weather-container"),
            )

            response = controller.chat(
                "capsule_1",
                {"message": "Check the weather", "files": []},
                "openai",
                "sk-test-0123456789",
            )

        self.assertEqual([assistant.id for assistant in runtime.context.assistants], ["hello-pulse", "weather-pulse"])
        self.assertEqual(runtime.context.thread_id, "local:local-space:capsule_1:default")
        self.assertEqual(response["team"], "Marketing")

    def test_team_identity_drift_stops_before_the_provider_call(self) -> None:
        class Runtime:
            def start(self, _context, _message):
                raise AssertionError("a changed Team must not reach the provider")

        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, Runtime())
            names = iter(("Marketing", "Renamed"))
            controller._validate_network = lambda _network, _cid: next(names)

            with self.assertRaises(local_app.ApiProblem) as caught:
                controller.chat(
                    "capsule_1",
                    {"message": "Hello", "files": []},
                    "openai",
                    "sk-test-0123456789",
                )

        self.assertEqual(caught.exception.code, "team-context-changed")

    def test_chat_executes_only_controller_owned_none_approval_power(self) -> None:
        class Runtime:
            def start(self, _context, _message):
                return brain_runtime_client.RuntimeTurn(
                    status="power-required",
                    reply="",
                    powers=(
                        brain_runtime_client.PowerRequest(
                            interrupt_id="power-1",
                            assistant_id="hello-pulse",
                            power="hello",
                            input={"name": "Captain"},
                            approval="none",
                        ),
                    ),
                )

            def resume(self, _context, results):
                if results != {"power-1": {"message": "Hello, Captain!"}}:
                    raise AssertionError("Power result did not return through the Controller")
                return brain_runtime_client.RuntimeTurn(status="completed", reply="Done", powers=())

        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, Runtime())
            invoked: list[tuple[str, str, object]] = []
            controller.invoke = lambda cid, assistant, power, payload: (
                invoked.append((cid, assistant, payload))
                or {"assistant": assistant, "power": power, "result": {"message": "Hello, Captain!"}}
            )
            response = controller.chat(
                "capsule_1",
                {"message": "Greet me", "files": []},
                "openai",
                "sk-test-0123456789",
            )

        self.assertEqual(invoked, [("capsule_1", "hello-pulse", {"name": "Captain"})])
        self.assertEqual(response, {"capsule": "capsule_1", "team": "Marketing", "reply": "Done"})

    def test_chat_fails_closed_before_a_power_that_requires_approval(self) -> None:
        class Runtime:
            def start(self, _context, _message):
                return brain_runtime_client.RuntimeTurn(
                    status="power-required",
                    reply="",
                    powers=(
                        brain_runtime_client.PowerRequest(
                            interrupt_id="power-1",
                            assistant_id="hello-pulse",
                            power="hello",
                            input={},
                            approval="each-run",
                        ),
                    ),
                )

        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, Runtime())
            spec = controller.registry["hello-pulse"]
            controller.registry["hello-pulse"] = replace(
                spec,
                powers={"hello": replace(spec.powers["hello"], approval="each-run")},
            )
            controller.invoke = lambda *_args: self.fail("approval-gated Power executed")
            with self.assertRaises(local_app.ApiProblem) as caught:
                controller.chat(
                    "capsule_1",
                    {"message": "Greet me", "files": []},
                    "openai",
                    "sk-test-0123456789",
                )
        self.assertEqual(caught.exception.code, "power-approval-required")

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
                        "capsule_1",
                        {"message": "Wait", "files": []},
                        "openai",
                        "sk-test-0123456789",
                    )
                except local_app.ApiProblem as exc:
                    failures.append(exc)

            worker = threading.Thread(target=turn)
            worker.start()
            self.assertTrue(started.wait(timeout=1))
            stopped = controller.stop_chat("capsule_1")
            release.set()
            worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertTrue(stopped["accepted"])
        self.assertFalse(stopped["confirmed"])
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], local_app.ApiProblem)
        self.assertEqual(failures[0].code, "chat-stopped")


if __name__ == "__main__":
    unittest.main()
