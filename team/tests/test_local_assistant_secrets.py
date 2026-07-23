from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import threading
from contextlib import closing
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM))
import assistant_secret_challenges
import assistant_secret_store
import brain_runtime_client
import local_app
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


class LocalAssistantSecretTests(LocalContractCase):
    def test_assistant_secret_put_route_reaches_rotation_contract(self) -> None:
        value = "replacement-secret-123"
        body = json.dumps(
            {
                "assistant_id": "shimpz-cloudflare",
                "values": [{"secret_id": "client-key", "value": value}],
            }
        ).encode()
        captured: dict[str, object] = {}

        class Controller:
            @staticmethod
            def replace_assistant_secrets(team_id, payload):
                captured.update(team_id=team_id, payload=payload)
                return {"team_id": team_id, "assistants": []}

        handler = object.__new__(local_app.Handler)
        handler.command = "PUT"
        handler.server = SimpleNamespace(controller=Controller())
        handler.headers = Message()
        handler.headers["Content-Type"] = "application/json"
        handler.headers["Content-Length"] = str(len(body))
        handler.rfile = BytesIO(body)

        status, response, operation, team_id, _assistant = handler._assistant_secret_route(
            ["v1", "teams", "team_1", "assistant-secrets"]
        )

        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(operation, "assistant-secret-replace")
        self.assertEqual(team_id, "team_1")
        self.assertEqual(captured["payload"]["values"][0]["value"], value)
        self.assertNotIn(value, json.dumps(response))

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
                "team_1",
                {"message": "Hello", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                "openai",
                key,
            )
            with self.assertRaises(local_app.ApiProblem):
                controller.chat(
                    "team_1",
                    {
                        "message": "Hello",
                        "files": [],
                        "assistant_ids": ["shimpz-cloudflare"],
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
        self.assertEqual([assistant.id for assistant in runtime.context.assistants], ["shimpz-cloudflare"])
        self.assertEqual(runtime.context.assistants[0].genesis, "Use only the declared Cloudflare Powers.")

    def test_local_chat_rechecks_pending_secrets_after_acquiring_its_slot(self) -> None:
        class Runtime:
            def start(self, _context, _message):
                raise AssertionError("a pending continuation started another turn")

        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, Runtime(), configure_secrets=False)
            challenge = controller.secret_challenges.create(
                "team_1",
                (
                    assistant_secret_challenges.SecretRequirement(
                        "shimpz-cloudflare",
                        "Shimpz Cloudflare",
                        ("list-zones",),
                        (("service-token", "Service Token", "Required."),),
                    ),
                ),
                object(),
            )
            current = mock.Mock(side_effect=(None, challenge))
            controller.secret_challenges.current = current

            response = controller.chat(
                "team_1",
                {"message": "Hello", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                "openai",
                "sk-test-0123456789",
            )

        self.assertEqual(response, controller._challenge_response(challenge))
        self.assertEqual(current.call_count, 2)

    def test_chat_collects_a_multi_secret_batch_before_any_power_side_effect(self) -> None:
        requests = (
            brain_runtime_client.PowerRequest(
                interrupt_id="lookup",
                assistant_id="shimpz-cloudflare",
                power="list-zones",
                input=LOOKUP_INPUT,
            ),
            brain_runtime_client.PowerRequest(
                interrupt_id="identity",
                assistant_id="shimpz-cloudflare",
                power="list-dns-records",
                input=DNS_INPUT,
            ),
        )

        class Runtime:
            def start(self, _context, _message):
                return brain_runtime_client.RuntimeTurn(status="power-required", reply="", powers=requests)

            def resume(self, _context, _results):
                raise AssertionError("a paused Power batch must not reach resume")

        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, Runtime(), configure_secrets=False)
            controller.invoke = lambda *_args: self.fail("a Power ran before every secret was available")

            response = controller.chat(
                "team_1",
                {"message": "Read my account", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                "openai",
                "sk-test-0123456789",
            )

            with closing(sqlite3.connect(controller.power_state.path)) as connection:
                pending_batches = connection.execute("SELECT COUNT(*) FROM batches").fetchone()
            self.assertFalse(controller.assistant_secrets.state_path.exists())
            self.assertFalse(controller.assistant_secrets.key_path.exists())

        self.assertEqual(response["status"], "secrets-required")
        self.assertEqual(response["turn_id"], response["challenge_id"])
        self.assertEqual(len(response["requirements"]), 1)
        requirement = response["requirements"][0]
        self.assertEqual(requirement["power_ids"], ["list-dns-records", "list-zones"])
        self.assertEqual(
            {secret["id"] for secret in requirement["secrets"]},
            set(TEST_SECRET_VALUES),
        )
        self.assertNotIn("access_token", repr(response))
        self.assertEqual(pending_batches, (0,))

    def test_oversized_secret_envelope_is_rejected_before_the_local_power_journal(self) -> None:
        request = brain_runtime_client.PowerRequest(
            interrupt_id="lookup",
            assistant_id="shimpz-cloudflare",
            power="list-zones",
            input=LOOKUP_INPUT,
        )

        class Runtime:
            def start(self, _context, _message):
                return brain_runtime_client.RuntimeTurn(status="power-required", reply="", powers=(request,))

            def resume(self, _context, _results):
                raise AssertionError("an oversized Power envelope must not reach resume")

        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, Runtime(), configure_secrets=True)
            controller.assistant_secrets.put_many(
                "team_1",
                "shimpz-cloudflare",
                {"service-token": "x" * assistant_secret_store.MAX_SECRET_BYTES},
            )
            controller.invoke = lambda *_args: self.fail("an oversized Power envelope executed")

            with self.assertRaises(local_app.ApiProblem) as caught:
                controller.chat(
                    "team_1",
                    {"message": "Find OpenAI", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                    "openai",
                    "sk-test-0123456789",
                )
            with closing(sqlite3.connect(controller.power_state.path)) as connection:
                pending_batches = connection.execute("SELECT COUNT(*) FROM batches").fetchone()

        self.assertEqual(caught.exception.code, "assistant-power-input-too-large")
        self.assertEqual(pending_batches, (0,))

    def test_secret_submission_is_exact_team_bound_and_single_use(self) -> None:
        request = brain_runtime_client.PowerRequest(
            interrupt_id="identity",
            assistant_id="shimpz-cloudflare",
            power="list-dns-records",
            input=DNS_INPUT,
        )

        class Runtime:
            def __init__(self) -> None:
                self.resumes: list[dict[str, object]] = []

            def start(self, _context, _message):
                return brain_runtime_client.RuntimeTurn(status="power-required", reply="", powers=(request,))

            def resume(self, _context, results):
                self.resumes.append(dict(results))
                return brain_runtime_client.RuntimeTurn(status="completed", reply="Connected.", powers=())

        runtime = Runtime()
        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, runtime, configure_secrets=False)
            invocations: list[tuple[str, str, object]] = []
            controller.invoke = lambda team_id, assistant_id, power_id, payload: (
                invocations.append((team_id, power_id, payload))
                or {"assistant": assistant_id, "power": power_id, "result": DNS_RESULT}
            )
            challenge = controller.chat(
                "team_1",
                {"message": "Who am I?", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                "openai",
                "sk-test-0123456789",
            )
            exact = self._secret_submission(challenge)
            values = exact["values"]
            invalid_submissions = (
                {**exact, "values": values[:-1]},
                {
                    **exact,
                    "values": [
                        *values,
                        {
                            "assistant_id": "shimpz-cloudflare",
                            "secret_id": "undeclared-secret",
                            "value": "must-not-be-stored",
                        },
                    ],
                },
                {**exact, "values": [values[0], *values]},
                {**exact, "unexpected": True},
            )
            for invalid in invalid_submissions:
                with self.subTest(invalid=invalid), self.assertRaises(local_app.ApiProblem) as rejected:
                    controller.submit_chat_secrets(
                        "team_1",
                        invalid,
                        "openai",
                        "sk-test-0123456789",
                    )
                self.assertEqual(rejected.exception.code, "invalid-assistant-secrets")
                self.assertIsNotNone(controller.secret_challenges.current("team_1"))

            original_put = controller.assistant_secrets.put_for_assistants
            controller.assistant_secrets.put_for_assistants = mock.Mock(
                side_effect=assistant_secret_store.AssistantSecretError("storage unavailable")
            )
            with self.assertRaises(local_app.ApiProblem) as unavailable:
                controller.submit_chat_secrets(
                    "team_1",
                    exact,
                    "openai",
                    "sk-test-0123456789",
                )
            controller.assistant_secrets.put_for_assistants = original_put
            self.assertEqual(unavailable.exception.code, "assistant-secret-state-unavailable")
            self.assertIsNotNone(controller.secret_challenges.current("team_1"))

            with self.assertRaises(local_app.ApiProblem) as isolated:
                controller.submit_chat_secrets(
                    "team_2",
                    exact,
                    "openai",
                    "sk-test-0123456789",
                )
            self.assertEqual(isolated.exception.code, "assistant-secret-challenge-expired")

            response = controller.submit_chat_secrets(
                "team_1",
                exact,
                "openai",
                "sk-test-0123456789",
            )
            with self.assertRaises(local_app.ApiProblem) as replay:
                controller.submit_chat_secrets(
                    "team_1",
                    exact,
                    "openai",
                    "sk-test-0123456789",
                )

            configured_for_other_team = controller.assistant_secrets.metadata(
                "team_2",
                "shimpz-cloudflare",
                tuple(TEST_SECRET_VALUES),
            )

        self.assertEqual(response["reply"], "Connected.")
        self.assertEqual(invocations, [("team_1", "list-dns-records", DNS_INPUT)])
        self.assertEqual(runtime.resumes, [{"identity": DNS_RESULT}])
        self.assertEqual(replay.exception.code, "assistant-secret-challenge-expired")
        self.assertTrue(all(not item.configured for item in configured_for_other_team))
        self.assertNotIn("must-not-be-stored", repr(response))

    def test_secret_continuation_rejects_context_drift_before_power_execution(self) -> None:
        request = brain_runtime_client.PowerRequest(
            interrupt_id="lookup",
            assistant_id="shimpz-cloudflare",
            power="list-zones",
            input=LOOKUP_INPUT,
        )

        class Runtime:
            def start(self, _context, _message):
                return brain_runtime_client.RuntimeTurn(status="power-required", reply="", powers=(request,))

            def resume(self, _context, _results):
                raise AssertionError("a drifted continuation must not reach resume")

        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, Runtime(), configure_secrets=False)
            controller.invoke = lambda *_args: self.fail("a drifted continuation executed a Power")
            challenge = controller.chat(
                "team_1",
                {"message": "Find OpenAI", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                "openai",
                "sk-test-0123456789",
            )
            controller._network = lambda _team_id: SimpleNamespace(id="b" * 64, name="team-network")

            with self.assertRaises(local_app.ApiProblem) as drifted:
                controller.submit_chat_secrets(
                    "team_1",
                    self._secret_submission(challenge),
                    "openai",
                    "sk-test-0123456789",
                )
            with closing(sqlite3.connect(controller.power_state.path)) as connection:
                pending_batches = connection.execute("SELECT COUNT(*) FROM batches").fetchone()

        self.assertEqual(drifted.exception.code, "team-context-changed")
        self.assertEqual(pending_batches, (0,))

    def test_secret_inventory_returns_only_team_scoped_masks_and_public_metadata(self) -> None:
        raw_secret = TEST_SECRET_VALUES["service-token"]
        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, object(), configure_secrets=False)
            controller.list_assistants = lambda _team_id: {
                "assistants": [{"assistant": "shimpz-cloudflare", "status": "running"}]
            }
            controller.assistant_secrets.put_many(
                "team_1",
                "shimpz-cloudflare",
                {"service-token": raw_secret},
            )

            own_inventory = controller.list_assistant_secrets("team_1")
            other_inventory = controller.list_assistant_secrets("team_2")

        encoded = repr(own_inventory)
        self.assertNotIn(raw_secret, encoded)
        self.assertNotIn("generation", encoded)
        self.assertNotIn("ciphertext", encoded)
        self.assertEqual(set(own_inventory), {"team_id", "assistants"})
        own_secrets = {item["id"]: item for item in own_inventory["assistants"][0]["secrets"]}
        other_secrets = {item["id"]: item for item in other_inventory["assistants"][0]["secrets"]}
        self.assertEqual(
            own_secrets["service-token"],
            {
                "id": "service-token",
                "name": "Service Token",
                "summary": "Test-only credential used to exercise the generic secret boundary.",
                "configured": True,
                "mask": assistant_secret_store.mask_secret(raw_secret),
            },
        )
        self.assertTrue(all(not item["configured"] and item["mask"] is None for item in other_secrets.values()))

    def test_secret_replacement_is_declared_atomic_and_returns_only_refreshed_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, object(), configure_secrets=True)
            controller.list_assistants = lambda _team_id: {
                "assistants": [{"assistant": "shimpz-cloudflare", "status": "running"}]
            }
            before = controller.assistant_secrets.resolve_many(
                "team_1",
                "shimpz-cloudflare",
                ("client-key", "client-secret"),
            )
            replacement = "replacement-api-key-123456789"
            response = controller.replace_assistant_secrets(
                "team_1",
                {
                    "assistant_id": "shimpz-cloudflare",
                    "values": [{"secret_id": "client-key", "value": replacement}],
                },
            )
            after = controller.assistant_secrets.resolve_many(
                "team_1",
                "shimpz-cloudflare",
                ("client-key", "client-secret"),
            )

            state_before_invalid = controller.assistant_secrets.state_path.read_bytes()
            for invalid in (
                {
                    "assistant_id": "shimpz-cloudflare",
                    "values": [
                        {"secret_id": "client-key", "value": "must-not-commit"},
                        {"secret_id": "undeclared", "value": "invalid"},
                    ],
                },
                {
                    "assistant_id": "shimpz-cloudflare",
                    "values": [{"secret_id": "client-key", "value": "line\nbreak"}],
                },
            ):
                with self.subTest(invalid=invalid), self.assertRaises(local_app.ApiProblem) as rejected:
                    controller.replace_assistant_secrets("team_1", invalid)
                self.assertEqual(rejected.exception.code, "invalid-assistant-secrets")
                self.assertEqual(controller.assistant_secrets.state_path.read_bytes(), state_before_invalid)

        self.assertEqual(before["client-secret"], after["client-secret"])
        self.assertNotEqual(before["client-key"], after["client-key"])
        self.assertEqual(after["client-key"], replacement)
        self.assertNotIn(replacement, repr(response))
        secret = next(item for item in response["assistants"][0]["secrets"] if item["id"] == "client-key")
        self.assertTrue(secret["configured"])
        self.assertEqual(secret["mask"], assistant_secret_store.mask_secret(replacement))

    def test_secret_rotation_is_excluded_while_a_chat_turn_is_active(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class Runtime:
            def start(self, _context, _message):
                started.set()
                if not release.wait(timeout=2):
                    raise AssertionError("test did not release chat")
                return brain_runtime_client.RuntimeTurn(status="completed", reply="Done.", powers=())

        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, Runtime(), configure_secrets=True)
            controller.list_assistants = lambda _team_id: {
                "assistants": [{"assistant": "shimpz-cloudflare", "status": "running"}]
            }
            before = controller.assistant_secrets.resolve_many(
                "team_1",
                "shimpz-cloudflare",
                ["client-key"],
            )
            results: list[dict[str, object]] = []

            def turn() -> None:
                results.append(
                    controller.chat(
                        "team_1",
                        {"message": "Wait", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                        "openai",
                        "sk-test-0123456789",
                    )
                )

            worker = threading.Thread(target=turn)
            worker.start()
            self.assertTrue(started.wait(timeout=2))
            with self.assertRaises(local_app.ApiProblem) as blocked:
                controller.replace_assistant_secrets(
                    "team_1",
                    {
                        "assistant_id": "shimpz-cloudflare",
                        "values": [{"secret_id": "client-key", "value": "must-not-win-123"}],
                    },
                )
            release.set()
            worker.join(timeout=2)
            after = controller.assistant_secrets.resolve_many(
                "team_1",
                "shimpz-cloudflare",
                ["client-key"],
            )

        self.assertFalse(worker.is_alive())
        self.assertEqual(blocked.exception.code, "chat-active")
        self.assertEqual(before, after)
        self.assertEqual(results[0]["reply"], "Done.")

    def test_rotation_invalidates_a_stale_jit_challenge_before_it_can_overwrite_values(self) -> None:
        request = brain_runtime_client.PowerRequest(
            interrupt_id="identity",
            assistant_id="shimpz-cloudflare",
            power="list-dns-records",
            input=DNS_INPUT,
        )

        class Runtime:
            def start(self, _context, _message):
                return brain_runtime_client.RuntimeTurn(status="power-required", reply="", powers=(request,))

            def resume(self, _context, _results):
                raise AssertionError("stale JIT challenge must never resume")

        replacements = {secret_id: f"rotated-{index}-credential" for index, secret_id in enumerate(TEST_SECRET_VALUES)}
        with tempfile.TemporaryDirectory() as directory:
            controller = self._chat_controller(directory, Runtime(), configure_secrets=False)
            controller.list_assistants = lambda _team_id: {
                "assistants": [{"assistant": "shimpz-cloudflare", "status": "running"}]
            }
            challenge = controller.chat(
                "team_1",
                {"message": "Who am I?", "files": [], "assistant_ids": ["shimpz-cloudflare"]},
                "openai",
                "sk-test-0123456789",
            )
            stale = self._secret_submission(challenge)
            controller.replace_assistant_secrets(
                "team_1",
                {
                    "assistant_id": "shimpz-cloudflare",
                    "values": [{"secret_id": secret_id, "value": value} for secret_id, value in replacements.items()],
                },
            )
            with self.assertRaises(local_app.ApiProblem) as rejected:
                controller.submit_chat_secrets(
                    "team_1",
                    stale,
                    "openai",
                    "sk-test-0123456789",
                )
            stored = controller.assistant_secrets.resolve_many(
                "team_1",
                "shimpz-cloudflare",
                tuple(replacements),
            )

        self.assertEqual(rejected.exception.code, "assistant-secret-challenge-expired")
        self.assertEqual(stored, replacements)
