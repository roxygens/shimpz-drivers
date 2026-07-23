from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM))
import brain_runtime_client
import local_app
from local_controller_harness import LocalContractCase
from local_support.chat_types import ActiveAssistant
from local_support.validation import MAX_CHAT_ASSISTANTS

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


class LocalChatScopeTests(LocalContractCase):
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
                ActiveAssistant(hello, "hello-container"),
                ActiveAssistant(account_helper, "account-helper-container"),
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
                [f"helper-{index}" for index in range(MAX_CHAT_ASSISTANTS + 1)],
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
            controller._active_chat_assistants = lambda _team_id, _network: (ActiveAssistant(spec, next(generations)),)

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
                ActiveAssistant(hello, "hello-container"),
                ActiveAssistant(account_helper, "account-helper-container"),
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
