from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM))
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


class LocalHumanInteractionTests(LocalContractCase):
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
