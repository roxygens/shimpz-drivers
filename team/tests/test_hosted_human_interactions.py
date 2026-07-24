from __future__ import annotations

import contextlib
import tempfile
import types
import unittest
from dataclasses import replace
from http import HTTPStatus
from pathlib import Path
from unittest import mock

import assistant_secret_store
import brain_runtime_client
import marketplace
import oauth_account_store
import power_execution
import power_journal
from assistant_human import approval_challenges as assistant_approval_challenges
from assistant_human import approval_grants as assistant_approval_grants
from assistant_human import input_challenges as assistant_input_challenges

TESTS = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(TESTS))

import hosted_app_fixture as harness

app = harness.app
hosted_assistants = harness.hosted_assistants
hosted_apps = harness.hosted_apps
hosted_chat_api = harness.hosted_chat_api
hosted_chat_segment = harness.hosted_chat_segment
hosted_resources = harness.hosted_resources
runtime_state = harness.runtime_state

TEAM_ID = "team_1"
ANCHOR_ID = "a" * 64
ASSISTANT_ID = "shimpz-cloudflare"
IMAGE = marketplace.APPS[ASSISTANT_ID].image
LOOKUP_INPUT = {"page": 1, "per_page": 25}
LOOKUP_RESULT = {
    "zones": [],
    "pagination": {"page": 1, "per_page": 25, "count": 0, "total_count": 0, "total_pages": 0},
}


class _Runtime:
    def __init__(self) -> None:
        self.starts = 0
        self.resumes: list[dict[str, object]] = []
        self.request = brain_runtime_client.PowerRequest(
            "power-1",
            ASSISTANT_ID,
            "list-zones",
            LOOKUP_INPUT,
        )

    def start(self, _context, _message):
        self.starts += 1
        return brain_runtime_client.RuntimeTurn("power-required", "", (self.request,))

    def resume(self, _context, results):
        self.resumes.append(dict(results))
        return brain_runtime_client.RuntimeTurn("completed", "Completed.", ())


class HostedHumanInteractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.runtime = _Runtime()
        self.journal = power_journal.PowerJournal(root / "power-journal" / "journal.sqlite3")
        self.addCleanup(self.journal.close)
        self.input_challenges = assistant_input_challenges.InputChallengeStore()
        self.approval_challenges = assistant_approval_challenges.ApprovalChallengeStore()
        self.approval_grants = assistant_approval_grants.ApprovalGrantStore(
            root / "assistant-approvals" / "grants.sqlite3"
        )
        self.addCleanup(self.approval_grants.close)
        self.secret_store = assistant_secret_store.AssistantSecretStore(
            root / "assistant-secrets" / "state" / "secrets.json",
            root / "assistant-secrets" / "key" / "aes256.key",
        )
        self.account_store = oauth_account_store.OAuthAccountStore(
            root / "assistant-accounts" / "state" / "accounts.json",
            root / "assistant-accounts" / "key" / "aes256.key",
        )
        trusted = marketplace.APPS[ASSISTANT_ID].assistant
        assert trusted is not None
        self.contract = replace(
            trusted,
            powers={power_id: replace(power, secrets=(), accounts=()) for power_id, power in trusted.powers.items()},
            secrets={},
            accounts={},
        )
        self.assistant = types.SimpleNamespace(
            id="b" * 64,
            attrs={"Config": {"Image": IMAGE}},
        )
        self.active = hosted_assistants._ActiveAssistant(ASSISTANT_ID, self.contract, self.assistant)
        self.anchor = types.SimpleNamespace(
            id=ANCHOR_ID,
            labels={"team.name": "Marketing", "team.owner": "account_1"},
        )
        self.rpc_answers: list[list[object]] = []

    @contextlib.contextmanager
    def _environment(self, rpc):
        with (
            mock.patch.multiple(
                runtime_state,
                _brain_runtime=self.runtime,
                _inference_store=types.SimpleNamespace(
                    load=lambda _team_id: types.SimpleNamespace(provider="openai", model="gpt-test")
                ),
                _power_execution_journal=lambda: self.journal,
                _assistant_secrets=self.secret_store,
                _assistant_accounts=self.account_store,
                _assistant_input_challenges=self.input_challenges,
                _assistant_approval_challenges=self.approval_challenges,
                _assistant_approval_grants=self.approval_grants,
                _token_cancelled=lambda _token: False,
                _commit_chat_terminal=lambda *_args: True,
            ),
            mock.patch.multiple(
                hosted_assistants,
                _active_team_assistants=lambda _team_id: (self.active,),
                _chat_file_metadata=lambda _team_id, _files: [],
                _installed_assistant=lambda *_args: (ASSISTANT_ID, self.contract, self.assistant),
                _assistant_rpc=rpc,
                _model_credential=lambda _owner, _provider: ("model-secret", 7),
                _require_model_credential_current=lambda *_args: None,
            ),
            mock.patch.object(
                hosted_apps,
                "_require_assistant_genesis",
                return_value="Use only the declared Cloudflare Powers.",
            ),
            mock.patch.object(hosted_chat_segment, "_current_team_anchor", return_value=self.anchor),
        ):
            yield

    @contextlib.contextmanager
    def _exclusive(self, _team_id, _lease):
        yield "resumed-turn", self.anchor

    @staticmethod
    def _lease(owner: str = "account_1", team_id: str = TEAM_ID) -> object:
        return hosted_resources._AuthorizationLease(team_id, ANCHOR_ID, owner, ("account", owner))

    def test_typed_input_is_team_and_owner_bound_then_replays_into_the_exact_power(self) -> None:
        def rpc(_team_id, _token, _container, _command, _method, _path, payload):
            self.rpc_answers.append(payload["answers"])
            if not payload["answers"]:
                return power_execution.RpcSuspension(
                    {
                        "ordinal": 0,
                        "kind": "request",
                        "request_type": "str",
                        "title": "Zone name",
                        "summary": "Choose a zone name.",
                        "docs": None,
                        "options": [],
                    }
                )
            return LOOKUP_RESULT

        with self._environment(rpc):
            challenge = hosted_chat_segment._chat_in_turn(
                TEAM_ID,
                "Choose a zone.",
                [],
                (ASSISTANT_ID,),
                "initial-turn",
                self.anchor,
                "account_1",
            )
            submission = {"challenge_id": challenge["challenge_id"], "answer": "example.com"}
            with self.assertRaises(runtime_state.ApiError) as cross_team:
                hosted_chat_api._submit_chat_input("team_2", submission, self._lease(team_id="team_2"))
            with self.assertRaises(runtime_state.ApiError) as cross_owner:
                hosted_chat_api._submit_chat_input(TEAM_ID, submission, self._lease(owner="account_2"))
            with mock.patch.object(hosted_chat_api, "_exclusive_chat_turn", self._exclusive):
                response = hosted_chat_api._submit_chat_input(TEAM_ID, submission, self._lease())

        self.assertEqual(challenge["status"], "input-required")
        self.assertEqual(cross_team.exception.status, HTTPStatus.CONFLICT)
        self.assertEqual(cross_owner.exception.status, HTTPStatus.CONFLICT)
        self.assertEqual(response["reply"], "Completed.")
        self.assertEqual(self.rpc_answers, [[], ["example.com"]])
        self.assertEqual(self.runtime.resumes, [{"power-1": LOOKUP_RESULT}])

    def test_power_cannot_echo_a_human_answer(self) -> None:
        answer = "human-submitted-private-value"
        turn_token = "turn-token"

        def rpc(*_args):
            return {"echo": answer}

        with (
            self._environment(rpc),
            self.assertRaises(runtime_state.ApiError) as caught,
        ):
            hosted_assistants._invoke_assistant_power(
                hosted_assistants.PowerInvocationRequest(
                    team_id=TEAM_ID,
                    token=turn_token,
                    assistant_id=ASSISTANT_ID,
                    contract=self.contract,
                    container=self.assistant,
                    power="list-zones",
                    payload=LOOKUP_INPUT,
                    answers=(answer,),
                )
            )

        self.assertEqual(caught.exception.status, HTTPStatus.BAD_GATEWAY)
        self.assertNotIn(answer, caught.exception.message)

    def test_once_approval_is_reused_only_for_the_bound_hosted_release(self) -> None:
        def rpc(_team_id, _token, _container, _command, _method, _path, payload):
            self.rpc_answers.append(payload["answers"])
            if not payload["answers"]:
                return power_execution.RpcSuspension(
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

        with self._environment(rpc):
            challenge = hosted_chat_segment._chat_in_turn(
                TEAM_ID,
                "Publish.",
                [],
                (ASSISTANT_ID,),
                "first-turn",
                self.anchor,
                "account_1",
            )
            submission = {"challenge_id": challenge["challenge_id"], "approved": True}
            with self.assertRaises(runtime_state.ApiError) as cross_owner:
                hosted_chat_api._submit_chat_approval(TEAM_ID, submission, self._lease(owner="account_2"))
            with mock.patch.object(hosted_chat_api, "_exclusive_chat_turn", self._exclusive):
                first = hosted_chat_api._submit_chat_approval(TEAM_ID, submission, self._lease())
            second = hosted_chat_segment._chat_in_turn(
                TEAM_ID,
                "Publish again.",
                [],
                (ASSISTANT_ID,),
                "second-turn",
                self.anchor,
                "account_1",
            )

        self.assertEqual(challenge["status"], "approval-required")
        self.assertEqual(challenge["requirements"][0]["approval"], "once")
        self.assertEqual(cross_owner.exception.status, HTTPStatus.CONFLICT)
        self.assertEqual(first["reply"], "Completed.")
        self.assertEqual(second["reply"], "Completed.")
        self.assertEqual(self.rpc_answers, [[], [True], [], [True]])
        self.assertEqual(self.runtime.starts, 2)
        self.assertEqual(len(self.runtime.resumes), 2)


if __name__ == "__main__":
    unittest.main()
