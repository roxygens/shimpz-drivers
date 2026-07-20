from __future__ import annotations

import contextlib
import sqlite3
import sys
import tempfile
import types
import unittest
from contextlib import closing
from http import HTTPStatus
from pathlib import Path
from unittest import mock

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

import test_r2_bridge as harness

app = harness.app
_patched = harness._patched

TEAM_ID = "team_1"
ANCHOR_ID = "a" * 64
ASSISTANT_ID = app.assistant_contract.ASSISTANT_ID
SECRET_VALUES = {
    "x-bearer-token": "bearer-test-value-123456789",
    "x-api-key": "consumer-key-value-123456789",
    "x-api-key-secret": "consumer-secret-value-123456789",
    "x-access-token": "access-token-value-123456789",
    "x-access-token-secret": "access-secret-value-123456789",
}


class _Runtime:
    def __init__(self) -> None:
        self.resume_calls = 0
        self.requests = (
            app.brain_runtime_client.PowerRequest(
                "public-read",
                ASSISTANT_ID,
                "public-user-lookup",
                {"username": "XDevelopers"},
                "none",
            ),
            app.brain_runtime_client.PowerRequest(
                "identity-read",
                ASSISTANT_ID,
                "identity-me",
                {},
                "none",
            ),
        )

    def start(self, _context, _message):
        return app.brain_runtime_client.RuntimeTurn("power-required", "", self.requests)

    def resume(self, _context, results):
        self.resume_calls += 1
        if set(results) != {"public-read", "identity-read"}:
            raise AssertionError("the complete Power batch must resume together")
        return app.brain_runtime_client.RuntimeTurn("completed", "X account connected.", ())


class _RouteHarness:
    def __init__(self, body: dict[str, object]) -> None:
        self.body = body
        self.sent: list[tuple[HTTPStatus, dict[str, object], bool]] = []

    def _read_body(self, *, max_bytes: int = app.MAX_JSON_BODY_BYTES) -> dict[str, object]:
        del max_bytes
        return self.body

    def _send_json(
        self,
        status: HTTPStatus,
        payload: dict[str, object],
        *,
        no_store: bool = False,
    ) -> None:
        self.sent.append((status, payload, no_store))

    def _stream_chat(self, *args, **kwargs) -> None:
        app.Handler._stream_chat(self, *args, **kwargs)


class HostedAssistantSecretTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.secret_store = app.assistant_secret_store.AssistantSecretStore(
            root / "state" / "secrets.json",
            root / "key" / "aes256.key",
        )
        self.challenge_store = app.assistant_secret_challenges.SecretChallengeStore()
        self.journal = app.power_journal.PowerJournal(root / "journal" / "journal.sqlite3")
        self.addCleanup(self.journal.close)
        self.runtime = _Runtime()
        contract = app.marketplace.APPS[ASSISTANT_ID].assistant
        assert contract is not None
        self.contract = contract
        self.assistant_container = types.SimpleNamespace(id="b" * 64)
        self.active = app._ActiveAssistant(ASSISTANT_ID, contract, self.assistant_container)
        self.anchor = types.SimpleNamespace(
            id=ANCHOR_ID,
            labels={"team.name": "Marketing", "team.owner": "account_1"},
        )
        self.rpc_calls: list[tuple[str, dict[str, object]]] = []

    def _rpc(self, _team_id, _token, _container, _command, _method, path, payload):
        self.rpc_calls.append((path, payload))
        return {"id": "123", "name": "X Developers", "username": "XDevelopers"}

    @contextlib.contextmanager
    def _environment(self):
        with _patched(
            _active_team_assistants=lambda _team_id: (self.active,),
            _require_assistant_genesis=lambda _container: "Use only the declared X Powers.",
            _chat_file_metadata=lambda _team_id, _files: [],
            _inference_store=types.SimpleNamespace(
                load=lambda _team_id: types.SimpleNamespace(provider="openai", model="gpt-test")
            ),
            _model_credential=lambda _owner, _provider: ("model-secret-not-an-assistant-secret", 7),
            _require_model_credential_current=lambda *_args: None,
            _current_team_anchor=lambda *_args: self.anchor,
            _brain_runtime=self.runtime,
            _power_execution_journal=lambda: self.journal,
            _assistant_secrets=self.secret_store,
            _assistant_secret_challenges=self.challenge_store,
            _installed_assistant=lambda *_args: (ASSISTANT_ID, self.contract, self.assistant_container),
            _assistant_rpc=self._rpc,
            _commit_chat_terminal=lambda *_args: True,
        ):
            yield

    @staticmethod
    def _submission(challenge: dict[str, object]) -> dict[str, object]:
        return {
            "challenge_id": challenge["challenge_id"],
            "values": [
                {
                    "assistant_id": ASSISTANT_ID,
                    "secret_id": secret_id,
                    "value": value,
                }
                for secret_id, value in SECRET_VALUES.items()
            ],
        }

    def test_missing_batch_pauses_then_resumes_with_exact_secret_envelopes(self) -> None:
        with self._environment():
            challenge = app._chat_in_turn(
                TEAM_ID,
                "Read the public profile and my connected identity.",
                [],
                (ASSISTANT_ID,),
                "initial-turn",
                self.anchor,
                "account_1",
            )

            self.assertEqual(challenge["status"], "secrets-required")
            self.assertEqual(self.runtime.resume_calls, 0)
            self.assertEqual(self.rpc_calls, [])
            requirement = challenge["requirements"][0]
            self.assertEqual(requirement["power_ids"], ["identity-me", "public-user-lookup"])
            self.assertEqual(
                {item["id"] for item in requirement["secrets"]},
                set(SECRET_VALUES),
            )
            serialized = app.json.dumps(challenge)
            for secret in SECRET_VALUES.values():
                self.assertNotIn(secret, serialized)
            self.assertNotIn("XDevelopers", serialized)

            @contextlib.contextmanager
            def exclusive(_team_id, _lease):
                yield "resumed-turn", self.anchor

            with _patched(_exclusive_chat_turn=exclusive):
                result = app._submit_chat_secrets(
                    TEAM_ID,
                    self._submission(challenge),
                    app._AuthorizationLease(TEAM_ID, ANCHOR_ID, "account_1", ("account", "account_1")),
                )

        self.assertEqual(result["reply"], "X account connected.")
        self.assertEqual(self.runtime.resume_calls, 1)
        self.assertEqual(len(self.rpc_calls), 2)
        payloads = dict(self.rpc_calls)
        self.assertEqual(
            payloads["/v1/powers/public-user-lookup"],
            {
                "input": {"username": "XDevelopers"},
                "secrets": {"x-bearer-token": SECRET_VALUES["x-bearer-token"]},
            },
        )

        self.assertEqual(
            payloads["/v1/powers/identity-me"],
            {
                "input": {},
                "secrets": {
                    secret_id: SECRET_VALUES[secret_id]
                    for secret_id in (
                        "x-api-key",
                        "x-api-key-secret",
                        "x-access-token",
                        "x-access-token-secret",
                    )
                },
            },
        )
        state = self.secret_store.state_path.read_text(encoding="utf-8")
        journal = self.journal.path.read_bytes()
        for secret in SECRET_VALUES.values():
            self.assertNotIn(secret, state)
            self.assertNotIn(secret.encode(), journal)
            self.assertNotIn(secret, app.json.dumps(result))
        inventory = app.assistant_secret_flow.inventory_payload(
            TEAM_ID,
            [app._hosted_secret_spec(self.active)],
            self.secret_store,
        )
        self.assertTrue(all(item["configured"] for item in inventory["assistants"][0]["secrets"]))
        self.assertTrue(all(item["mask"] for item in inventory["assistants"][0]["secrets"]))

        with self._environment(), self.assertRaises(app.ApiError) as replay:
            app._submit_chat_secrets(
                TEAM_ID,
                self._submission(challenge),
                app._AuthorizationLease(TEAM_ID, ANCHOR_ID, "account_1", ("account", "account_1")),
            )
        self.assertEqual(replay.exception.status, HTTPStatus.CONFLICT)

    def test_oversized_secret_envelope_is_rejected_before_the_power_journal(self) -> None:
        oversized = dict(SECRET_VALUES)
        oversized["x-bearer-token"] = "x" * app.assistant_secret_store.MAX_SECRET_BYTES
        self.secret_store.put_many(TEAM_ID, ASSISTANT_ID, oversized)

        with self._environment(), self.assertRaises(app.ApiError) as caught:
            app._chat_in_turn(
                TEAM_ID,
                "Read the public profile and my connected identity.",
                [],
                (ASSISTANT_ID,),
                "oversized-turn",
                self.anchor,
                "account_1",
            )
        with closing(sqlite3.connect(self.journal.path)) as connection:
            batches = connection.execute("SELECT COUNT(*) FROM batches").fetchone()

        self.assertEqual(caught.exception.status, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        self.assertEqual(batches, (0,))
        self.assertEqual(self.rpc_calls, [])

    def test_invalid_submission_is_rejected_before_claim_or_storage(self) -> None:
        with self._environment():
            challenge = app._chat_in_turn(
                TEAM_ID,
                "Read the public profile and my connected identity.",
                [],
                (ASSISTANT_ID,),
                "initial-turn",
                self.anchor,
                "account_1",
            )
            invalid = self._submission(challenge)
            values = invalid["values"]
            assert isinstance(values, list)
            invalid["values"] = [
                *values,
                {"assistant_id": ASSISTANT_ID, "secret_id": "undeclared", "value": "attacker-value"},
            ]
            with self.assertRaises(app.ApiError) as caught:
                app._submit_chat_secrets(
                    TEAM_ID,
                    invalid,
                    app._AuthorizationLease(TEAM_ID, ANCHOR_ID, "account_1", ("account", "account_1")),
                )

        self.assertEqual(caught.exception.status, HTTPStatus.UNPROCESSABLE_ENTITY)
        self.assertIsNotNone(self.challenge_store.current(TEAM_ID))
        self.assertFalse(self.secret_store.state_path.exists())
        self.assertNotIn("attacker-value", caught.exception.message)

    def test_storage_failure_keeps_the_one_use_challenge_retryable(self) -> None:
        with self._environment():
            challenge = app._chat_in_turn(
                TEAM_ID,
                "Read the public profile and my connected identity.",
                [],
                (ASSISTANT_ID,),
                "initial-turn",
                self.anchor,
                "account_1",
            )

            @contextlib.contextmanager
            def exclusive(_team_id, _lease):
                yield "resumed-turn", self.anchor

            original = self.secret_store.put_for_assistants
            self.secret_store.put_for_assistants = mock.Mock(
                side_effect=app.assistant_secret_store.AssistantSecretError("storage unavailable")
            )
            with _patched(_exclusive_chat_turn=exclusive), self.assertRaises(app.ApiError) as caught:
                app._submit_chat_secrets(
                    TEAM_ID,
                    self._submission(challenge),
                    app._AuthorizationLease(TEAM_ID, ANCHOR_ID, "account_1", ("account", "account_1")),
                )
            self.secret_store.put_for_assistants = original

        self.assertEqual(caught.exception.status, HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertEqual(self.challenge_store.current(TEAM_ID).id, challenge["challenge_id"])
        self.assertFalse(self.secret_store.state_path.exists())

    def test_secret_bearing_assistant_output_is_rejected_without_echo(self) -> None:
        secret = SECRET_VALUES["x-bearer-token"]
        self.secret_store.put_many(TEAM_ID, ASSISTANT_ID, {"x-bearer-token": secret})
        with (
            _patched(
                _assistant_secrets=self.secret_store,
                _installed_assistant=lambda *_args: (ASSISTANT_ID, self.contract, self.assistant_container),
                _assistant_rpc=lambda *_args, **_kwargs: {
                    "id": "123",
                    "name": secret,
                    "username": "XDevelopers",
                },
            ),
            self.assertRaises(app.ApiError) as caught,
        ):
            app._invoke_assistant_power(
                TEAM_ID,
                "turn-token",
                ASSISTANT_ID,
                self.contract,
                self.assistant_container,
                "public-user-lookup",
                {"username": "XDevelopers"},
            )
        self.assertEqual(caught.exception.status, HTTPStatus.BAD_GATEWAY)
        self.assertNotIn(secret, caught.exception.message)

    def test_chat_route_uses_428_no_store_and_uninstall_purges_secrets(self) -> None:
        challenge = {
            "team_id": TEAM_ID,
            "status": "secrets-required",
            "challenge_id": "f" * 32,
            "turn_id": "f" * 32,
            "requirements": [],
        }
        handler = _RouteHarness({"message": "hello", "files": [], "assistant_ids": [ASSISTANT_ID]})
        with (
            _patched(_chat=lambda *_args: challenge),
            mock.patch.object(app, "_enforce_rate"),
        ):
            app.Handler._route_chat(
                handler,
                "POST",
                ["v1", "teams", TEAM_ID, "chat"],
                TEAM_ID,
                ("account", "account_1"),
                object(),
            )
        self.assertEqual(handler.sent, [(HTTPStatus.PRECONDITION_REQUIRED, challenge, True)])

        self.secret_store.put_many(TEAM_ID, ASSISTANT_ID, {"x-bearer-token": "stored-secret-value"})
        pending = self.challenge_store.create(
            TEAM_ID,
            (
                app.assistant_secret_challenges.SecretRequirement(
                    ASSISTANT_ID,
                    "Shimpz Assistant",
                    ("public-user-lookup",),
                    (("x-bearer-token", "X Bearer Token", "Required."),),
                ),
            ),
            object(),
        )
        with _patched(
            _assistant_secrets=self.secret_store,
            _assistant_secret_challenges=self.challenge_store,
            _require_current_authorization=lambda *_args, **_kwargs: None,
            _teardown_app=lambda *_args, **_kwargs: app._CleanupResult(True, True),
        ):
            result = app._uninstall_app(TEAM_ID, ASSISTANT_ID, object())
        self.assertTrue(result["uninstalled"])
        self.assertIsNone(self.challenge_store.current(TEAM_ID))
        self.assertTrue(pending.id)
        self.assertFalse(self.secret_store.metadata(TEAM_ID, ASSISTANT_ID, ("x-bearer-token",))[0].configured)

    def test_chat_rechecks_a_pending_secret_challenge_after_acquiring_its_slot(self) -> None:
        requirement = app.assistant_secret_challenges.SecretRequirement(
            ASSISTANT_ID,
            "Shimpz Assistant",
            ("public-user-lookup",),
            (("x-bearer-token", "X Bearer Token", "Required."),),
        )
        pending = app.assistant_secret_challenges.PendingSecretChallenge(
            "f" * 32,
            TEAM_ID,
            1.0,
            (requirement,),
            object(),
        )
        current = mock.Mock(side_effect=(None, pending))

        @contextlib.contextmanager
        def exclusive(_team_id, _lease):
            yield "turn-token", self.anchor

        with _patched(
            _assistant_secret_challenges=types.SimpleNamespace(current=current),
            _exclusive_chat_turn=exclusive,
            _chat_in_turn=lambda *_args: self.fail("a pending continuation started another turn"),
        ):
            result = app._chat(TEAM_ID, "hello", [], (ASSISTANT_ID,), types.SimpleNamespace(owner="account_1"))

        self.assertEqual(result, app.assistant_secret_flow.challenge_payload(pending))
        self.assertEqual(current.call_count, 2)

    def test_stream_rechecks_pending_secrets_before_sending_any_stream_bytes(self) -> None:
        requirement = app.assistant_secret_challenges.SecretRequirement(
            ASSISTANT_ID,
            "Shimpz Assistant",
            ("public-user-lookup",),
            (("x-bearer-token", "X Bearer Token", "Required."),),
        )
        pending = app.assistant_secret_challenges.PendingSecretChallenge(
            "e" * 32,
            TEAM_ID,
            1.0,
            (requirement,),
            object(),
        )
        current = mock.Mock(side_effect=(None, pending))
        handler = _RouteHarness({"message": "hello", "files": [], "assistant_ids": [ASSISTANT_ID]})

        @contextlib.contextmanager
        def exclusive(_team_id, _lease):
            yield "turn-token", self.anchor

        with (
            _patched(
                _assistant_secret_challenges=types.SimpleNamespace(current=current),
                _exclusive_chat_turn=exclusive,
            ),
            mock.patch.object(app, "_enforce_rate"),
        ):
            app.Handler._route_chat(
                handler,
                "POST",
                ["v1", "teams", TEAM_ID, "chat", "stream"],
                TEAM_ID,
                ("account", "account_1"),
                types.SimpleNamespace(owner="account_1"),
            )

        self.assertEqual(
            handler.sent,
            [
                (
                    HTTPStatus.PRECONDITION_REQUIRED,
                    app.assistant_secret_flow.challenge_payload(pending),
                    True,
                )
            ],
        )
        self.assertEqual(current.call_count, 2)


if __name__ == "__main__":
    unittest.main()
