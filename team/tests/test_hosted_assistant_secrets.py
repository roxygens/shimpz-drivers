from __future__ import annotations

import contextlib
import json
import sqlite3
import sys
import tempfile
import types
import unittest
from contextlib import closing
from dataclasses import replace
from http import HTTPStatus
from pathlib import Path
from unittest import mock

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

import hosted_app_fixture as harness

app = harness.app
_patched = harness._patched
hosted_apps = harness.hosted_apps
hosted_assistants = harness.hosted_assistants
hosted_resources = harness.hosted_resources
runtime_state = harness.runtime_state

TEAM_ID = "team_1"
ANCHOR_ID = "a" * 64
ASSISTANT_ID = "shimpz-cloudflare"
ZONE_INPUT = {"page": 1, "per_page": 25}
DNS_INPUT = {"zone_id": "a" * 32, "page": 1, "per_page": 25}


def _zones(name: str = "example.com") -> dict[str, object]:
    return {
        "zones": [
            {
                "id": "a" * 32,
                "name": name,
                "status": "active",
                "type": "full",
                "paused": False,
                "account": {"id": "b" * 32, "name": "Shimpz"},
            }
        ],
        "pagination": {"page": 1, "per_page": 25, "count": 1, "total_count": 1, "total_pages": 1},
    }


def _records() -> dict[str, object]:
    return {
        "records": [],
        "pagination": {"page": 1, "per_page": 25, "count": 0, "total_count": 0, "total_pages": 0},
    }


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
                "list-zones",
                ZONE_INPUT,
            ),
            app.brain_runtime_client.PowerRequest(
                "identity-read",
                ASSISTANT_ID,
                "list-dns-records",
                DNS_INPUT,
            ),
        )

    def start(self, _context, _message):
        return app.brain_runtime_client.RuntimeTurn("power-required", "", self.requests)

    def resume(self, _context, results):
        self.resume_calls += 1
        if set(results) != {"public-read", "identity-read"}:
            raise AssertionError("the complete Power batch must resume together")
        return app.brain_runtime_client.RuntimeTurn("completed", "Cloudflare account connected.", ())


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
        trusted_contract = app.marketplace.APPS[ASSISTANT_ID].assistant
        assert trusted_contract is not None
        secret_contract = {
            secret_id: app.marketplace.SecretSpec(secret_id.replace("-", " ").title(), "Test credential.")
            for secret_id in SECRET_VALUES
        }
        self.contract = replace(
            trusted_contract,
            powers={
                power_id: replace(
                    power,
                    secrets=(tuple(SECRET_VALUES)[:1] if power_id == "list-zones" else tuple(SECRET_VALUES)[1:]),
                    accounts=(),
                )
                for power_id, power in trusted_contract.powers.items()
            },
            secrets=secret_contract,
            accounts={},
        )
        self.assistant_container = types.SimpleNamespace(id="b" * 64)
        self.active = app._ActiveAssistant(ASSISTANT_ID, self.contract, self.assistant_container)
        self.anchor = types.SimpleNamespace(
            id=ANCHOR_ID,
            labels={"team.name": "Marketing", "team.owner": "account_1"},
        )
        self.rpc_calls: list[tuple[str, dict[str, object]]] = []

    def _rpc(self, _team_id, _token, _container, _command, _method, path, payload):
        self.rpc_calls.append((path, payload))
        return _zones() if path.endswith("list-zones") else _records()

    @contextlib.contextmanager
    def _environment(self):
        with _patched(
            _active_team_assistants=lambda _team_id: (self.active,),
            _require_assistant_genesis=lambda _container: "Use only the declared Cloudflare Powers.",
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
            self.assertEqual(requirement["power_ids"], ["list-dns-records", "list-zones"])
            self.assertEqual(
                {item["id"] for item in requirement["secrets"]},
                set(SECRET_VALUES),
            )
            serialized = json.dumps(challenge)
            for secret in SECRET_VALUES.values():
                self.assertNotIn(secret, serialized)
            self.assertNotIn("private-value", serialized)

            @contextlib.contextmanager
            def exclusive(_team_id, _lease):
                yield "resumed-turn", self.anchor

            with _patched(_exclusive_chat_turn=exclusive):
                result = app._submit_chat_secrets(
                    TEAM_ID,
                    self._submission(challenge),
                    app._AuthorizationLease(TEAM_ID, ANCHOR_ID, "account_1", ("account", "account_1")),
                )

        self.assertEqual(result["reply"], "Cloudflare account connected.")
        self.assertEqual(self.runtime.resume_calls, 1)
        self.assertEqual(len(self.rpc_calls), 2)
        payloads = dict(self.rpc_calls)
        self.assertEqual(
            payloads["/v1/powers/list-zones"],
            {
                "input": ZONE_INPUT,
                "secrets": {"x-bearer-token": SECRET_VALUES["x-bearer-token"]},
                "accounts": {},
                "answers": [],
            },
        )

        self.assertEqual(
            payloads["/v1/powers/list-dns-records"],
            {
                "input": DNS_INPUT,
                "secrets": {
                    secret_id: SECRET_VALUES[secret_id]
                    for secret_id in (
                        "x-api-key",
                        "x-api-key-secret",
                        "x-access-token",
                        "x-access-token-secret",
                    )
                },
                "accounts": {},
                "answers": [],
            },
        )
        state = self.secret_store.state_path.read_text(encoding="utf-8")
        journal = self.journal.path.read_bytes()
        for secret in SECRET_VALUES.values():
            self.assertNotIn(secret, state)
            self.assertNotIn(secret.encode(), journal)
            self.assertNotIn(secret, json.dumps(result))
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
        turn_token = "turn-token"
        self.secret_store.put_many(TEAM_ID, ASSISTANT_ID, {"x-bearer-token": secret})
        with (
            _patched(
                _assistant_secrets=self.secret_store,
                _installed_assistant=lambda *_args: (ASSISTANT_ID, self.contract, self.assistant_container),
                _assistant_rpc=lambda *_args, **_kwargs: _zones(secret),
            ),
            self.assertRaises(app.ApiError) as caught,
        ):
            app._invoke_assistant_power(
                app.PowerInvocationRequest(
                    team_id=TEAM_ID,
                    token=turn_token,
                    assistant_id=ASSISTANT_ID,
                    contract=self.contract,
                    container=self.assistant_container,
                    power="list-zones",
                    payload=ZONE_INPUT,
                )
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
            app.Handler._route_chat_turn(
                handler,
                types.SimpleNamespace(
                    team_id=TEAM_ID,
                    principal=("account", "account_1"),
                    lease=object(),
                ),
                stream=False,
            )
        self.assertEqual(handler.sent, [(HTTPStatus.PRECONDITION_REQUIRED, challenge, True)])

        self.secret_store.put_many(TEAM_ID, ASSISTANT_ID, {"x-bearer-token": "stored-secret-value"})
        pending = self.challenge_store.create(
            TEAM_ID,
            (
                app.assistant_secret_challenges.SecretRequirement(
                    ASSISTANT_ID,
                    "Shimpz Cloudflare",
                    ("list-zones",),
                    (("x-bearer-token", "X Bearer Token", "Required."),),
                ),
            ),
            object(),
        )
        with (
            mock.patch.object(runtime_state, "_assistant_secrets", self.secret_store),
            mock.patch.object(runtime_state, "_assistant_secret_challenges", self.challenge_store),
            mock.patch.object(hosted_resources, "_require_current_authorization", return_value=None),
            mock.patch.object(
                hosted_apps,
                "_teardown_app",
                return_value=app._CleanupResult(True, True),
            ),
        ):
            result = app._uninstall_app(TEAM_ID, ASSISTANT_ID, object())
        self.assertTrue(result["uninstalled"])
        self.assertIsNone(self.challenge_store.current(TEAM_ID))
        self.assertTrue(pending.id)
        self.assertFalse(self.secret_store.metadata(TEAM_ID, ASSISTANT_ID, ("x-bearer-token",))[0].configured)

    def test_hosted_rotation_is_atomic_masked_and_invalidates_a_stale_challenge(self) -> None:
        declared = {
            "primary-token": app.marketplace.SecretSpec("Primary token", "Primary credential."),
            "secondary-token": app.marketplace.SecretSpec("Secondary token", "Secondary credential."),
        }
        contract = app.marketplace.AssistantContract("assistant-rpc", {}, declared)
        container = types.SimpleNamespace(id="c" * 64)
        active = app._ActiveAssistant(ASSISTANT_ID, contract, container)
        spec = app._hosted_secret_spec(active)
        original = {
            "primary-token": "primary-original-credential",
            "secondary-token": "secondary-original-credential",
        }
        replacement = "primary-rotated-credential"
        self.secret_store.put_many(TEAM_ID, ASSISTANT_ID, original)
        pending = self.challenge_store.create(
            TEAM_ID,
            (
                app.assistant_secret_challenges.SecretRequirement(
                    ASSISTANT_ID,
                    "Shimpz Cloudflare",
                    ("read-account",),
                    (("primary-token", "Primary token", "Primary credential."),),
                ),
            ),
            object(),
        )
        lease = app._AuthorizationLease(TEAM_ID, ANCHOR_ID, "account_1", ("account", "account_1"))
        invalid = {
            "assistant_id": ASSISTANT_ID,
            "values": [
                {"secret_id": "primary-token", "value": "must-not-commit"},
                {"secret_id": "undeclared-token", "value": "attacker-controlled"},
            ],
        }
        with _patched(
            _require_current_authorization=lambda *_args, **_kwargs: self.anchor,
            _installed_assistant=lambda *_args: (ASSISTANT_ID, contract, container),
            _installed_assistant_secret_specs=lambda _team_id: (spec,),
            _assistant_secrets=self.secret_store,
            _assistant_secret_challenges=self.challenge_store,
        ):
            with self.assertRaises(app.ApiError) as rejected:
                app._replace_assistant_secrets(TEAM_ID, invalid, lease)
            self.assertEqual(rejected.exception.status, HTTPStatus.UNPROCESSABLE_ENTITY)
            self.assertEqual(self.challenge_store.current(TEAM_ID).id, pending.id)

            response = app._replace_assistant_secrets(
                TEAM_ID,
                {
                    "assistant_id": ASSISTANT_ID,
                    "values": [{"secret_id": "primary-token", "value": replacement}],
                },
                lease,
            )

        stored = self.secret_store.resolve_many(TEAM_ID, ASSISTANT_ID, tuple(declared))
        self.assertEqual(stored["primary-token"], replacement)
        self.assertEqual(stored["secondary-token"], original["secondary-token"])
        self.assertIsNone(self.challenge_store.current(TEAM_ID))
        serialized = json.dumps(response)
        for value in (*original.values(), replacement, "must-not-commit", "attacker-controlled"):
            self.assertNotIn(value, serialized)
        metadata = {item["id"]: item for item in response["assistants"][0]["secrets"]}
        self.assertEqual(metadata["primary-token"]["mask"], app.assistant_secret_store.mask_secret(replacement))
        self.assertTrue(metadata["secondary-token"]["configured"])

    def test_hosted_rotation_is_rejected_before_authorization_or_storage_during_chat(self) -> None:
        self.secret_store.put_many(TEAM_ID, ASSISTANT_ID, {"primary-token": "original-credential"})
        before = self.secret_store.state_path.read_bytes()
        lock = app._chat_lock_for(TEAM_ID)
        self.assertTrue(lock.acquire(blocking=False))
        try:
            with (
                _patched(
                    _require_current_authorization=lambda *_args, **_kwargs: self.fail("authorization ran during chat"),
                    _assistant_secrets=self.secret_store,
                ),
                self.assertRaises(app.ApiError) as rejected,
            ):
                app._replace_assistant_secrets(TEAM_ID, {}, object())
        finally:
            lock.release()

        self.assertEqual(rejected.exception.status, HTTPStatus.CONFLICT)
        self.assertEqual(self.secret_store.state_path.read_bytes(), before)

    def test_hosted_rotation_route_is_rate_limited_and_never_cacheable(self) -> None:
        body = {
            "assistant_id": ASSISTANT_ID,
            "values": [{"secret_id": "primary-token", "value": "private-value"}],
        }
        response = {"team_id": TEAM_ID, "assistants": []}
        handler = _RouteHarness(body)
        principal = ("account", "account_1")
        with (
            mock.patch.object(app, "_enforce_rate") as enforce,
            mock.patch.object(app, "_replace_assistant_secrets", return_value=response) as replace_secrets,
            mock.patch.object(app.audit, "log"),
        ):
            lease = object()
            app.Handler._route_assistant_secret_replace(
                handler,
                types.SimpleNamespace(team_id=TEAM_ID, principal=principal, lease=lease),
            )

        enforce.assert_called_once_with("secret", principal)
        replace_secrets.assert_called_once_with(TEAM_ID, body, lease)
        self.assertEqual(handler.sent, [(HTTPStatus.OK, response, True)])

    def test_idempotent_install_prunes_obsolete_secrets_after_admission(self) -> None:
        declared = {"retained-token": app.marketplace.SecretSpec("Retained token", "Still declared.")}
        contract = app.marketplace.AssistantContract("assistant-rpc", {}, declared)
        spec = types.SimpleNamespace(
            assistant=contract,
            image="registry.example/shimpz-cloudflare@sha256:" + ("d" * 64),
            port=8080,
            health_path="/health",
        )
        container = types.SimpleNamespace(
            id="e" * 64,
            labels={
                "team.app.driver": "1",
                "team.id": TEAM_ID,
                "team.app": ASSISTANT_ID,
                "team.owner": "account_1",
            },
            attrs={"Config": {"Image": spec.image}},
            status="running",
            reload=lambda: None,
        )
        self.secret_store.put_many(
            TEAM_ID,
            ASSISTANT_ID,
            {
                "retained-token": "retained-secret-value",
                "obsolete-token": "obsolete-secret-value",
            },
        )
        self.challenge_store.create(
            TEAM_ID,
            (
                app.assistant_secret_challenges.SecretRequirement(
                    ASSISTANT_ID,
                    "Shimpz Cloudflare",
                    ("old-power",),
                    (("obsolete-token", "Obsolete token", "Removed."),),
                ),
            ),
            object(),
        )
        with (
            mock.patch.object(runtime_state, "_lock_for", side_effect=lambda _team_id: contextlib.nullcontext()),
            mock.patch.object(runtime_state, "_assistant_secrets", self.secret_store),
            mock.patch.object(runtime_state, "_assistant_secret_challenges", self.challenge_store),
            mock.patch.object(
                hosted_resources,
                "_require_current_authorization",
                return_value=types.SimpleNamespace(labels={"team.name": "Marketing"}),
            ),
            mock.patch.object(hosted_resources, "_prepare_marketplace_image", return_value=None),
            mock.patch.object(hosted_resources, "_get_container", return_value=container),
            mock.patch.object(hosted_resources, "_require_team_isolation", return_value=None),
            mock.patch.object(hosted_assistants, "_admit_app_contract", return_value=()),
            mock.patch.object(hosted_apps, "_validate_admitted_egress", return_value="admitted-token"),
            mock.patch.object(hosted_apps, "_validate_assistant_proxy_environment", return_value=None),
            mock.patch.object(hosted_apps, "_app_ready_now", return_value=(True, "running")),
        ):
            result = app._install_app(
                TEAM_ID,
                ASSISTANT_ID,
                spec,
                "account_1",
                types.SimpleNamespace(owner="account_1"),
            )

        metadata = {
            item.id: item
            for item in self.secret_store.metadata(
                TEAM_ID,
                ASSISTANT_ID,
                ("retained-token", "obsolete-token"),
            )
        }
        self.assertFalse(result["installed"])
        self.assertTrue(metadata["retained-token"].configured)
        self.assertFalse(metadata["obsolete-token"].configured)
        self.assertIsNone(self.challenge_store.current(TEAM_ID))

    def test_team_secret_teardown_cancels_continuations_and_deletes_all_records(self) -> None:
        self.secret_store.put_many(TEAM_ID, ASSISTANT_ID, {"retained-token": "retained-secret-value"})
        self.challenge_store.create(
            TEAM_ID,
            (
                app.assistant_secret_challenges.SecretRequirement(
                    ASSISTANT_ID,
                    "Shimpz Cloudflare",
                    ("read",),
                    (("retained-token", "Retained token", "Required."),),
                ),
            ),
            object(),
        )
        with _patched(
            _assistant_secrets=self.secret_store,
            _assistant_secret_challenges=self.challenge_store,
        ):
            complete = app._teardown_assistant_secrets(TEAM_ID)

        self.assertTrue(complete)
        self.assertIsNone(self.challenge_store.current(TEAM_ID))
        self.assertFalse(self.secret_store.metadata(TEAM_ID, ASSISTANT_ID, ("retained-token",))[0].configured)

    def test_chat_rechecks_a_pending_secret_challenge_after_acquiring_its_slot(self) -> None:
        requirement = app.assistant_secret_challenges.SecretRequirement(
            ASSISTANT_ID,
            "Shimpz Cloudflare",
            ("list-zones",),
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
            "Shimpz Cloudflare",
            ("list-zones",),
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
            app.Handler._route_chat_turn(
                handler,
                types.SimpleNamespace(
                    team_id=TEAM_ID,
                    principal=("account", "account_1"),
                    lease=types.SimpleNamespace(owner="account_1"),
                ),
                stream=True,
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
