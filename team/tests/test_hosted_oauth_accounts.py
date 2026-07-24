from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

import hosted_app_fixture as harness

app = harness.app
_patched = harness._patched
runtime_state = harness.runtime_state

TEAM_ID = "team_1"
ASSISTANT_ID = "shimpz-cloudflare"
SCOPES = ("dns.read", "zone.read")
ACCESS_TOKEN = "-".join(("hosted", "access", "token", "value", "123456789"))
ANCHOR_ID = "a" * 64
ZONE_INPUT = {"page": 1, "per_page": 25}


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


class _Runtime:
    def __init__(self) -> None:
        self.start_calls = 0
        self.resume_calls = 0
        self.request = app.brain_runtime_client.PowerRequest(
            "lookup",
            ASSISTANT_ID,
            "list-zones",
            ZONE_INPUT,
        )

    def start(self, _context, _message):
        self.start_calls += 1
        return app.brain_runtime_client.RuntimeTurn("power-required", "", (self.request,))

    def resume(self, _context, results):
        self.resume_calls += 1
        if set(results) != {"lookup"}:
            raise AssertionError("the admitted Power must resume once")
        return app.brain_runtime_client.RuntimeTurn("completed", "Connected lookup complete.", ())


class HostedOAuthAccountTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.store = app.oauth_account_store.OAuthAccountStore(
            root / "state" / "accounts.json",
            root / "key" / "aes256.key",
        )
        trusted = app.marketplace.APPS[ASSISTANT_ID].assistant
        assert trusted is not None
        self.contract = replace(
            trusted,
            powers={
                power_id: replace(
                    power,
                    secrets=(),
                    accounts=("cloudflare",) if power_id == "list-zones" else (),
                )
                for power_id, power in trusted.powers.items()
            },
            secrets={},
            accounts={"cloudflare": app.marketplace.AccountSpec("cloudflare", SCOPES)},
        )
        self.container = types.SimpleNamespace(id="b" * 64)
        self.active = app._ActiveAssistant(ASSISTANT_ID, self.contract, self.container)

    def _connect(self) -> None:
        self.store.put(
            TEAM_ID,
            ASSISTANT_ID,
            "cloudflare",
            "cloudflare",
            SCOPES,
            app.oauth_http_client.OAuthTokenSet(ACCESS_TOKEN, "refresh-token-value-123456789", SCOPES, 3600),
        )

    def test_refresh_uses_the_configured_hosted_oauth_client(self) -> None:
        token_set = app.oauth_http_client.OAuthTokenSet(ACCESS_TOKEN, "new-refresh-token", SCOPES, 3600)
        oauth_http = mock.Mock()
        oauth_http.refresh.return_value = token_set
        client_secret = "-".join(("hosted", "client", "secret", "value"))
        refresh_token = "-".join(("old", "refresh", "token", "value"))

        with mock.patch.multiple(
            runtime_state,
            _oauth_http=oauth_http,
            _cloudflare_oauth_client_id="client-id",
            _cloudflare_oauth_client_secret=client_secret,
        ):
            result = app._refresh_oauth_account("cloudflare", SCOPES, refresh_token, None)

        self.assertIs(result, token_set)
        oauth_http.refresh.assert_called_once_with(
            provider_id="cloudflare",
            client_id="client-id",
            client_secret=client_secret,
            refresh_token=refresh_token,
            scopes=SCOPES,
        )

    def test_inventory_is_status_only_and_private_token_reaches_only_declared_power(self) -> None:
        self._connect()
        captured: list[dict[str, object]] = []
        inspected = []
        inspect_memo: dict[str, dict[str, dict]] = {}
        turn_token = "turn-token"

        def rpc(_team_id, _token, _container, _command, _method, _path, payload):
            captured.append(payload)
            return _zones()

        def installed(_team_id, _assistant_id, current_inspect_memo=None):
            inspected.append(current_inspect_memo)
            return ASSISTANT_ID, self.contract, self.container

        with (
            mock.patch.object(runtime_state, "_assistant_accounts", self.store),
            mock.patch.multiple(
                harness.hosted_assistants,
                _installed_assistant=installed,
                _assistant_rpc=rpc,
            ),
        ):
            result = app._invoke_assistant_power(
                app.PowerInvocationRequest(
                    team_id=TEAM_ID,
                    token=turn_token,
                    assistant_id=ASSISTANT_ID,
                    contract=self.contract,
                    container=self.container,
                    power="list-zones",
                    payload=ZONE_INPUT,
                    inspect_memo=inspect_memo,
                )
            )
            payload = app.assistant_account_flow.inventory_payload(
                TEAM_ID,
                [app._hosted_secret_spec(self.active)],
                self.store,
            )

        self.assertEqual(result["result"]["zones"][0]["name"], "example.com")
        self.assertEqual(len(inspected), 1)
        self.assertIs(inspected[0], inspect_memo)
        self.assertEqual(
            captured,
            [
                {
                    "input": ZONE_INPUT,
                    "secrets": {},
                    "accounts": {
                        "cloudflare": {"type": "oauth2-bearer", "access_token": ACCESS_TOKEN},
                    },
                    "answers": [],
                }
            ],
        )
        serialized = json.dumps(payload)
        self.assertNotIn(ACCESS_TOKEN, serialized)
        self.assertNotIn("refresh-token", serialized)
        self.assertNotIn("generation", serialized)
        self.assertEqual(payload["accounts"][0]["status"], "connected")

    def test_account_token_exposure_is_rejected_without_echoing_it(self) -> None:
        self._connect()
        turn_token = "turn-token"
        with (
            mock.patch.object(runtime_state, "_assistant_accounts", self.store),
            mock.patch.multiple(
                harness.hosted_assistants,
                _installed_assistant=lambda *_args: (ASSISTANT_ID, self.contract, self.container),
                _assistant_rpc=lambda *_args, **_kwargs: _zones(ACCESS_TOKEN),
            ),
            self.assertRaises(app.ApiError) as caught,
        ):
            app._invoke_assistant_power(
                app.PowerInvocationRequest(
                    team_id=TEAM_ID,
                    token=turn_token,
                    assistant_id=ASSISTANT_ID,
                    contract=self.contract,
                    container=self.container,
                    power="list-zones",
                    payload=ZONE_INPUT,
                )
            )

        self.assertEqual(caught.exception.status, app.HTTPStatus.BAD_GATEWAY)
        self.assertNotIn(ACCESS_TOKEN, caught.exception.message)

    def test_admitted_contract_prunes_removed_accounts_and_cancels_paused_turn(self) -> None:
        self._connect()
        challenge_store = app.assistant_account_challenges.AccountChallengeStore()
        requirement = app.assistant_account_challenges.AccountRequirement(
            ASSISTANT_ID,
            "Shimpz Cloudflare",
            ("list-zones",),
            (("cloudflare", "cloudflare", SCOPES),),
        )
        challenge_store.create(TEAM_ID, (requirement,), object())
        without_accounts = replace(
            app.marketplace.APPS[ASSISTANT_ID],
            assistant=replace(self.contract, accounts={}),
        )

        with (
            mock.patch.object(runtime_state, "_assistant_accounts", self.store),
            mock.patch.object(runtime_state, "_assistant_account_challenges", challenge_store),
        ):
            app._retain_admitted_assistant_accounts(TEAM_ID, ASSISTANT_ID, without_accounts)

        self.assertIsNone(challenge_store.current(TEAM_ID))
        self.assertEqual(self.store.metadata(TEAM_ID, ASSISTANT_ID, {}), ())
        self.assertNotIn(ACCESS_TOKEN, self.store.state_path.read_text(encoding="utf-8"))

    def test_account_resume_can_pause_for_secrets_before_any_power_runs(self) -> None:
        private_contract = replace(
            self.contract,
            powers={
                power_id: replace(
                    power,
                    secrets=("lookup-key",) if power_id == "list-zones" else (),
                )
                for power_id, power in self.contract.powers.items()
            },
            secrets={"lookup-key": app.marketplace.SecretSpec("Lookup key", "Required for this lookup.")},
        )
        active = app._ActiveAssistant(ASSISTANT_ID, private_contract, self.container)
        anchor = types.SimpleNamespace(
            id=ANCHOR_ID,
            labels={"team.name": "Marketing", "team.owner": "account_1"},
        )
        runtime = _Runtime()
        account_challenges = app.assistant_account_challenges.AccountChallengeStore()
        secret_challenges = app.assistant_secret_challenges.SecretChallengeStore()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            secret_store = app.assistant_secret_store.AssistantSecretStore(
                root / "secret-state" / "secrets.json",
                root / "secret-key" / "aes256.key",
            )
            journal = app.power_journal.PowerJournal(root / "journal" / "journal.sqlite3")
            self.addCleanup(journal.close)
            rpc_calls: list[dict[str, object]] = []

            def rpc(_team_id, _token, _container, _command, _method, _path, payload):
                rpc_calls.append(payload)
                return _zones()

            @contextlib.contextmanager
            def exclusive(_team_id, _lease):
                yield "resumed-turn", anchor

            with (
                _patched(
                    _active_team_assistants=lambda _team_id: (active,),
                    _installed_assistant=lambda *_args: (ASSISTANT_ID, private_contract, self.container),
                    _require_assistant_genesis=lambda _container: "Use only the declared X Power.",
                    _chat_file_metadata=lambda _team_id, _files: [],
                    _inference_store=types.SimpleNamespace(
                        load=lambda _team_id: types.SimpleNamespace(provider="openai", model="gpt-test")
                    ),
                    _model_credential=lambda _owner, _provider: ("model-key-value", 7),
                    _require_model_credential_current=lambda *_args: None,
                    _current_team_anchor=lambda *_args: anchor,
                    _brain_runtime=runtime,
                    _power_execution_journal=lambda: journal,
                    _assistant_accounts=self.store,
                    _assistant_account_challenges=account_challenges,
                    _assistant_secrets=secret_store,
                    _assistant_secret_challenges=secret_challenges,
                    _commit_chat_terminal=lambda *_args: True,
                ),
                mock.patch.multiple(
                    runtime_state,
                    _assistant_accounts=self.store,
                    _assistant_account_challenges=account_challenges,
                    _assistant_secrets=secret_store,
                    _assistant_secret_challenges=secret_challenges,
                ),
                mock.patch.multiple(
                    harness.hosted_assistants,
                    _installed_assistant=lambda *_args: (ASSISTANT_ID, private_contract, self.container),
                    _assistant_rpc=rpc,
                ),
            ):
                account_prompt = app._chat_in_turn(
                    TEAM_ID,
                    "Look up Cloudflare.",
                    [],
                    (ASSISTANT_ID,),
                    "initial-turn",
                    anchor,
                    "account_1",
                )
                self.assertEqual(account_prompt["status"], "accounts-required")
                self.assertEqual(runtime.start_calls, 1)
                self.assertEqual(runtime.resume_calls, 0)
                self.assertEqual(rpc_calls, [])

                self._connect()
                with _patched(_exclusive_chat_turn=exclusive):
                    secret_prompt = app._resume_chat_accounts(
                        TEAM_ID,
                        account_prompt["challenge_id"],
                        app._AuthorizationLease(
                            TEAM_ID,
                            ANCHOR_ID,
                            "account_1",
                            ("account", "account_1"),
                        ),
                    )

            self.assertEqual(secret_prompt["status"], "secrets-required")
            self.assertEqual(runtime.start_calls, 1)
            self.assertEqual(runtime.resume_calls, 0)
            self.assertEqual(rpc_calls, [])
            self.assertIsNone(account_challenges.current(TEAM_ID))
            self.assertIsNotNone(secret_challenges.current(TEAM_ID))

    def test_authorize_and_callback_expose_no_oauth_private_material(self) -> None:
        challenge_store = app.assistant_account_challenges.AccountChallengeStore()
        continuation = app.chat_orchestrator.ChatContinuation(
            app.brain_runtime_client.RuntimeTurn("power-required", "", ()),
            (),
            (),
            0,
        )
        pending = app._PendingHostedChat(
            continuation,
            (ASSISTANT_ID,),
            (),
            "account_1",
            ("identity",),
        )
        challenge = challenge_store.create(
            TEAM_ID,
            (
                app.assistant_account_challenges.AccountRequirement(
                    ASSISTANT_ID,
                    "Shimpz Cloudflare",
                    ("list-zones",),
                    (("cloudflare", "cloudflare", SCOPES),),
                ),
            ),
            pending,
        )
        fake_service = types.SimpleNamespace(
            authorization_url=lambda current, session: (
                "https://x.com/i/oauth2/authorize?state=opaque"
                if current is challenge and session == "browser-session-binding-value"
                else None
            ),
            complete=lambda state, code, session, resolver: types.SimpleNamespace(
                team_id=TEAM_ID,
                assistant_id=ASSISTANT_ID,
                account_id="cloudflare",
                provider="cloudflare",
                scopes=SCOPES,
                generation=9,
            ),
            disconnect=lambda *_args: True,
        )
        lease = app._AuthorizationLease(
            TEAM_ID,
            ANCHOR_ID,
            "account_1",
            ("account", "account_1"),
        )
        with _patched(
            _assistant_account_challenges=challenge_store,
            _oauth_accounts=fake_service,
            _require_current_authorization=lambda *_args, **_kwargs: object(),
            _authorize=lambda *_args, **_kwargs: lease,
        ):
            started = app._start_oauth_account(
                TEAM_ID,
                challenge.id,
                "browser-session-binding-value",
                lease,
            )
            completed = app._complete_oauth_account(
                {
                    "state": "provider-state-value",
                    "code": "provider-code-value",
                    "session_binding": "browser-session-binding-value",
                },
                ("account", "account_1"),
            )
            with self.assertRaises(app.ApiError) as extra_field:
                app._complete_oauth_account(
                    {
                        "state": "provider-state-value",
                        "code": "provider-code-value",
                        "session_binding": "browser-session-binding-value",
                        "redirect": "https://attacker.test",
                    },
                    ("account", "account_1"),
                )

        self.assertEqual(started, {"authorization_url": "https://x.com/i/oauth2/authorize?state=opaque"})
        self.assertEqual(extra_field.exception.status, app.HTTPStatus.UNPROCESSABLE_ENTITY)
        self.assertEqual(
            completed,
            {
                "connected": True,
                "team_id": TEAM_ID,
                "assistant_id": ASSISTANT_ID,
                "account_id": "cloudflare",
                "provider": "cloudflare",
                "scopes": list(SCOPES),
                "challenge_id": challenge.id,
            },
        )
        serialized = json.dumps({"started": started, "completed": completed})
        for forbidden in (
            "provider-code-value",
            "browser-session-binding-value",
            "access_token",
            "refresh_token",
            "code_verifier",
            "client_id",
            "generation",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_team_teardown_cancels_account_turn_and_purges_tokens(self) -> None:
        self._connect()
        challenges = app.assistant_account_challenges.AccountChallengeStore()
        challenges.create(
            TEAM_ID,
            (
                app.assistant_account_challenges.AccountRequirement(
                    ASSISTANT_ID,
                    "Shimpz Cloudflare",
                    ("list-zones",),
                    (("cloudflare", "cloudflare", SCOPES),),
                ),
            ),
            object(),
        )
        with (
            mock.patch.object(runtime_state, "_assistant_accounts", self.store),
            mock.patch.object(runtime_state, "_assistant_account_challenges", challenges),
        ):
            self.assertTrue(app._teardown_assistant_accounts(TEAM_ID))

        self.assertIsNone(challenges.current(TEAM_ID))
        self.assertEqual(self.store.metadata(TEAM_ID, ASSISTANT_ID, self.contract.accounts)[0].status, "missing")


if __name__ == "__main__":
    unittest.main()
