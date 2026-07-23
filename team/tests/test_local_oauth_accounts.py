from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM))

import assistant_account_challenges
import brain_runtime_client
import chat_orchestrator
import chat_turn_engine
import inference_config
import local_app
import local_registry
import oauth_account_service
import oauth_account_store
import oauth_broker_client
from local_support.chat_segment import SegmentRequest

TEST_ACCESS_TOKEN = "oauth-access-test-token-123456789"
TEST_REFRESH_TOKEN = "oauth-refresh-test-token-123456789"


class LocalOAuthAccountTests(unittest.TestCase):
    @staticmethod
    def _registry() -> dict[str, local_registry.AssistantSpec]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": 2,
                        "images": {
                            "shimpz-cloudflare": "example.invalid/cloudflare@sha256:" + ("b" * 64),
                        },
                    }
                ),
                encoding="utf-8",
            )
            return local_registry.load_registry(path)

    def test_controller_accepts_injected_account_state(self) -> None:
        injected_store = SimpleNamespace()
        injected_challenges = assistant_account_challenges.AccountChallengeStore()
        controller = local_app.LocalController(
            SimpleNamespace(info=lambda: {"SecurityOptions": ["name=seccomp"], "NCPU": 2}),
            "local-space",
            self._registry(),
            SimpleNamespace(),
            inference_store=SimpleNamespace(),
            brain_runtime=SimpleNamespace(),
            power_state=SimpleNamespace(),
            assistant_secrets=SimpleNamespace(),
            secret_challenges=SimpleNamespace(),
            assistant_accounts=injected_store,
            account_challenges=injected_challenges,
            oauth_service=SimpleNamespace(),
            approval_challenges=SimpleNamespace(),
            approval_grants=SimpleNamespace(),
        )

        self.assertIs(controller.assistant_accounts, injected_store)
        self.assertIs(controller.account_challenges, injected_challenges)

    def test_team_account_teardown_prevents_same_id_resurrection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = object.__new__(local_app.LocalController)
            controller.assistant_accounts = oauth_account_store.OAuthAccountStore(
                Path(directory) / "state" / "accounts.json",
                Path(directory) / "key" / "aes256.key",
            )
            controller.assistant_accounts.put(
                "team_1",
                "shimpz-cloudflare",
                "cloudflare",
                "cloudflare",
                ("zone.read",),
                SimpleNamespace(
                    access_token=TEST_ACCESS_TOKEN,
                    refresh_token=TEST_REFRESH_TOKEN,
                    scopes=("zone.read",),
                    expires_in=3600,
                ),
            )

            controller._delete_team_account_state("team_1")
            recreated = controller.assistant_accounts.metadata(
                "team_1",
                "shimpz-cloudflare",
                {"cloudflare": {"provider": "cloudflare", "scopes": ("zone.read",)}},
            )

        self.assertEqual(recreated[0].status, "missing")

    def test_account_inventory_is_exact_and_never_contains_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = object.__new__(local_app.LocalController)
            controller._locks = tuple(threading.RLock() for _ in range(64))
            controller.registry = self._registry()
            controller.assistant_accounts = oauth_account_store.OAuthAccountStore(
                Path(directory) / "state" / "accounts.json",
                Path(directory) / "key" / "aes256.key",
            )
            controller.list_assistants = lambda _team: {
                "assistants": [{"assistant": "shimpz-cloudflare", "status": "running"}]
            }

            payload = controller.list_assistant_accounts("team_1")

        self.assertEqual(set(payload), {"team_id", "accounts"})
        self.assertEqual(payload["team_id"], "team_1")
        self.assertEqual(
            payload["accounts"],
            [
                {
                    "assistant_id": "shimpz-cloudflare",
                    "assistant_name": "Shimpz Cloudflare",
                    "id": "cloudflare",
                    "provider": "cloudflare",
                    "name": "Cloudflare",
                    "summary": (
                        "Connect your Cloudflare account so this Assistant can use only its reviewed read permissions."
                    ),
                    "scopes": ["dns.read", "offline_access", "zone.read"],
                    "status": "missing",
                    "account": None,
                    "expires_at": None,
                }
            ],
        )
        encoded = repr(payload)
        self.assertNotIn("access_token", encoded)
        self.assertNotIn("refresh_token", encoded)
        self.assertNotIn("generation", encoded)

    def test_account_inventory_route_has_one_exact_internal_shape(self) -> None:
        expected = {"team_id": "team_1", "accounts": []}
        handler = object.__new__(local_app.Handler)
        handler.command = "GET"
        handler.server = SimpleNamespace(controller=SimpleNamespace(list_assistant_accounts=lambda team_id: expected))

        route = handler._assistant_account_route(["v1", "teams", "team_1", "assistant-accounts"])

        self.assertEqual(
            route,
            (HTTPStatus.OK, expected, "assistant-account-list", "team_1", None),
        )

    def test_local_controller_builds_only_the_hosted_broker_boundary(self) -> None:
        transport = SimpleNamespace()
        broker = SimpleNamespace()
        service = SimpleNamespace()
        pkce = SimpleNamespace()
        accounts = SimpleNamespace()

        with (
            mock.patch.dict(
                os.environ,
                {
                    "SHIMPZ_OAUTH_BROKER_PROXY_HOST": "oauth-broker-proxy",
                    "SHIMPZ_OAUTH_BROKER_PROXY_TOKEN": "a" * 64,
                    "SHIMPZ_OAUTH_CALLBACK_MODE": "loopback",
                },
            ),
            mock.patch.object(oauth_broker_client, "FixedBrokerTransport", return_value=transport) as transport_type,
            mock.patch.object(oauth_broker_client, "OAuthBrokerClient", return_value=broker) as broker_type,
            mock.patch.object(
                oauth_account_service,
                "BrokeredOAuthAccountService",
                return_value=service,
            ) as service_type,
        ):
            controller = local_app.LocalController(
                SimpleNamespace(info=lambda: {"SecurityOptions": ["name=seccomp"], "NCPU": 2}),
                "local-space",
                self._registry(),
                SimpleNamespace(),
                inference_store=SimpleNamespace(),
                brain_runtime=SimpleNamespace(),
                power_state=SimpleNamespace(),
                assistant_secrets=SimpleNamespace(),
                secret_challenges=SimpleNamespace(),
                assistant_accounts=accounts,
                account_challenges=SimpleNamespace(),
                oauth_pkce=pkce,
                approval_challenges=SimpleNamespace(),
                approval_grants=SimpleNamespace(),
            )

        transport_type.assert_called_once_with(proxy_host="oauth-broker-proxy", proxy_token="a" * 64)
        broker_type.assert_called_once_with(transport=transport, callback_mode="loopback")
        service_type.assert_called_once_with(challenge=pkce, store=accounts, broker=broker)
        self.assertIs(controller.oauth_broker, broker)
        self.assertIs(controller.oauth_service, service)

    def test_authorization_and_callback_delegate_to_one_brokered_service(self) -> None:
        requirement = assistant_account_challenges.AccountRequirement(
            assistant_id="shimpz-cloudflare",
            assistant_name="Shimpz Cloudflare",
            power_ids=("list-zones",),
            accounts=(("cloudflare", "cloudflare", ("dns.read", "offline_access", "zone.read")),),
        )
        challenges = assistant_account_challenges.AccountChallengeStore()
        pending = challenges.create("team_1", (requirement,), {"private": "continuation"})
        calls: list[tuple[str, object]] = []

        class Service:
            def authorization_url(self, challenge, session_binding):
                calls.append(("start", (challenge, session_binding)))
                return (
                    "https://shimpz.com/api/oauth/cloudflare/start?state="
                    + "s" * 43
                    + "&code_challenge="
                    + "c" * 43
                    + "&scope=dns.read+offline_access+zone.read"
                )

            def complete(self, state, claim, session_binding, resolver):
                calls.append(("complete", (state, claim, session_binding, resolver)))
                return oauth_account_service.OAuthAccountCompletion(
                    "team_1",
                    "shimpz-cloudflare",
                    "cloudflare",
                    "cloudflare",
                    ("dns.read", "offline_access", "zone.read"),
                    1,
                )

        controller = object.__new__(local_app.LocalController)
        controller.account_challenges = challenges
        controller.oauth_service = Service()
        controller._current_account_declaration = lambda *_args: None

        started = controller.start_assistant_account_authorization(
            "team_1",
            pending.id,
            "browser-session-private-123456789",
        )
        completed = controller.complete_cloudflare_oauth_callback(
            state="s" * 43,
            claim="a" * 64,
            session_binding="browser-session-private-123456789",
        )

        self.assertEqual(set(started), {"authorization_url"})
        self.assertEqual(
            completed,
            {
                "connected": True,
                "team_id": "team_1",
                "assistant_id": "shimpz-cloudflare",
                "account_id": "cloudflare",
            },
        )
        self.assertEqual([call[0] for call in calls], ["start", "complete"])

    def test_internal_oauth_routes_are_closed_and_exact(self) -> None:
        controller = SimpleNamespace(
            start_assistant_account_authorization=lambda team, challenge, binding: {
                "authorization_url": f"https://shimpz.com/{team}/{challenge}/{binding}"
            },
            complete_cloudflare_oauth_callback=lambda **_values: {
                "connected": True,
                "team_id": "team_1",
                "assistant_id": "shimpz-cloudflare",
                "account_id": "cloudflare",
            },
            disconnect_assistant_account=lambda *_values: {"disconnected": True},
        )
        handler = object.__new__(local_app.Handler)
        handler.server = SimpleNamespace(controller=controller)
        handler._body = lambda **_kwargs: {"session_binding": "browser-session-private-123456789"}
        handler.command = "POST"

        authorize = handler._assistant_account_route(
            [
                "v1",
                "teams",
                "team_1",
                "assistant-accounts",
                "challenges",
                "a" * 32,
                "authorize",
            ]
        )
        self.assertEqual(authorize[0], HTTPStatus.OK)
        self.assertEqual(authorize[2], "assistant-account-authorize")

        handler._body = lambda **_kwargs: {
            "state": "s" * 43,
            "claim": "a" * 64,
            "session_binding": "browser-session-private-123456789",
        }
        callback = handler._fixed_route(["v1", "oauth", "cloudflare", "callback"])
        self.assertEqual(callback[0], HTTPStatus.OK)
        self.assertEqual(callback[2], "assistant-account-complete")

        handler.command = "DELETE"
        disconnected = handler._assistant_account_route(
            [
                "v1",
                "teams",
                "team_1",
                "assistant-accounts",
                "shimpz-cloudflare",
                "cloudflare",
            ]
        )
        self.assertEqual(disconnected[1], {"disconnected": True})
        self.assertEqual(disconnected[2], "assistant-account-disconnect")

    def test_chat_pauses_before_any_power_when_account_is_missing(self) -> None:
        spec = self._registry()["shimpz-cloudflare"]
        request = brain_runtime_client.PowerRequest(
            interrupt_id="call-1",
            assistant_id=spec.assistant_id,
            power="list-zones",
            input={"page": 1, "per_page": 25},
        )

        class Runtime:
            def start(self, _context, _message):
                return brain_runtime_client.RuntimeTurn("power-required", "", (request,))

            def resume(self, _context, _results):  # pragma: no cover - must stay unreachable
                raise AssertionError("Power batch must not execute before OAuth consent")

        with tempfile.TemporaryDirectory() as directory:
            controller = object.__new__(local_app.LocalController)
            controller.space_id = "local-space"
            controller.brain_runtime = Runtime()
            controller.power_state = SimpleNamespace()
            controller.assistant_accounts = oauth_account_store.OAuthAccountStore(
                Path(directory) / "state" / "accounts.json",
                Path(directory) / "key" / "aes256.key",
            )
            controller.assistant_secrets = SimpleNamespace(
                metadata=lambda *_args: (_ for _ in ()).throw(
                    AssertionError("secret gate must run only after the account gate")
                )
            )
            controller.approval_grants = SimpleNamespace()
            active = local_app._ActiveAssistant(spec, "b" * 64)
            setup = (
                "Team One",
                "c" * 64,
                (active,),
                [],
                inference_config.InferenceConfig("openai", "gpt-5-nano"),
            )
            controller._chat_setup = lambda *_args: setup
            controller._active_assistant_genesis = lambda _active: "Use reviewed Powers only."
            controller._chat_cancelled = lambda _token: False
            controller._invoke_chat_power = lambda *_args: (_ for _ in ()).throw(
                AssertionError("Power must not execute before OAuth consent")
            )
            turn_token = "turn-token"

            result = controller._run_chat_segment(
                SegmentRequest(
                    team_id="team_1",
                    file_ids=[],
                    assistant_ids=(spec.assistant_id,),
                    provider="openai",
                    api_key="test-api-key",
                    token=turn_token,
                    message="List my Cloudflare zones",
                )
            )

        self.assertIsInstance(result.outcome, chat_orchestrator.ChatSuspension)
        self.assertEqual(len(result.accounts), 1)
        self.assertEqual(result.accounts[0].accounts[0][0], "cloudflare")
        self.assertEqual((result.secrets, result.inputs, result.approvals, result.answer_logs), ((), (), (), ()))

    def test_account_resume_is_one_use_and_returns_completed_turn(self) -> None:
        registry = self._registry()
        spec = registry["shimpz-cloudflare"]
        request = brain_runtime_client.PowerRequest(
            interrupt_id="call-1",
            assistant_id=spec.assistant_id,
            power="list-zones",
            input={"page": 1, "per_page": 25},
        )
        continuation = chat_orchestrator.ChatContinuation(
            turn=brain_runtime_client.RuntimeTurn("power-required", "", (request,)),
            seen_interrupts=(),
            invoked=(),
            round_index=0,
        )
        requirements = (
            assistant_account_challenges.AccountRequirement(
                assistant_id=spec.assistant_id,
                assistant_name=spec.name,
                power_ids=("list-zones",),
                accounts=(("cloudflare", "cloudflare", spec.accounts["cloudflare"].scopes),),
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            controller = object.__new__(local_app.LocalController)
            controller.registry = registry
            controller._locks = tuple(threading.RLock() for _ in range(64))
            controller._active_chat_guard = threading.Lock()
            controller._chat_locks = {}
            controller._active_chat_tokens = {}
            controller._active_power_containers = {}
            controller._cancelled_chat_tokens = set()
            controller.account_challenges = assistant_account_challenges.AccountChallengeStore()
            controller.oauth_pkce = SimpleNamespace(cancel_team=lambda _team: 0)
            controller.assistant_accounts = oauth_account_store.OAuthAccountStore(
                Path(directory) / "state" / "accounts.json",
                Path(directory) / "key" / "aes256.key",
            )
            config = inference_config.InferenceConfig("openai", "gpt-5-nano")
            active = local_app._ActiveAssistant(spec, "b" * 64)
            setup = ("Team One", "c" * 64, (active,), [], config)
            identity = controller._chat_identity(*setup)
            controller._chat_setup = lambda *_args: setup
            pending = local_app._PendingLocalChat(
                continuation=continuation,
                assistant_ids=(spec.assistant_id,),
                file_ids=(),
                provider="openai",
                identity=identity,
            )
            challenge = controller.account_challenges.create("team_1", requirements, pending)
            controller.assistant_accounts.put(
                "team_1",
                spec.assistant_id,
                "cloudflare",
                "cloudflare",
                spec.accounts["cloudflare"].scopes,
                SimpleNamespace(
                    access_token="a" * 32,
                    refresh_token="r" * 32,
                    scopes=spec.accounts["cloudflare"].scopes,
                    expires_in=3600,
                ),
            )
            controller._run_chat_segment = lambda *_args, **_kwargs: chat_turn_engine.SegmentResult(
                "Team One",
                identity,
                chat_orchestrator.ChatOutcome("Done", ()),
                (),
                (),
                (),
                (),
                (),
            )

            response = controller.resume_chat_accounts(
                "team_1",
                {"challenge_id": challenge.id},
                "openai",
                "test-api-key",
            )

            self.assertEqual(response, {"team_id": "team_1", "team_name": "Team One", "reply": "Done"})
            self.assertIsNone(controller.account_challenges.current("team_1"))
            with self.assertRaises(local_app.ApiProblem) as replay:
                controller.resume_chat_accounts(
                    "team_1",
                    {"challenge_id": challenge.id},
                    "openai",
                    "test-api-key",
                )
            self.assertEqual(replay.exception.code, "assistant-account-challenge-expired")

    def test_chat_account_routes_are_exact(self) -> None:
        pending = {"team_id": "team_1", "status": "accounts-required"}
        completed = {"team_id": "team_1", "team_name": "Team One", "reply": "Done"}
        controller = SimpleNamespace(
            pending_chat_accounts=lambda team_id: pending,
            resume_chat_accounts=lambda team_id, body, provider, api_key: completed,
        )
        handler = object.__new__(local_app.Handler)
        handler.server = SimpleNamespace(controller=controller)
        handler._model_credential_headers = lambda: ("openai", "test-api-key")
        handler._body = lambda **_kwargs: {"challenge_id": "a" * 32}

        handler.command = "GET"
        self.assertEqual(
            handler._chat_route(["v1", "teams", "team_1", "chat", "accounts"]),
            (HTTPStatus.OK, pending, "chat-account-pending", "team_1", None),
        )
        handler.command = "POST"
        self.assertEqual(
            handler._chat_route(["v1", "teams", "team_1", "chat", "accounts"]),
            (HTTPStatus.OK, completed, "chat-account-submit", "team_1", None),
        )


if __name__ == "__main__":
    unittest.main()
