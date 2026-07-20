from __future__ import annotations

import tempfile
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

import test_r2_bridge as harness

app = harness.app
_patched = harness._patched

TEAM_ID = "team_1"
ASSISTANT_ID = app.assistant_contract.ASSISTANT_ID
SCOPES = ("tweet.read", "users.read")
ACCESS_TOKEN = "hosted-access-token-value-123456789"


class HostedOAuthConnectionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.store = app.oauth_connection_store.OAuthConnectionStore(
            root / "state" / "connections.json",
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
                    connections=("x",) if power_id == "public-user-lookup" else (),
                )
                for power_id, power in trusted.powers.items()
            },
            secrets={},
            connections={"x": app.marketplace.ConnectionSpec("x", SCOPES)},
        )
        self.container = types.SimpleNamespace(id="b" * 64)
        self.active = app._ActiveAssistant(ASSISTANT_ID, self.contract, self.container)

    def _connect(self) -> None:
        self.store.put(
            TEAM_ID,
            ASSISTANT_ID,
            "x",
            "x",
            SCOPES,
            app.oauth_http_client.OAuthTokenSet(ACCESS_TOKEN, "refresh-token-value-123456789", SCOPES, 3600),
        )

    def test_inventory_is_status_only_and_private_token_reaches_only_declared_power(self) -> None:
        self._connect()
        captured: list[dict[str, object]] = []

        def rpc(_team_id, _token, _container, _command, _method, _path, payload):
            captured.append(payload)
            return {"id": "123", "name": "X Developers", "username": "XDevelopers"}

        with _patched(
            _assistant_connections=self.store,
            _installed_assistant=lambda *_args: (ASSISTANT_ID, self.contract, self.container),
            _assistant_rpc=rpc,
        ):
            result = app._invoke_assistant_power(
                TEAM_ID,
                "turn-token",
                ASSISTANT_ID,
                self.contract,
                self.container,
                "public-user-lookup",
                {"username": "XDevelopers"},
            )
            payload = app.assistant_connection_flow.inventory_payload(
                TEAM_ID,
                [app._hosted_secret_spec(self.active)],
                self.store,
            )

        self.assertEqual(result["result"]["username"], "XDevelopers")
        self.assertEqual(
            captured,
            [
                {
                    "input": {"username": "XDevelopers"},
                    "secrets": {},
                    "connections": {
                        "x": {"type": "oauth2-bearer", "access_token": ACCESS_TOKEN},
                    },
                }
            ],
        )
        serialized = app.json.dumps(payload)
        self.assertNotIn(ACCESS_TOKEN, serialized)
        self.assertNotIn("refresh-token", serialized)
        self.assertNotIn("generation", serialized)
        self.assertEqual(payload["connections"][0]["status"], "connected")

    def test_connection_token_exposure_is_rejected_without_echoing_it(self) -> None:
        self._connect()
        with (
            _patched(
                _assistant_connections=self.store,
                _installed_assistant=lambda *_args: (ASSISTANT_ID, self.contract, self.container),
                _assistant_rpc=lambda *_args, **_kwargs: {"id": "123", "name": ACCESS_TOKEN},
            ),
            self.assertRaises(app.ApiError) as caught,
        ):
            app._invoke_assistant_power(
                TEAM_ID,
                "turn-token",
                ASSISTANT_ID,
                self.contract,
                self.container,
                "public-user-lookup",
                {"username": "XDevelopers"},
            )

        self.assertEqual(caught.exception.status, app.HTTPStatus.BAD_GATEWAY)
        self.assertNotIn(ACCESS_TOKEN, caught.exception.message)

    def test_admitted_contract_prunes_removed_connections_and_cancels_paused_turn(self) -> None:
        self._connect()
        challenge_store = app.assistant_connection_challenges.ConnectionChallengeStore()
        requirement = app.assistant_connection_challenges.ConnectionRequirement(
            ASSISTANT_ID,
            "Shimpz Assistant",
            ("public-user-lookup",),
            (("x", "x", SCOPES),),
        )
        challenge_store.create(TEAM_ID, (requirement,), object())
        without_connections = replace(
            app.marketplace.APPS[ASSISTANT_ID],
            assistant=replace(self.contract, connections={}),
        )

        with _patched(
            _assistant_connections=self.store,
            _assistant_connection_challenges=challenge_store,
        ):
            app._retain_admitted_assistant_connections(TEAM_ID, ASSISTANT_ID, without_connections)

        self.assertIsNone(challenge_store.current(TEAM_ID))
        self.assertEqual(self.store.metadata(TEAM_ID, ASSISTANT_ID, {}), ())
        self.assertNotIn(ACCESS_TOKEN, self.store.state_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
