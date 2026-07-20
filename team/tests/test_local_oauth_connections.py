from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM))

import assistant_connection_challenges
import local_app
import local_registry
import oauth_connection_store


class LocalOAuthConnectionTests(unittest.TestCase):
    @staticmethod
    def _registry() -> dict[str, local_registry.AssistantSpec]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "shimpz_assistant_image": "example.invalid/shimpz@sha256:" + ("a" * 64),
                    }
                ),
                encoding="utf-8",
            )
            return local_registry.load_registry(path)

    def test_controller_accepts_injected_connection_state(self) -> None:
        injected_store = SimpleNamespace()
        injected_challenges = assistant_connection_challenges.ConnectionChallengeStore()
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
            assistant_connections=injected_store,
            connection_challenges=injected_challenges,
            approval_challenges=SimpleNamespace(),
            approval_grants=SimpleNamespace(),
        )

        self.assertIs(controller.assistant_connections, injected_store)
        self.assertIs(controller.connection_challenges, injected_challenges)

    def test_connection_inventory_is_exact_and_never_contains_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = object.__new__(local_app.LocalController)
            controller._locks = tuple(threading.RLock() for _ in range(64))
            controller.registry = self._registry()
            controller.assistant_connections = oauth_connection_store.OAuthConnectionStore(
                Path(directory) / "state" / "connections.json",
                Path(directory) / "key" / "aes256.key",
            )
            controller.list_assistants = lambda _team: {
                "assistants": [{"assistant": "shimpz-assistant", "status": "running"}]
            }

            payload = controller.list_assistant_connections("team_1")

        self.assertEqual(set(payload), {"team_id", "connections"})
        self.assertEqual(payload["team_id"], "team_1")
        self.assertEqual(
            payload["connections"],
            [
                {
                    "assistant_id": "shimpz-assistant",
                    "assistant_name": "Shimpz Assistant",
                    "id": "x",
                    "provider": "x",
                    "name": "X",
                    "summary": "Connect your X account so this Assistant can use only its reviewed X permissions.",
                    "scopes": ["offline.access", "tweet.read", "tweet.write", "users.read"],
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

    def test_connection_inventory_route_has_one_exact_internal_shape(self) -> None:
        expected = {"team_id": "team_1", "connections": []}
        handler = object.__new__(local_app.Handler)
        handler.command = "GET"
        handler.server = SimpleNamespace(
            controller=SimpleNamespace(list_assistant_connections=lambda team_id: expected)
        )

        route = handler._assistant_connection_route(
            ["v1", "teams", "team_1", "assistant-connections"]
        )

        self.assertEqual(
            route,
            (HTTPStatus.OK, expected, "assistant-connection-list", "team_1", None),
        )


if __name__ == "__main__":
    unittest.main()
