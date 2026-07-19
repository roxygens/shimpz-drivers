from __future__ import annotations

import threading
import unittest
from email.message import Message
from http import HTTPStatus
from types import SimpleNamespace

import local_app


class LocalAssistantHelpTests(unittest.TestCase):
    @staticmethod
    def _controller(markdown: str) -> tuple[local_app.LocalController, list[tuple[str, str, object]]]:
        controller = object.__new__(local_app.LocalController)
        spec = SimpleNamespace(assistant_id="example-assistant")
        controller.registry = {"example-assistant": spec}
        controller._locks = tuple(threading.RLock() for _ in range(64))
        controller._network = lambda _team_id: SimpleNamespace(name="team-network")
        container = SimpleNamespace(status="running", reload=lambda: None)
        controller._assistant_container = lambda _team_id, _assistant_id: container
        controller._validate_container = lambda *_args: None
        calls: list[tuple[str, str, object]] = []
        controller._rpc = lambda _container, _spec, method, path, payload: (
            calls.append((method, path, payload)) or {"markdown": markdown}
        )
        return controller, calls

    def test_help_requires_an_installed_running_assistant_and_fixed_rpc(self) -> None:
        controller, calls = self._controller("# Example\n\nAsk a simple question.")

        result = controller.assistant_help("team_1", "example-assistant")

        self.assertEqual(
            result,
            {
                "assistant": "example-assistant",
                "markdown": "# Example\n\nAsk a simple question.",
            },
        )
        self.assertEqual(calls, [("GET", "/v1/help", {})])
        controller._rpc = lambda *_args: {"markdown": "x" * (32 * 1024 + 1)}
        with self.assertRaises(local_app.ApiProblem) as caught:
            controller.assistant_help("team_1", "example-assistant")
        self.assertEqual(caught.exception.status, HTTPStatus.BAD_GATEWAY)
        self.assertEqual(caught.exception.code, "invalid-assistant-help")

    def test_help_route_is_exact_and_has_no_request_body(self) -> None:
        controller = SimpleNamespace(
            assistant_help=lambda team_id, assistant_id: {
                "assistant": assistant_id,
                "markdown": f"# {team_id}/{assistant_id}",
            }
        )
        handler = object.__new__(local_app.Handler)
        handler.command = "GET"
        handler.path = "/v1/teams/team_1/assistants/example-assistant/help"
        handler.headers = Message()
        handler.server = SimpleNamespace(controller=controller)

        status, payload, operation, team_id, assistant_id = handler._route()

        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(
            payload,
            {
                "assistant": "example-assistant",
                "markdown": "# team_1/example-assistant",
            },
        )
        self.assertEqual((operation, team_id, assistant_id), ("assistant-help", "team_1", "example-assistant"))


if __name__ == "__main__":
    unittest.main()
