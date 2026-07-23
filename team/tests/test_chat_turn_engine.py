"""Characterize the shared hosted/local chat-turn decision engine."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import local_app
from test_hosted_app import app


class SharedChatTurnEngineTest(unittest.TestCase):
    def test_hosted_and_local_wrappers_return_the_same_shared_decision(self) -> None:
        context = SimpleNamespace(thread_id="thread-1")
        batch = SimpleNamespace(
            prepare=lambda _requests: None,
            invoke=lambda _request: {},
            delivered=lambda _requests: None,
        )
        outcome = app.chat_orchestrator.ChatOutcome(reply="same decision", powers=())
        controller = object.__new__(local_app.LocalController)
        controller.brain_runtime = object()

        def callback(*_args) -> bool:
            return False

        with (
            mock.patch.object(app.chat_turn_engine, "drive", return_value=outcome) as hosted_shared,
            mock.patch.object(local_app.chat_turn_engine, "drive", return_value=outcome) as local_shared,
        ):
            hosted = app._drive_hosted_chat(
                context,
                "hello",
                [],
                None,
                callback,
                batch,
                callback,
                "token-1",
                lambda: None,
            )
            local = controller._drive_local_chat(
                context,
                "hello",
                [],
                None,
                callback,
                batch,
                callback,
                callback,
                callback,
                lambda: False,
                lambda: None,
            )

        self.assertIs(hosted, outcome)
        self.assertIs(local, outcome)
        calls = hosted_shared.call_args_list + local_shared.call_args_list
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertEqual(call.args[2:5], ("hello", [], None))


if __name__ == "__main__":
    unittest.main()
