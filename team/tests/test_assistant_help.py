from __future__ import annotations

import sys
import types
import unittest
from http import HTTPStatus
from pathlib import Path
from unittest import mock

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

import test_r2_bridge as harness

app = harness.app
_patched = harness._patched


class _RouteHarness:
    def __init__(self) -> None:
        self.sent: list[tuple[HTTPStatus, dict, bool]] = []

    def _send_json(self, status: HTTPStatus, payload: dict, *, no_store: bool = False) -> None:
        self.sent.append((status, payload, no_store))


class HostedAssistantHelpTests(unittest.TestCase):
    def test_help_uses_only_the_fixed_rpc_and_closed_markdown_contract(self) -> None:
        lease = object()
        contract = types.SimpleNamespace(rpc_command="/usr/local/bin/shimpz-assistant-rpc")
        container = types.SimpleNamespace(id="b" * 64)
        calls: list[tuple[object, ...]] = []

        def rpc(*args, **kwargs):
            calls.append((*args, kwargs))
            return {"markdown": "# Shimpz Assistant\n\nAsk about weather."}

        with _patched(
            _require_current_authorization=lambda team_id, current_lease: calls.append(
                ("authorize", team_id, current_lease)
            ),
            _installed_assistant=lambda _team_id, _assistant_id: ("shimpz-assistant", contract, container),
            _assistant_rpc_exchange=rpc,
        ):
            result = app._assistant_help("team_1", "shimpz-assistant", lease)

        self.assertEqual(
            result,
            {
                "assistant": "shimpz-assistant",
                "markdown": "# Shimpz Assistant\n\nAsk about weather.",
            },
        )
        self.assertEqual(calls[0], ("authorize", "team_1", lease))
        self.assertEqual(
            calls[1],
            (
                "team_1",
                container,
                "/usr/local/bin/shimpz-assistant-rpc",
                "GET",
                "/v1/help",
                {},
                {"token": None, "operation": "Assistant Help"},
            ),
        )

        with (
            _patched(
                _require_current_authorization=lambda *_args: None,
                _installed_assistant=lambda *_args: ("shimpz-assistant", contract, container),
                _assistant_rpc_exchange=lambda *_args, **_kwargs: {"markdown": "x" * (32 * 1024 + 1)},
            ),
            self.assertRaises(app.ApiError) as caught,
        ):
            app._assistant_help("team_1", "shimpz-assistant", lease)
        self.assertEqual(caught.exception.status, HTTPStatus.BAD_GATEWAY)

    def test_help_route_is_exact_and_disables_caching(self) -> None:
        handler = _RouteHarness()
        lease = object()
        with mock.patch.object(
            app,
            "_assistant_help",
            return_value={"assistant": "shimpz-assistant", "markdown": "# Help"},
        ) as assistant_help:
            app.Handler._route_assistants(
                handler,
                "GET",
                ["v1", "teams", "team_1", "assistants", "shimpz-assistant", "help"],
                "team_1",
                lease,
            )

        assistant_help.assert_called_once_with("team_1", "shimpz-assistant", lease)
        self.assertEqual(
            handler.sent,
            [
                (
                    HTTPStatus.OK,
                    {
                        "assistant": "shimpz-assistant",
                        "markdown": "# Help",
                        "trace_id": "trace",
                    },
                    True,
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
