from __future__ import annotations

import sys
import types
import unittest
from email.message import Message
from http import HTTPStatus
from pathlib import Path
from unittest import mock

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

import assistant_help
import hosted_app_fixture as harness

app = harness.app
_patched = harness._patched


class AssistantHelpContractTests(unittest.TestCase):
    def test_help_payload_and_locales_are_closed_and_bounded(self) -> None:
        self.assertEqual(
            assistant_help.validate_payload({"markdown": "# Help\n\nOlá!"}),
            {"markdown": "# Help\n\nOlá!"},
        )
        for payload in (
            {"markdown": ""},
            {"markdown": "x" * (assistant_help.MAX_HELP_BYTES + 1)},
            {"assistant": "shimpz-cloudflare", "markdown": "ok"},
            [],
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                assistant_help.validate_payload(payload)
        self.assertEqual(
            assistant_help.HELP_LOCALES,
            frozenset({"en", "pt", "es", "zh", "fr", "de", "ja", "ar"}),
        )
        for locale in assistant_help.HELP_LOCALES:
            self.assertEqual(assistant_help.validate_locale(locale), locale)
        for locale in ("pt-BR", "EN", "", None):
            with self.subTest(locale=locale), self.assertRaises(ValueError):
                assistant_help.validate_locale(locale)


class _RouteHarness:
    def __init__(self) -> None:
        self.sent: list[tuple[HTTPStatus, dict, bool]] = []
        self.headers = Message()

    def _send_json(self, status: HTTPStatus, payload: dict, *, no_store: bool = False) -> None:
        self.sent.append((status, payload, no_store))


class HostedAssistantHelpTests(unittest.TestCase):
    def test_help_uses_only_the_fixed_rpc_and_closed_markdown_contract(self) -> None:
        lease = object()
        contract = types.SimpleNamespace(rpc_command="/usr/local/bin/shimpz-cloudflare-rpc")
        container = types.SimpleNamespace(id="b" * 64)
        calls: list[object] = []

        def rpc(request):
            calls.append(request)
            return {"markdown": "# Shimpz Cloudflare\n\nAsk about weather."}

        with _patched(
            _require_current_authorization=lambda team_id, current_lease: calls.append(
                ("authorize", team_id, current_lease)
            ),
            _installed_assistant=lambda _team_id, _assistant_id: ("shimpz-cloudflare", contract, container),
            _assistant_rpc_exchange=rpc,
        ):
            result = app._assistant_help("team_1", "shimpz-cloudflare", lease, "pt")

        self.assertEqual(
            result,
            {
                "assistant": "shimpz-cloudflare",
                "markdown": "# Shimpz Cloudflare\n\nAsk about weather.",
            },
        )
        self.assertEqual(calls[0], ("authorize", "team_1", lease))
        self.assertEqual(
            calls[1],
            app.AssistantRpcRequest(
                team_id="team_1",
                container=container,
                command="/usr/local/bin/shimpz-cloudflare-rpc",
                method="GET",
                path="/v1/help/pt",
                payload={},
                token=None,
                operation="Assistant Help",
                detect_unsupported_path=True,
            ),
        )

        with (
            _patched(
                _require_current_authorization=lambda *_args: None,
                _installed_assistant=lambda *_args: ("shimpz-cloudflare", contract, container),
                _assistant_rpc_exchange=lambda *_args, **_kwargs: {"markdown": "x" * (32 * 1024 + 1)},
            ),
            self.assertRaises(app.ApiError) as caught,
        ):
            app._assistant_help("team_1", "shimpz-cloudflare", lease, "pt")
        self.assertEqual(caught.exception.status, HTTPStatus.BAD_GATEWAY)

        calls.clear()
        with self.assertRaises(app.ApiError) as caught:
            app._assistant_help("team_1", "shimpz-cloudflare", lease, "pt-BR")
        self.assertEqual(caught.exception.status, HTTPStatus.BAD_REQUEST)
        self.assertEqual(calls, [])

    def test_help_falls_back_only_when_the_localized_rpc_path_is_unsupported(self) -> None:
        lease = object()
        contract = types.SimpleNamespace(rpc_command="/usr/local/bin/shimpz-cloudflare-rpc")
        container = types.SimpleNamespace(id="b" * 64)
        paths: list[str] = []

        def rpc(request):
            paths.append(request.path)
            if request.path == "/v1/help/pt":
                self.assertTrue(request.detect_unsupported_path)
                raise app._UnsupportedAssistantRpcPathError(request.path)
            self.assertFalse(request.detect_unsupported_path)
            return {"markdown": "# English fallback"}

        with _patched(
            _require_current_authorization=lambda *_args: None,
            _installed_assistant=lambda *_args: ("shimpz-cloudflare", contract, container),
            _assistant_rpc_exchange=rpc,
        ):
            result = app._assistant_help("team_1", "shimpz-cloudflare", lease, "pt")

        self.assertEqual(result["markdown"], "# English fallback")
        self.assertEqual(paths, ["/v1/help/pt", "/v1/help"])

        paths.clear()

        def fail_rpc(request):
            paths.append(request.path)
            raise app.ApiError(HTTPStatus.BAD_GATEWAY, "Assistant Help failed")

        with (
            _patched(
                _require_current_authorization=lambda *_args: None,
                _installed_assistant=lambda *_args: ("shimpz-cloudflare", contract, container),
                _assistant_rpc_exchange=fail_rpc,
            ),
            self.assertRaises(app.ApiError),
        ):
            app._assistant_help("team_1", "shimpz-cloudflare", lease, "pt")
        self.assertEqual(paths, ["/v1/help/pt"])

    def test_help_route_is_exact_and_disables_caching(self) -> None:
        handler = _RouteHarness()
        lease = object()
        with mock.patch.object(
            app,
            "_assistant_help",
            return_value={"assistant": "shimpz-cloudflare", "markdown": "# Help"},
        ) as assistant_help:
            app.Handler._route_assistants(
                handler,
                "GET",
                ["v1", "teams", "team_1", "assistants", "shimpz-cloudflare", "help", "ja"],
                "team_1",
                lease,
            )

        assistant_help.assert_called_once_with("team_1", "shimpz-cloudflare", lease, "ja")
        self.assertEqual(
            handler.sent,
            [
                (
                    HTTPStatus.OK,
                    {
                        "assistant": "shimpz-cloudflare",
                        "markdown": "# Help",
                        "trace_id": "trace",
                    },
                    True,
                )
            ],
        )

    def test_legacy_help_route_maps_only_to_english(self) -> None:
        handler = _RouteHarness()
        lease = object()
        with mock.patch.object(
            app,
            "_assistant_help",
            return_value={"assistant": "shimpz-cloudflare", "markdown": "# Help"},
        ) as assistant_help:
            app.Handler._route_assistants(
                handler,
                "GET",
                ["v1", "teams", "team_1", "assistants", "shimpz-cloudflare", "help"],
                "team_1",
                lease,
            )
        assistant_help.assert_called_once_with("team_1", "shimpz-cloudflare", lease, "en")

    def test_help_route_rejects_query_before_rpc(self) -> None:
        handler = _RouteHarness()
        handler.path = "/v1/teams/team_1/assistants/shimpz-cloudflare/help/en?fallback=pt"
        with (
            _patched(_authorize=lambda *_args: object()),
            mock.patch.object(app, "_assistant_help") as assistant_help,
            self.assertRaises(app.ApiError) as caught,
        ):
            app.Handler._route(handler, "GET", ("operator", None))
        self.assertEqual(caught.exception.status, HTTPStatus.BAD_REQUEST)
        assistant_help.assert_not_called()


if __name__ == "__main__":
    unittest.main()
