"""Focused contract tests for the provider-isolated Codex device-login bridge."""

from __future__ import annotations

import threading
import unittest
from http import HTTPStatus
from unittest import mock

import test_r2_bridge as fixture


app = fixture.app
_patched = fixture._patched


class _CodexContainer:
    id = "codex-capsule-container"
    status = "running"
    labels = {"capsule.brain": "codex"}

    def reload(self) -> None:
        return None


class CodexLoginBridgeTests(unittest.TestCase):
    def test_adapter_exposes_only_fixed_device_auth_operations(self) -> None:
        login = app._BRAIN_ADAPTERS["codex"].interactive_login
        self.assertIsNotNone(login)
        self.assertEqual(login.mode, "device_code")
        self.assertEqual(login.start_command, ("shimpz-codex-auth", "device-login"))
        self.assertEqual(login.cancel_command, ("shimpz-codex-auth", "device-cancel"))
        self.assertIsNone(login.submit_command)
        self.assertNotIn("auth.json", " ".join((*login.start_command, *login.cancel_command)))

    def test_device_info_accepts_only_the_closed_official_shape(self) -> None:
        valid = '{"pending":false,"url":"https://auth.openai.com/codex/device","user_code":"AB12-CDE34"}'
        with _patched(_brain_exec=lambda *_args: (0, valid)):
            self.assertEqual(
                app._codex_device_info(object()),
                {
                    "pending": False,
                    "url": "https://auth.openai.com/codex/device",
                    "user_code": "AB12-CDE34",
                },
            )
        malformed = (
            '{"pending":false,"url":"https://evil.example/codex/device","user_code":"AB12-CDE34"}',
            '{"pending":false,"url":"https://auth.openai.com/codex/device","user_code":"bad code"}',
            '{"pending":false,"url":"https://auth.openai.com/codex/device","user_code":"AB12-CDE34","extra":1}',
            "[]",
        )
        for payload in malformed:
            with self.subTest(payload=payload), _patched(_brain_exec=lambda *_args, value=payload: (0, value)):
                with self.assertRaises(app.ApiError) as caught:
                    app._codex_device_info(object())
                self.assertEqual(caught.exception.status, HTTPStatus.BAD_GATEWAY)

    def test_status_requires_both_oauth_and_a_succeeded_device_writer(self) -> None:
        def status_for(auth_type: str | None, state: str):
            configured = auth_type is not None
            responses = iter(
                (
                    (
                        0,
                        '{"provider":"codex","configured":'
                        + str(configured).lower()
                        + ',"auth_type":'
                        + (f'"{auth_type}"' if auth_type else "null")
                        + "}",
                    ),
                    (0, f'{{"state":"{state}"}}'),
                )
            )
            with _patched(_brain_exec=lambda *_args: next(responses)):
                return app._codex_login_status(object())

        waiting, verdict = status_for("oauth", "waiting")
        self.assertEqual(waiting, {"loggedIn": False, "state": "waiting"})
        self.assertEqual(verdict, {"ok": False})
        succeeded, verdict = status_for("oauth", "succeeded")
        self.assertEqual(succeeded, {"loggedIn": True, "state": "succeeded"})
        self.assertEqual(verdict, {"ok": True})
        api_key, verdict = status_for("api_key", "succeeded")
        self.assertEqual(api_key, {"loggedIn": False, "state": "succeeded"})
        self.assertEqual(verdict, {"ok": False})

        invalid = iter(
            (
                (0, '{"provider":"codex","configured":false,"auth_type":"oauth"}'),
                (0, '{"state":"succeeded"}'),
            )
        )
        with _patched(_brain_exec=lambda *_args: next(invalid)):
            with self.assertRaises(app.ApiError) as caught:
                app._codex_login_status(object())
        self.assertEqual(caught.exception.status, HTTPStatus.BAD_GATEWAY)

    def test_device_code_is_never_accepted_by_shimpz(self) -> None:
        container = _CodexContainer()
        lease = object()
        with (
            _patched(
                _lock_for=lambda _cid: threading.Lock(),
                _require_current_authorization=lambda cid, supplied: container,
            ),
            mock.patch.object(app, "_brain_exec_stdin") as secret_transport,
        ):
            with self.assertRaises(app.ApiError) as caught:
                app._capsule_login_code("capsule_1", {"code": "AB12-CDE34"}, lease)
        self.assertEqual(caught.exception.status, HTTPStatus.CONFLICT)
        self.assertIn("provider website", caught.exception.message)
        secret_transport.assert_not_called()

    def test_cancel_is_capsule_authorized_and_uses_only_the_fixed_adapter_command(self) -> None:
        container = _CodexContainer()
        lease = object()
        events: list[object] = []

        def authorize(cid, supplied):
            events.append(("authorize", cid, supplied))
            return container

        def execute(supplied, command):
            self.assertIs(supplied, container)
            events.append(("exec", command))
            return 0, '{"cancelled":true}'

        with _patched(
            _lock_for=lambda _cid: threading.Lock(),
            _require_current_authorization=authorize,
            _chat_lock_for=lambda _cid: threading.Lock(),
            _require_no_durable_chat_turn=lambda supplied: events.append(("durable", supplied.id)),
            _release_finished_credential_mutation=lambda cid, supplied: events.append(("release", cid, supplied.id))
            or True,
            _brain_exec=execute,
        ):
            result = app._capsule_login_cancel("capsule_1", lease)
        self.assertEqual(
            result,
            {
                "capsule": "capsule_1",
                "provider": "codex",
                "mode": "device_code",
                "cancelled": True,
            },
        )
        self.assertIn(("authorize", "capsule_1", lease), events)
        self.assertEqual(events.count(("release", "capsule_1", container.id)), 2)
        self.assertIn(("exec", ["shimpz-codex-auth", "device-cancel"]), events)


if __name__ == "__main__":
    unittest.main()
