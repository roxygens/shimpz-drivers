"""Direct contracts for the hosted Controller authorization chain."""

from __future__ import annotations

import sys
import unittest
from email.message import Message
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hosted_app_fixture import app, hosted_resources

TEAM_ID = "team_1"
CONTAINER_ID = "a" * 64


def _container(*, container_id: str = CONTAINER_ID, owner: str = "account_1") -> SimpleNamespace:
    labels = {"team.id": TEAM_ID, "team.owner": owner}
    return SimpleNamespace(id=container_id, labels=labels, attrs={"Config": {"Labels": labels}})


def _routable_container() -> SimpleNamespace:
    name = app.manifests.team_container_name(TEAM_ID)
    labels = {
        "team.driver": "1",
        "team.id": TEAM_ID,
        "team.owner": "account_1",
    }
    return SimpleNamespace(
        id=CONTAINER_ID,
        name=name,
        status="running",
        labels=labels,
        attrs={"Name": f"/{name}", "Config": {"Labels": labels}},
    )


def _lease(
    *,
    container_id: str = CONTAINER_ID,
    owner: str = "account_1",
    principal: tuple[str, str | None] = ("account", "account_1"),
) -> app._AuthorizationLease:
    return app._AuthorizationLease(TEAM_ID, container_id, owner, principal)


class HostedAuthorizationTests(unittest.TestCase):
    def test_current_lease_rejects_recreated_or_reowned_team(self) -> None:
        cases = (
            _container(container_id="b" * 64),
            _container(owner="account_2"),
        )

        for current in cases:
            with (
                self.subTest(current=current),
                mock.patch.object(
                    hosted_resources,
                    "_get_container",
                    side_effect=lambda _name, current=current: current,
                ),
                mock.patch.object(hosted_resources, "_cleanup_record", return_value=None),
                mock.patch.object(app.network_policy, "brain_identity_valid", return_value=True),
                self.assertRaises(app.ApiError) as caught,
            ):
                app._require_current_authorization(TEAM_ID, _lease(), require_isolation=False)

            self.assertEqual(caught.exception.status, HTTPStatus.NOT_FOUND)

    def test_account_scope_failure_is_indistinguishable_from_missing_team(self) -> None:
        current = _container()
        with (
            mock.patch.object(hosted_resources, "_get_container", return_value=current),
            mock.patch.object(hosted_resources, "_cleanup_record", return_value=None),
            mock.patch.object(app.network_policy, "brain_identity_valid", return_value=True),
            self.assertRaises(app.ApiError) as wrong_owner,
        ):
            app._require_current_authorization(
                TEAM_ID,
                _lease(principal=("account", "account_2")),
                require_isolation=False,
            )

        with (
            mock.patch.object(hosted_resources, "_get_container", return_value=None),
            self.assertRaises(app.ApiError) as missing,
        ):
            app._authorize(TEAM_ID, ("account", "account_2"))

        self.assertEqual(
            (wrong_owner.exception.status, wrong_owner.exception.message),
            (missing.exception.status, missing.exception.message),
        )

    def test_operator_bypasses_ownership_but_not_container_identity(self) -> None:
        current = _container(owner="account_1")
        with mock.patch.object(app.network_policy, "brain_identity_valid", return_value=True):
            lease = app._authorize_container(TEAM_ID, ("operator", None), current)
        self.assertEqual(lease.owner, "account_1")

        with (
            mock.patch.object(hosted_resources, "_get_container", return_value=current),
            mock.patch.object(hosted_resources, "_cleanup_record", return_value=None),
            mock.patch.object(app.network_policy, "brain_identity_valid", return_value=True),
        ):
            self.assertIs(
                app._require_current_authorization(TEAM_ID, lease, require_isolation=False),
                current,
            )

        with (
            mock.patch.object(hosted_resources, "_get_container", return_value=current),
            mock.patch.object(hosted_resources, "_cleanup_record", return_value=None),
            mock.patch.object(app.network_policy, "brain_identity_valid", return_value=False),
            self.assertRaises(app.ApiError) as invalid_identity,
        ):
            app._require_current_authorization(TEAM_ID, lease, require_isolation=False)
        self.assertEqual(invalid_identity.exception.status, HTTPStatus.NOT_FOUND)

    def test_pending_cleanup_blocks_normal_use_but_allows_teardown(self) -> None:
        current = _container()
        cleanup = SimpleNamespace(nonce="cleanup")
        with (
            mock.patch.object(hosted_resources, "_get_container", return_value=current),
            mock.patch.object(hosted_resources, "_cleanup_record", return_value=cleanup),
            mock.patch.object(app.network_policy, "brain_identity_valid", return_value=True),
            self.assertRaises(app.ApiError) as blocked,
        ):
            app._require_current_authorization(TEAM_ID, _lease(), require_isolation=False)
        self.assertEqual(blocked.exception.status, HTTPStatus.CONFLICT)

        with (
            mock.patch.object(hosted_resources, "_get_container", return_value=current),
            mock.patch.object(hosted_resources, "_cleanup_record", return_value=cleanup),
            mock.patch.object(app.network_policy, "brain_identity_valid", return_value=True),
        ):
            self.assertIs(
                app._require_current_authorization(
                    TEAM_ID,
                    _lease(),
                    require_isolation=False,
                    allow_pending_cleanup=True,
                ),
                current,
            )

    @staticmethod
    def _handler(*headers: tuple[str, str]) -> app.Handler:
        handler = object.__new__(app.Handler)
        handler.headers = Message()
        for name, value in headers:
            handler.headers.add_header(name, value)
        return handler

    def test_principal_accepts_only_operator_or_verified_account_credentials(self) -> None:
        operator = self._handler(("Authorization", "Bearer operator-token"))
        account = self._handler(("X-Shimpz-Account", "account-session"))
        wrong = self._handler(("Authorization", "Bearer wrong"))

        self.assertEqual(operator._principal(), ("operator", None))
        with mock.patch.object(app.hosted_http.accounts_client, "verify", return_value="account_1") as verify:
            self.assertEqual(account._principal(), ("account", "account_1"))
        verify.assert_called_once_with("account-session")

        sent: list[tuple[HTTPStatus, dict]] = []
        wrong.client_address = ("198.51.100.10", 1234)
        wrong.path = "/v1/teams"
        wrong._send_json = lambda status, payload: sent.append((status, payload))
        wrong._route = mock.Mock()
        wrong._dispatch("GET")

        self.assertEqual(sent, [(HTTPStatus.FORBIDDEN, {"error": "invalid or missing credentials"})])
        wrong._route.assert_not_called()

    def test_sensitive_route_drives_the_real_current_authorization_chain(self) -> None:
        container = _routable_container()
        handler = self._handler()
        handler.path = f"/v1/teams/{TEAM_ID}/apps"
        sent: list[tuple[HTTPStatus, dict]] = []
        handler._send_json = lambda status, payload, **_kwargs: sent.append((status, payload))

        with (
            mock.patch.object(hosted_resources, "_get_container", return_value=container),
            mock.patch.object(hosted_resources, "_cleanup_record", return_value=None),
            mock.patch.object(
                hosted_resources,
                "_require_current_authorization",
                wraps=hosted_resources._require_current_authorization,
            ) as require_current,
        ):
            handler._route("GET", ("account", "account_1"))

        self.assertEqual(sent, [(HTTPStatus.OK, {"team_id": TEAM_ID, "apps": []})])
        require_current.assert_called_once()
        self.assertEqual(require_current.call_args.args[0], TEAM_ID)
        self.assertEqual(require_current.call_args.args[1].container_id, CONTAINER_ID)
        self.assertEqual(require_current.call_args.kwargs, {"require_isolation": False})


if __name__ == "__main__":
    unittest.main()
