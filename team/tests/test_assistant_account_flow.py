from __future__ import annotations

import sys
import time
import unittest
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM))

import assistant_account_challenges
import assistant_account_flow
import brain_runtime_client
from local_registry import AccountSpec, AssistantSpec, PowerSpec


@dataclass(frozen=True)
class _Active:
    spec: AssistantSpec


@dataclass(frozen=True)
class _Account:
    id: str
    username: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class _Metadata:
    id: str
    provider: str
    scopes: tuple[str, ...]
    status: str
    account: _Account | None
    expires_at: int | None
    generation: int
    access_token: str = "-".join(("must", "never", "be", "public"))
    refresh_token: str = "-".join(("must", "never", "be", "public"))


class _Store:
    def __init__(
        self,
        rows: dict[tuple[str, str], _Metadata],
        tokens: dict[tuple[str, str], str] | None = None,
    ) -> None:
        self.rows = rows
        self.tokens = tokens or {}
        self.resolved: list[tuple[object, ...]] = []

    def metadata(self, team_id: object, assistant_id: object, declarations: object) -> tuple[_Metadata, ...]:
        assert isinstance(assistant_id, str)
        assert isinstance(declarations, dict)
        return tuple(self.rows[(assistant_id, account_id)] for account_id in declarations)

    def resolve(
        self,
        team_id: object,
        assistant_id: object,
        account_id: object,
        provider: object,
        scopes: object,
        refresh_callback: object,
    ) -> str:
        assert isinstance(assistant_id, str)
        assert isinstance(account_id, str)
        assert callable(refresh_callback)
        self.resolved.append((team_id, assistant_id, account_id, provider, scopes, refresh_callback))
        return self.tokens[(assistant_id, account_id)]


def _spec() -> AssistantSpec:
    read_scopes = ("dns.read", "zone.read")
    write_scopes = ("dns.read", "offline_access", "zone.read")
    return AssistantSpec(
        assistant_id="cloudflare-assistant",
        name="Cloudflare Assistant",
        summary="test",
        image="example.invalid/x@sha256:" + ("a" * 64),
        rpc_command="/app/rpc",
        health_path="/healthz",
        powers={
            "read-profile": PowerSpec(
                "POST",
                "/read-profile",
                "Read one external profile.",
                {},
                {},
                (),
                ("cloudflare-read",),
            ),
            "publish-post": PowerSpec(
                "POST",
                "/publish-post",
                "Publish one approved external update.",
                {},
                {},
                (),
                ("cloudflare-write",),
            ),
        },
        secrets={},
        allowed_hosts=("api.cloudflare.com",),
        accounts={
            "cloudflare-read": AccountSpec("cloudflare", read_scopes),
            "cloudflare-write": AccountSpec("cloudflare", write_scopes),
        },
    )


def _request(power: str, interrupt_id: str) -> brain_runtime_client.PowerRequest:
    return brain_runtime_client.PowerRequest(interrupt_id, "cloudflare-assistant", power, {})


def _cloudflare_spec() -> AssistantSpec:
    return AssistantSpec(
        assistant_id="shimpz-cloudflare",
        name="Shimpz Cloudflare",
        summary="test",
        image="example.invalid/cloudflare@sha256:" + ("b" * 64),
        rpc_command="/app/rpc",
        health_path="/healthz",
        powers={
            "list-zones": PowerSpec(
                "POST",
                "/v1/powers/list-zones",
                "List a bounded page of Cloudflare zones and domains.",
                {},
                {},
                (),
                ("cloudflare",),
            )
        },
        secrets={},
        allowed_hosts=("api.cloudflare.com",),
        accounts={
            "cloudflare": AccountSpec("cloudflare", ("dns.read", "offline_access", "zone.read")),
        },
    )


class AssistantAccountFlowTests(unittest.TestCase):
    def test_batch_collects_every_unusable_account_before_any_power(self) -> None:
        expiry = int(time.time()) + 3600
        spec = _spec()
        store = _Store(
            {
                ("cloudflare-assistant", "cloudflare-read"): _Metadata(
                    "cloudflare-read",
                    "cloudflare",
                    tuple(sorted(spec.accounts["cloudflare-read"].scopes)),
                    "connected",
                    _Account("123", "reader", "Reader"),
                    expiry,
                    1,
                ),
                ("cloudflare-assistant", "cloudflare-write"): _Metadata(
                    "cloudflare-write",
                    "cloudflare",
                    tuple(sorted(spec.accounts["cloudflare-write"].scopes)),
                    "refresh-required",
                    _Account("123", "reader", "Reader"),
                    expiry,
                    2,
                ),
            }
        )

        requirements = assistant_account_flow.requirements_for_batch(
            "team_1",
            {"cloudflare-assistant": _Active(spec)},
            (_request("read-profile", "one"), _request("publish-post", "two")),
            store,
        )

        self.assertEqual(len(requirements), 1)
        self.assertEqual(requirements[0].power_ids, ("publish-post",))
        self.assertEqual(
            requirements[0].accounts,
            (("cloudflare-write", "cloudflare", ("dns.read", "offline_access", "zone.read")),),
        )

    def test_challenge_is_exact_bounded_public_metadata(self) -> None:
        spec = _spec()
        requirement = assistant_account_challenges.AccountRequirement(
            "cloudflare-assistant",
            "Cloudflare Assistant",
            ("publish-post",),
            (("cloudflare-write", "cloudflare", ("dns.read", "offline_access", "zone.read")),),
        )
        challenge = assistant_account_challenges.PendingAccountChallenge(
            "a" * 32,
            "team_1",
            time.monotonic() + 300,
            (requirement,),
            {"input": "must-never-be-public"},
        )

        payload = assistant_account_flow.challenge_payload(
            challenge,
            {"cloudflare-assistant": _Active(spec)},
        )

        self.assertEqual(
            set(payload),
            {"team_id", "status", "turn_id", "challenge_id", "expires_in", "requirements"},
        )
        self.assertEqual(payload["status"], "accounts-required")
        self.assertIn(payload["expires_in"], {299, 300})
        self.assertEqual(
            payload["requirements"],
            [
                {
                    "assistant_id": "cloudflare-assistant",
                    "assistant_name": "Cloudflare Assistant",
                    "account_id": "cloudflare-write",
                    "provider": "cloudflare",
                    "name": "Cloudflare",
                    "summary": (
                        "Connect your Cloudflare account so this Assistant can use only its reviewed read permissions."
                    ),
                    "scopes": ["dns.read", "offline_access", "zone.read"],
                    "powers": [
                        {
                            "id": "publish-post",
                            "name": "Publish Post",
                            "summary": "Publish one approved external update.",
                        }
                    ],
                }
            ],
        )
        self.assertNotIn("must-never-be-public", repr(payload))
        self.assertNotIn("access_token", repr(payload))

    def test_cloudflare_challenge_projects_reviewed_oauth_metadata(self) -> None:
        spec = _cloudflare_spec()
        requirement = assistant_account_challenges.AccountRequirement(
            "shimpz-cloudflare",
            "Shimpz Cloudflare",
            ("list-zones",),
            (("cloudflare", "cloudflare", ("dns.read", "offline_access", "zone.read")),),
        )
        challenge = assistant_account_challenges.PendingAccountChallenge(
            "b" * 32,
            "team_1",
            time.monotonic() + 300,
            (requirement,),
            {"input": "must-never-be-public"},
        )

        payload = assistant_account_flow.challenge_payload(
            challenge,
            {"shimpz-cloudflare": _Active(spec)},
        )

        self.assertEqual(
            payload["requirements"],
            [
                {
                    "assistant_id": "shimpz-cloudflare",
                    "assistant_name": "Shimpz Cloudflare",
                    "account_id": "cloudflare",
                    "provider": "cloudflare",
                    "name": "Cloudflare",
                    "summary": (
                        "Connect your Cloudflare account so this Assistant can use only its reviewed read permissions."
                    ),
                    "scopes": ["dns.read", "offline_access", "zone.read"],
                    "powers": [
                        {
                            "id": "list-zones",
                            "name": "List Zones",
                            "summary": "List a bounded page of Cloudflare zones and domains.",
                        }
                    ],
                }
            ],
        )
        self.assertNotIn("must-never-be-public", repr(payload))

    def test_inventory_flattens_status_without_token_or_generation_fields(self) -> None:
        spec = _spec()
        expiry = 1_800_000_000
        store = _Store(
            {
                ("cloudflare-assistant", "cloudflare-read"): _Metadata(
                    "cloudflare-read",
                    "cloudflare",
                    tuple(sorted(spec.accounts["cloudflare-read"].scopes)),
                    "missing",
                    None,
                    None,
                    0,
                ),
                ("cloudflare-assistant", "cloudflare-write"): _Metadata(
                    "cloudflare-write",
                    "cloudflare",
                    tuple(sorted(spec.accounts["cloudflare-write"].scopes)),
                    "refresh-required",
                    _Account("123", "juliano", "Juliano"),
                    expiry,
                    4,
                ),
            }
        )

        payload = assistant_account_flow.inventory_payload("team_1", [spec], store)

        self.assertEqual(set(payload), {"accounts"})
        self.assertEqual(payload["accounts"][0]["status"], "missing")
        self.assertEqual(payload["accounts"][1]["status"], "expired")
        self.assertEqual(
            payload["accounts"][1]["account"],
            {"id": "123", "name": "Juliano", "username": "juliano"},
        )
        self.assertEqual(
            payload["accounts"][1]["expires_at"],
            datetime.fromtimestamp(expiry, UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        )
        encoded = repr(payload)
        for forbidden in ("access_token", "refresh_token", "must-never-be-public", "generation"):
            self.assertNotIn(forbidden, encoded)

    def test_private_resolution_returns_only_the_selected_power_account(self) -> None:
        spec = _spec()
        token = "-".join(("private", "access", "token", "123456"))
        store = _Store({}, {("cloudflare-assistant", "cloudflare-write"): token})
        refresh_calls: list[tuple[str, tuple[str, ...], str, str | None]] = []

        accounts = assistant_account_flow.resolve_power_accounts(
            "team_1",
            spec,
            "publish-post",
            store,
            lambda provider, scopes, refresh, lease: refresh_calls.append((provider, scopes, refresh, lease)),
        )

        self.assertEqual(
            accounts,
            {"cloudflare-write": {"type": "oauth2-bearer", "access_token": token}},
        )
        self.assertEqual(len(store.resolved), 1)
        callback = store.resolved[0][-1]
        callback("private-refresh-token-123", "private-broker-lease-123")
        self.assertEqual(
            refresh_calls,
            [
                (
                    "cloudflare",
                    ("dns.read", "offline_access", "zone.read"),
                    "private-refresh-token-123",
                    "private-broker-lease-123",
                )
            ],
        )

    def test_flow_fails_closed_on_drift_sensitive_public_fields_and_invalid_tokens(self) -> None:
        spec = _spec()
        drifted = _Store(
            {
                ("cloudflare-assistant", "cloudflare-read"): _Metadata(
                    "cloudflare-read",
                    "cloudflare",
                    ("dns.read",),
                    "connected",
                    None,
                    int(time.time()) + 60,
                    1,
                )
            }
        )
        with self.assertRaises(assistant_account_flow.AccountFlowError):
            assistant_account_flow.requirements_for_batch(
                "team_1",
                {"cloudflare-assistant": _Active(spec)},
                (_request("read-profile", "one"),),
                drifted,
            )
        with self.assertRaises(assistant_account_flow.AccountFlowError):
            assistant_account_flow._assert_public_payload({"access_token": "private"})

        invalid_token_store = _Store({}, {("cloudflare-assistant", "cloudflare-read"): "short"})
        with self.assertRaises(assistant_account_flow.AccountFlowError):
            assistant_account_flow.resolve_power_accounts(
                "team_1",
                spec,
                "read-profile",
                invalid_token_store,
                lambda _provider, _scopes, _refresh: object(),
            )


if __name__ == "__main__":
    unittest.main()
