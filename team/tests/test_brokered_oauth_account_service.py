from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import assistant_account_challenges
import oauth_account_service
import oauth_account_store
import oauth_broker_client
import oauth_pkce_challenges

SCOPES = ("dns.read", "offline_access", "zone.read")
SESSION = "browser-session-private-123456789"
CLAIM = "a" * 64
ACCESS = "access-token-private-123456789"
REFRESH = "refresh-token-private-123456789"
LEASE = f"l1.1999999999.{'b' * 43}.{'c' * 43}.{'d' * 43}"
DECLARATION = {"provider": "cloudflare", "scopes": SCOPES}


class Transport:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def request(self, **request) -> oauth_broker_client.BrokerHTTPResponse:
        self.requests.append(request)
        operation = urlsplit(str(request["url"])).path.rsplit("/", 1)[-1]
        payload = (
            {"revoked": True}
            if operation == "revoke"
            else {
                "access_token": ACCESS,
                "refresh_token": REFRESH,
                "expires_in": 3600,
                "scopes": list(SCOPES),
                "broker_lease": LEASE,
            }
        )
        return oauth_broker_client.BrokerHTTPResponse(
            200,
            "application/json",
            json.dumps(payload, separators=(",", ":")).encode(),
        )


def pending() -> assistant_account_challenges.PendingAccountChallenge:
    requirement = assistant_account_challenges.AccountRequirement(
        assistant_id="shimpz-cloudflare",
        assistant_name="Shimpz Cloudflare",
        power_ids=("list-zones",),
        accounts=(("cloudflare", "cloudflare", SCOPES),),
    )
    return assistant_account_challenges.AccountChallengeStore().create(
        "team_1",
        (requirement,),
        {"private": "continuation"},
    )


class BrokeredOAuthAccountServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.store = oauth_account_store.OAuthAccountStore(
            root / "state" / "accounts.json",
            root / "key" / "aes256.key",
            clock=lambda: 1_000_000_000,
        )
        self.transport = Transport()
        self.service = oauth_account_service.BrokeredOAuthAccountService(
            challenge=oauth_pkce_challenges.OAuthPKCEChallengeStore(),
            store=self.store,
            broker=oauth_broker_client.OAuthBrokerClient(self.transport),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_full_flow_uses_broker_and_never_requires_client_credentials(self) -> None:
        url = self.service.authorization_url(pending(), SESSION)
        state = parse_qs(urlsplit(url).query, strict_parsing=True)["state"][0]

        completion = self.service.complete(
            state,
            CLAIM,
            SESSION,
            lambda _team, _assistant, _account: DECLARATION,
        )

        self.assertEqual(
            (completion.team_id, completion.assistant_id, completion.account_id),
            ("team_1", "shimpz-cloudflare", "cloudflare"),
        )
        self.assertEqual(len(self.transport.requests), 1)
        claim = json.loads(self.transport.requests[0]["body"])
        self.assertEqual(set(claim), {"claim", "state", "code_verifier"})
        self.assertNotIn("client", repr(claim))
        self.assertNotIn("secret", repr(claim))
        self.assertEqual(
            self.store.metadata(
                "team_1",
                "shimpz-cloudflare",
                {"cloudflare": DECLARATION},
            )[0].status,
            "connected",
        )

        self.assertTrue(
            self.service.disconnect(
                "team_1",
                "shimpz-cloudflare",
                "cloudflare",
            )
        )
        self.assertEqual(
            [urlsplit(str(item["url"])).path for item in self.transport.requests],
            ["/api/oauth/cloudflare/claim", "/api/oauth/cloudflare/revoke"],
        )
        revoked = json.loads(self.transport.requests[-1]["body"])
        self.assertEqual(revoked, {"token": REFRESH, "broker_lease": LEASE})

    def test_wrong_session_and_declaration_drift_consume_no_broker_claim(self) -> None:
        for declaration in (
            DECLARATION,
            {"provider": "cloudflare", "scopes": ("zone.read",)},
        ):
            start_session = SESSION if declaration is DECLARATION else "second-browser-session-private-123456789"
            url = self.service.authorization_url(pending(), start_session)
            state = parse_qs(urlsplit(url).query, strict_parsing=True)["state"][0]
            session = "other-browser-session-private-123" if declaration is DECLARATION else start_session
            with self.assertRaises(oauth_account_service.OAuthAccountServiceError):
                self.service.complete(
                    state,
                    CLAIM,
                    session,
                    lambda _team, _assistant, _account, value=declaration: value,
                )
        self.assertEqual(self.transport.requests, [])


if __name__ == "__main__":
    unittest.main()
