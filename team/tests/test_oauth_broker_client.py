from __future__ import annotations

import json
import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

import oauth_broker_client

SCOPES = ("dns.read", "offline_access", "zone.read")
STATE = "s" * 43
CHALLENGE = "c" * 43
CLAIM = "a" * 64
ACCESS = "access-token-private-123456789"
REFRESH = "refresh-token-private-123456789"
LEASE = f"l1.1999999999.{'b' * 43}.{'c' * 43}.{'d' * 43}"


class Transport:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def request(self, **request) -> oauth_broker_client.BrokerHTTPResponse:
        self.requests.append(request)
        operation = urlsplit(str(request["url"])).path.rsplit("/", 1)[-1]
        if operation == "revoke":
            payload = {"revoked": True}
        else:
            payload = {
                "access_token": ACCESS,
                "refresh_token": REFRESH,
                "expires_in": 3600,
                "scopes": list(SCOPES),
                "broker_lease": LEASE,
            }
        return oauth_broker_client.BrokerHTTPResponse(
            200,
            "application/json",
            json.dumps(payload, separators=(",", ":")).encode(),
        )


class OAuthBrokerClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transport = Transport()
        self.client = oauth_broker_client.OAuthBrokerClient(self.transport)

    def test_start_url_is_fixed_and_contains_only_public_pkce_fields(self) -> None:
        url = self.client.authorization_url(
            provider_id="cloudflare",
            state=STATE,
            code_challenge=CHALLENGE,
            scopes=SCOPES,
        )
        parsed = urlsplit(url)
        self.assertEqual(
            (parsed.scheme, parsed.netloc, parsed.path),
            ("https", "shimpz.com", "/api/oauth/cloudflare/start"),
        )
        self.assertEqual(
            parse_qs(parsed.query, strict_parsing=True),
            {
                "state": [STATE],
                "code_challenge": [CHALLENGE],
                "scope": [" ".join(SCOPES)],
                "callback": ["loopback"],
            },
        )
        self.assertNotIn("client", url)
        self.assertEqual(self.transport.requests, [])

    def test_hosted_callback_mode_is_named_and_closed(self) -> None:
        client = oauth_broker_client.OAuthBrokerClient(self.transport, callback_mode="hosted")
        url = client.authorization_url(
            provider_id="cloudflare",
            state=STATE,
            code_challenge=CHALLENGE,
            scopes=SCOPES,
        )
        self.assertEqual(parse_qs(urlsplit(url).query)["callback"], ["hosted"])
        with self.assertRaises(oauth_broker_client.OAuthBrokerClientError):
            oauth_broker_client.OAuthBrokerClient(self.transport, callback_mode="https://evil.example")

    def test_fixed_transport_uses_only_the_authenticated_broker_proxy(self) -> None:
        response = Mock(
            status=200,
            read=Mock(return_value=b"{}"),
            getheader=Mock(
                side_effect=lambda name, default=None: "application/json" if name == "Content-Type" else default
            ),
        )
        connection = Mock(getresponse=Mock(return_value=response))
        token = "a" * 64
        with patch.object(oauth_broker_client.http.client, "HTTPSConnection", return_value=connection) as connect:
            transport = oauth_broker_client.FixedBrokerTransport(
                proxy_host="oauth-broker-proxy",
                proxy_token=token,
            )
            result = transport.request(
                url="https://shimpz.com/api/oauth/cloudflare/claim",
                headers={"Content-Type": "application/json"},
                body=b"{}",
            )

        self.assertEqual(result.status, 200)
        connect.assert_called_once_with("oauth-broker-proxy", 8889, timeout=10)
        tunnel = connection.set_tunnel.call_args
        self.assertEqual(tunnel.args, ("shimpz.com", 443))
        self.assertRegex(tunnel.kwargs["headers"]["Proxy-Authorization"], r"^Basic [A-Za-z0-9+/]+=*$")
        connection.request.assert_called_once_with(
            "POST",
            "/api/oauth/cloudflare/claim",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )

    def test_fixed_transport_rejects_partial_or_foreign_proxy_configuration(self) -> None:
        invalid = (
            {"proxy_host": "oauth-broker-proxy"},
            {"proxy_token": "a" * 64},
            {"proxy_host": "evil.example", "proxy_token": "a" * 64},
            {"proxy_host": "oauth-broker-proxy", "proxy_token": "short"},
        )
        for values in invalid:
            with self.subTest(values=set(values)), self.assertRaises(oauth_broker_client.OAuthBrokerClientError):
                oauth_broker_client.FixedBrokerTransport(**values)

    def test_claim_refresh_and_revoke_use_only_fixed_broker_operations(self) -> None:
        claimed = self.client.claim(
            provider_id="cloudflare",
            claim=CLAIM,
            state=STATE,
            code_verifier="v" * 43,
            scopes=SCOPES,
        )
        refreshed = self.client.refresh(
            provider_id="cloudflare",
            refresh_token=REFRESH,
            broker_lease=LEASE,
            scopes=SCOPES,
        )
        self.client.revoke(
            provider_id="cloudflare",
            token=ACCESS,
            broker_lease=LEASE,
        )

        self.assertEqual(claimed.broker_lease, LEASE)
        self.assertEqual(refreshed.access_token, ACCESS)
        self.assertEqual(
            [urlsplit(str(request["url"])).path for request in self.transport.requests],
            [
                "/api/oauth/cloudflare/claim",
                "/api/oauth/cloudflare/refresh",
                "/api/oauth/cloudflare/revoke",
            ],
        )
        self.assertTrue(
            all(
                request["headers"]
                == {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": "shimpz-local-controller/1",
                }
                for request in self.transport.requests
            )
        )
        self.assertTrue(all(b"client_secret" not in request["body"] for request in self.transport.requests))

    def test_invalid_provider_and_response_shapes_fail_without_reflection(self) -> None:
        with self.assertRaises(oauth_broker_client.OAuthBrokerClientError):
            self.client.authorization_url(
                provider_id="https://evil.example",
                state=STATE,
                code_challenge=CHALLENGE,
                scopes=SCOPES,
            )

        private = "private-broker-response-123456789"

        class InvalidTransport:
            def request(self, **_request) -> oauth_broker_client.BrokerHTTPResponse:
                return oauth_broker_client.BrokerHTTPResponse(
                    200,
                    "application/json",
                    json.dumps({"unexpected": private}).encode(),
                )

        client = oauth_broker_client.OAuthBrokerClient(InvalidTransport())
        with self.assertRaises(oauth_broker_client.OAuthBrokerClientError) as captured:
            client.claim(
                provider_id="cloudflare",
                claim=CLAIM,
                state=STATE,
                code_verifier="v" * 43,
                scopes=SCOPES,
            )
        self.assertNotIn(private, f"{captured.exception!r} {captured.exception}")


if __name__ == "__main__":
    unittest.main()
