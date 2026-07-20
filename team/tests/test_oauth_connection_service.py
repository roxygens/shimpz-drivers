from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import assistant_connection_challenges
import oauth_connection_service
import oauth_connection_store
import oauth_http_client
import oauth_pkce_challenges

CLIENT_ID = "public-client-123456789"
CODE = "authorization-code-private-123456789"
SESSION = "browser-session-private-123456789"
OTHER_SESSION = "other-browser-session-123456789"
SCOPES = ("offline.access", "tweet.read", "tweet.write", "users.read")
DECLARATION = {"provider": "x", "scopes": SCOPES}
ACCESS = "access-token-private-123456789"
REFRESH = "refresh-token-private-987654321"


class SyntheticTransport:
    def __init__(self, response: oauth_http_client.OAuthHTTPResponse | None = None) -> None:
        self.response = response or oauth_http_client.OAuthHTTPResponse(
            200,
            "application/json",
            json.dumps(
                {
                    "access_token": ACCESS,
                    "refresh_token": REFRESH,
                    "token_type": "bearer",
                    "expires_in": 3600,
                    "scope": " ".join(SCOPES),
                }
            ).encode(),
        )
        self.requests: list[dict[str, object]] = []

    def request(self, **request) -> oauth_http_client.OAuthHTTPResponse:
        self.requests.append(request)
        return self.response


def requirement(
    assistant: str = "shimpz-assistant",
    *,
    provider: str = "x",
    scopes: tuple[str, ...] = SCOPES,
) -> assistant_connection_challenges.ConnectionRequirement:
    return assistant_connection_challenges.ConnectionRequirement(
        assistant_id=assistant,
        assistant_name=assistant,
        power_ids=("identity-me",),
        connections=(("x", provider, scopes),),
    )


def pending(
    *requirements: assistant_connection_challenges.ConnectionRequirement,
    team: str = "team_1",
) -> assistant_connection_challenges.PendingConnectionChallenge:
    return assistant_connection_challenges.ConnectionChallengeStore().create(
        team,
        tuple(requirements or (requirement(),)),
        {"private": "paused user input"},
    )


class OAuthConnectionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.store = oauth_connection_store.OAuthConnectionStore(
            root / "state" / "connections.json",
            root / "key" / "aes256.key",
            clock=lambda: 1_000_000_000,
        )
        self.challenges = oauth_pkce_challenges.OAuthPKCEChallengeStore()
        self.transport = SyntheticTransport()
        self.http = oauth_http_client.OAuthHTTPClient(self.transport)
        self.service = oauth_connection_service.OAuthConnectionService(
            client_id=CLIENT_ID,
            redirect_uri=oauth_http_client.LOCAL_REDIRECT_URI,
            challenge=self.challenges,
            store=self.store,
            http=self.http,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _state(url: str) -> str:
        return parse_qs(urlsplit(url).query, strict_parsing=True)["state"][0]

    def _complete(self, state: str, *, session: str = SESSION):
        return self.service.complete(
            state,
            CODE,
            session,
            lambda _team, _assistant, _connection: DECLARATION,
        )

    def test_trusted_url_selects_first_deterministic_unconfigured_requirement(self) -> None:
        self.store.put(
            "team_1",
            "a-assistant",
            "x",
            "x",
            SCOPES,
            oauth_http_client.OAuthTokenSet(ACCESS, REFRESH, SCOPES, 3600),
        )
        flow = pending(requirement("z-assistant"), requirement("a-assistant"))

        url = self.service.authorization_url(flow, SESSION)
        parsed = urlsplit(url)
        query = parse_qs(parsed.query, strict_parsing=True)
        self.assertEqual((parsed.scheme, parsed.netloc, parsed.path), ("https", "x.com", "/i/oauth2/authorize"))
        self.assertEqual(query["redirect_uri"], [oauth_http_client.LOCAL_REDIRECT_URI])
        self.assertEqual(query["client_id"], [CLIENT_ID])
        self.assertEqual(query["code_challenge_method"], ["S256"])
        self.assertEqual(query["scope"], [" ".join(SCOPES)])

        completed = self._complete(query["state"][0])
        self.assertEqual(
            (completed.team_id, completed.assistant_id, completed.connection_id),
            ("team_1", "z-assistant", "x"),
        )
        self.assertEqual(completed.provider, "x")
        self.assertEqual(completed.generation, 1)
        for private in (CODE, ACCESS, REFRESH, CLIENT_ID, query["state"][0], "verifier"):
            self.assertNotIn(private, repr(completed))
        metadata = self.store.metadata("team_1", "z-assistant", {"x": DECLARATION})[0]
        self.assertEqual(metadata.status, "connected")
        self.assertIsNone(metadata.account)

    def test_wrong_session_does_not_consume_but_success_and_replay_are_one_use(self) -> None:
        state = self._state(self.service.authorization_url(pending(requirement()), SESSION))
        with self.assertRaises(oauth_connection_service.OAuthConnectionServiceError):
            self._complete(state, session=OTHER_SESSION)
        self.assertEqual(self.transport.requests, [])

        completed = self._complete(state)
        self.assertEqual(completed.connection_id, "x")
        self.assertEqual(len(self.transport.requests), 1)
        with self.assertRaises(oauth_connection_service.OAuthConnectionServiceError):
            self._complete(state)
        self.assertEqual(len(self.transport.requests), 1)

    def test_install_or_scope_drift_consumes_state_before_any_exchange(self) -> None:
        drifted = (
            None,
            {"provider": "x", "scopes": ("tweet.read", "users.read")},
        )
        for current in drifted:
            with self.subTest(current=current):
                state = self._state(self.service.authorization_url(pending(requirement()), SESSION))
                with self.assertRaises(oauth_connection_service.OAuthConnectionServiceError):
                    self.service.complete(
                        state,
                        CODE,
                        SESSION,
                        lambda _team, _assistant, _connection, value=current: value,
                    )
                self.assertEqual(self.transport.requests, [])
                with self.assertRaises(oauth_connection_service.OAuthConnectionServiceError):
                    self._complete(state)

    def test_provider_scope_and_configuration_injection_fail_closed(self) -> None:
        for malicious in (
            requirement(provider="https://evil.example/token"),
            requirement(scopes=("tweet.read", "https://evil.example")),
        ):
            with (
                self.subTest(malicious=malicious),
                self.assertRaises(oauth_connection_service.OAuthConnectionServiceError),
            ):
                self.service.authorization_url(pending(malicious), SESSION)
        self.assertEqual(self.challenges.cancel_all(), 0)
        self.assertEqual(self.transport.requests, [])

        malformed = assistant_connection_challenges.PendingConnectionChallenge(
            id="0" * 32,
            team_id="team_1",
            expires_at=0,
            requirements=(requirement(),),
            payload=None,
        )
        with self.assertRaises(oauth_connection_service.OAuthConnectionServiceError):
            self.service.authorization_url(malformed, SESSION)

        lazy = oauth_connection_service.OAuthConnectionService(
            client_id=None,
            redirect_uri=oauth_http_client.LOCAL_REDIRECT_URI,
            challenge=self.challenges,
            store=self.store,
            http=self.http,
        )
        self.assertNotIn(CLIENT_ID, repr(self.service))
        with self.assertRaisesRegex(
            oauth_connection_service.OAuthConnectionServiceError,
            "not configured",
        ):
            lazy.authorization_url(pending(requirement()), SESSION)
        with self.assertRaises(oauth_connection_service.OAuthConnectionServiceError):
            oauth_connection_service.OAuthConnectionService(
                client_id=CLIENT_ID,
                redirect_uri="https://evil.example/callback",
                challenge=self.challenges,
                store=self.store,
                http=self.http,
            )

    def test_expired_stored_connection_can_start_fresh_authorization(self) -> None:
        root = Path(self.temporary.name)
        now = [1_000]
        store = oauth_connection_store.OAuthConnectionStore(
            root / "expired-state" / "connections.json",
            root / "expired-key" / "aes256.key",
            clock=lambda: now[0],
        )
        store.put(
            "team_1",
            "shimpz-assistant",
            "x",
            "x",
            SCOPES,
            oauth_http_client.OAuthTokenSet(ACCESS, REFRESH, SCOPES, 30),
        )
        now[0] = 1_031
        service = oauth_connection_service.OAuthConnectionService(
            client_id=CLIENT_ID,
            redirect_uri=oauth_http_client.LOCAL_REDIRECT_URI,
            challenge=oauth_pkce_challenges.OAuthPKCEChallengeStore(),
            store=store,
            http=self.http,
        )
        url = service.authorization_url(pending(requirement()), SESSION)
        self.assertEqual(urlsplit(url).hostname, "x.com")

    def test_provider_response_and_callback_errors_never_reflect_private_values(self) -> None:
        leaked = "provider-private-response-123456789"
        transport = SyntheticTransport(
            oauth_http_client.OAuthHTTPResponse(
                200,
                "application/json",
                json.dumps(
                    {
                        "access_token": leaked,
                        "token_type": "bearer",
                        "expires_in": 3600,
                        "scope": " ".join(SCOPES),
                        "unexpected": leaked,
                    }
                ).encode(),
            )
        )
        service = oauth_connection_service.OAuthConnectionService(
            client_id=CLIENT_ID,
            redirect_uri=oauth_http_client.LOCAL_REDIRECT_URI,
            challenge=self.challenges,
            store=self.store,
            http=oauth_http_client.OAuthHTTPClient(transport),
        )
        state = self._state(service.authorization_url(pending(requirement()), SESSION))
        with self.assertRaises(oauth_connection_service.OAuthConnectionServiceError) as captured:
            service.complete(
                state,
                CODE,
                SESSION,
                lambda _team, _assistant, _connection: DECLARATION,
            )
        rendered = f"{captured.exception!r} {captured.exception}"
        for private in (leaked, ACCESS, REFRESH, CODE, CLIENT_ID, state, "verifier"):
            self.assertNotIn(private, rendered)

        next_state = self._state(service.authorization_url(pending(requirement()), SESSION))
        callback_secret = "manifest-parser-private-value-123456789"  # noqa: S105 -- synthetic redaction canary
        with self.assertRaises(oauth_connection_service.OAuthConnectionServiceError) as callback:
            service.complete(
                next_state,
                CODE,
                SESSION,
                lambda _team, _assistant, _connection: (_ for _ in ()).throw(RuntimeError(callback_secret)),
            )
        self.assertNotIn(callback_secret, f"{callback.exception!r} {callback.exception}")

    def test_all_configured_and_disconnect_are_local_only(self) -> None:
        state = self._state(self.service.authorization_url(pending(requirement()), SESSION))
        self._complete(state)
        requests = len(self.transport.requests)
        with self.assertRaises(oauth_connection_service.OAuthConnectionUnavailableError):
            self.service.authorization_url(pending(requirement()), SESSION)
        self.assertTrue(self.service.disconnect("team_1", "shimpz-assistant", "x"))
        self.assertFalse(self.service.disconnect("team_1", "shimpz-assistant", "x"))
        self.assertEqual(len(self.transport.requests), requests)
        self.assertEqual(
            self.store.metadata("team_1", "shimpz-assistant", {"x": DECLARATION})[0].status,
            "missing",
        )


if __name__ == "__main__":
    unittest.main()
