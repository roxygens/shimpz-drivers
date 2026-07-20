from __future__ import annotations

import base64
import hashlib
import unittest
from unittest import mock

import oauth_pkce_challenges

SESSION_ONE = "session-binding-one-123456789"
SESSION_TWO = "session-binding-two-123456789"
SCOPES = ("offline.access", "tweet.read", "tweet.write", "users.read")


def create(
    store: oauth_pkce_challenges.OAuthPKCEChallengeStore,
    *,
    session: str = SESSION_ONE,
    team: str = "team_1",
    assistant: str = "shimpz-assistant",
    connection: str = "x",
):
    return store.create(
        session_binding=session,
        team_id=team,
        assistant_id=assistant,
        connection_id=connection,
        provider_id="x",
        scopes=SCOPES,
    )


class OAuthPKCEChallengeTests(unittest.TestCase):
    def test_s256_verifier_is_private_bound_and_single_use(self) -> None:
        store = oauth_pkce_challenges.OAuthPKCEChallengeStore()
        challenge = create(store)

        self.assertEqual(challenge.code_challenge_method, "S256")
        self.assertNotIn("verifier", repr(challenge).lower())
        self.assertNotIn(SESSION_ONE, repr(store._pending))
        for mismatched in (
            {"session_binding": SESSION_TWO},
            {"team_id": "team_2"},
            {"assistant_id": "other-assistant"},
            {"connection_id": "other"},
        ):
            binding = {
                "session_binding": SESSION_ONE,
                "team_id": "team_1",
                "assistant_id": "shimpz-assistant",
                "connection_id": "x",
            }
            binding.update(mismatched)
            with self.subTest(mismatched=mismatched), self.assertRaises(
                oauth_pkce_challenges.OAuthChallengeNotFoundError
            ):
                store.claim(state=challenge.state, **binding)

        exchange = store.claim(
            state=challenge.state,
            session_binding=SESSION_ONE,
            team_id="team_1",
            assistant_id="shimpz-assistant",
            connection_id="x",
        )
        expected = base64.urlsafe_b64encode(
            hashlib.sha256(exchange.code_verifier.encode("ascii")).digest()
        ).rstrip(b"=").decode("ascii")
        self.assertEqual(expected, challenge.code_challenge)
        self.assertEqual(exchange.provider_id, "x")
        self.assertEqual(exchange.scopes, tuple(sorted(SCOPES)))
        self.assertEqual(
            (exchange.team_id, exchange.assistant_id, exchange.connection_id),
            ("team_1", "shimpz-assistant", "x"),
        )
        with self.assertRaises(oauth_pkce_challenges.OAuthChallengeNotFoundError):
            store.claim(
                state=challenge.state,
                session_binding=SESSION_ONE,
                team_id="team_1",
                assistant_id="shimpz-assistant",
                connection_id="x",
            )

    def test_callback_recovers_private_binding_only_for_the_starting_browser(self) -> None:
        store = oauth_pkce_challenges.OAuthPKCEChallengeStore()
        challenge = create(store)

        with self.assertRaises(oauth_pkce_challenges.OAuthChallengeNotFoundError):
            store.claim_callback(state=challenge.state, session_binding=SESSION_TWO)
        exchange = store.claim_callback(state=challenge.state, session_binding=SESSION_ONE)
        self.assertEqual(exchange.provider_id, "x")
        self.assertEqual(exchange.team_id, "team_1")
        self.assertEqual(exchange.assistant_id, "shimpz-assistant")
        self.assertEqual(exchange.connection_id, "x")
        self.assertNotIn(SESSION_ONE, repr(exchange))
        with self.assertRaises(oauth_pkce_challenges.OAuthChallengeNotFoundError):
            store.claim_callback(state=challenge.state, session_binding=SESSION_ONE)

    def test_expiry_and_binding_collision_fail_closed(self) -> None:
        store = oauth_pkce_challenges.OAuthPKCEChallengeStore(ttl_seconds=30)
        with mock.patch.object(oauth_pkce_challenges.time, "monotonic", return_value=100.0):
            challenge = create(store)
            with self.assertRaisesRegex(oauth_pkce_challenges.OAuthChallengeError, "pending"):
                create(store)
        with (
            mock.patch.object(oauth_pkce_challenges.time, "monotonic", return_value=130.0),
            self.assertRaises(oauth_pkce_challenges.OAuthChallengeNotFoundError),
        ):
            store.claim(
                state=challenge.state,
                session_binding=SESSION_ONE,
                team_id="team_1",
                assistant_id="shimpz-assistant",
                connection_id="x",
            )

    def test_global_session_and_team_caps_are_independent(self) -> None:
        global_store = oauth_pkce_challenges.OAuthPKCEChallengeStore(
            capacity=1,
            per_session=1,
            per_team=1,
        )
        create(global_store)
        with self.assertRaisesRegex(oauth_pkce_challenges.OAuthChallengeError, "capacity"):
            create(global_store, session=SESSION_TWO, team="team_2")

        session_store = oauth_pkce_challenges.OAuthPKCEChallengeStore(
            capacity=3,
            per_session=1,
            per_team=3,
        )
        create(session_store)
        with self.assertRaisesRegex(oauth_pkce_challenges.OAuthChallengeError, "session"):
            create(session_store, team="team_2")

        team_store = oauth_pkce_challenges.OAuthPKCEChallengeStore(
            capacity=3,
            per_session=3,
            per_team=1,
        )
        create(team_store)
        with self.assertRaisesRegex(oauth_pkce_challenges.OAuthChallengeError, "Team"):
            create(team_store, session=SESSION_TWO, assistant="other-assistant")

    def test_cancel_is_scoped_and_invalid_inputs_never_create_state(self) -> None:
        store = oauth_pkce_challenges.OAuthPKCEChallengeStore(capacity=4, per_session=4, per_team=4)
        first = create(store)
        second = create(store, session=SESSION_TWO, team="team_2")
        self.assertEqual(store.cancel_session(SESSION_ONE), 1)
        with self.assertRaises(oauth_pkce_challenges.OAuthChallengeNotFoundError):
            store.claim(
                state=first.state,
                session_binding=SESSION_ONE,
                team_id="team_1",
                assistant_id="shimpz-assistant",
                connection_id="x",
            )
        self.assertEqual(store.cancel_team("team_2"), 1)
        self.assertEqual(store.cancel_all(), 0)
        with self.assertRaises(oauth_pkce_challenges.OAuthChallengeNotFoundError):
            store.claim(
                state=second.state,
                session_binding=SESSION_TWO,
                team_id="team_2",
                assistant_id="shimpz-assistant",
                connection_id="x",
            )

        for invalid in (
            {"session_binding": "short"},
            {"team_id": "../team"},
            {"assistant_id": "Assistant"},
            {"connection_id": "x/evil"},
            {"provider_id": "evil"},
            {"scopes": ("dm.read",)},
        ):
            arguments = {
                "session_binding": SESSION_ONE,
                "team_id": "team_1",
                "assistant_id": "shimpz-assistant",
                "connection_id": "x",
                "provider_id": "x",
                "scopes": SCOPES,
            }
            arguments.update(invalid)
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                store.create(**arguments)
        self.assertEqual(store.cancel_all(), 0)


if __name__ == "__main__":
    unittest.main()
