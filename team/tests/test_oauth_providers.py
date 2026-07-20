from __future__ import annotations

import unittest

import oauth_providers


class OAuthProviderTests(unittest.TestCase):
    def test_x_provider_is_controller_owned_and_uses_only_pkce_s256(self) -> None:
        provider = oauth_providers.resolve("x")

        self.assertEqual(provider.authorization_endpoint, "https://x.com/i/oauth2/authorize")
        self.assertEqual(provider.token_endpoint, "https://api.x.com/2/oauth2/token")
        self.assertEqual(provider.revocation_endpoint, "https://api.x.com/2/oauth2/revoke")
        self.assertEqual(provider.api_hosts, ("api.x.com",))
        self.assertEqual(provider.pkce_method, "S256")
        self.assertEqual(provider.client_auth_method, "client_secret_basic")
        self.assertEqual(
            provider.allowed_scopes,
            {"offline.access", "tweet.read", "tweet.write", "users.read"},
        )
        with self.assertRaises(TypeError):
            oauth_providers.PROVIDERS["evil"] = provider

    def test_connection_scopes_are_canonical_and_limited_to_the_trusted_provider(self) -> None:
        intent = oauth_providers.connection_intent(
            "x",
            ("users.read", "tweet.write", "offline.access", "tweet.read"),
        )
        self.assertEqual(intent.provider.id, "x")
        self.assertEqual(
            intent.scopes,
            ("offline.access", "tweet.read", "tweet.write", "users.read"),
        )

        invalid = (
            ("unknown", ("tweet.read",)),
            ("X", ("tweet.read",)),
            ("x", ()),
            ("x", "tweet.read"),
            ("x", ("tweet.read", "tweet.read")),
            ("x", ("dm.read",)),
            ("x", ("tweet/read",)),
            ("x", tuple("scope" for _ in range(oauth_providers.MAX_REQUESTED_SCOPES + 1))),
        )
        for provider_id, scopes in invalid:
            with self.subTest(provider=provider_id, scopes=scopes), self.assertRaises(
                oauth_providers.OAuthProviderError
            ):
                oauth_providers.connection_intent(provider_id, scopes)


if __name__ == "__main__":
    unittest.main()
