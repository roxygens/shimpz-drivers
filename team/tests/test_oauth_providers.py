from __future__ import annotations

import unittest

import oauth_providers


class OAuthProviderTests(unittest.TestCase):
    def test_cloudflare_provider_is_core_owned_and_uses_confidential_pkce(self) -> None:
        provider = oauth_providers.resolve("cloudflare")

        self.assertEqual(provider.authorization_endpoint, "https://dash.cloudflare.com/oauth2/auth")
        self.assertEqual(provider.token_endpoint, "https://dash.cloudflare.com/oauth2/token")
        self.assertEqual(provider.revocation_endpoint, "https://dash.cloudflare.com/oauth2/revoke")
        self.assertEqual(provider.api_hosts, ("api.cloudflare.com",))
        self.assertEqual(provider.pkce_method, "S256")
        self.assertEqual(provider.client_auth_method, "client_secret_basic")
        self.assertEqual(
            provider.allowed_scopes,
            {"dns.read", "offline_access", "zone.read"},
        )
        self.assertEqual(set(oauth_providers.PROVIDERS), {"cloudflare", "x"})
        with self.assertRaises(TypeError):
            oauth_providers.PROVIDERS["evil"] = provider

    def test_connection_scopes_are_canonical_and_limited_to_the_trusted_provider(self) -> None:
        intent = oauth_providers.account_intent(
            "cloudflare",
            ("zone.read", "offline_access", "dns.read"),
        )
        self.assertEqual(intent.provider.id, "cloudflare")
        self.assertEqual(
            intent.scopes,
            ("dns.read", "offline_access", "zone.read"),
        )

        invalid = (
            ("unknown", ("zone.read",)),
            ("Cloudflare", ("zone.read",)),
            ("cloudflare", ()),
            ("cloudflare", "zone.read"),
            ("cloudflare", ("zone.read", "zone.read")),
            ("cloudflare", ("dns.write",)),
            ("cloudflare", ("zone/read",)),
            ("cloudflare", tuple("scope" for _ in range(oauth_providers.MAX_REQUESTED_SCOPES + 1))),
        )
        for provider_id, scopes in invalid:
            with (
                self.subTest(provider=provider_id, scopes=scopes),
                self.assertRaises(oauth_providers.OAuthProviderError),
            ):
                oauth_providers.account_intent(provider_id, scopes)


if __name__ == "__main__":
    unittest.main()
