from __future__ import annotations

import json
import secrets
import unittest
from unittest import mock

import brain_credentials_client
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


def _delivery(account_id: str, provider: str, recipient: str, secret: str) -> dict[str, object]:
    recipient_bytes = brain_credentials_client._b64decode(recipient)
    sender = x25519.X25519PrivateKey.generate()
    sender_public = sender.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    salt = secrets.token_bytes(brain_credentials_client.DELIVERY_SALT_BYTES)
    nonce = secrets.token_bytes(brain_credentials_client.DELIVERY_NONCE_BYTES)
    aad = brain_credentials_client._delivery_aad(
        account_id,
        provider,
        "api_key",
        recipient_bytes,
        sender_public,
    )
    shared_key = sender.exchange(x25519.X25519PublicKey.from_public_bytes(recipient_bytes))
    key = HKDF(
        algorithm=hashes.SHA256(),
        length=brain_credentials_client.DELIVERY_KEY_BYTES,
        salt=salt,
        info=aad,
    ).derive(shared_key)
    ciphertext = AESGCM(key).encrypt(nonce, secret.encode(), aad)
    return {
        "v": brain_credentials_client.DELIVERY_VERSION,
        "alg": brain_credentials_client.DELIVERY_ALGORITHM,
        "sender_public_key": brain_credentials_client._b64encode(sender_public),
        "salt": brain_credentials_client._b64encode(salt),
        "nonce": brain_credentials_client._b64encode(nonce),
        "ciphertext": brain_credentials_client._b64encode(ciphertext),
    }


class BrainCredentialsClientTests(unittest.TestCase):
    def test_resolve_delivers_only_an_api_key_in_memory(self):
        account_id = "account-1"
        provider = "openai"
        secret = secrets.token_urlsafe(32)
        requests: list[tuple[str, dict]] = []

        def post(_base_url, path, payload, _token_file):
            requests.append((path, payload))
            if path == "/v1/internal/brains/resolve":
                return 200, {
                    "auth_type": "api_key",
                    "secret_ref": {"opaque": "envelope"},
                    "generation": 4,
                }
            self.assertEqual(path, "/v1/deliver")
            return 200, {
                "delivery": _delivery(
                    account_id,
                    provider,
                    payload["recipient_public_key"],
                    secret,
                )
            }

        with mock.patch.object(brain_credentials_client, "_post", side_effect=post):
            credential = brain_credentials_client.resolve(account_id, provider)

        self.assertEqual(credential, ("api_key", secret, 4))
        self.assertEqual(
            [path for path, _payload in requests],
            [
                "/v1/internal/brains/resolve",
                "/v1/deliver",
            ],
        )
        self.assertNotIn(secret, json.dumps(requests))

    def test_legacy_providers_and_oauth_metadata_fail_closed(self):
        for provider in ("claude-code", "codex"):
            with self.subTest(provider=provider), mock.patch.object(brain_credentials_client, "_post") as post:
                with self.assertRaises(brain_credentials_client.BrainCredentialError):
                    brain_credentials_client.resolve("account-1", provider)
                post.assert_not_called()

        metadata = {
            "auth_type": "oauth",
            "secret_ref": {"opaque": "legacy-envelope"},
            "generation": 1,
        }
        with mock.patch.object(brain_credentials_client, "_post", return_value=(200, metadata)) as post:
            with self.assertRaises(brain_credentials_client.BrainCredentialError):
                brain_credentials_client.resolve("account-1", "anthropic")
            post.assert_called_once()

    def test_generation_check_keeps_revocation_authority_in_accounts(self):
        with mock.patch.object(
            brain_credentials_client,
            "_post",
            side_effect=((200, {"valid": True}), (409, {"valid": False})),
        ):
            self.assertTrue(brain_credentials_client.generation_is_current("account-1", "openai", 3))
            self.assertFalse(brain_credentials_client.generation_is_current("account-1", "openai", 3))

        with mock.patch.object(brain_credentials_client, "_post") as post:
            with self.assertRaises(brain_credentials_client.BrainCredentialError):
                brain_credentials_client.generation_is_current("account-1", "codex", 3)
            post.assert_not_called()

    def test_volume_archive_helpers_are_not_part_of_the_runtime_contract(self):
        for legacy_name in ("credential_file", "credential_archive", "resolve_archive"):
            with self.subTest(name=legacy_name):
                self.assertFalse(hasattr(brain_credentials_client, legacy_name))


if __name__ == "__main__":
    unittest.main()
