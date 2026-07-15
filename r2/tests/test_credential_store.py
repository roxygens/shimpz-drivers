from __future__ import annotations

import hashlib
import json
import stat
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from credential_store import (
    CredentialConflictError,
    CredentialNotFoundError,
    CredentialRevokedError,
    CredentialStore,
    CredentialStoreError,
    CredentialValidationError,
)


def bundle(*, bucket: str = "capsule-bucket", secret: str = "b" * 64) -> dict[str, str]:
    return {
        "account_id": "a" * 32,
        "access_key_id": "ACCESS_KEY_ID_1234567890",
        "secret_access_key": secret,
        "bucket": bucket,
    }


class CredentialStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.state_path = root / "state" / "state.json"
        self.key_path = root / "keyring" / "aes256.key"
        self.store = CredentialStore(self.state_path, self.key_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def create(self, **overrides: object):
        arguments: dict[str, object] = {
            "capsule_id": "capsule_one",
            "credential_id": "primary-r2",
            "profile_id": "s3-access-key",
            "label": "Primary R2",
            "values": bundle(),
            "idempotency_key": "create_primary_r2_000001",
        }
        arguments.update(overrides)
        return self.store.create(**arguments)

    def state(self) -> dict:
        return json.loads(self.state_path.read_text())

    def test_create_is_idempotent_private_and_metadata_only(self) -> None:
        first = self.create()
        second = self.create()

        self.assertEqual(first, second)
        self.assertEqual(first.generation, 1)
        self.assertEqual(self.store.list_metadata("capsule_one"), (first,))
        self.assertEqual(
            set(first.public()),
            {
                "capsule_id",
                "credential_id",
                "profile_id",
                "label",
                "generation",
                "status",
                "created_at",
                "updated_at",
            },
        )
        state_bytes = self.state_path.read_bytes()
        for value in bundle().values():
            self.assertNotIn(value.encode(), state_bytes)
        self.assertNotIn(b"create_primary_r2_000001", state_bytes)
        self.assertEqual(stat.S_IMODE(self.state_path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.key_path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.state_path.parent.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(self.key_path.parent.stat().st_mode), 0o700)
        self.assertNotEqual(self.state_path.parent, self.key_path.parent)
        self.assertNotIn(bundle()["secret_access_key"], repr(self.store.resolve("capsule_one", "primary-r2")))

    def test_create_scopes_hmac_idempotency_and_rejects_changed_payload(self) -> None:
        self.create()
        with self.assertRaises(CredentialConflictError):
            self.create(values=bundle(bucket="different-bucket"))
        second = self.create(
            capsule_id="capsule_two",
            credential_id="secondary-r2",
            idempotency_key="create_primary_r2_000001",
        )
        self.assertEqual(second.capsule_id, "capsule_two")
        state = self.state()["capsules"]
        first_hash = state["capsule_one"]["primary-r2"]["idempotency_hash"]
        second_hash = state["capsule_two"]["secondary-r2"]["idempotency_hash"]
        self.assertNotEqual(first_hash, second_hash)
        self.assertNotEqual(first_hash, hashlib.sha256(b"create_primary_r2_000001").hexdigest())
        with self.assertRaises(CredentialConflictError):
            self.create(credential_id="other-r2")

    def test_bundle_and_identifiers_are_closed(self) -> None:
        with self.assertRaises(CredentialValidationError):
            self.create(values={**bundle(), "unexpected": "value"})
        with self.assertRaises(CredentialValidationError):
            self.create(capsule_id="../capsule")
        with self.assertRaises(CredentialValidationError):
            self.create(label=" padded ")
        with self.assertRaises(CredentialValidationError):
            self.create(values=bundle(bucket="UPPERCASE"))

    def test_rotation_uses_new_nonce_and_compare_and_swap(self) -> None:
        self.create()
        old_nonce = self.state()["capsules"]["capsule_one"]["primary-r2"]["envelope"]["nonce"]
        rotated = self.store.rotate(
            "capsule_one",
            "primary-r2",
            1,
            "s3-access-key",
            "Rotated R2",
            bundle(secret="c" * 64),
        )
        new_nonce = self.state()["capsules"]["capsule_one"]["primary-r2"]["envelope"]["nonce"]

        self.assertEqual(rotated.generation, 2)
        self.assertEqual(rotated.label, "Rotated R2")
        self.assertNotEqual(old_nonce, new_nonce)
        self.assertEqual(self.store.resolve("capsule_one", "primary-r2").value("secret_access_key"), "c" * 64)
        with self.assertRaises(CredentialConflictError):
            self.store.rotate(
                "capsule_one",
                "primary-r2",
                1,
                "s3-access-key",
                "Stale",
                bundle(),
            )
        with self.assertRaises(CredentialValidationError):
            self.store.rotate("capsule_one", "primary-r2", 2, "other", "Invalid", bundle())

    def test_aad_prevents_envelope_swaps_and_cross_capsule_access(self) -> None:
        self.create()
        self.create(
            capsule_id="capsule_two",
            credential_id="secondary-r2",
            label="Secondary R2",
            values=bundle(bucket="second-bucket", secret="d" * 64),
            idempotency_key="create_secondary_r2_0002",
        )
        with self.assertRaises(CredentialNotFoundError):
            self.store.resolve("capsule_two", "primary-r2")

        state = self.state()
        first = state["capsules"]["capsule_one"]["primary-r2"]
        second = state["capsules"]["capsule_two"]["secondary-r2"]
        first["envelope"], second["envelope"] = second["envelope"], first["envelope"]
        self.state_path.write_text(json.dumps(state, sort_keys=True, separators=(",", ":")))
        with self.assertRaises(CredentialStoreError):
            self.store.resolve("capsule_one", "primary-r2")
        with self.assertRaises(CredentialStoreError):
            self.store.resolve("capsule_two", "secondary-r2")

    def test_remove_destroys_envelope_and_is_retry_idempotent(self) -> None:
        self.create()
        removed = self.store.remove("capsule_one", "primary-r2", 1)
        retried = self.store.remove("capsule_one", "primary-r2", 1)

        self.assertEqual(removed, retried)
        self.assertEqual(removed.generation, 2)
        self.assertEqual(removed.status, "revoked")
        self.assertIsNone(self.state()["capsules"]["capsule_one"]["primary-r2"]["envelope"])
        self.assertEqual(self.store.list_metadata("capsule_one"), ())
        self.assertEqual(self.store.capsule_record_count("capsule_one"), 1)
        with self.assertRaises(CredentialRevokedError):
            self.store.resolve("capsule_one", "primary-r2")
        self.assertEqual(self.store.remove("capsule_one", "primary-r2", 2), removed)
        self.store.purge_revoked("capsule_one", "primary-r2", 2)
        with self.assertRaises(CredentialNotFoundError):
            self.store.resolve("capsule_one", "primary-r2")

    def test_concurrent_idempotent_create_commits_once(self) -> None:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: self.create(), range(24)))

        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(len(self.state()["capsules"]["capsule_one"]), 1)

    def test_unsafe_state_mode_and_missing_key_fail_closed(self) -> None:
        self.create()
        self.state_path.chmod(0o644)
        with self.assertRaises(CredentialStoreError):
            self.store.resolve("capsule_one", "primary-r2")
        self.state_path.chmod(0o600)

        self.key_path.unlink()
        with self.assertRaises(CredentialStoreError):
            self.store.resolve("capsule_one", "primary-r2")
        self.assertFalse(self.key_path.exists(), "a missing key must never be silently replaced")

    def test_health_authenticates_state_and_requires_its_key(self) -> None:
        self.store.check_health()
        self.create()
        self.store.check_health()
        self.key_path.write_bytes(b"z" * 32)
        with self.assertRaises(CredentialStoreError):
            self.store.check_health()

        self.key_path.unlink()
        with self.assertRaises(CredentialStoreError):
            self.store.check_health()

        root = Path(self.temporary.name) / "empty"
        empty = CredentialStore(root / "state" / "state.json", root / "key" / "aes256.key")
        metadata = empty.create(
            "capsule_one",
            "temporary-r2",
            "s3-access-key",
            "Temporary",
            bundle(),
            "temporary_create_000001",
        )
        removed = empty.remove("capsule_one", metadata.credential_id, 1)
        empty.purge_revoked("capsule_one", metadata.credential_id, removed.generation)
        empty.key_path.unlink()
        with self.assertRaises(CredentialStoreError):
            empty.check_health()


if __name__ == "__main__":
    unittest.main()
