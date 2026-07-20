from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM))

import assistant_secret_store


class AssistantSecretStoreTests(unittest.TestCase):
    def _store(self, root: Path) -> assistant_secret_store.AssistantSecretStore:
        return assistant_secret_store.AssistantSecretStore(
            root / "state" / "secrets.json",
            root / "key" / "aes256.key",
        )

    def test_values_are_encrypted_and_only_masks_are_listed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            raw = "consumer-key-material-123456789"

            stored = store.put_many("team_1", "x-assistant", {"x-api-key": raw})

            self.assertEqual(stored[0].mask, "co…89")
            self.assertEqual(stored[0].generation, 1)
            self.assertEqual(store.resolve_many("team_1", "x-assistant", ["x-api-key"]), {"x-api-key": raw})
            metadata = store.metadata("team_1", "x-assistant", ["x-api-key", "x-token"])
            self.assertEqual(
                [(item.id, item.configured, item.mask, item.generation) for item in metadata],
                [
                    ("x-api-key", True, "co…89", 1),
                    ("x-token", False, None, None),
                ],
            )
            state = (root / "state" / "secrets.json").read_text(encoding="utf-8")
            key = (root / "key" / "aes256.key").read_bytes()
            self.assertNotIn(raw, state)
            self.assertNotIn(raw.encode(), key)
            self.assertEqual(stat.S_IMODE((root / "state" / "secrets.json").stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE((root / "key" / "aes256.key").stat().st_mode), 0o600)

    def test_mask_policy_discloses_less_for_shorter_values(self) -> None:
        self.assertEqual(assistant_secret_store.mask_secret("1234567"), "••••")
        self.assertEqual(assistant_secret_store.mask_secret("12345678"), "1…8")
        self.assertEqual(assistant_secret_store.mask_secret("1" * 16), "11…11")
        self.assertEqual(assistant_secret_store.mask_secret("1" * 32), "111…111")
        self.assertEqual(assistant_secret_store.mask_secret("1" * 64), "1111…1111")
        for length in (8, 16, 32, 64, 1_024):
            with self.subTest(length=length):
                mask = assistant_secret_store.mask_secret("x" * length)
                self.assertLessEqual(len(mask.replace("…", "")), 8)

    def test_multi_secret_update_is_atomic_and_rotates_generations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            store.put_many("team_1", "x-assistant", {"first-secret": "abcdefgh", "second-secret": "ijklmnop"})
            store.put_many("team_1", "x-assistant", {"first-secret": "qrstuvwx"})
            state = json.loads((root / "state" / "secrets.json").read_text(encoding="utf-8"))
            records = state["teams"]["team_1"]["x-assistant"]
            self.assertEqual(records["first-secret"]["generation"], 2)
            self.assertEqual(records["second-secret"]["generation"], 1)
            self.assertEqual(
                store.resolve_many("team_1", "x-assistant", ["first-secret", "second-secret"]),
                {"first-secret": "qrstuvwx", "second-secret": "ijklmnop"},
            )

    def test_release_pruning_removes_obsolete_records_without_touching_declared_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(Path(directory))
            store.put_many(
                "team_1",
                "x-assistant",
                {"kept-secret": "abcdefgh", "removed-secret": "ijklmnop"},
            )

            self.assertTrue(store.retain_declared("team_1", "x-assistant", ["kept-secret"]))
            self.assertEqual(
                store.resolve_many("team_1", "x-assistant", ["kept-secret"]),
                {"kept-secret": "abcdefgh"},
            )
            self.assertFalse(store.metadata("team_1", "x-assistant", ["removed-secret"])[0].configured)
            self.assertFalse(store.retain_declared("team_1", "x-assistant", ["kept-secret"]))

    def test_invalid_batch_does_not_change_existing_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            store.put_many("team_1", "x-assistant", {"first-secret": "abcdefgh"})
            state_path = root / "state" / "secrets.json"
            original_state = state_path.read_bytes()

            with self.assertRaises(assistant_secret_store.AssistantSecretValidationError):
                store.put_many(
                    "team_1",
                    "x-assistant",
                    {"first-secret": "qrstuvwx", "second-secret": "line\nbreak"},
                )

            self.assertEqual(state_path.read_bytes(), original_state)
            self.assertEqual(
                store.resolve_many("team_1", "x-assistant", ["first-secret"]),
                {"first-secret": "abcdefgh"},
            )

    def test_missing_key_is_never_replaced_while_envelopes_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            store.put_many("team_1", "x-assistant", {"x-token": "abcdefgh"})
            state_path = root / "state" / "secrets.json"
            key_path = root / "key" / "aes256.key"
            original_state = state_path.read_bytes()
            key_path.unlink()

            with self.assertRaises(assistant_secret_store.AssistantSecretError):
                store.put_many("team_1", "x-assistant", {"x-token": "ijklmnop"})

            self.assertFalse(key_path.exists())
            self.assertEqual(state_path.read_bytes(), original_state)

    def test_missing_secret_fails_without_echoing_identifiers_or_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(Path(directory))
            with self.assertRaises(assistant_secret_store.AssistantSecretMissingError) as caught:
                store.resolve_many("team_1", "x-assistant", ["private-token"])
            self.assertEqual(caught.exception.missing, ("private-token",))
            self.assertNotIn("private-token", str(caught.exception))

    def test_tampering_and_key_substitution_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            store.put_many("team_1", "x-assistant", {"x-token": "abcdefghijk"})
            state_path = root / "state" / "secrets.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            envelope = state["teams"]["team_1"]["x-assistant"]["x-token"]["envelope"]
            envelope["ciphertext"] = envelope["ciphertext"][:-4] + "AAAA"
            state_path.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
            state_path.chmod(0o600)
            with self.assertRaises(assistant_secret_store.AssistantSecretError):
                store.resolve_many("team_1", "x-assistant", ["x-token"])

            store.put_many("team_1", "x-assistant", {"x-token": "abcdefghijk"})
            (root / "key" / "aes256.key").write_bytes(os.urandom(32))
            (root / "key" / "aes256.key").chmod(0o600)
            with self.assertRaises(assistant_secret_store.AssistantSecretError):
                store.resolve_many("team_1", "x-assistant", ["x-token"])

    def test_envelopes_are_cryptographically_bound_to_their_team(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            store.put_many("team_1", "x-assistant", {"x-token": "team-one-secret"})
            store.put_many("team_2", "x-assistant", {"x-token": "team-two-secret"})
            self.assertEqual(
                store.resolve_many("team_1", "x-assistant", ["x-token"]),
                {"x-token": "team-one-secret"},
            )
            self.assertEqual(
                store.resolve_many("team_2", "x-assistant", ["x-token"]),
                {"x-token": "team-two-secret"},
            )

            state_path = root / "state" / "secrets.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["teams"]["team_2"]["x-assistant"]["x-token"] = state["teams"]["team_1"]["x-assistant"]["x-token"]
            state_path.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
            state_path.chmod(0o600)

            with self.assertRaises(assistant_secret_store.AssistantSecretError):
                store.resolve_many("team_2", "x-assistant", ["x-token"])

    def test_permissions_symlinks_and_invalid_values_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            store.put_many("team_1", "x-assistant", {"x-token": "abcdefghijk"})
            (root / "state" / "secrets.json").chmod(0o644)
            with self.assertRaises(assistant_secret_store.AssistantSecretError):
                store.metadata("team_1", "x-assistant", ["x-token"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "state").mkdir(mode=0o700)
            target = root / "target.json"
            target.write_text('{"schema":1,"teams":{}}', encoding="utf-8")
            target.chmod(0o600)
            (root / "state" / "secrets.json").symlink_to(target)
            with self.assertRaises(assistant_secret_store.AssistantSecretError):
                self._store(root).metadata("team_1", "x-assistant", ["x-token"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            invalid_values = (
                " padded",
                "padded ",
                "line\nbreak",
                "bidirectional\u202eoverride",
                "",
                "x" * (assistant_secret_store.MAX_SECRET_BYTES + 1),
            )
            for invalid in invalid_values:
                with (
                    self.subTest(invalid=invalid[:20]),
                    self.assertRaises(assistant_secret_store.AssistantSecretValidationError),
                ):
                    store.put_many("team_1", "x-assistant", {"x-token": invalid})

    def test_secret_id_iterables_are_bounded_while_consumed(self) -> None:
        def unlimited_ids():
            index = 0
            while True:
                yield f"x-{index}"
                index += 1

        with tempfile.TemporaryDirectory() as directory:
            store = self._store(Path(directory))
            with self.assertRaises(assistant_secret_store.AssistantSecretValidationError):
                store.metadata("team_1", "x-assistant", unlimited_ids())

    def test_assistant_and_team_deletion_purge_envelopes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(Path(directory))
            store.put_many("team_1", "first-assistant", {"x-token": "abcdefgh"})
            store.put_many("team_1", "second-assistant", {"x-token": "ijklmnop"})
            self.assertTrue(store.delete_assistant("team_1", "first-assistant"))
            self.assertFalse(store.metadata("team_1", "first-assistant", ["x-token"])[0].configured)
            self.assertTrue(store.delete_team("team_1"))
            self.assertFalse(store.metadata("team_1", "second-assistant", ["x-token"])[0].configured)

    def test_space_reset_purges_every_encrypted_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(Path(directory))
            store.put_many("team_1", "first-assistant", {"api-key": "secret-one-1234"})
            store.put_many("team_2", "second-assistant", {"token": "secret-two-5678"})

            self.assertTrue(store.delete_all())
            self.assertFalse(store.delete_all())
            self.assertFalse(store.metadata("team_1", "first-assistant", ["api-key"])[0].configured)
            self.assertFalse(store.metadata("team_2", "second-assistant", ["token"])[0].configured)


if __name__ == "__main__":
    unittest.main()
