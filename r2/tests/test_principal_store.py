from __future__ import annotations

import hashlib
import json
import stat
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from principal_store import PrincipalError, PrincipalStore


class PrincipalStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.state_path = Path(self.temporary.name) / "principals" / "principals.json"
        self.store = PrincipalStore(self.state_path)
        self.token = "1" * 64

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_principal_is_hashed_scoped_and_fail_closed(self) -> None:
        self.store.provision("capsule_one", self.token)
        self.assertEqual(self.store.resolve(self.token, "capsule_one"), "capsule_one")
        with self.assertRaises(PrincipalError):
            self.store.resolve(self.token, "capsule_two")
        with self.assertRaises(PrincipalError):
            self.store.resolve("2" * 64, "capsule_one")

        serialized = self.state_path.read_text()
        self.assertNotIn(self.token, serialized)
        self.assertIn(hashlib.sha256(self.token.encode()).hexdigest(), serialized)
        self.assertEqual(stat.S_IMODE(self.state_path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.state_path.parent.stat().st_mode), 0o700)

    def test_retire_finalize_history_blocks_replay_and_allows_only_a_new_token(self) -> None:
        replacement = "2" * 64
        self.store.provision("capsule_one", self.token)
        with self.assertRaises(PrincipalError):
            self.store.provision("capsule_one", replacement)
        self.store.retire(self.token, "capsule_one")
        self.store.retire(self.token, "capsule_one")
        with self.assertRaises(PrincipalError):
            self.store.resolve(self.token, "capsule_one")
        self.assertEqual(self.store.resolve(self.token, "capsule_one", allow_retired=True), "capsule_one")
        self.store.finalize("capsule_one")
        self.store.finalize("capsule_one")
        with self.assertRaises(PrincipalError):
            self.store.provision("capsule_one", self.token)
        self.store.provision("capsule_one", replacement)
        self.assertEqual(self.store.resolve(replacement, "capsule_one"), "capsule_one")
        self.store.retire(replacement, "capsule_one")
        self.store.finalize("capsule_one")
        with self.assertRaises(PrincipalError):
            self.store.provision("capsule_one", self.token)
        state = json.loads(self.state_path.read_text())["principals"]
        self.assertEqual(len(state), 2)
        self.assertEqual({record["status"] for record in state.values()}, {"finalized"})

    def test_finalize_refuses_an_active_principal(self) -> None:
        self.store.provision("capsule_one", self.token)
        with self.assertRaises(PrincipalError):
            self.store.finalize("capsule_one")

    def test_retire_waits_for_in_flight_authorization_and_closes_the_gate(self) -> None:
        self.store.provision("capsule_one", self.token)
        entered = threading.Event()
        release = threading.Event()
        retired = threading.Event()

        def use_principal() -> None:
            with self.store.authorized(self.token, "capsule_one"):
                entered.set()
                self.assertTrue(release.wait(1))

        def retire_principal() -> None:
            self.store.retire(self.token, "capsule_one")
            retired.set()

        with ThreadPoolExecutor(max_workers=2) as executor:
            use = executor.submit(use_principal)
            self.assertTrue(entered.wait(1))
            retire = executor.submit(retire_principal)
            self.assertFalse(retired.wait(0.05))
            release.set()
            use.result()
            retire.result()

        with self.assertRaises(PrincipalError), self.store.authorized(self.token, "capsule_one"):
            self.fail("retired bearer unexpectedly passed the lifecycle gate")


if __name__ == "__main__":
    unittest.main()
