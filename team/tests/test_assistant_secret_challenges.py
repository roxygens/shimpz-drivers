from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM))

import assistant_secret_challenges


def requirement() -> assistant_secret_challenges.SecretRequirement:
    return assistant_secret_challenges.SecretRequirement(
        assistant_id="x-assistant",
        assistant_name="X Assistant",
        power_ids=("identity-me",),
        secrets=(("x-api-key", "X API Key", "Identifies the X application."),),
    )


class SecretChallengeStoreTests(unittest.TestCase):
    def test_challenge_is_team_bound_and_single_use(self) -> None:
        store = assistant_secret_challenges.SecretChallengeStore()
        challenge = store.create("team_1", (requirement(),), {"continuation": "opaque"})

        self.assertEqual(store.get("team_1", challenge.id), challenge)
        with self.assertRaises(assistant_secret_challenges.SecretChallengeNotFoundError):
            store.get("team_2", challenge.id)
        self.assertEqual(store.claim("team_1", challenge.id), challenge)
        with self.assertRaises(assistant_secret_challenges.SecretChallengeNotFoundError):
            store.claim("team_1", challenge.id)

    def test_one_team_cannot_accumulate_waiting_turns(self) -> None:
        store = assistant_secret_challenges.SecretChallengeStore()
        store.create("team_1", (requirement(),), object())
        with self.assertRaisesRegex(assistant_secret_challenges.SecretChallengeError, "already"):
            store.create("team_1", (requirement(),), object())

    def test_expiry_removes_payload_without_persisting_it(self) -> None:
        store = assistant_secret_challenges.SecretChallengeStore(ttl_seconds=30)
        with patch.object(assistant_secret_challenges.time, "monotonic", return_value=100.0):
            challenge = store.create("team_1", (requirement(),), {"private": "memory-only"})
        with patch.object(assistant_secret_challenges.time, "monotonic", return_value=131.0):
            self.assertIsNone(store.current("team_1"))
            with self.assertRaises(assistant_secret_challenges.SecretChallengeNotFoundError):
                store.get("team_1", challenge.id)

    def test_capacity_and_manual_cancel_fail_closed(self) -> None:
        store = assistant_secret_challenges.SecretChallengeStore(capacity=1)
        store.create("team_1", (requirement(),), object())
        with self.assertRaisesRegex(assistant_secret_challenges.SecretChallengeError, "capacity"):
            store.create("team_2", (requirement(),), object())
        self.assertTrue(store.cancel_team("team_1"))
        self.assertFalse(store.cancel_team("team_1"))
        self.assertIsNotNone(store.create("team_2", (requirement(),), object()))

    def test_space_reset_drops_every_pending_continuation(self) -> None:
        store = assistant_secret_challenges.SecretChallengeStore(capacity=2)
        first = store.create("team_1", (requirement(),), {"private": "one"})
        second = store.create("team_2", (requirement(),), {"private": "two"})

        self.assertEqual(store.cancel_all(), 2)
        self.assertEqual(store.cancel_all(), 0)
        with self.assertRaises(assistant_secret_challenges.SecretChallengeNotFoundError):
            store.get("team_1", first.id)
        with self.assertRaises(assistant_secret_challenges.SecretChallengeNotFoundError):
            store.get("team_2", second.id)


if __name__ == "__main__":
    unittest.main()
