from __future__ import annotations

import unittest
from unittest import mock

import assistant_account_challenges


def requirement() -> assistant_account_challenges.AccountRequirement:
    return assistant_account_challenges.AccountRequirement(
        assistant_id="shimpz-cloudflare",
        assistant_name="Shimpz Cloudflare",
        power_ids=("protected-action", "read-profile"),
        accounts=(
            (
                "x",
                "x",
                ("offline.access", "tweet.read", "tweet.write", "users.read"),
            ),
        ),
    )


class AssistantAccountChallengeTests(unittest.TestCase):
    def test_challenge_is_team_bound_single_use_and_keeps_payload_private(self) -> None:
        store = assistant_account_challenges.AccountChallengeStore()
        private = {"continuation": "private user input"}
        challenge = store.create("team_1", (requirement(),), private)

        self.assertNotIn("private user input", repr(store._by_team))
        with self.assertRaises(assistant_account_challenges.AccountChallengeNotFoundError):
            store.get("team_2", challenge.id)
        claimed = store.claim("team_1", challenge.id)
        self.assertIs(claimed.payload, private)
        with self.assertRaises(assistant_account_challenges.AccountChallengeNotFoundError):
            store.claim("team_1", challenge.id)

    def test_one_pending_turn_per_team_and_global_capacity_fail_closed(self) -> None:
        store = assistant_account_challenges.AccountChallengeStore(capacity=2)
        store.create("team_1", (requirement(),), object())
        with self.assertRaisesRegex(
            assistant_account_challenges.AccountChallengeError,
            "already",
        ):
            store.create("team_1", (requirement(),), object())
        store.create("team_2", (requirement(),), object())
        with self.assertRaisesRegex(
            assistant_account_challenges.AccountChallengeError,
            "capacity",
        ):
            store.create("team_3", (requirement(),), object())

    def test_expiry_cancel_and_invalid_identifiers_remove_no_other_team(self) -> None:
        store = assistant_account_challenges.AccountChallengeStore(ttl_seconds=30)
        with mock.patch.object(assistant_account_challenges.time, "monotonic", return_value=1.0):
            expired = store.create("team_1", (requirement(),), object())
        with (
            mock.patch.object(assistant_account_challenges.time, "monotonic", return_value=31.0),
            self.assertRaises(assistant_account_challenges.AccountChallengeNotFoundError),
        ):
            store.get("team_1", expired.id)

        active = store.create("team_2", (requirement(),), object())
        for team, identifier in (("../team", active.id), ("team_2", "not-a-challenge")):
            with self.subTest(team=team, identifier=identifier), self.assertRaises(RuntimeError):
                store.get(team, identifier)
        self.assertTrue(store.cancel_team("team_2"))
        self.assertFalse(store.cancel_team("team_2"))
        self.assertEqual(store.cancel_all(), 0)

    def test_empty_requirements_and_invalid_limits_are_rejected(self) -> None:
        with self.assertRaises(assistant_account_challenges.AccountChallengeError):
            assistant_account_challenges.AccountChallengeStore().create("team_1", (), object())
        for options in (
            {"capacity": 0},
            {"capacity": True},
            {"ttl_seconds": 29},
            {"ttl_seconds": 901},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                assistant_account_challenges.AccountChallengeStore(**options)


if __name__ == "__main__":
    unittest.main()
