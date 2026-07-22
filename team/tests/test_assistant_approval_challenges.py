from __future__ import annotations

import unittest
from unittest import mock

import assistant_approval_challenges
import assistant_approval_flow


def requirement(interrupt_id: str = "interrupt-1") -> assistant_approval_challenges.ApprovalRequirement:
    return assistant_approval_challenges.ApprovalRequirement(
        interrupt_id=interrupt_id,
        assistant_id="shimpz-cloudflare",
        assistant_name="Shimpz Cloudflare",
        power_id="protected-action",
        power_summary="Publish one approved Post.",
        input_json='{"text":"Approved Post"}',
        approval="each-run",
        assistant_image="ghcr.io/theshimpz/shimpz-cloudflare@sha256:" + "a" * 64,
    )


class ApprovalChallengeTests(unittest.TestCase):
    def test_challenge_is_team_bound_one_use_and_hides_runtime_identifiers(self) -> None:
        store = assistant_approval_challenges.ApprovalChallengeStore(ttl_seconds=30)
        challenge = store.create("marketing", (requirement(),), payload={"private": "continuation"})

        payload = assistant_approval_flow.challenge_payload(challenge)
        self.assertEqual(payload["status"], "approval-required")
        self.assertEqual(payload["requirements"][0]["input"], {"text": "Approved Post"})
        self.assertNotIn("interrupt", repr(payload))
        self.assertNotIn("continuation", repr(payload))
        with self.assertRaises(assistant_approval_challenges.ApprovalChallengeNotFoundError):
            store.get("sales", challenge.id)

        claimed = store.claim("marketing", challenge.id)
        self.assertIs(claimed, challenge)
        with self.assertRaises(assistant_approval_challenges.ApprovalChallengeNotFoundError):
            store.claim("marketing", challenge.id)

    def test_expired_or_non_affirmative_challenge_fails_closed(self) -> None:
        store = assistant_approval_challenges.ApprovalChallengeStore(ttl_seconds=30)
        with mock.patch.object(assistant_approval_challenges.time, "monotonic", return_value=10.0):
            challenge = store.create("marketing", (requirement(),), payload=None)
        with mock.patch.object(assistant_approval_challenges.time, "monotonic", return_value=40.0):
            self.assertIsNone(store.current("marketing"))

        for body in (
            {"challenge_id": challenge.id, "approved": False},
            {"challenge_id": challenge.id, "approved": True, "extra": True},
            {"challenge_id": "0" * 32, "approved": True},
        ):
            with self.subTest(body=body), self.assertRaises(assistant_approval_flow.ApprovalFlowError):
                assistant_approval_flow.approved_interrupts(challenge, body)

    def test_affirmative_submission_grants_only_exact_interrupts(self) -> None:
        challenge = assistant_approval_challenges.PendingApprovalChallenge(
            id="a" * 32,
            team_id="marketing",
            expires_at=100.0,
            requirements=(requirement("one"), requirement("two")),
            payload=None,
        )
        self.assertEqual(
            assistant_approval_flow.approved_interrupts(
                challenge,
                {"challenge_id": challenge.id, "approved": True},
            ),
            frozenset({"one", "two"}),
        )


if __name__ == "__main__":
    unittest.main()
