from __future__ import annotations

import unittest
from unittest import mock

import brain_runtime_client
import chat_orchestrator
from assistant_human import approval_challenges as assistant_approval_challenges
from assistant_human import approval_flow as assistant_approval_flow


def requirement(interrupt_id: str = "interrupt-1") -> assistant_approval_challenges.ApprovalRequirement:
    return assistant_approval_challenges.ApprovalRequirement(
        interrupt_id=interrupt_id,
        assistant_id="shimpz-cloudflare",
        assistant_name="Shimpz Cloudflare",
        power_id="protected-action",
        assistant_image="ghcr.io/theshimpz/shimpz-cloudflare@sha256:" + "a" * 64,
        ordinal=0,
        title="Publish Post",
        summary="Publish one approved Post.",
        docs="https://docs.shimpz.com/",
        runs="always",
    )


class ApprovalChallengeTests(unittest.TestCase):
    def test_sdk_sentinel_is_bound_to_the_exact_replay_ordinal(self) -> None:
        request = brain_runtime_client.PowerRequest(
            interrupt_id="interrupt-1",
            assistant_id="shimpz-cloudflare",
            power="protected-action",
            input={},
        )
        interaction = chat_orchestrator.HumanInteraction(
            request,
            {
                "ordinal": 1,
                "kind": "approval",
                "request_type": "bool",
                "title": "Publish Post",
                "summary": "Publish one approved Post.",
                "docs": None,
                "options": [],
                "runs": "once",
            },
        )

        bound = assistant_approval_flow.requirement(
            interaction,
            "Shimpz Cloudflare",
            "ghcr.io/theshimpz/shimpz-cloudflare@sha256:" + "a" * 64,
            1,
        )

        self.assertEqual((bound.ordinal, bound.runs), (1, "once"))
        for answer_count, update in (
            (0, {}),
            (1, {"runs": "sometimes"}),
            (1, {"options": ["yes"]}),
            (1, {"unexpected": True}),
        ):
            with self.subTest(answer_count=answer_count, update=update):
                changed = dict(interaction.payload)
                changed.update(update)
                invalid = chat_orchestrator.HumanInteraction(request, changed)
                with self.assertRaises(assistant_approval_flow.ApprovalFlowError):
                    assistant_approval_flow.requirement(
                        invalid,
                        "Shimpz Cloudflare",
                        "ghcr.io/theshimpz/shimpz-cloudflare@sha256:" + "a" * 64,
                        answer_count,
                    )

    def test_challenge_is_team_bound_one_use_and_hides_runtime_identifiers(self) -> None:
        store = assistant_approval_challenges.ApprovalChallengeStore(ttl_seconds=30)
        challenge = store.create("marketing", (requirement(),), payload={"private": "continuation"})

        payload = assistant_approval_flow.challenge_payload(challenge)
        self.assertEqual(payload["status"], "approval-required")
        self.assertEqual(payload["requirements"][0]["title"], "Publish Post")
        self.assertEqual(payload["requirements"][0]["summary"], "Publish one approved Post.")
        self.assertEqual(payload["requirements"][0]["docs"], "https://docs.shimpz.com/")
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
                assistant_approval_flow.submitted_answer(challenge, body)

    def test_affirmative_submission_binds_exactly_one_call_site(self) -> None:
        challenge = assistant_approval_challenges.PendingApprovalChallenge(
            id="a" * 32,
            team_id="marketing",
            expires_at=100.0,
            requirements=(requirement("one"),),
            payload=None,
        )
        self.assertEqual(
            assistant_approval_flow.submitted_answer(
                challenge,
                {"challenge_id": challenge.id, "approved": True},
            ),
            True,
        )
        malformed = assistant_approval_challenges.PendingApprovalChallenge(
            id="b" * 32,
            team_id="marketing",
            expires_at=100.0,
            requirements=(requirement("one"), requirement("two")),
            payload=None,
        )
        with self.assertRaises(assistant_approval_flow.ApprovalFlowError):
            assistant_approval_flow.submitted_answer(
                malformed,
                {"challenge_id": malformed.id, "approved": True},
            )


if __name__ == "__main__":
    unittest.main()
