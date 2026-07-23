from __future__ import annotations

import math
import unittest

import brain_runtime_client
import chat_orchestrator
from assistant_human import input_challenges, input_flow

IMAGE = "ghcr.io/theshimpz/assistant@sha256:" + "a" * 64


def interaction(request_type: str, *, options: list[str] | None = None) -> chat_orchestrator.HumanInteraction:
    request = brain_runtime_client.PowerRequest(
        "interrupt-1",
        "assistant",
        "collect",
        {},
    )
    return chat_orchestrator.HumanInteraction(
        request,
        {
            "ordinal": 0,
            "kind": "request",
            "request_type": request_type,
            "title": "Choose",
            "summary": "Provide one value.",
            "docs": None,
            "options": options or [],
        },
    )


class InputFlowTests(unittest.TestCase):
    def challenge(self, request_type: str, *, options: list[str] | None = None):
        requirement = input_flow.requirement(interaction(request_type, options=options), IMAGE, 0)
        return input_challenges.PendingInputChallenge(
            "a" * 32,
            "team_1",
            100.0,
            requirement,
            {"private": "continuation"},
        )

    def test_challenge_is_team_bound_one_use_and_public(self) -> None:
        store = input_challenges.InputChallengeStore(ttl_seconds=30)
        requirement = input_flow.requirement(interaction("choice", options=["one", "two"]), IMAGE, 0)
        challenge = store.create("team_1", requirement, {"private": "continuation"})

        payload = input_flow.challenge_payload(challenge)

        self.assertEqual(payload["status"], "input-required")
        self.assertEqual(payload["request"]["type"], "choice")
        self.assertEqual(payload["request"]["options"], ["one", "two"])
        self.assertNotIn("interrupt", repr(payload))
        self.assertNotIn("continuation", repr(payload))
        with self.assertRaises(input_challenges.InputChallengeNotFoundError):
            store.get("team_2", challenge.id)
        self.assertIs(store.claim("team_1", challenge.id), challenge)
        with self.assertRaises(input_challenges.InputChallengeNotFoundError):
            store.claim("team_1", challenge.id)

    def test_accepts_each_typed_answer(self) -> None:
        cases = (
            ("str", None, "Ada"),
            ("int", None, 3),
            ("float", None, 3.5),
            ("float", None, 3),
            ("bool", None, True),
            ("choice", ["one", "two"], "two"),
            ("choices", ["one", "two"], ["two", "one"]),
        )
        for request_type, options, answer in cases:
            with self.subTest(request_type=request_type, answer=answer):
                challenge = self.challenge(request_type, options=options)
                self.assertEqual(
                    input_flow.submitted_answer(
                        challenge,
                        {"challenge_id": challenge.id, "answer": answer},
                    ),
                    answer,
                )

    def test_rejects_wrong_type_range_unknown_option_and_extra_fields(self) -> None:
        cases = (
            (self.challenge("int"), "3"),
            (self.challenge("int"), True),
            (self.challenge("float"), math.inf),
            (self.challenge("str"), "x" * (input_flow.MAX_ANSWER_CHARS + 1)),
            (self.challenge("choice", options=["one"]), "two"),
            (self.challenge("choices", options=["one"]), ["one", "one"]),
        )
        for challenge, answer in cases:
            with self.subTest(answer=answer), self.assertRaises(input_flow.InputFlowError):
                input_flow.submitted_answer(
                    challenge,
                    {"challenge_id": challenge.id, "answer": answer},
                )
        with self.assertRaises(input_flow.InputFlowError):
            input_flow.submitted_answer(
                self.challenge("bool"),
                {"challenge_id": "a" * 32, "answer": True, "extra": True},
            )

    def test_rejects_malformed_or_out_of_order_sentinel(self) -> None:
        malformed = interaction("choice", options=["one", "one"])
        with self.assertRaises(input_flow.InputFlowError):
            input_flow.requirement(malformed, IMAGE, 0)
        with self.assertRaises(input_flow.InputFlowError):
            input_flow.requirement(interaction("str"), IMAGE, 1)


if __name__ == "__main__":
    unittest.main()
