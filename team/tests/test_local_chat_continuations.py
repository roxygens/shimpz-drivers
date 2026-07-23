from __future__ import annotations

import sys
import unittest
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM))

import assistant_account_challenges
import assistant_secret_challenges
import brain_runtime_client
import chat_orchestrator
import inference_config
import local_chat_continuation_store
import local_chat_continuations
from assistant_human import approval_challenges, input_challenges

IMAGE = "registry.example/assistant@sha256:" + "b" * 64
TURN = brain_runtime_client.RuntimeTurn(
    status="power-required",
    reply="",
    powers=(
        brain_runtime_client.PowerRequest(
            interrupt_id="power-1",
            assistant_id="demo-assistant",
            power="publish",
            input={"message": "private Power input"},
        ),
    ),
)


def pending() -> local_chat_continuations.PendingLocalChat:
    return local_chat_continuations.PendingLocalChat(
        continuation=chat_orchestrator.ChatContinuation(
            turn=TURN,
            seen_interrupts=("older-power",),
            invoked=(chat_orchestrator.InvokedPower("demo-assistant", "lookup"),),
            round_index=1,
        ),
        assistant_ids=("demo-assistant",),
        file_ids=("a" * 32,),
        provider="openai",
        identity=(
            "Demo Team",
            "network-id",
            (("demo-assistant", IMAGE, "container-id"),),
            [
                {
                    "id": "a" * 32,
                    "name": "brief.txt",
                    "media_type": "text/plain",
                    "size": 42,
                }
            ],
            inference_config.normalize("openai", "gpt-5.5"),
        ),
        answer_logs=(("power-1", ("private human answer", True)),),
    )


class LocalChatContinuationCodecTests(unittest.TestCase):
    def _round_trip(self, kind: str, requirements: tuple[object, ...]) -> None:
        bindings, payload = local_chat_continuations.encode(kind, requirements, pending())
        stored = local_chat_continuation_store.StoredContinuation(
            "team_1",
            kind,
            "c" * 32,
            1_300,
            1,
            bindings,
            payload,
        )
        decoded = local_chat_continuations.decode(stored)
        self.assertEqual(decoded.kind, kind)
        self.assertEqual(decoded.requirements, requirements)
        self.assertEqual(decoded.pending, pending())

    def test_round_trips_every_local_suspension_kind(self) -> None:
        cases = {
            "accounts": (
                assistant_account_challenges.AccountRequirement(
                    "demo-assistant",
                    "Demo Assistant",
                    ("publish",),
                    (("cloudflare", "cloudflare", ("dns.read", "zone.read")),),
                ),
            ),
            "secrets": (
                assistant_secret_challenges.SecretRequirement(
                    "demo-assistant",
                    "Demo Assistant",
                    ("publish",),
                    (("api-key", "API key", "Credential used for publishing."),),
                ),
            ),
            "input": (
                input_challenges.InputRequirement(
                    "power-1",
                    "demo-assistant",
                    "publish",
                    IMAGE,
                    2,
                    "choice",
                    "Audience",
                    "Choose the publication audience.",
                    "https://docs.example/audience",
                    ("private", "public"),
                ),
            ),
            "approval": (
                approval_challenges.ApprovalRequirement(
                    "power-1",
                    "demo-assistant",
                    "Demo Assistant",
                    "publish",
                    IMAGE,
                    2,
                    "Publish update",
                    "Publish this update now?",
                    None,
                    "once",
                ),
            ),
        }
        for kind, requirements in cases.items():
            with self.subTest(kind=kind):
                self._round_trip(kind, requirements)

    def test_rejects_release_binding_and_decrypted_shape_drift(self) -> None:
        requirement = (
            input_challenges.InputRequirement(
                "power-1",
                "demo-assistant",
                "publish",
                IMAGE,
                2,
                "str",
                "Message",
                "Enter the message.",
                None,
                (),
            ),
        )
        bindings, payload = local_chat_continuations.encode("input", requirement, pending())
        drifted = local_chat_continuation_store.StoredContinuation(
            "team_1",
            "input",
            "c" * 32,
            1_300,
            1,
            ("demo-assistant/publish/" + IMAGE + "/3",),
            payload,
        )
        with self.assertRaisesRegex(
            local_chat_continuations.ContinuationCodecError,
            "binding changed",
        ):
            local_chat_continuations.decode(drifted)

        malformed = local_chat_continuation_store.StoredContinuation(
            "team_1",
            "input",
            "c" * 32,
            1_300,
            1,
            bindings,
            b'{"schema":1,"kind":"input","requirements":[],"pending":{}}',
        )
        with self.assertRaises(local_chat_continuations.ContinuationCodecError):
            local_chat_continuations.decode(malformed)


if __name__ == "__main__":
    unittest.main()
