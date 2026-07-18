from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

CAPSULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAPSULE))

import assistant_chat


class AssistantChatContractTests(unittest.TestCase):
    def test_prompt_contains_only_file_metadata_and_message(self) -> None:
        prompt = assistant_chat.build_prompt(
            "Say hello to Ada",
            [
                {
                    "id": "a" * 32,
                    "name": "brief.txt",
                    "media_type": "text/plain",
                    "size": 12,
                    "sha256": "must-not-enter-model-context",
                }
            ],
        )
        decoded = json.loads(prompt)
        self.assertEqual(set(decoded), {"files", "message"})
        self.assertEqual(set(decoded["files"][0]), {"id", "name", "media_type", "size"})
        self.assertNotIn("must-not-enter-model-context", prompt)

    def test_exact_direct_message_and_power_decisions_are_accepted(self) -> None:
        direct = '{"kind":"message","message":"Hello","power":"","input":"{}"}'
        self.assertEqual(
            assistant_chat.parse_decision(direct, max_message_chars=100, max_input_bytes=100),
            ("message", "Hello", "", {}),
        )
        power = '{"kind":"power","message":"","power":"hello","input":"{\\"name\\":\\"Ada\\"}"}'
        self.assertEqual(
            assistant_chat.parse_decision(power, max_message_chars=100, max_input_bytes=100),
            ("power", "", "hello", {"name": "Ada"}),
        )

    def test_ambient_or_malformed_authority_fails_closed(self) -> None:
        invalid = (
            '{"kind":"message","message":"Hello","power":"","input":"{}","shell":"id"}',
            '{"kind":"message","message":"","power":"","input":"{}"}',
            '{"kind":"message","message":"Hello","power":"shell","input":"{}"}',
            '{"kind":"power","message":"","power":"../shell","input":"{}"}',
            '{"kind":"power","message":"","power":"hello","input":"[]"}',
            '{"kind":"power","message":"explain","power":"hello","input":"{}"}',
        )
        for decision in invalid:
            with self.subTest(decision=decision), self.assertRaises(assistant_chat.ChatContractError):
                assistant_chat.parse_decision(decision, max_message_chars=100, max_input_bytes=100)

    def test_input_and_message_limits_are_enforced_before_power_validation(self) -> None:
        message = '{"kind":"message","message":"long","power":"","input":"{}"}'
        power = '{"kind":"power","message":"","power":"hello","input":"{\\"value\\":123}"}'
        with self.assertRaises(assistant_chat.ChatContractError):
            assistant_chat.parse_decision(message, max_message_chars=3, max_input_bytes=100)
        with self.assertRaises(assistant_chat.ChatContractError):
            assistant_chat.parse_decision(power, max_message_chars=100, max_input_bytes=4)


if __name__ == "__main__":
    unittest.main()
