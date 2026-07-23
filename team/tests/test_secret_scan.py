"""Characterization tests for the Assistant output secret-echo scan."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import local_app
from test_hosted_app import app

SECRET = "private-test-credential-123456789"


class SecretScanTests(unittest.TestCase):
    @staticmethod
    def _decisions(value: object, secrets: dict[str, str]) -> tuple[bool, bool]:
        return (
            app._contains_secret(value, secrets),
            local_app.LocalController._contains_secret(value, secrets),
        )

    def test_literal_secret_in_nested_result_or_key_is_caught(self) -> None:
        for result in (
            {"result": ["prefix", {"value": f"leaked:{SECRET}"}]},
            {SECRET: "value"},
        ):
            with self.subTest(result=result):
                self.assertEqual(self._decisions(result, {"service-token": SECRET}), (True, True))

    def test_benign_results_and_empty_secret_values_pass(self) -> None:
        self.assertEqual(
            self._decisions({"result": ["safe", 42, None]}, {"service-token": SECRET}),
            (False, False),
        )
        self.assertEqual(self._decisions({"result": "safe"}, {"unset": ""}), (False, False))

    def test_excessive_nesting_fails_closed(self) -> None:
        result: object = "safe"
        for _ in range(34):
            result = [result]

        self.assertEqual(self._decisions(result, {"service-token": SECRET}), (True, True))

    def test_transformed_values_are_outside_the_literal_scan_contract(self) -> None:
        reversed_secret = SECRET[::-1]

        self.assertEqual(self._decisions({"result": reversed_secret}, {"service-token": SECRET}), (False, False))


if __name__ == "__main__":
    unittest.main()
