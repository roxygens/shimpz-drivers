from __future__ import annotations

import unittest

import assistant_contract


class AssistantContractTests(unittest.TestCase):
    def test_reference_identity_egress_and_powers_are_closed(self) -> None:
        self.assertEqual(assistant_contract.ASSISTANT_ID, "shimpz-assistant")
        self.assertEqual(assistant_contract.ASSISTANT_NAME, "Shimpz Assistant")
        self.assertEqual(
            assistant_contract.ASSISTANT_ALLOWED_HOSTS,
            ("api.x.com",),
        )
        powers = assistant_contract.power_contracts()
        self.assertEqual(set(powers), {"public-user-lookup", "identity-me", "create-post", "delete-post"})
        self.assertEqual(set(assistant_contract.secret_contracts()), {
            "x-bearer-token", "x-api-key", "x-api-key-secret", "x-access-token", "x-access-token-secret"
        })
        for power_id, power in powers.items():
            self.assertEqual(power["approval"], "each-run" if power_id in {"create-post", "delete-post"} else "none")
            self.assertFalse(power["input_schema"]["additionalProperties"])
            self.assertFalse(power["output_schema"]["additionalProperties"])
            self.assertTrue(power["secrets"])

    def test_each_power_normalizes_only_its_declared_input(self) -> None:
        self.assertEqual(
            assistant_contract.validate_power_input(
                "shimpz-assistant",
                "public-user-lookup",
                {"username": "XDevelopers"},
            ),
            {"username": "XDevelopers"},
        )
        self.assertEqual(
            assistant_contract.validate_power_input(
                "shimpz-assistant",
                "identity-me",
                {},
            ),
            {},
        )
        self.assertEqual(
            assistant_contract.validate_power_input(
                "shimpz-assistant",
                "create-post",
                {"text": "Hello from Shimpz"},
            ),
            {"text": "Hello from Shimpz"},
        )
        for power, payload in (
            ("public-user-lookup", {"username": "invalid-user"}),
            ("identity-me", {"extra": True}),
            ("create-post", {"text": "x" * 281}),
            ("delete-post", {"id": "not-a-snowflake"}),
        ):
            with self.subTest(power=power), self.assertRaises(ValueError):
                assistant_contract.validate_power_input("shimpz-assistant", power, payload)

    def test_power_outputs_reject_undeclared_or_unsafe_values(self) -> None:
        current = {"id": "2244994945", "name": "X Developers", "username": "XDevelopers"}
        self.assertEqual(
            assistant_contract.validate_power_output("shimpz-assistant", "identity-me", current),
            current,
        )
        for payload in (current | {"extra": True}, current | {"id": "not-numeric"}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                assistant_contract.validate_power_output("shimpz-assistant", "identity-me", payload)
        self.assertEqual(
            assistant_contract.validate_power_output("shimpz-assistant", "delete-post", {"deleted": True}),
            {"deleted": True},
        )
        with self.assertRaises(ValueError):
            assistant_contract.validate_power_output("shimpz-assistant", "delete-post", {"deleted": False})

    def test_help_is_exact_utf8_markdown_bounded_to_32_kib(self) -> None:
        self.assertEqual(
            assistant_contract.validate_help_payload({"markdown": "# Help\n\nOlá!"}),
            {"markdown": "# Help\n\nOlá!"},
        )
        for payload in (
            {"markdown": "x" * (assistant_contract.MAX_HELP_BYTES + 1)},
            {"markdown": "unsafe\x00text"},
            {"markdown": "ok", "html": "<script>"},
            {"assistant": "shimpz-assistant", "markdown": "ok"},
            {},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                assistant_contract.validate_help_payload(payload)

    def test_help_locales_are_a_fixed_exact_allowlist(self) -> None:
        self.assertEqual(
            assistant_contract.HELP_LOCALES,
            {"en", "pt", "es", "zh", "fr", "de", "ja", "ar"},
        )
        for locale in assistant_contract.HELP_LOCALES:
            with self.subTest(locale=locale):
                self.assertEqual(assistant_contract.validate_help_locale(locale), locale)
        for locale in ("EN", "pt-BR", "../en", "en?fallback=pt", "", None):
            with self.subTest(locale=locale), self.assertRaises(ValueError):
                assistant_contract.validate_help_locale(locale)


if __name__ == "__main__":
    unittest.main()
