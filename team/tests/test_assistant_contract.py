from __future__ import annotations

import unittest

import assistant_contract


class AssistantContractTests(unittest.TestCase):
    def test_reference_identity_egress_and_powers_are_closed(self) -> None:
        self.assertEqual(assistant_contract.ASSISTANT_ID, "shimpz-assistant")
        self.assertEqual(assistant_contract.ASSISTANT_NAME, "Shimpz Assistant")
        self.assertEqual(
            assistant_contract.ASSISTANT_EGRESS,
            ("api.open-meteo.com", "geocoding-api.open-meteo.com"),
        )
        powers = assistant_contract.power_contracts()
        self.assertEqual(set(powers), {"search-location", "current-weather", "daily-forecast"})
        for power in powers.values():
            self.assertEqual(power["approval"], "none")
            self.assertFalse(power["input_schema"]["additionalProperties"])
            self.assertFalse(power["output_schema"]["additionalProperties"])

    def test_each_power_normalizes_only_its_declared_input(self) -> None:
        self.assertEqual(
            assistant_contract.validate_power_input(
                "shimpz-assistant",
                "search-location",
                {"query": " Lisbon "},
            ),
            {"query": "Lisbon", "limit": 5},
        )
        self.assertEqual(
            assistant_contract.validate_power_input(
                "shimpz-assistant",
                "current-weather",
                {"latitude": 38.72, "longitude": -9.14},
            ),
            {"latitude": 38.72, "longitude": -9.14},
        )
        self.assertEqual(
            assistant_contract.validate_power_input(
                "shimpz-assistant",
                "daily-forecast",
                {"latitude": 38.72, "longitude": -9.14},
            ),
            {"latitude": 38.72, "longitude": -9.14, "days": 7},
        )
        for power, payload in (
            ("search-location", {"query": 12}),
            ("current-weather", {"latitude": True, "longitude": 0}),
            ("daily-forecast", {"latitude": 0, "longitude": 0, "days": 17}),
        ):
            with self.subTest(power=power), self.assertRaises(ValueError):
                assistant_contract.validate_power_input("shimpz-assistant", power, payload)

    def test_power_outputs_reject_undeclared_or_unsafe_values(self) -> None:
        current = {
            "observed_at": "2026-07-18T10:00",
            "temperature_c": 22.5,
            "apparent_temperature_c": 22.0,
            "wind_speed_kmh": 11.2,
            "weather_code": 1,
            "timezone": "Europe/Lisbon",
        }
        self.assertEqual(
            assistant_contract.validate_power_output("shimpz-assistant", "current-weather", current),
            current,
        )
        for payload in (current | {"extra": True}, current | {"temperature_c": float("nan")}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                assistant_contract.validate_power_output("shimpz-assistant", "current-weather", payload)

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


if __name__ == "__main__":
    unittest.main()
