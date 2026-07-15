from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import healthcheck


class HealthcheckTests(unittest.TestCase):
    def test_requires_both_storage_health_and_protected_gate(self) -> None:
        with mock.patch.object(healthcheck, "probe", side_effect=[200, 403]) as probe:
            self.assertEqual(healthcheck.main(), 0)
        self.assertEqual([call.args[0] for call in probe.call_args_list], ["/healthz", "/v1/r2/list"])

    def test_rejects_degraded_storage_or_open_operational_route(self) -> None:
        for statuses in ([503, 403], [200, 200]):
            with self.subTest(statuses=statuses), mock.patch.object(healthcheck, "probe", side_effect=statuses):
                self.assertEqual(healthcheck.main(), 1)

    def test_transport_failure_is_unhealthy(self) -> None:
        with mock.patch.object(healthcheck, "probe", side_effect=OSError):
            self.assertEqual(healthcheck.main(), 1)


if __name__ == "__main__":
    unittest.main()
