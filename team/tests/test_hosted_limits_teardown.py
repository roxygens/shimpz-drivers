"""Hosted rate, capacity, and durable teardown state contracts."""

from __future__ import annotations

import sys
import tempfile
import unittest
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hosted_app_fixture import hosted_lifecycle, hosted_resources, runtime_state

cleanup_state = hosted_lifecycle.cleanup_state
pgdriver_client = hosted_lifecycle.pgdriver_client


class HostedLimitAndTeardownTests(unittest.TestCase):
    def tearDown(self) -> None:
        runtime_state._capacity_reservations.clear()

    def test_fixed_window_rate_limiter_allows_denies_and_resets(self) -> None:
        limiter = runtime_state._FixedWindowRateLimiter(limit=2, window_seconds=10)

        self.assertEqual(limiter.consume("account", now=0), 0)
        self.assertEqual(limiter.consume("account", now=1), 0)
        self.assertEqual(limiter.consume("other", now=2), 0)
        self.assertEqual(limiter.consume("account", now=2), 8)
        self.assertEqual(limiter.consume("account", now=9.9), 1)
        self.assertEqual(limiter.consume("account", now=10), 0)

    def test_capacity_reservations_count_inflight_memory_and_release(self) -> None:
        empty_usage = hosted_resources._MemoryUsage(total=0, by_owner={})
        with (
            mock.patch.object(hosted_resources, "_memory_usage", side_effect=lambda **_kwargs: empty_usage),
            mock.patch.object(runtime_state, "GLOBAL_MEMORY_BUDGET_BYTES", 100),
            mock.patch.object(runtime_state, "OWNER_MEMORY_BUDGET_BYTES", 100),
            hosted_resources._reserve_capacity("team:one", "account_1", 60, team_slot=False),
        ):
            self.assertIn("team:one", runtime_state._capacity_reservations)
            with (
                self.assertRaises(runtime_state.ApiError) as exhausted,
                hosted_resources._reserve_capacity("app:one:extra", "account_1", 41, team_slot=False),
            ):
                self.fail("an over-budget reservation was admitted")
            self.assertEqual(exhausted.exception.status, HTTPStatus.TOO_MANY_REQUESTS)

        self.assertEqual(runtime_state._capacity_reservations, {})

    def test_capacity_rejects_duplicate_inflight_resource(self) -> None:
        empty_usage = hosted_resources._MemoryUsage(total=0, by_owner={})
        with (
            mock.patch.object(hosted_resources, "_memory_usage", side_effect=lambda **_kwargs: empty_usage),
            mock.patch.object(runtime_state, "GLOBAL_MEMORY_BUDGET_BYTES", 100),
            mock.patch.object(runtime_state, "OWNER_MEMORY_BUDGET_BYTES", 100),
            hosted_resources._reserve_capacity("team:one", "account_1", 10, team_slot=False),
            self.assertRaises(runtime_state.ApiError) as duplicate,
            hosted_resources._reserve_capacity("team:one", "account_1", 10, team_slot=False),
        ):
            self.fail("a duplicate reservation was admitted")

        self.assertEqual(duplicate.exception.status, HTTPStatus.CONFLICT)
        self.assertEqual(runtime_state._capacity_reservations, {})

    def test_capacity_inventory_runs_outside_reservation_lock(self) -> None:
        lock_observations: list[tuple[str, bool]] = []

        def physical_teams(**_kwargs):
            lock_observations.append(("teams", runtime_state._capacity_lock.locked()))
            return []

        def memory_usage(**_kwargs):
            lock_observations.append(("memory", runtime_state._capacity_lock.locked()))
            return hosted_resources._MemoryUsage(total=0, by_owner={})

        with (
            mock.patch.object(hosted_resources, "_physical_teams", side_effect=physical_teams),
            mock.patch.object(hosted_resources, "_memory_usage", side_effect=memory_usage),
            hosted_resources._reserve_capacity("team:one", "account_1", 10, team_slot=True),
        ):
            self.assertIn("team:one", runtime_state._capacity_reservations)

        self.assertEqual(lock_observations, [("teams", False), ("memory", False)])

    def test_teardown_advances_and_removes_real_durable_cleanup_record(self) -> None:
        events: list[object] = []
        brain = SimpleNamespace(id="a" * 64)

        def phase(name: str, result=True):
            def run(*_args, **_kwargs):
                events.append(name)
                return result

            return run

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(cleanup_state, "STATE_DIR", Path(directory)),
            mock.patch.multiple(
                hosted_lifecycle,
                _owned_teardown_brain=lambda *_args: (True, brain),
                _stop_teardown_brain=phase("stop"),
                _teardown_apps=phase("apps"),
                _teardown_storage=phase("storage"),
                _teardown_inference=phase("inference"),
                _teardown_assistant_secrets=phase("secrets"),
                _teardown_assistant_accounts=phase("accounts"),
                _teardown_network_planes=phase("networks"),
                _remove_teardown_brain=phase("brain"),
                _teardown_volumes=phase("volumes"),
            ),
            mock.patch.object(pgdriver_client, "drop_team", side_effect=phase("database")),
            mock.patch.object(pgdriver_client, "finalize_team_drop", side_effect=phase("finalize")),
        ):
            result = hosted_lifecycle._teardown("team_1", owner="account_1", brain_id=brain.id)
            pending = cleanup_state.load("team_1")

        self.assertTrue(result.complete)
        self.assertIsNone(pending)
        self.assertEqual(
            events,
            [
                "stop",
                "apps",
                "storage",
                "inference",
                "secrets",
                "accounts",
                "networks",
                "brain",
                "volumes",
                "database",
                "finalize",
            ],
        )

    def test_teardown_failure_preserves_owner_bound_cleanup_record(self) -> None:
        brain = SimpleNamespace(id="b" * 64)
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(cleanup_state, "STATE_DIR", Path(directory)),
            mock.patch.multiple(
                hosted_lifecycle,
                _owned_teardown_brain=lambda *_args: (True, brain),
                _stop_teardown_brain=lambda _brain: True,
                _teardown_apps=lambda _team_id: False,
            ),
        ):
            result = hosted_lifecycle._teardown("team_1", owner="account_1", brain_id=brain.id)
            pending = cleanup_state.load("team_1")

        self.assertFalse(result.complete)
        self.assertIsNotNone(pending)
        self.assertEqual((pending.owner, pending.brain_id, pending.db_dropped), ("account_1", brain.id, False))


if __name__ == "__main__":
    unittest.main()
