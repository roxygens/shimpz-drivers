from __future__ import annotations

import hashlib
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import power_journal


def operation(interrupt_id: str, value: str) -> power_journal.Operation:
    return power_journal.Operation(interrupt_id, hashlib.sha256(value.encode()).hexdigest())


class PowerJournalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "private" / "journal.sqlite3"
        self.first = operation("interrupt-1", "validated-input-1")
        self.second = operation("interrupt-2", "validated-input-2")

    def journal(self, **limits: int) -> power_journal.PowerJournal:
        journal = power_journal.PowerJournal(self.path, **limits)
        self.addCleanup(journal.close)
        return journal

    def test_reopen_returns_canonical_cached_result_without_reexecution(self) -> None:
        journal = self.journal()
        batch = journal.prepare_batch("generation-1", "thread-1", [self.first])
        self.assertEqual(journal.begin(batch, self.first), power_journal.Execution(True, None))
        journal.complete(batch, self.first, {"z": [2, 1], "a": "ok"})
        journal.close()

        reopened = self.journal()
        same = reopened.prepare_batch("generation-1", "thread-1", [self.first])

        self.assertEqual(same, batch)
        self.assertEqual(
            reopened.begin(same, self.first),
            power_journal.Execution(False, {"a": "ok", "z": [2, 1]}),
        )
        self.assertEqual(stat.S_IMODE(self.path.parent.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_executing_operation_is_uncertain_after_reopen(self) -> None:
        journal = self.journal()
        batch = journal.prepare_batch("generation-1", "thread-1", [self.first])
        self.assertTrue(journal.begin(batch, self.first).execute)
        journal.close()

        reopened = self.journal()
        same = reopened.prepare_batch("generation-1", "thread-1", [self.first])
        with self.assertRaises(power_journal.PowerJournalUncertainError):
            reopened.begin(same, self.first)

    def test_changed_pending_batch_and_changed_completed_result_fail_closed(self) -> None:
        journal = self.journal()
        batch = journal.prepare_batch("generation-1", "thread-1", [self.first])

        with self.assertRaises(power_journal.PowerJournalConflictError):
            journal.prepare_batch("generation-1", "thread-1", [self.second])
        with self.assertRaises(power_journal.PowerJournalConflictError):
            journal.prepare_batch("generation-1", "other-thread", [self.first])

        journal.begin(batch, self.first)
        journal.complete(batch, self.first, {"answer": 1})
        journal.complete(batch, self.first, {"answer": 1})
        with self.assertRaises(power_journal.PowerJournalConflictError):
            journal.complete(batch, self.first, {"answer": 2})

    def test_delivery_requires_all_results_then_allows_the_next_batch(self) -> None:
        journal = self.journal(max_generations=1)
        batch = journal.prepare_batch("generation-1", "thread-1", [self.first, self.second])
        journal.begin(batch, self.first)
        journal.complete(batch, self.first, {"first": True})
        with self.assertRaises(power_journal.PowerJournalConflictError):
            journal.delivered(batch)

        journal.begin(batch, self.second)
        journal.complete(batch, self.second, {"second": True})
        journal.delivered(batch)
        journal.delivered(batch)

        next_batch = journal.prepare_batch("generation-1", "thread-1", [operation("interrupt-3", "input-3")])
        self.assertNotEqual(next_batch.fingerprint, batch.fingerprint)
        with self.assertRaises(power_journal.PowerJournalConflictError):
            journal.delivered(batch)
        with self.assertRaises(power_journal.PowerJournalConflictError):
            journal.begin(batch, self.first)

        next_operation = next_batch.operations[0]
        journal.begin(next_batch, next_operation)
        journal.complete(next_batch, next_operation, {"third": True})
        journal.delivered(next_batch)
        journal.prepare_batch("generation-2", "thread-2", [self.first])

    def test_corrupt_database_and_noncanonical_cache_fail_closed(self) -> None:
        journal = self.journal()
        batch = journal.prepare_batch("generation-1", "thread-1", [self.first])
        journal.begin(batch, self.first)
        journal.complete(batch, self.first, {"answer": 1})
        journal.close()

        connection = sqlite3.connect(self.path)
        connection.execute(
            "UPDATE operations SET result = ? WHERE interrupt_id = ?",
            (b'{"answer": 1}', self.first.interrupt_id),
        )
        connection.commit()
        connection.close()

        reopened = self.journal()
        with self.assertRaises(power_journal.PowerJournalCorruptionError):
            reopened.begin(batch, self.first)
        reopened.close()

        self.path.write_bytes(b"not-a-sqlite-database")
        self.path.chmod(0o600)
        with self.assertRaises(power_journal.PowerJournalCorruptionError):
            power_journal.PowerJournal(self.path)

    def test_capacity_and_json_limits_are_enforced_without_raw_inputs(self) -> None:
        journal = self.journal(max_generations=1, max_operations=1, max_result_bytes=16)
        batch = journal.prepare_batch("generation-1", "thread-secret", [self.first])
        with self.assertRaises(power_journal.PowerJournalConflictError):
            journal.prepare_batch("generation-2", "thread-2", [self.second])
        with self.assertRaises(power_journal.PowerJournalConflictError):
            journal.prepare_batch("generation-1", "thread-secret", [self.first, self.second])

        journal.begin(batch, self.first)
        with self.assertRaises(power_journal.PowerJournalConflictError):
            journal.complete(batch, self.first, {"secret": "raw-input-must-not-fit"})

        raw = self.path.read_bytes()
        self.assertNotIn(b"thread-secret", raw)
        self.assertNotIn(b"validated-input-1", raw)

    def test_purge_removes_even_uncertain_generation_and_frees_capacity(self) -> None:
        journal = self.journal(max_generations=1)
        batch = journal.prepare_batch("generation-1", "thread-1", [self.first])
        journal.begin(batch, self.first)

        journal.purge("generation-1")
        journal.purge("generation-1")
        replacement = journal.prepare_batch("generation-2", "thread-2", [self.second])

        self.assertTrue(journal.begin(replacement, self.second).execute)
        with self.assertRaises(power_journal.PowerJournalConflictError):
            journal.begin(batch, self.first)

    def test_unsafe_file_and_symlink_paths_are_rejected(self) -> None:
        self.path.parent.mkdir(mode=0o700)
        self.path.write_bytes(b"")
        self.path.chmod(0o644)
        with self.assertRaises(power_journal.PowerJournalCorruptionError):
            power_journal.PowerJournal(self.path)

        self.path.unlink()
        victim = Path(self.temporary.name) / "victim"
        victim.write_bytes(b"unchanged")
        self.path.symlink_to(victim)
        with self.assertRaises(power_journal.PowerJournalCorruptionError):
            power_journal.PowerJournal(self.path)
        self.assertEqual(victim.read_bytes(), b"unchanged")

    def test_new_database_file_and_parent_entry_are_fsynced(self) -> None:
        synced_modes: list[int] = []
        real_fsync = power_journal.os.fsync

        def observe(descriptor: int) -> None:
            synced_modes.append(power_journal.os.fstat(descriptor).st_mode)
            real_fsync(descriptor)

        with mock.patch.object(power_journal.os, "fsync", side_effect=observe):
            journal = self.journal()

        self.assertIsNotNone(journal)
        self.assertEqual(len(synced_modes), 2)
        self.assertTrue(stat.S_ISREG(synced_modes[0]))
        self.assertTrue(stat.S_ISDIR(synced_modes[1]))

    def test_hardlinked_database_fails_closed(self) -> None:
        journal = self.journal()
        journal.close()
        linked = self.path.parent / "journal-copy.sqlite3"
        linked.hardlink_to(self.path)

        with self.assertRaises(power_journal.PowerJournalCorruptionError):
            power_journal.PowerJournal(self.path)

    def test_foreign_parent_or_database_owner_fails_closed(self) -> None:
        journal = self.journal()
        journal.close()

        effective_uid = power_journal.os.geteuid()
        with (
            mock.patch.object(power_journal.os, "geteuid", return_value=effective_uid + 1),
            self.assertRaises(power_journal.PowerJournalCorruptionError),
        ):
            power_journal.PowerJournal(self.path)

        real_lstat = Path.lstat

        def foreign_database(path: Path) -> power_journal.os.stat_result:
            metadata = real_lstat(path)
            if path == self.path:
                fields = list(metadata)
                fields[4] = metadata.st_uid + 1
                return power_journal.os.stat_result(fields)
            return metadata

        with (
            mock.patch.object(Path, "lstat", autospec=True, side_effect=foreign_database),
            self.assertRaises(power_journal.PowerJournalCorruptionError),
        ):
            power_journal.PowerJournal(self.path)


if __name__ == "__main__":
    unittest.main()
