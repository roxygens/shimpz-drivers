from __future__ import annotations

import os
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path

import assistant_approval_grants

IMAGE_V1 = "ghcr.io/theshimpz/shimpz-assistant@sha256:" + "a" * 64
IMAGE_V2 = "ghcr.io/theshimpz/shimpz-assistant@sha256:" + "b" * 64


class ApprovalGrantStoreTests(unittest.TestCase):
    def test_once_grant_survives_restart_but_is_bound_to_the_exact_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state" / "grants.sqlite3"
            grant = assistant_approval_grants.Grant("team_1", "shimpz-assistant", "create-post", IMAGE_V1)
            store = assistant_approval_grants.ApprovalGrantStore(path)
            store.grant_many((grant,))
            store.close()

            reopened = assistant_approval_grants.ApprovalGrantStore(path)
            self.addCleanup(reopened.close)
            self.assertTrue(reopened.is_granted("team_1", "shimpz-assistant", "create-post", IMAGE_V1))
            self.assertFalse(reopened.is_granted("team_1", "shimpz-assistant", "create-post", IMAGE_V2))
            self.assertFalse(reopened.is_granted("team_2", "shimpz-assistant", "create-post", IMAGE_V1))
            self.assertEqual(reopened.list_team("team_1"), (grant,))
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertFalse(path.with_name(path.name + "-wal").exists())

    def test_revocation_is_scoped_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = assistant_approval_grants.ApprovalGrantStore(Path(directory) / "grants.sqlite3")
            self.addCleanup(store.close)
            grants = (
                assistant_approval_grants.Grant("team_1", "first-assistant", "create-post", IMAGE_V1),
                assistant_approval_grants.Grant("team_1", "second-assistant", "create-post", IMAGE_V1),
                assistant_approval_grants.Grant("team_2", "first-assistant", "create-post", IMAGE_V1),
            )
            store.grant_many(grants)

            self.assertEqual(store.revoke_assistant("team_1", "first-assistant"), 1)
            self.assertEqual(store.revoke_assistant("team_1", "first-assistant"), 0)
            self.assertEqual(store.revoke_team("team_1"), 1)
            self.assertEqual(store.list_team("team_2"), (grants[2],))
            self.assertEqual(store.revoke_all(), 1)
            self.assertEqual(store.list_team("team_2"), ())

    def test_capacity_invalid_ids_and_unsafe_files_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grants.sqlite3"
            store = assistant_approval_grants.ApprovalGrantStore(path, max_grants=1)
            self.addCleanup(store.close)
            store.grant_many((assistant_approval_grants.Grant("team_1", "first-assistant", "create-post", IMAGE_V1),))
            with self.assertRaises(assistant_approval_grants.ApprovalGrantError):
                store.grant_many(
                    (assistant_approval_grants.Grant("team_1", "second-assistant", "create-post", IMAGE_V1),)
                )
            with self.assertRaises(assistant_approval_grants.ApprovalGrantError):
                store.is_granted("../team", "first-assistant", "create-post", IMAGE_V1)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grants.sqlite3"
            path.write_bytes(b"not sqlite")
            path.chmod(0o644)
            with self.assertRaises(assistant_approval_grants.ApprovalGrantError):
                assistant_approval_grants.ApprovalGrantStore(path)

        with tempfile.TemporaryDirectory() as directory:
            real_parent = Path(directory) / "real"
            real_parent.mkdir(mode=0o700)
            linked_parent = Path(directory) / "linked"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaises(assistant_approval_grants.ApprovalGrantError):
                assistant_approval_grants.ApprovalGrantStore(linked_parent / "grants.sqlite3")

        with tempfile.TemporaryDirectory() as directory:
            unsafe_parent = Path(directory) / "unsafe"
            unsafe_parent.mkdir(mode=0o755)
            with self.assertRaises(assistant_approval_grants.ApprovalGrantError):
                assistant_approval_grants.ApprovalGrantStore(unsafe_parent / "grants.sqlite3")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grants.sqlite3"
            target = Path(directory) / "target.sqlite3"
            target.write_bytes(os.urandom(16))
            target.chmod(0o600)
            path.symlink_to(target)
            with self.assertRaises(assistant_approval_grants.ApprovalGrantError):
                assistant_approval_grants.ApprovalGrantStore(path)

    def test_failed_begin_does_not_attempt_a_masking_rollback(self) -> None:
        class FailBegin:
            @staticmethod
            def execute(_statement: str):
                raise sqlite3.OperationalError("begin failed")

        with tempfile.TemporaryDirectory() as directory:
            store = assistant_approval_grants.ApprovalGrantStore(Path(directory) / "grants.sqlite3")
            connection = store._connection
            store._connection = FailBegin()
            try:
                with self.assertRaisesRegex(
                    assistant_approval_grants.ApprovalGrantError,
                    "could not be stored",
                ):
                    store.grant_many(
                        (
                            assistant_approval_grants.Grant(
                                "team_1",
                                "first-assistant",
                                "create-post",
                                IMAGE_V1,
                            ),
                        )
                    )
            finally:
                store._connection = connection
                store.close()


if __name__ == "__main__":
    unittest.main()
