from __future__ import annotations

import concurrent.futures
import stat
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

CAPSULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAPSULE))

import capsule_storage


class CapsuleStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "capsules"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_round_trip_is_opaque_and_isolated_by_capsule(self) -> None:
        storage = capsule_storage.CapsuleStorage(self.root, limit_bytes=128)
        first = storage.put("alpha", "brief.txt", b"confidential", "text/plain")
        second = storage.put("beta", "brief.txt", b"different", "text/plain")

        metadata, content = storage.get("alpha", first["id"])
        self.assertEqual(content, b"confidential")
        self.assertEqual(metadata["name"], "brief.txt")
        self.assertNotEqual(first["id"], second["id"])
        with self.assertRaises(capsule_storage.StorageNotFoundError):
            storage.get("beta", first["id"])

        alpha_directory = self.root / "alpha"
        self.assertEqual({path.name for path in alpha_directory.iterdir()}, {"files.sqlite3"})
        self.assertFalse((alpha_directory / "brief.txt").exists())
        self.assertEqual(stat.S_IMODE(alpha_directory.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((alpha_directory / "files.sqlite3").stat().st_mode), 0o600)

    def test_exact_content_quota_is_transactional(self) -> None:
        storage = capsule_storage.CapsuleStorage(self.root, limit_bytes=10)
        storage.put("alpha", "first.bin", b"1234")
        exact = storage.put("alpha", "second.bin", b"567890")
        self.assertEqual(exact["used_bytes"], 10)
        self.assertEqual(exact["remaining_bytes"], 0)

        with self.assertRaises(capsule_storage.StorageQuotaError):
            storage.put("alpha", "overflow.bin", b"x")
        listing = storage.list("alpha")
        self.assertEqual(listing["used_bytes"], 10)
        self.assertEqual(len(listing["files"]), 2)

    def test_concurrent_writes_cannot_overbook_quota(self) -> None:
        storage = capsule_storage.CapsuleStorage(self.root, limit_bytes=10)

        def write(index: int) -> bool:
            try:
                storage.put("alpha", f"{index}.bin", b"123456")
            except capsule_storage.StorageQuotaError:
                return False
            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(write, range(2)))

        self.assertEqual(sorted(results), [False, True])
        self.assertEqual(storage.list("alpha")["used_bytes"], 6)

    def test_delete_releases_logical_quota_and_destroy_is_scoped(self) -> None:
        storage = capsule_storage.CapsuleStorage(self.root, limit_bytes=8)
        alpha = storage.put("alpha", "alpha.bin", b"12345678")
        beta = storage.put("beta", "beta.bin", b"abcdefgh")

        deleted = storage.delete("alpha", alpha["id"])
        self.assertTrue(deleted["deleted"])
        self.assertEqual(deleted["used_bytes"], 0)
        storage.put("alpha", "replacement.bin", b"87654321")

        self.assertTrue(storage.destroy("alpha"))
        self.assertFalse(storage.destroy("alpha"))
        self.assertEqual(storage.get("beta", beta["id"])[1], b"abcdefgh")

    def test_file_count_and_metadata_are_bounded(self) -> None:
        storage = capsule_storage.CapsuleStorage(
            self.root,
            limit_bytes=capsule_storage.MAX_FILES + 1,
        )
        for index in range(capsule_storage.MAX_FILES):
            storage.put("alpha", f"{index}.txt", b"x", "text/plain")
        with self.assertRaises(capsule_storage.StorageQuotaError):
            storage.put("alpha", "one-too-many.txt", b"x")

        invalid_names = ("", " ../secret", "../secret", "nested/file", "line\nfeed")
        for name in invalid_names:
            with self.subTest(name=name), self.assertRaises(capsule_storage.StorageError):
                storage.put("beta", name, b"x")
        with self.assertRaises(capsule_storage.StorageError):
            storage.put("beta", "safe.txt", b"x", "text/plain; charset=utf-8")

    def test_database_page_ceiling_and_integrity_check_fail_closed(self) -> None:
        storage = capsule_storage.CapsuleStorage(self.root, limit_bytes=64)
        stored = storage.put("alpha", "safe.bin", b"safe")
        with closing(storage._connect("alpha", create=False)) as connection:
            page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
            expected = (
                64 + capsule_storage.DATABASE_HEADROOM_BYTES + page_size - 1
            ) // page_size
            self.assertEqual(int(connection.execute("PRAGMA max_page_count").fetchone()[0]), expected)
            connection.execute("UPDATE files SET content=? WHERE id=?", (b"evil", stored["id"]))

        with self.assertRaises(capsule_storage.StorageError):
            storage.get("alpha", stored["id"])

    def test_unsafe_storage_shapes_are_rejected(self) -> None:
        self.root.mkdir(mode=0o700)
        capsule = self.root / "alpha"
        capsule.mkdir(mode=0o700)
        target = self.root / "target"
        target.write_bytes(b"outside")
        (capsule / "files.sqlite3").symlink_to(target)

        storage = capsule_storage.CapsuleStorage(self.root, limit_bytes=64)
        with self.assertRaises(capsule_storage.StorageError):
            storage.put("alpha", "safe.bin", b"safe")
        self.assertEqual(target.read_bytes(), b"outside")


if __name__ == "__main__":
    unittest.main()
