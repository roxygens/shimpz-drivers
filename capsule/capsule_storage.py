"""Controller-owned, quota-bounded file storage for one Capsule.

The Brain and Assistant containers never mount this directory.  Files are opaque
blobs reached only through named controller operations, so uploaded bytes cannot
be executed or traversed as paths by tenant workloads.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
import sqlite3
import stat
import time
from contextlib import closing
from pathlib import Path

DEFAULT_LIMIT_BYTES = 100 * 1024 * 1024
DATABASE_HEADROOM_BYTES = 8 * 1024 * 1024
MAX_FILES = 256
MAX_FILENAME_BYTES = 255
MAX_MEDIA_TYPE_BYTES = 127
_CAPSULE_ID = re.compile(r"[a-z0-9_]{1,40}")
_FILE_ID = re.compile(r"[a-f0-9]{32}")
_MEDIA_TYPE = re.compile(r"[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*")


class StorageError(RuntimeError):
    """The storage boundary could not prove a safe operation."""


class StorageQuotaError(StorageError):
    """The requested write would exceed the Capsule's fixed content quota."""


class StorageNotFoundError(StorageError):
    """The requested opaque file id does not belong to this Capsule."""


def _capsule_id(value: object) -> str:
    if not isinstance(value, str) or _CAPSULE_ID.fullmatch(value) is None:
        raise StorageError("invalid Capsule id")
    return value


def _file_id(value: object) -> str:
    if not isinstance(value, str) or _FILE_ID.fullmatch(value) is None:
        raise StorageNotFoundError("file not found")
    return value


def _filename(value: object) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise StorageError("filename must be non-empty and trimmed")
    if len(value.encode("utf-8")) > MAX_FILENAME_BYTES:
        raise StorageError("filename is too long")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise StorageError("filename must not contain a path")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise StorageError("filename contains control characters")
    return value


def _media_type(value: object) -> str:
    if value in {None, ""}:
        return "application/octet-stream"
    if (
        not isinstance(value, str)
        or len(value.encode("ascii", "ignore")) != len(value)
        or len(value) > MAX_MEDIA_TYPE_BYTES
        or _MEDIA_TYPE.fullmatch(value.lower()) is None
    ):
        raise StorageError("invalid media type")
    return value.lower()


class CapsuleStorage:
    """One SQLite blob database per Capsule with a durable page and content ceiling."""

    def __init__(self, root: Path, *, limit_bytes: int = DEFAULT_LIMIT_BYTES) -> None:
        if isinstance(limit_bytes, bool) or not isinstance(limit_bytes, int) or limit_bytes < 1:
            raise StorageError("storage limit must be a positive integer")
        self.root = root
        self.limit_bytes = limit_bytes
        self._ensure_root()

    def _ensure_root(self) -> None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.root.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_nlink < 2
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise StorageError("storage root has unsafe ownership or permissions")

    def _capsule_dir(self, capsule_id: str, *, create: bool) -> Path:
        capsule_id = _capsule_id(capsule_id)
        directory = self.root / capsule_id
        if create:
            directory.mkdir(mode=0o700, exist_ok=True)
        try:
            info = directory.lstat()
        except FileNotFoundError as exc:
            raise StorageNotFoundError("Capsule storage not found") from exc
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise StorageError("Capsule storage has unsafe ownership or permissions")
        return directory

    def _database_path(self, capsule_id: str, *, create: bool) -> Path:
        path = self._capsule_dir(capsule_id, create=create) / "files.sqlite3"
        if create:
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(path, flags, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
        if path.exists() or path.is_symlink():
            info = path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) & 0o077
            ):
                raise StorageError("Capsule storage database has an unsafe shape")
        return path

    def _connect(self, capsule_id: str, *, create: bool) -> sqlite3.Connection:
        path = self._database_path(capsule_id, create=create)
        if not create and not path.exists():
            raise StorageNotFoundError("Capsule storage not found")
        connection = sqlite3.connect(path, timeout=5, isolation_level=None)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA secure_delete=ON")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY CHECK(length(id) = 32),
                    name TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    size INTEGER NOT NULL CHECK(size > 0),
                    sha256 TEXT NOT NULL CHECK(length(sha256) = 64),
                    created_at INTEGER NOT NULL,
                    content BLOB NOT NULL
                ) STRICT
                """
            )
            page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
            page_limit = (self.limit_bytes + DATABASE_HEADROOM_BYTES + page_size - 1) // page_size
            applied = int(connection.execute(f"PRAGMA max_page_count={page_limit}").fetchone()[0])
            if applied != page_limit:
                raise StorageError("Capsule storage page limit could not be applied")
            connection.execute("PRAGMA user_version=1")
            path.chmod(0o600)
        except BaseException:
            connection.close()
            raise
        else:
            return connection

    @staticmethod
    def _usage(connection: sqlite3.Connection) -> tuple[int, int]:
        count, used = connection.execute("SELECT count(*), coalesce(sum(size), 0) FROM files").fetchone()
        return int(count), int(used)

    def put(self, capsule_id: str, name: object, content: bytes, media_type: object = None) -> dict[str, object]:
        safe_name = _filename(name)
        safe_media_type = _media_type(media_type)
        if not isinstance(content, bytes) or not content:
            raise StorageError("file must contain bytes")
        if len(content) > self.limit_bytes:
            raise StorageQuotaError("Capsule storage quota exceeded")
        file_id = secrets.token_hex(16)
        digest = hashlib.sha256(content).hexdigest()
        try:
            with closing(self._connect(capsule_id, create=True)) as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    count, used = self._usage(connection)
                    if count >= MAX_FILES:
                        raise StorageQuotaError("Capsule file count limit reached")
                    if used + len(content) > self.limit_bytes:
                        raise StorageQuotaError("Capsule storage quota exceeded")
                    connection.execute(
                        "INSERT INTO files(id,name,media_type,size,sha256,created_at,content) VALUES(?,?,?,?,?,?,?)",
                        (file_id, safe_name, safe_media_type, len(content), digest, int(time.time()), content),
                    )
                    connection.execute("COMMIT")
                except BaseException:
                    connection.execute("ROLLBACK")
                    raise
        except sqlite3.DatabaseError as exc:
            if "full" in str(exc).lower():
                raise StorageQuotaError("Capsule storage quota exceeded") from exc
            raise StorageError("Capsule storage transaction failed") from exc
        used += len(content)
        return {
            "id": file_id,
            "name": safe_name,
            "media_type": safe_media_type,
            "size": len(content),
            "sha256": digest,
            "used_bytes": used,
            "limit_bytes": self.limit_bytes,
            "remaining_bytes": self.limit_bytes - used,
        }

    def list(self, capsule_id: str) -> dict[str, object]:
        try:
            with closing(self._connect(capsule_id, create=False)) as connection:
                rows = connection.execute(
                    "SELECT id,name,media_type,size,sha256,created_at FROM files ORDER BY created_at,id"
                ).fetchall()
                _count, used = self._usage(connection)
        except StorageNotFoundError:
            rows = []
            used = 0
        return {
            "files": [
                {
                    "id": row[0],
                    "name": row[1],
                    "media_type": row[2],
                    "size": row[3],
                    "sha256": row[4],
                    "created_at": row[5],
                }
                for row in rows
            ],
            "used_bytes": used,
            "limit_bytes": self.limit_bytes,
            "remaining_bytes": self.limit_bytes - used,
        }

    def get(self, capsule_id: str, file_id: object) -> tuple[dict[str, object], bytes]:
        safe_id = _file_id(file_id)
        try:
            with closing(self._connect(capsule_id, create=False)) as connection:
                row = connection.execute(
                    "SELECT name,media_type,size,sha256,created_at,content FROM files WHERE id=?",
                    (safe_id,),
                ).fetchone()
        except StorageNotFoundError:
            row = None
        if row is None:
            raise StorageNotFoundError("file not found")
        content = bytes(row[5])
        if len(content) != row[2] or hashlib.sha256(content).hexdigest() != row[3]:
            raise StorageError("stored file failed its integrity check")
        return (
            {
                "id": safe_id,
                "name": row[0],
                "media_type": row[1],
                "size": row[2],
                "sha256": row[3],
                "created_at": row[4],
            },
            content,
        )

    def delete(self, capsule_id: str, file_id: object) -> dict[str, object]:
        safe_id = _file_id(file_id)
        try:
            with closing(self._connect(capsule_id, create=False)) as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    cursor = connection.execute("DELETE FROM files WHERE id=?", (safe_id,))
                    _count, used = self._usage(connection)
                    connection.execute("COMMIT")
                except BaseException:
                    connection.execute("ROLLBACK")
                    raise
        except StorageNotFoundError:
            cursor = None
            used = 0
        return {
            "id": safe_id,
            "deleted": cursor is not None and cursor.rowcount == 1,
            "used_bytes": used,
            "limit_bytes": self.limit_bytes,
            "remaining_bytes": self.limit_bytes - used,
        }

    def destroy(self, capsule_id: str) -> bool:
        capsule_id = _capsule_id(capsule_id)
        directory = self.root / capsule_id
        try:
            info = directory.lstat()
        except FileNotFoundError:
            return False
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise StorageError("Capsule storage has unsafe ownership or permissions")
        shutil.rmtree(directory)
        return True
