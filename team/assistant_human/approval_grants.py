"""Durable release-bound grants for in-body approvals declared with runs="once"."""

from __future__ import annotations

import os
import re
import sqlite3
import stat
import threading
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 2
APPLICATION_ID = 0x53484147  # SHAG
MAX_GRANTS = 8192
DEFAULT_PATH = Path("/var/lib/shimpz-local/assistant-approvals/grants.sqlite3")

_TEAM_ID = re.compile(r"[a-z0-9_]{1,40}\Z")
_COMPONENT_ID = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*\Z")
_IMAGE = re.compile(r"[^\s\x00-\x1f\x7f]{1,512}@sha256:[0-9a-f]{64}\Z")


class ApprovalGrantError(RuntimeError):
    """Durable approval state could not be proven or updated safely."""


@dataclass(frozen=True, slots=True)
class Grant:
    team_id: str
    assistant_id: str
    power_id: str
    image: str
    ordinal: int


def _identity(team_id: object, assistant_id: object, power_id: object, image: object, ordinal: object) -> Grant:
    if not isinstance(team_id, str) or _TEAM_ID.fullmatch(team_id) is None:
        raise ApprovalGrantError("approval grant Team is invalid")
    if not isinstance(assistant_id, str) or len(assistant_id) > 80 or _COMPONENT_ID.fullmatch(assistant_id) is None:
        raise ApprovalGrantError("approval grant Assistant is invalid")
    if not isinstance(power_id, str) or len(power_id) > 80 or _COMPONENT_ID.fullmatch(power_id) is None:
        raise ApprovalGrantError("approval grant Power is invalid")
    if not isinstance(image, str) or _IMAGE.fullmatch(image) is None:
        raise ApprovalGrantError("approval grant release is invalid")
    if type(ordinal) is not int or not 0 <= ordinal <= 63:
        raise ApprovalGrantError("approval grant call-site ordinal is invalid")
    return Grant(team_id, assistant_id, power_id, image, ordinal)


class ApprovalGrantStore:
    """Serialize bounded grants in one owner-only SQLite database."""

    def __init__(self, path: Path = DEFAULT_PATH, *, max_grants: int = MAX_GRANTS) -> None:
        if type(max_grants) is not int or not 1 <= max_grants <= 65536:
            raise ValueError("approval grant capacity is invalid")
        self.path = Path(path)
        self.max_grants = max_grants
        self._guard = threading.RLock()
        self._closed = False
        initialize = self._prepare_file()
        try:
            self._connection = sqlite3.connect(
                self.path,
                isolation_level=None,
                check_same_thread=False,
                timeout=5.0,
            )
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA trusted_schema = OFF")
            self._connection.execute("PRAGMA busy_timeout = 5000")
            mode = self._connection.execute("PRAGMA journal_mode = DELETE").fetchone()
            if mode != ("delete",):
                raise ApprovalGrantError("approval grants could not enable their durable mode")
            self._connection.execute("PRAGMA synchronous = FULL")
            if initialize:
                self._create_schema()
            self._validate_schema()
            self.path.chmod(0o600)
        except (OSError, sqlite3.Error, ApprovalGrantError) as exc:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            if isinstance(exc, ApprovalGrantError):
                raise
            raise ApprovalGrantError("approval grants could not be opened safely") from exc

    def _prepare_file(self) -> bool:
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            parent = self.path.parent.stat(follow_symlinks=False)
            if (
                not stat.S_ISDIR(parent.st_mode)
                or stat.S_ISLNK(parent.st_mode)
                or parent.st_uid != os.geteuid()
                or stat.S_IMODE(parent.st_mode) != 0o700
            ):
                raise ApprovalGrantError("approval grant parent is not a private directory")
            try:
                metadata = self.path.lstat()
            except FileNotFoundError:
                descriptor = os.open(
                    self.path,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                )
                try:
                    metadata = os.fstat(descriptor)
                    self._validate_file(metadata)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
                return True
            self._validate_file(metadata)
        except OSError as exc:
            raise ApprovalGrantError("approval grant path is unavailable") from exc
        return metadata.st_size == 0

    @staticmethod
    def _validate_file(metadata: os.stat_result) -> None:
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise ApprovalGrantError("approval grant file failed its ownership contract")

    def _create_schema(self) -> None:
        self._connection.executescript(
            f"""
            PRAGMA application_id = {APPLICATION_ID};
            PRAGMA user_version = {SCHEMA_VERSION};
            CREATE TABLE grants (
                team_id TEXT NOT NULL,
                assistant_id TEXT NOT NULL,
                power_id TEXT NOT NULL,
                image TEXT NOT NULL,
                ordinal INTEGER NOT NULL CHECK (ordinal BETWEEN 0 AND 63),
                PRIMARY KEY (team_id, assistant_id, power_id, image, ordinal)
            ) WITHOUT ROWID;
            """
        )

    def _validate_schema(self) -> None:
        application_id = self._connection.execute("PRAGMA application_id").fetchone()[0]
        version = self._connection.execute("PRAGMA user_version").fetchone()[0]
        columns = tuple(row[1] for row in self._connection.execute("PRAGMA table_info(grants)"))
        if (
            application_id != APPLICATION_ID
            or version != SCHEMA_VERSION
            or columns
            != (
                "team_id",
                "assistant_id",
                "power_id",
                "image",
                "ordinal",
            )
        ):
            raise ApprovalGrantError("approval grant schema is invalid")

    def _ensure_open(self) -> None:
        if self._closed:
            raise ApprovalGrantError("approval grant store is closed")

    def is_granted(
        self,
        team_id: object,
        assistant_id: object,
        power_id: object,
        image: object,
        ordinal: object,
    ) -> bool:
        grant = _identity(team_id, assistant_id, power_id, image, ordinal)
        with self._guard:
            self._ensure_open()
            try:
                row = self._connection.execute(
                    """SELECT 1 FROM grants
                       WHERE team_id = ? AND assistant_id = ? AND power_id = ? AND image = ? AND ordinal = ?""",
                    (grant.team_id, grant.assistant_id, grant.power_id, grant.image, grant.ordinal),
                ).fetchone()
            except sqlite3.Error as exc:
                raise ApprovalGrantError("approval grant could not be read") from exc
        return row == (1,)

    def grant_many(self, grants: Iterable[Grant]) -> None:
        items = tuple(grants)
        if not items or any(not isinstance(item, Grant) for item in items):
            raise ApprovalGrantError("approval grant batch is invalid")
        canonical = tuple(
            _identity(item.team_id, item.assistant_id, item.power_id, item.image, item.ordinal) for item in items
        )
        if not canonical or len(set(canonical)) != len(canonical):
            raise ApprovalGrantError("approval grant batch is invalid")
        with self._guard:
            self._ensure_open()
            begun = False
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                begun = True
                current = self._connection.execute("SELECT COUNT(*) FROM grants").fetchone()[0]
                missing = sum(
                    self._connection.execute(
                        """SELECT 1 FROM grants
                           WHERE team_id = ? AND assistant_id = ? AND power_id = ? AND image = ? AND ordinal = ?""",
                        (item.team_id, item.assistant_id, item.power_id, item.image, item.ordinal),
                    ).fetchone()
                    is None
                    for item in canonical
                )
                if current + missing > self.max_grants:
                    raise ApprovalGrantError("approval grant capacity reached")
                self._connection.executemany(
                    """INSERT OR IGNORE INTO grants
                       (team_id, assistant_id, power_id, image, ordinal) VALUES (?, ?, ?, ?, ?)""",
                    ((item.team_id, item.assistant_id, item.power_id, item.image, item.ordinal) for item in canonical),
                )
                self._connection.execute("COMMIT")
            except ApprovalGrantError:
                if begun:
                    with suppress(sqlite3.Error):
                        self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                if begun:
                    with suppress(sqlite3.Error):
                        self._connection.execute("ROLLBACK")
                raise ApprovalGrantError("approval grants could not be stored") from exc

    def list_team(self, team_id: object) -> tuple[Grant, ...]:
        team = _identity(team_id, "assistant", "power", "release@sha256:" + "0" * 64, 0).team_id
        with self._guard:
            self._ensure_open()
            try:
                rows = self._connection.execute(
                    """SELECT team_id, assistant_id, power_id, image, ordinal
                       FROM grants WHERE team_id = ? ORDER BY assistant_id, power_id, image, ordinal""",
                    (team,),
                ).fetchall()
            except sqlite3.Error as exc:
                raise ApprovalGrantError("approval grants could not be listed") from exc
        return tuple(_identity(*row) for row in rows)

    def revoke_assistant(self, team_id: object, assistant_id: object) -> int:
        team = _identity(team_id, assistant_id, "power", "release@sha256:" + "0" * 64, 0)
        return self._delete_assistant(team.team_id, team.assistant_id)

    def revoke_team(self, team_id: object) -> int:
        team = _identity(team_id, "assistant", "power", "release@sha256:" + "0" * 64, 0).team_id
        with self._guard:
            self._ensure_open()
            try:
                cursor = self._connection.execute("DELETE FROM grants WHERE team_id = ?", (team,))
            except sqlite3.Error as exc:
                raise ApprovalGrantError("approval grants could not be revoked") from exc
        return cursor.rowcount

    def revoke_all(self) -> int:
        with self._guard:
            self._ensure_open()
            try:
                cursor = self._connection.execute("DELETE FROM grants")
            except sqlite3.Error as exc:
                raise ApprovalGrantError("approval grants could not be revoked") from exc
        return cursor.rowcount

    def _delete_assistant(self, team_id: str, assistant_id: str) -> int:
        with self._guard:
            self._ensure_open()
            try:
                cursor = self._connection.execute(
                    "DELETE FROM grants WHERE team_id = ? AND assistant_id = ?",
                    (team_id, assistant_id),
                )
            except sqlite3.Error as exc:
                raise ApprovalGrantError("approval grants could not be revoked") from exc
        return cursor.rowcount

    def close(self) -> None:
        with self._guard:
            if not self._closed:
                self._connection.close()
                self._closed = True
