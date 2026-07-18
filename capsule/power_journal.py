"""Crash-safe, bounded idempotency journal for Assistant Power side effects.

The journal stores only caller-provided fingerprints and bounded Power results. Raw
Power inputs never cross this boundary. An operation durably enters ``executing``
before its side effect starts; finding it there again is intentionally an uncertain
outcome and fails closed instead of risking a duplicate side effect.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import threading
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1
APPLICATION_ID = 0x53484A31  # SHJ1
MAX_GENERATIONS = 1024
MAX_OPERATIONS = 64
MAX_RESULT_BYTES = 32 * 1024
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 4096

_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
_FINGERPRINT_RE = re.compile(r"[a-f0-9]{64}\Z")
_STATES = frozenset({"prepared", "executing", "completed"})


class PowerJournalError(RuntimeError):
    """The Power journal could not safely prove the requested transition."""


class PowerJournalConflictError(PowerJournalError):
    """Durable state does not match the caller's immutable batch contract."""


class PowerJournalUncertainError(PowerJournalError):
    """A side effect may already have happened and must not be executed again."""


class PowerJournalCorruptionError(PowerJournalError):
    """The journal or a persisted record violated its closed schema."""


@dataclass(frozen=True, slots=True)
class Operation:
    """Opaque Power identity; the fingerprint commits to the validated request."""

    interrupt_id: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class Batch:
    """Immutable handle for one Brain suspension in a Team generation."""

    generation: str
    fingerprint: str
    operations: tuple[Operation, ...]


@dataclass(frozen=True, slots=True)
class Execution:
    """Decision returned before invocation, or a previously committed JSON result."""

    execute: bool
    result: object | None = None


def _positive_limit(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _safe_id(value: object, name: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise PowerJournalConflictError(f"{name} is invalid")
    return value


def _operation(value: object) -> Operation:
    if not isinstance(value, Operation):
        raise PowerJournalConflictError("operation is invalid")
    _safe_id(value.interrupt_id, "operation interrupt id")
    if not isinstance(value.fingerprint, str) or _FINGERPRINT_RE.fullmatch(value.fingerprint) is None:
        raise PowerJournalConflictError("operation fingerprint is invalid")
    return value


def _walk_json(value: object, *, depth: int = 0, budget: list[int] | None = None) -> None:
    if budget is None:
        budget = [MAX_JSON_NODES]
    budget[0] -= 1
    if budget[0] < 0 or depth > MAX_JSON_DEPTH:
        raise PowerJournalConflictError("Power result exceeds the JSON structure limit")
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PowerJournalConflictError("Power result contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _walk_json(item, depth=depth + 1, budget=budget)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PowerJournalConflictError("Power result object keys must be strings")
            _walk_json(item, depth=depth + 1, budget=budget)
        return
    raise PowerJournalConflictError("Power result must contain only JSON values")


def _canonical_result(value: object, max_bytes: int) -> bytes:
    _walk_json(value)
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        raise PowerJournalConflictError("Power result is not canonical JSON") from exc
    if len(encoded) > max_bytes:
        raise PowerJournalConflictError("Power result exceeds the durable size limit")
    return encoded


class PowerJournal:
    """Serialize durable Power transitions through one private SQLite database."""

    def __init__(
        self,
        path: Path,
        *,
        max_generations: int = MAX_GENERATIONS,
        max_operations: int = MAX_OPERATIONS,
        max_result_bytes: int = MAX_RESULT_BYTES,
    ) -> None:
        self.path = Path(path)
        self.max_generations = _positive_limit(max_generations, "max_generations")
        self.max_operations = _positive_limit(max_operations, "max_operations")
        self.max_result_bytes = _positive_limit(max_result_bytes, "max_result_bytes")
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
            self._configure()
            if initialize:
                self._create_schema()
            self._validate_schema()
            self.path.chmod(0o600)
        except (OSError, sqlite3.Error, PowerJournalError) as exc:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            if isinstance(exc, PowerJournalError):
                raise
            raise PowerJournalCorruptionError("Power journal could not be opened safely") from exc

    def _prepare_file(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            parent = self.path.parent.lstat()
            if not stat.S_ISDIR(parent.st_mode) or stat.S_ISLNK(parent.st_mode) or parent.st_uid != os.geteuid():
                raise PowerJournalCorruptionError("Power journal parent is not a private directory")
            self.path.parent.chmod(0o700)
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
                    self._validate_file_metadata(metadata)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                directory = os.open(
                    self.path.parent,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                )
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
                return True
            self._validate_file_metadata(metadata)
            if metadata.st_mode & 0o077:
                raise PowerJournalCorruptionError("Power journal file permissions are not private")
        except OSError as exc:
            raise PowerJournalCorruptionError("Power journal private path is unavailable") from exc
        else:
            return metadata.st_size == 0

    @staticmethod
    def _validate_file_metadata(metadata: os.stat_result) -> None:
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
        ):
            raise PowerJournalCorruptionError("Power journal path has unsafe ownership or links")

    def _configure(self) -> None:
        self._connection.execute("PRAGMA trusted_schema = OFF")
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        mode = self._connection.execute("PRAGMA journal_mode = DELETE").fetchone()
        if mode != ("delete",):
            raise PowerJournalCorruptionError("Power journal could not enable its durable mode")
        self._connection.execute("PRAGMA synchronous = FULL")

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE batches (
                generation TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                operation_count INTEGER NOT NULL CHECK (operation_count > 0)
            ) WITHOUT ROWID;
            CREATE TABLE operations (
                generation TEXT NOT NULL,
                ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
                interrupt_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('prepared', 'executing', 'completed')),
                result BLOB,
                PRIMARY KEY (generation, interrupt_id),
                UNIQUE (generation, ordinal),
                FOREIGN KEY (generation) REFERENCES batches(generation) ON DELETE CASCADE,
                CHECK ((state = 'completed' AND result IS NOT NULL) OR
                       (state != 'completed' AND result IS NULL))
            ) WITHOUT ROWID;
            PRAGMA application_id = 1397246513;
            PRAGMA user_version = 1;
            COMMIT;
            """
        )

    def _validate_schema(self) -> None:
        try:
            check = self._connection.execute("PRAGMA quick_check").fetchall()
            application_id = self._connection.execute("PRAGMA application_id").fetchone()
            version = self._connection.execute("PRAGMA user_version").fetchone()
            tables = {
                row[0]
                for row in self._connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            batch_columns = [row[1] for row in self._connection.execute("PRAGMA table_info(batches)")]
            operation_columns = [row[1] for row in self._connection.execute("PRAGMA table_info(operations)")]
            foreign_keys = self._connection.execute("PRAGMA foreign_key_check").fetchall()
        except sqlite3.Error as exc:
            raise PowerJournalCorruptionError("Power journal integrity could not be verified") from exc
        if (
            check != [("ok",)]
            or application_id != (APPLICATION_ID,)
            or version != (SCHEMA_VERSION,)
            or tables != {"batches", "operations"}
            or batch_columns != ["generation", "fingerprint", "operation_count"]
            or operation_columns != ["generation", "ordinal", "interrupt_id", "fingerprint", "state", "result"]
            or foreign_keys
        ):
            raise PowerJournalCorruptionError("Power journal schema or contents are invalid")

    def _transaction(self) -> None:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as exc:
            raise PowerJournalError("Power journal transaction could not start") from exc

    def _commit(self) -> None:
        try:
            self._connection.execute("COMMIT")
        except sqlite3.Error as exc:
            raise PowerJournalError("Power journal transaction could not commit") from exc

    def _rollback(self) -> None:
        with suppress(sqlite3.Error):
            self._connection.execute("ROLLBACK")

    def _ensure_open(self) -> None:
        if self._closed:
            raise PowerJournalError("Power journal is closed")

    def _batch(self, generation: object, thread_id: object, operations: Sequence[Operation]) -> Batch:
        safe_generation = _safe_id(generation, "generation")
        safe_thread = _safe_id(thread_id, "thread id")
        if isinstance(operations, (str, bytes)):
            raise PowerJournalConflictError("operations are invalid")
        try:
            selected = tuple(_operation(item) for item in operations)
        except TypeError as exc:
            raise PowerJournalConflictError("operations are invalid") from exc
        if not selected or len(selected) > self.max_operations:
            raise PowerJournalConflictError("Power batch exceeds the operation count limit")
        if len({item.interrupt_id for item in selected}) != len(selected):
            raise PowerJournalConflictError("Power batch repeats an interrupt id")
        payload = json.dumps(
            {
                "generation": safe_generation,
                "operations": [[item.interrupt_id, item.fingerprint] for item in selected],
                "thread": hashlib.sha256(safe_thread.encode("utf-8")).hexdigest(),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return Batch(safe_generation, hashlib.sha256(payload).hexdigest(), selected)

    @staticmethod
    def _validate_handle(batch: object) -> Batch:
        if not isinstance(batch, Batch):
            raise PowerJournalConflictError("Power batch handle is invalid")
        _safe_id(batch.generation, "generation")
        if not isinstance(batch.fingerprint, str) or _FINGERPRINT_RE.fullmatch(batch.fingerprint) is None:
            raise PowerJournalConflictError("Power batch fingerprint is invalid")
        if not isinstance(batch.operations, tuple) or not batch.operations:
            raise PowerJournalConflictError("Power batch operations are invalid")
        for operation in batch.operations:
            _operation(operation)
        if len({item.interrupt_id for item in batch.operations}) != len(batch.operations):
            raise PowerJournalConflictError("Power batch repeats an interrupt id")
        return batch

    def _load_batch(self, batch: Batch) -> list[tuple[object, ...]]:
        try:
            row = self._connection.execute(
                "SELECT fingerprint, operation_count FROM batches WHERE generation = ?",
                (batch.generation,),
            ).fetchone()
            operations = self._connection.execute(
                """SELECT ordinal, interrupt_id, fingerprint, state, result
                   FROM operations WHERE generation = ? ORDER BY ordinal""",
                (batch.generation,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise PowerJournalCorruptionError("Power journal batch could not be read") from exc
        if row is None:
            raise PowerJournalConflictError("Power batch is no longer current")
        fingerprint, operation_count = row
        expected = [(item.interrupt_id, item.fingerprint) for item in batch.operations]
        actual = [(row[1], row[2]) for row in operations]
        if (
            fingerprint != batch.fingerprint
            or type(operation_count) is not int
            or operation_count != len(batch.operations)
            or actual != expected
            or [row[0] for row in operations] != list(range(len(operations)))
            or any(row[3] not in _STATES for row in operations)
        ):
            raise PowerJournalConflictError("Power batch changed or is corrupt")
        return operations

    def prepare_batch(
        self,
        generation: str,
        thread_id: str,
        operations: Sequence[Operation],
    ) -> Batch:
        batch = self._batch(generation, thread_id, operations)
        with self._guard:
            self._ensure_open()
            self._transaction()
            try:
                row = self._connection.execute(
                    "SELECT fingerprint FROM batches WHERE generation = ?",
                    (batch.generation,),
                ).fetchone()
                if row == (batch.fingerprint,):
                    self._load_batch(batch)
                elif row is not None:
                    raise PowerJournalConflictError("another Power batch is pending for this generation")
                else:
                    count = self._connection.execute("SELECT COUNT(*) FROM batches").fetchone()
                    if count is None or type(count[0]) is not int:
                        raise PowerJournalCorruptionError("Power journal capacity is invalid")
                    if count[0] >= self.max_generations:
                        raise PowerJournalConflictError("Power journal generation capacity is exhausted")
                    self._connection.execute(
                        "INSERT INTO batches VALUES (?, ?, ?)",
                        (batch.generation, batch.fingerprint, len(batch.operations)),
                    )
                    self._connection.executemany(
                        "INSERT INTO operations VALUES (?, ?, ?, ?, 'prepared', NULL)",
                        [
                            (
                                batch.generation,
                                ordinal,
                                operation.interrupt_id,
                                operation.fingerprint,
                            )
                            for ordinal, operation in enumerate(batch.operations)
                        ],
                    )
                self._commit()
            except (sqlite3.Error, PowerJournalError) as exc:
                self._rollback()
                if isinstance(exc, PowerJournalError):
                    raise
                raise PowerJournalError("Power batch could not be prepared") from exc
            else:
                return batch

    def begin(self, batch: Batch, operation: Operation) -> Execution:
        batch = self._validate_handle(batch)
        operation = _operation(operation)
        if operation not in batch.operations:
            raise PowerJournalConflictError("operation does not belong to this Power batch")
        with self._guard:
            self._ensure_open()
            self._transaction()
            try:
                operations = self._load_batch(batch)
                persisted = next(row for row in operations if row[1] == operation.interrupt_id)
                state, raw_result = persisted[3], persisted[4]
                if state == "executing":
                    raise PowerJournalUncertainError(
                        "Power execution outcome is uncertain; refusing a duplicate side effect"
                    )
                if state == "completed":
                    result = self._decode_result(raw_result)
                    self._commit()
                    return Execution(execute=False, result=result)
                if raw_result is not None:
                    raise PowerJournalCorruptionError("Power operation has an invalid durable state")
                self._connection.execute(
                    """UPDATE operations SET state = 'executing'
                       WHERE generation = ? AND interrupt_id = ? AND state = 'prepared'""",
                    (batch.generation, operation.interrupt_id),
                )
                if self._connection.execute("SELECT changes()").fetchone() != (1,):
                    raise PowerJournalConflictError("Power operation changed before execution")
                self._commit()
                return Execution(execute=True)
            except (sqlite3.Error, PowerJournalError, StopIteration) as exc:
                self._rollback()
                if isinstance(exc, PowerJournalError):
                    raise
                if isinstance(exc, StopIteration):
                    raise PowerJournalCorruptionError("Power operation is missing") from exc
                raise PowerJournalError("Power execution could not begin") from exc

    def _decode_result(self, raw: object) -> object:
        if not isinstance(raw, bytes) or len(raw) > self.max_result_bytes:
            raise PowerJournalCorruptionError("cached Power result is invalid")
        try:
            result = json.loads(raw)
            if _canonical_result(result, self.max_result_bytes) != raw:
                raise ValueError("result is not canonical")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, PowerJournalError) as exc:
            if isinstance(exc, PowerJournalCorruptionError):
                raise
            raise PowerJournalCorruptionError("cached Power result is invalid") from exc
        return result

    def complete(self, batch: Batch, operation: Operation, result: object) -> None:
        batch = self._validate_handle(batch)
        operation = _operation(operation)
        if operation not in batch.operations:
            raise PowerJournalConflictError("operation does not belong to this Power batch")
        encoded = _canonical_result(result, self.max_result_bytes)
        with self._guard:
            self._ensure_open()
            self._transaction()
            try:
                operations = self._load_batch(batch)
                persisted = next(row for row in operations if row[1] == operation.interrupt_id)
                state, existing = persisted[3], persisted[4]
                if state == "completed":
                    if existing != encoded:
                        raise PowerJournalConflictError("Power result changed after completion")
                    self._commit()
                    return
                if state != "executing" or existing is not None:
                    raise PowerJournalConflictError("Power operation was not executing")
                self._connection.execute(
                    """UPDATE operations SET state = 'completed', result = ?
                       WHERE generation = ? AND interrupt_id = ? AND state = 'executing'""",
                    (encoded, batch.generation, operation.interrupt_id),
                )
                if self._connection.execute("SELECT changes()").fetchone() != (1,):
                    raise PowerJournalConflictError("Power operation changed before completion")
                self._commit()
            except (sqlite3.Error, PowerJournalError, StopIteration) as exc:
                self._rollback()
                if isinstance(exc, PowerJournalError):
                    raise
                if isinstance(exc, StopIteration):
                    raise PowerJournalCorruptionError("Power operation is missing") from exc
                raise PowerJournalError("Power result could not be committed") from exc

    def delivered(self, batch: Batch) -> None:
        batch = self._validate_handle(batch)
        with self._guard:
            self._ensure_open()
            self._transaction()
            try:
                row = self._connection.execute(
                    "SELECT fingerprint FROM batches WHERE generation = ?",
                    (batch.generation,),
                ).fetchone()
                if row is None:
                    self._commit()
                    return
                if row != (batch.fingerprint,):
                    raise PowerJournalConflictError("a newer Power batch replaced this delivery handle")
                operations = self._load_batch(batch)
                if any(row[3] != "completed" for row in operations):
                    raise PowerJournalConflictError("Power batch cannot be delivered before every result exists")
                for operation in operations:
                    self._decode_result(operation[4])
                self._connection.execute(
                    "DELETE FROM batches WHERE generation = ? AND fingerprint = ?",
                    (batch.generation, batch.fingerprint),
                )
                if self._connection.execute("SELECT changes()").fetchone() != (1,):
                    raise PowerJournalConflictError("Power batch changed before delivery")
                self._commit()
            except (sqlite3.Error, PowerJournalError) as exc:
                self._rollback()
                if isinstance(exc, PowerJournalError):
                    raise
                raise PowerJournalError("Power batch delivery could not be committed") from exc

    def purge(self, generation: str) -> None:
        safe_generation = _safe_id(generation, "generation")
        with self._guard:
            self._ensure_open()
            self._transaction()
            try:
                self._connection.execute("DELETE FROM batches WHERE generation = ?", (safe_generation,))
                self._commit()
            except sqlite3.Error as exc:
                self._rollback()
                raise PowerJournalError("Power generation could not be purged") from exc

    def close(self) -> None:
        with self._guard:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> PowerJournal:
        self._ensure_open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
