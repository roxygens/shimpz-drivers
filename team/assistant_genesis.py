"""Bounded, immutable Assistant Genesis admission and controller-side pre-cache."""

from __future__ import annotations

import io
import tarfile
import threading
from collections import OrderedDict
from contextlib import ExitStack

GENESIS_PATH = "/opt/shimpz-assistant/GENESIS.md"
MAX_GENESIS_BYTES = 128 * 1024
MAX_ARCHIVE_BYTES = MAX_GENESIS_BYTES + (32 * 1024)
DEFAULT_CACHE_ENTRIES = 256


class GenesisError(RuntimeError):
    """An immutable Assistant package did not expose a safe Genesis contract."""


def canonical_genesis(raw: bytes) -> str:
    """Decode and canonicalize one bounded Markdown document for a model prompt."""
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= MAX_GENESIS_BYTES:
        raise GenesisError("Assistant Genesis has an invalid size")
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GenesisError("Assistant Genesis is not UTF-8") from exc
    value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if (
        not value
        or len(value.encode("utf-8")) > MAX_GENESIS_BYTES
        or any(not character.isprintable() and character not in {"\n", "\t"} for character in value)
    ):
        raise GenesisError("Assistant Genesis contains invalid text")
    return value


def _bounded_archive(chunks) -> bytes:
    archive = bytearray()
    try:
        with ExitStack() as cleanup:
            close = getattr(chunks, "close", None)
            if callable(close):
                cleanup.callback(close)
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise GenesisError("Assistant Genesis archive is invalid")
                archive.extend(chunk)
                if len(archive) > MAX_ARCHIVE_BYTES:
                    raise GenesisError("Assistant Genesis archive is too large")
    except GenesisError:
        raise
    except Exception as exc:
        raise GenesisError("Assistant Genesis archive is unavailable") from exc
    return bytes(archive)


def read_container_genesis(container) -> str:
    """Read the fixed regular file from a read-only, digest-bound Assistant root."""
    try:
        chunks, metadata = container.get_archive(GENESIS_PATH)
    except Exception as exc:
        raise GenesisError("Assistant Genesis is unavailable") from exc
    if not isinstance(metadata, dict):
        raise GenesisError("Assistant Genesis metadata is invalid")
    size = metadata.get("size")
    mode = metadata.get("mode")
    name = metadata.get("name")
    if (
        name != "GENESIS.md"
        or not isinstance(size, int)
        or isinstance(size, bool)
        or not 1 <= size <= MAX_GENESIS_BYTES
        or not isinstance(mode, int)
        or isinstance(mode, bool)
        or mode != 0o444
    ):
        raise GenesisError("Assistant Genesis metadata is invalid")

    archive = _bounded_archive(chunks)

    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            members = bundle.getmembers()
            if (
                len(members) != 1
                or members[0].name not in {"GENESIS.md", "./GENESIS.md"}
                or not members[0].isreg()
                or members[0].size != size
                or members[0].mode & 0o777 != 0o444
            ):
                raise GenesisError("Assistant Genesis archive is invalid")
            extracted = bundle.extractfile(members[0])
            if extracted is None:
                raise GenesisError("Assistant Genesis archive is invalid")
            raw = extracted.read(MAX_GENESIS_BYTES + 1)
    except GenesisError:
        raise
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise GenesisError("Assistant Genesis archive is invalid") from exc
    if len(raw) != size:
        raise GenesisError("Assistant Genesis archive is invalid")
    return canonical_genesis(raw)


class GenesisCache:
    """Read Genesis once per immutable container generation with a bounded LRU."""

    def __init__(self, max_entries: int = DEFAULT_CACHE_ENTRIES) -> None:
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries < 1:
            raise ValueError("Genesis cache size must be positive")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, container) -> str:
        container_id = getattr(container, "id", None)
        if (
            not isinstance(container_id, str)
            or not container_id
            or len(container_id) > 256
            or any(not character.isalnum() and character not in {"-", "_", "."} for character in container_id)
        ):
            raise GenesisError("Assistant container identity is invalid")
        with self._lock:
            cached = self._entries.get(container_id)
            if cached is not None:
                self._entries.move_to_end(container_id)
                return cached
            genesis = read_container_genesis(container)
            self._entries[container_id] = genesis
            self._entries.move_to_end(container_id)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            return genesis

    def discard(self, container_id: object) -> None:
        if isinstance(container_id, str):
            with self._lock:
                self._entries.pop(container_id, None)
