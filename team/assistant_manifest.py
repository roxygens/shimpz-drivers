"""Immutable Assistant manifest admission for reviewed outbound hosts."""

from __future__ import annotations

import io
import ipaddress
import re
import tarfile
import threading
import tomllib
from collections import OrderedDict
from collections.abc import Iterable
from contextlib import ExitStack

MANIFEST_PATH = "/opt/shimpz-assistant/shimpz.assistant.toml"
MAX_MANIFEST_BYTES = 256 * 1024
MAX_ARCHIVE_BYTES = MAX_MANIFEST_BYTES + (32 * 1024)
MAX_ALLOWED_HOSTS = 32
DEFAULT_CACHE_ENTRIES = 256
_PUBLIC_HOST_RE = re.compile(
    r"(?=.{1,253}\Z)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?\Z"
)
_NON_PUBLIC_HOST_SUFFIXES = (
    ".arpa",
    ".example",
    ".home",
    ".internal",
    ".invalid",
    ".lan",
    ".local",
    ".localdomain",
    ".localhost",
    ".onion",
    ".test",
)


class ManifestError(RuntimeError):
    """An immutable Assistant package did not expose its reviewed network intent."""


def canonical_allowed_hosts(value: object) -> tuple[str, ...]:
    """Return one deterministic list of exact public DNS host names."""
    if not isinstance(value, list | tuple) or len(value) > MAX_ALLOWED_HOSTS:
        raise ManifestError("Assistant allowed_hosts is invalid")
    hosts: list[str] = []
    for host in value:
        if not isinstance(host, str) or not 1 <= len(host) <= 253 or not host.isascii() or host != host.lower():
            raise ManifestError("Assistant allowed_hosts is invalid")
        if _PUBLIC_HOST_RE.fullmatch(host) is None or host.endswith(_NON_PUBLIC_HOST_SUFFIXES):
            raise ManifestError("Assistant allowed_hosts is invalid")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise ManifestError("Assistant allowed_hosts is invalid")
        hosts.append(host)
    if len(set(hosts)) != len(hosts):
        raise ManifestError("Assistant allowed_hosts is invalid")
    return tuple(sorted(hosts))


def parse_allowed_hosts(raw: bytes) -> tuple[str, ...]:
    """Parse only the bounded network-intent field from one UTF-8 TOML manifest."""
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= MAX_MANIFEST_BYTES:
        raise ManifestError("Assistant manifest has an invalid size")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestError("Assistant manifest is not UTF-8") from exc
    if any(not character.isprintable() and character not in {"\n", "\r", "\t"} for character in text):
        raise ManifestError("Assistant manifest contains invalid text")
    try:
        manifest = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError("Assistant manifest is invalid TOML") from exc
    if not isinstance(manifest, dict) or "allowed_hosts" not in manifest:
        raise ManifestError("Assistant manifest does not declare allowed_hosts")
    return canonical_allowed_hosts(manifest["allowed_hosts"])


def _bounded_archive(chunks: Iterable[bytes]) -> bytes:
    archive = bytearray()
    try:
        with ExitStack() as cleanup:
            close = getattr(chunks, "close", None)
            if callable(close):
                cleanup.callback(close)
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise ManifestError("Assistant manifest archive is invalid")
                archive.extend(chunk)
                if len(archive) > MAX_ARCHIVE_BYTES:
                    raise ManifestError("Assistant manifest archive is too large")
    except ManifestError:
        raise
    except Exception as exc:
        raise ManifestError("Assistant manifest archive is unavailable") from exc
    return bytes(archive)


def read_container_allowed_hosts(container) -> tuple[str, ...]:
    """Read allowed_hosts from the fixed regular manifest in a digest-bound root."""
    try:
        chunks, metadata = container.get_archive(MANIFEST_PATH)
    except Exception as exc:
        raise ManifestError("Assistant manifest is unavailable") from exc
    if not isinstance(metadata, dict):
        raise ManifestError("Assistant manifest metadata is invalid")
    size = metadata.get("size")
    mode = metadata.get("mode")
    name = metadata.get("name")
    if (
        name != "shimpz.assistant.toml"
        or not isinstance(size, int)
        or isinstance(size, bool)
        or not 1 <= size <= MAX_MANIFEST_BYTES
        or not isinstance(mode, int)
        or isinstance(mode, bool)
        or mode != 0o444
    ):
        raise ManifestError("Assistant manifest metadata is invalid")

    archive = _bounded_archive(chunks)
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            members = bundle.getmembers()
            if (
                len(members) != 1
                or members[0].name not in {"shimpz.assistant.toml", "./shimpz.assistant.toml"}
                or not members[0].isreg()
                or members[0].size != size
                or members[0].mode & 0o777 != 0o444
            ):
                raise ManifestError("Assistant manifest archive is invalid")
            extracted = bundle.extractfile(members[0])
            if extracted is None:
                raise ManifestError("Assistant manifest archive is invalid")
            raw = extracted.read(MAX_MANIFEST_BYTES + 1)
    except ManifestError:
        raise
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ManifestError("Assistant manifest archive is invalid") from exc
    if len(raw) != size:
        raise ManifestError("Assistant manifest archive is invalid")
    return parse_allowed_hosts(raw)


class AllowedHostsCache:
    """Admit reviewed hosts once per immutable container generation with a bounded LRU."""

    def __init__(self, max_entries: int = DEFAULT_CACHE_ENTRIES) -> None:
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries < 1:
            raise ValueError("Assistant manifest cache size must be positive")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, tuple[str, ...]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, container, reviewed_hosts: object) -> tuple[str, ...]:
        """Return the declared list only when it exactly matches controller review."""
        container_id = getattr(container, "id", None)
        if (
            not isinstance(container_id, str)
            or not container_id
            or len(container_id) > 256
            or any(not character.isalnum() and character not in {"-", "_", "."} for character in container_id)
        ):
            raise ManifestError("Assistant container identity is invalid")
        reviewed = canonical_allowed_hosts(reviewed_hosts)
        with self._lock:
            declared = self._entries.get(container_id)
            if declared is None:
                declared = read_container_allowed_hosts(container)
                self._entries[container_id] = declared
                while len(self._entries) > self._max_entries:
                    self._entries.popitem(last=False)
            else:
                self._entries.move_to_end(container_id)
        if declared != reviewed:
            raise ManifestError("Assistant allowed_hosts does not match its reviewed contract")
        return declared

    def discard(self, container_id: object) -> None:
        if isinstance(container_id, str):
            with self._lock:
                self._entries.pop(container_id, None)
