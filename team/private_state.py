"""Shared fail-closed plumbing for encrypted controller-owned state."""

from __future__ import annotations

import base64
import os
import secrets
import stat
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def empty_state() -> dict[str, object]:
    return {"schema": 1, "teams": {}}


@dataclass(frozen=True, slots=True)
class PrivateState:
    error_class: type[RuntimeError]
    malformed_state: str
    malformed_envelope: str
    maximum_encoded_part: int

    def decode_part(
        self,
        value: object,
        *,
        expected: int | None = None,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> bytes:
        if not isinstance(value, str) or len(value) > self.maximum_encoded_part:
            raise self.error_class(self.malformed_envelope)
        try:
            decoded = base64.b64decode(value, validate=True)
        except (ValueError, TypeError) as exc:
            raise self.error_class(self.malformed_envelope) from exc
        if (
            (expected is not None and len(decoded) != expected)
            or (minimum is not None and len(decoded) < minimum)
            or (maximum is not None and len(decoded) > maximum)
        ):
            raise self.error_class(self.malformed_envelope)
        return decoded

    def read_private_file(self, path: Path, maximum: int, label: str) -> bytes | None:
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise self.error_class(f"{label} is unavailable") from exc
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_size > maximum
            ):
                raise self.error_class(f"{label} failed its ownership contract")
            payload = bytearray()
            while len(payload) <= maximum:
                chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - len(payload)))
                if not chunk:
                    break
                payload.extend(chunk)
            if len(payload) > maximum:
                raise self.error_class(f"{label} exceeds its fixed byte limit")
            return bytes(payload)
        finally:
            os.close(descriptor)

    def atomic_write(self, path: Path, payload: bytes, label: str) -> None:
        self._require_private_parent(path.parent, label)
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
        descriptor = -1
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written < 1:
                    raise OSError("short private write")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            temporary.replace(path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as exc:
            raise self.error_class(f"{label} could not be persisted") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            with suppress(FileNotFoundError):
                temporary.unlink()

    def key(self, path: Path, label: str, *, allow_create: bool = False) -> bytes:
        payload = self.read_private_file(path, 32, label)
        if payload is None:
            if not allow_create:
                raise self.error_class(f"{label} is unavailable")
            payload = AESGCM.generate_key(bit_length=256)
            self.atomic_write(path, payload, label)
        if len(payload) != 32:
            raise self.error_class(f"{label} is invalid")
        return payload

    def records(
        self,
        state: dict[str, object],
        team_id: str,
        assistant_id: str,
        *,
        create: bool,
    ) -> dict[str, object]:
        teams = self._teams(state)
        assistants = teams.get(team_id)
        if assistants is None:
            if not create:
                return {}
            assistants = {}
            teams[team_id] = assistants
        elif not isinstance(assistants, dict):
            raise self.error_class(self.malformed_state)
        records = assistants.get(assistant_id)
        if records is None:
            if not create:
                return {}
            records = {}
            assistants[assistant_id] = records
        elif not isinstance(records, dict):
            raise self.error_class(self.malformed_state)
        return records

    def has_records(self, state: Mapping[str, object]) -> bool:
        teams = self._teams(state)
        for assistants in teams.values():
            if not isinstance(assistants, dict):
                raise self.error_class(self.malformed_state)
            for records in assistants.values():
                if not isinstance(records, dict):
                    raise self.error_class(self.malformed_state)
                if records:
                    return True
        return False

    def prune_empty_records(self, state: dict[str, object], team_id: str, assistant_id: str) -> None:
        teams = self._teams(state)
        assistants = teams.get(team_id)
        if not isinstance(assistants, dict):
            raise self.error_class(self.malformed_state)
        records = assistants.get(assistant_id)
        if records is not None and not isinstance(records, dict):
            raise self.error_class(self.malformed_state)
        if isinstance(records, dict) and not records:
            assistants.pop(assistant_id)
        if not assistants:
            teams.pop(team_id)

    def delete_assistant(self, state: dict[str, object], team_id: str, assistant_id: str) -> bool:
        teams = self._teams(state)
        assistants = teams.get(team_id)
        if assistants is None:
            return False
        if not isinstance(assistants, dict):
            raise self.error_class(self.malformed_state)
        removed = assistants.pop(assistant_id, None) is not None
        if removed and not assistants:
            teams.pop(team_id)
        return removed

    def delete_team(self, state: dict[str, object], team_id: str) -> bool:
        return self._teams(state).pop(team_id, None) is not None

    def _require_private_parent(self, path: Path, label: str) -> None:
        try:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            metadata = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise self.error_class(f"{label} directory is unavailable") from exc
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise self.error_class(f"{label} directory failed its ownership contract")

    def _teams(self, state: Mapping[str, object]) -> dict[str, object]:
        teams = state.get("teams")
        if not isinstance(teams, dict):
            raise self.error_class(self.malformed_state)
        return teams
