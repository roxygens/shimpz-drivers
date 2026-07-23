"""Bounded, process-local continuations for just-in-time Assistant secrets.

Pending Power inputs can contain user data, so this MVP deliberately keeps them
out of persistent storage.  A controller restart expires the challenge and the
user can safely retry; no Power in a paused batch has started yet.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MAX_PENDING_CHALLENGES = 32
DEFAULT_TTL_SECONDS = 300
_CHALLENGE_ID = re.compile(r"[0-9a-f]{32}")
_TEAM_ID = re.compile(r"[a-z0-9_]{1,40}")


class SecretChallengeError(RuntimeError):
    """A pending secret continuation is unavailable or conflicts."""


class SecretChallengeNotFoundError(SecretChallengeError):
    """The opaque challenge is unknown, expired, or belongs to another Team."""


@dataclass(frozen=True, slots=True)
class SecretRequirement:
    assistant_id: str
    assistant_name: str
    power_ids: tuple[str, ...]
    secrets: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True, slots=True)
class PendingSecretChallenge:
    id: str
    team_id: str
    expires_at: float
    requirements: tuple[SecretRequirement, ...]
    payload: Any


def _team_id(value: object) -> str:
    if not isinstance(value, str) or _TEAM_ID.fullmatch(value) is None:
        raise SecretChallengeError("Team id is invalid")
    return value


def _challenge_id(value: object) -> str:
    if not isinstance(value, str) or _CHALLENGE_ID.fullmatch(value) is None:
        raise SecretChallengeNotFoundError("secret challenge is unavailable")
    return value


class SecretChallengeStore:
    def __init__(self, *, capacity: int = MAX_PENDING_CHALLENGES, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        if type(capacity) is not int or not 1 <= capacity <= 1024:
            raise ValueError("secret challenge capacity is invalid")
        if type(ttl_seconds) is not int or not 30 <= ttl_seconds <= 900:
            raise ValueError("secret challenge TTL is invalid")
        self._capacity = capacity
        self._ttl = ttl_seconds
        self._pending: dict[str, PendingSecretChallenge] = {}
        self._by_team: dict[str, str] = {}
        self._lock = threading.Lock()

    def _expire(self, now: float) -> None:
        expired = [challenge_id for challenge_id, item in self._pending.items() if item.expires_at <= now]
        for challenge_id in expired:
            challenge = self._pending.pop(challenge_id)
            if self._by_team.get(challenge.team_id) == challenge_id:
                self._by_team.pop(challenge.team_id, None)

    def create(
        self,
        team_id: object,
        requirements: tuple[SecretRequirement, ...],
        payload: Any,
    ) -> PendingSecretChallenge:
        team = _team_id(team_id)
        if not requirements:
            raise SecretChallengeError("secret challenge requires metadata")
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            if team in self._by_team:
                raise SecretChallengeError("Team already has a pending secret challenge")
            if len(self._pending) >= self._capacity:
                raise SecretChallengeError("secret challenge capacity reached")
            challenge_id = secrets.token_hex(16)
            while challenge_id in self._pending:
                challenge_id = secrets.token_hex(16)
            challenge = PendingSecretChallenge(
                id=challenge_id,
                team_id=team,
                expires_at=now + self._ttl,
                requirements=requirements,
                payload=payload,
            )
            self._pending[challenge_id] = challenge
            self._by_team[team] = challenge_id
            return challenge

    def get(self, team_id: object, challenge_id: object) -> PendingSecretChallenge:
        team = _team_id(team_id)
        identifier = _challenge_id(challenge_id)
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            challenge = self._pending.get(identifier)
            if challenge is None or challenge.team_id != team:
                raise SecretChallengeNotFoundError("secret challenge is unavailable")
            return challenge

    def restore(
        self,
        team_id: object,
        challenge_id: object,
        remaining_seconds: object,
        requirements: tuple[SecretRequirement, ...],
        payload: Any,
    ) -> PendingSecretChallenge:
        """Rehydrate one authenticated durable challenge without extending its TTL."""
        team = _team_id(team_id)
        identifier = _challenge_id(challenge_id)
        if type(remaining_seconds) is not int or not 1 <= remaining_seconds <= self._ttl or not requirements:
            raise SecretChallengeError("secret challenge restore is invalid")
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            if team in self._by_team or identifier in self._pending:
                raise SecretChallengeError("Team already has a pending secret challenge")
            if len(self._pending) >= self._capacity:
                raise SecretChallengeError("secret challenge capacity reached")
            challenge = PendingSecretChallenge(
                identifier,
                team,
                now + remaining_seconds,
                requirements,
                payload,
            )
            self._pending[identifier] = challenge
            self._by_team[team] = identifier
            return challenge

    def current(self, team_id: object) -> PendingSecretChallenge | None:
        team = _team_id(team_id)
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            identifier = self._by_team.get(team)
            return self._pending.get(identifier) if identifier is not None else None

    def claim(self, team_id: object, challenge_id: object) -> PendingSecretChallenge:
        team = _team_id(team_id)
        identifier = _challenge_id(challenge_id)
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            challenge = self._pending.get(identifier)
            if challenge is None or challenge.team_id != team:
                raise SecretChallengeNotFoundError("secret challenge is unavailable")
            self._pending.pop(identifier)
            if self._by_team.get(team) == identifier:
                self._by_team.pop(team, None)
            return challenge

    def claim_after(
        self,
        team_id: object,
        challenge_id: object,
        commit: Callable[[PendingSecretChallenge], None],
    ) -> PendingSecretChallenge:
        """Consume one challenge only after its bounded controller transaction commits."""
        team = _team_id(team_id)
        identifier = _challenge_id(challenge_id)
        if not callable(commit):
            raise SecretChallengeError("secret challenge commit is invalid")
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            challenge = self._pending.get(identifier)
            if challenge is None or challenge.team_id != team:
                raise SecretChallengeNotFoundError("secret challenge is unavailable")
            commit(challenge)
            self._pending.pop(identifier)
            if self._by_team.get(team) == identifier:
                self._by_team.pop(team, None)
            return challenge

    def cancel_team(self, team_id: object) -> bool:
        team = _team_id(team_id)
        with self._lock:
            identifier = self._by_team.pop(team, None)
            return self._pending.pop(identifier, None) is not None if identifier is not None else False

    def cancel_all(self) -> int:
        """Drop every process-local continuation during a Space reset."""
        with self._lock:
            removed = len(self._pending)
            self._pending.clear()
            self._by_team.clear()
            return removed
