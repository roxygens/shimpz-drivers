"""Bounded, process-local continuations for just-in-time OAuth accounts.

Pending Power input remains memory-only. Provider authorization has its own
session-bound PKCE state; this challenge only preserves the paused Team turn and
the public account requirements needed to resume it once.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any

MAX_PENDING_CHALLENGES = 32
DEFAULT_TTL_SECONDS = 300
_CHALLENGE_ID = re.compile(r"[0-9a-f]{32}\Z")
_TEAM_ID = re.compile(r"[a-z0-9_]{1,40}\Z")


class AccountChallengeError(RuntimeError):
    """A pending account continuation is unavailable or conflicts."""


class AccountChallengeNotFoundError(AccountChallengeError):
    """The opaque challenge expired, was consumed, or belongs to another Team."""


@dataclass(frozen=True, slots=True)
class AccountRequirement:
    assistant_id: str
    assistant_name: str
    power_ids: tuple[str, ...]
    accounts: tuple[tuple[str, str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class PendingAccountChallenge:
    id: str
    team_id: str
    expires_at: float
    requirements: tuple[AccountRequirement, ...]
    payload: Any


def _team_id(value: object) -> str:
    if not isinstance(value, str) or _TEAM_ID.fullmatch(value) is None:
        raise AccountChallengeError("Team id is invalid")
    return value


def _challenge_id(value: object) -> str:
    if not isinstance(value, str) or _CHALLENGE_ID.fullmatch(value) is None:
        raise AccountChallengeNotFoundError("account challenge is unavailable")
    return value


class AccountChallengeStore:
    """Keep one short-lived, one-use paused account turn per Team."""

    def __init__(
        self,
        *,
        capacity: int = MAX_PENDING_CHALLENGES,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        if type(capacity) is not int or not 1 <= capacity <= 1024:
            raise ValueError("account challenge capacity is invalid")
        if type(ttl_seconds) is not int or not 30 <= ttl_seconds <= 900:
            raise ValueError("account challenge TTL is invalid")
        self._capacity = capacity
        self._ttl = ttl_seconds
        self._pending: dict[str, PendingAccountChallenge] = {}
        self._by_team: dict[str, str] = {}
        self._lock = threading.Lock()

    def _expire(self, now: float) -> None:
        for identifier in tuple(
            identifier for identifier, challenge in self._pending.items() if challenge.expires_at <= now
        ):
            challenge = self._pending.pop(identifier)
            if self._by_team.get(challenge.team_id) == identifier:
                self._by_team.pop(challenge.team_id, None)

    def create(
        self,
        team_id: object,
        requirements: tuple[AccountRequirement, ...],
        payload: Any,
    ) -> PendingAccountChallenge:
        team = _team_id(team_id)
        if not requirements:
            raise AccountChallengeError("account challenge requires metadata")
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            if team in self._by_team:
                raise AccountChallengeError("Team already has a pending account challenge")
            if len(self._pending) >= self._capacity:
                raise AccountChallengeError("account challenge capacity reached")
            identifier = secrets.token_hex(16)
            while identifier in self._pending:
                identifier = secrets.token_hex(16)
            challenge = PendingAccountChallenge(
                id=identifier,
                team_id=team,
                expires_at=now + self._ttl,
                requirements=requirements,
                payload=payload,
            )
            self._pending[identifier] = challenge
            self._by_team[team] = identifier
            return challenge

    def get(self, team_id: object, challenge_id: object) -> PendingAccountChallenge:
        team = _team_id(team_id)
        identifier = _challenge_id(challenge_id)
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            challenge = self._pending.get(identifier)
            if challenge is None or challenge.team_id != team:
                raise AccountChallengeNotFoundError("account challenge is unavailable")
            return challenge

    def restore(
        self,
        team_id: object,
        challenge_id: object,
        remaining_seconds: object,
        requirements: tuple[AccountRequirement, ...],
        payload: Any,
    ) -> PendingAccountChallenge:
        """Rehydrate one authenticated durable challenge without extending its TTL."""
        team = _team_id(team_id)
        identifier = _challenge_id(challenge_id)
        if type(remaining_seconds) is not int or not 1 <= remaining_seconds <= self._ttl or not requirements:
            raise AccountChallengeError("account challenge restore is invalid")
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            if team in self._by_team or identifier in self._pending:
                raise AccountChallengeError("Team already has a pending account challenge")
            if len(self._pending) >= self._capacity:
                raise AccountChallengeError("account challenge capacity reached")
            challenge = PendingAccountChallenge(
                identifier,
                team,
                now + remaining_seconds,
                requirements,
                payload,
            )
            self._pending[identifier] = challenge
            self._by_team[team] = identifier
            return challenge

    def current(self, team_id: object) -> PendingAccountChallenge | None:
        team = _team_id(team_id)
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            identifier = self._by_team.get(team)
            return self._pending.get(identifier) if identifier is not None else None

    def claim(self, team_id: object, challenge_id: object) -> PendingAccountChallenge:
        team = _team_id(team_id)
        identifier = _challenge_id(challenge_id)
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            challenge = self._pending.get(identifier)
            if challenge is None or challenge.team_id != team:
                raise AccountChallengeNotFoundError("account challenge is unavailable")
            self._pending.pop(identifier)
            if self._by_team.get(team) == identifier:
                self._by_team.pop(team, None)
            return challenge

    def cancel_team(self, team_id: object) -> bool:
        team = _team_id(team_id)
        with self._lock:
            identifier = self._by_team.pop(team, None)
            return self._pending.pop(identifier, None) is not None if identifier else False

    def cancel_all(self) -> int:
        with self._lock:
            removed = len(self._pending)
            self._pending.clear()
            self._by_team.clear()
            return removed
