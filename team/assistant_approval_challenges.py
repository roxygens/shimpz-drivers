"""One-use, Team-bound continuations for explicit Assistant Power approval."""

from __future__ import annotations

import re
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any

MAX_PENDING_CHALLENGES = 32
DEFAULT_TTL_SECONDS = 300
_CHALLENGE_ID = re.compile(r"[0-9a-f]{32}")
_TEAM_ID = re.compile(r"[a-z0-9_]{1,40}")


class ApprovalChallengeError(RuntimeError):
    """A pending approval continuation is unavailable or conflicts."""


class ApprovalChallengeNotFoundError(ApprovalChallengeError):
    """The opaque challenge is unknown, expired, consumed, or belongs to another Team."""


@dataclass(frozen=True, slots=True)
class ApprovalRequirement:
    interrupt_id: str
    assistant_id: str
    assistant_name: str
    power_id: str
    power_summary: str
    input_json: str
    approval: str
    assistant_image: str


@dataclass(frozen=True, slots=True)
class PendingApprovalChallenge:
    id: str
    team_id: str
    expires_at: float
    requirements: tuple[ApprovalRequirement, ...]
    payload: Any


def _team_id(value: object) -> str:
    if not isinstance(value, str) or _TEAM_ID.fullmatch(value) is None:
        raise ApprovalChallengeError("Team id is invalid")
    return value


def _challenge_id(value: object) -> str:
    if not isinstance(value, str) or _CHALLENGE_ID.fullmatch(value) is None:
        raise ApprovalChallengeNotFoundError("approval challenge is unavailable")
    return value


class ApprovalChallengeStore:
    """Keep approval grants memory-only and consume them before continuation."""

    def __init__(self, *, capacity: int = MAX_PENDING_CHALLENGES, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        if type(capacity) is not int or not 1 <= capacity <= 1024:
            raise ValueError("approval challenge capacity is invalid")
        if type(ttl_seconds) is not int or not 30 <= ttl_seconds <= 900:
            raise ValueError("approval challenge TTL is invalid")
        self._capacity = capacity
        self._ttl = ttl_seconds
        self._pending: dict[str, PendingApprovalChallenge] = {}
        self._by_team: dict[str, str] = {}
        self._lock = threading.Lock()

    def _expire(self, now: float) -> None:
        expired = [identifier for identifier, item in self._pending.items() if item.expires_at <= now]
        for identifier in expired:
            challenge = self._pending.pop(identifier)
            if self._by_team.get(challenge.team_id) == identifier:
                self._by_team.pop(challenge.team_id, None)

    def create(
        self,
        team_id: object,
        requirements: tuple[ApprovalRequirement, ...],
        payload: Any,
    ) -> PendingApprovalChallenge:
        team = _team_id(team_id)
        if not requirements:
            raise ApprovalChallengeError("approval challenge requires metadata")
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            if team in self._by_team:
                raise ApprovalChallengeError("Team already has a pending approval challenge")
            if len(self._pending) >= self._capacity:
                raise ApprovalChallengeError("approval challenge capacity reached")
            identifier = secrets.token_hex(16)
            while identifier in self._pending:
                identifier = secrets.token_hex(16)
            challenge = PendingApprovalChallenge(
                id=identifier,
                team_id=team,
                expires_at=now + self._ttl,
                requirements=requirements,
                payload=payload,
            )
            self._pending[identifier] = challenge
            self._by_team[team] = identifier
            return challenge

    def get(self, team_id: object, challenge_id: object) -> PendingApprovalChallenge:
        team = _team_id(team_id)
        identifier = _challenge_id(challenge_id)
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            challenge = self._pending.get(identifier)
            if challenge is None or challenge.team_id != team:
                raise ApprovalChallengeNotFoundError("approval challenge is unavailable")
            return challenge

    def current(self, team_id: object) -> PendingApprovalChallenge | None:
        team = _team_id(team_id)
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            identifier = self._by_team.get(team)
            return self._pending.get(identifier) if identifier is not None else None

    def claim(self, team_id: object, challenge_id: object) -> PendingApprovalChallenge:
        team = _team_id(team_id)
        identifier = _challenge_id(challenge_id)
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            challenge = self._pending.get(identifier)
            if challenge is None or challenge.team_id != team:
                raise ApprovalChallengeNotFoundError("approval challenge is unavailable")
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
        with self._lock:
            removed = len(self._pending)
            self._pending.clear()
            self._by_team.clear()
            return removed
