"""Bounded, one-use OAuth Authorization Code + PKCE continuations.

The browser receives only ``state`` and the S256 challenge. The verifier stays
process-local and is released once, after the callback proves the same session,
Team, Assistant, and connection binding that started the flow.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import threading
import time
from dataclasses import dataclass

import oauth_providers

DEFAULT_TTL_SECONDS = 300
MAX_PENDING_CHALLENGES = 128
MAX_PENDING_PER_SESSION = 4
MAX_PENDING_PER_TEAM = 16
_STATE = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_COMPONENT_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_TEAM_ID = re.compile(r"[a-z0-9_]{1,40}\Z")


class OAuthChallengeError(RuntimeError):
    """A PKCE challenge could not be created or safely managed."""


class OAuthChallengeNotFoundError(OAuthChallengeError):
    """The challenge expired, was consumed, or does not match its binding."""


@dataclass(frozen=True, slots=True)
class OAuthAuthorizationChallenge:
    """Public values safe to place in the provider authorization request."""

    state: str
    provider_id: str
    scopes: tuple[str, ...]
    code_challenge: str
    code_challenge_method: str
    expires_in: int


@dataclass(frozen=True, slots=True)
class OAuthExchange:
    """Private values returned only after one exact callback claim."""

    provider_id: str
    scopes: tuple[str, ...]
    code_verifier: str


@dataclass(frozen=True, slots=True)
class _PendingChallenge:
    public: OAuthAuthorizationChallenge
    session_digest: bytes
    team_id: str
    assistant_id: str
    connection_id: str
    code_verifier: str
    expires_at: float


def _session_digest(value: object) -> bytes:
    if isinstance(value, str):
        try:
            encoded = value.encode("ascii")
        except UnicodeError as exc:
            raise OAuthChallengeError("OAuth session binding is invalid") from exc
    elif isinstance(value, bytes):
        encoded = value
    else:
        raise OAuthChallengeError("OAuth session binding is invalid")
    if not 16 <= len(encoded) <= 2048 or any(byte <= 32 or byte >= 127 for byte in encoded):
        raise OAuthChallengeError("OAuth session binding is invalid")
    return hashlib.sha256(b"shimpz-oauth-session-v1\0" + encoded).digest()


def _team_id(value: object) -> str:
    if not isinstance(value, str) or _TEAM_ID.fullmatch(value) is None:
        raise OAuthChallengeError("OAuth Team binding is invalid")
    return value


def _component_id(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) > 64 or _COMPONENT_ID.fullmatch(value) is None:
        raise OAuthChallengeError(f"OAuth {label} binding is invalid")
    return value


def _state(value: object) -> str:
    if not isinstance(value, str) or _STATE.fullmatch(value) is None:
        raise OAuthChallengeNotFoundError("OAuth challenge is unavailable")
    return value


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


class OAuthPKCEChallengeStore:
    """Keep PKCE verifiers memory-only, bounded, short-lived, and one-use."""

    def __init__(
        self,
        *,
        capacity: int = MAX_PENDING_CHALLENGES,
        per_session: int = MAX_PENDING_PER_SESSION,
        per_team: int = MAX_PENDING_PER_TEAM,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        if type(capacity) is not int or not 1 <= capacity <= 1024:
            raise ValueError("OAuth challenge capacity is invalid")
        if type(per_session) is not int or not 1 <= per_session <= capacity:
            raise ValueError("OAuth per-session challenge capacity is invalid")
        if type(per_team) is not int or not 1 <= per_team <= capacity:
            raise ValueError("OAuth per-Team challenge capacity is invalid")
        if type(ttl_seconds) is not int or not 30 <= ttl_seconds <= 900:
            raise ValueError("OAuth challenge TTL is invalid")
        self._capacity = capacity
        self._per_session = per_session
        self._per_team = per_team
        self._ttl = ttl_seconds
        self._pending: dict[str, _PendingChallenge] = {}
        self._by_binding: dict[tuple[bytes, str, str, str], str] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _binding(
        session_binding: object,
        team_id: object,
        assistant_id: object,
        connection_id: object,
    ) -> tuple[bytes, str, str, str]:
        return (
            _session_digest(session_binding),
            _team_id(team_id),
            _component_id(assistant_id, "Assistant"),
            _component_id(connection_id, "connection"),
        )

    def _remove(self, state: str) -> _PendingChallenge | None:
        challenge = self._pending.pop(state, None)
        if challenge is not None:
            binding = (
                challenge.session_digest,
                challenge.team_id,
                challenge.assistant_id,
                challenge.connection_id,
            )
            if self._by_binding.get(binding) == state:
                self._by_binding.pop(binding, None)
        return challenge

    def _expire(self, now: float) -> None:
        for state in tuple(
            state for state, challenge in self._pending.items() if challenge.expires_at <= now
        ):
            self._remove(state)

    def create(
        self,
        *,
        session_binding: object,
        team_id: object,
        assistant_id: object,
        connection_id: object,
        provider_id: object,
        scopes: object,
    ) -> OAuthAuthorizationChallenge:
        binding = self._binding(session_binding, team_id, assistant_id, connection_id)
        intent = oauth_providers.connection_intent(provider_id, scopes)
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            if binding in self._by_binding:
                raise OAuthChallengeError("OAuth connection already has a pending challenge")
            if len(self._pending) >= self._capacity:
                raise OAuthChallengeError("OAuth challenge capacity reached")
            session_count = sum(
                hmac.compare_digest(item.session_digest, binding[0]) for item in self._pending.values()
            )
            team_count = sum(item.team_id == binding[1] for item in self._pending.values())
            if session_count >= self._per_session:
                raise OAuthChallengeError("OAuth session challenge capacity reached")
            if team_count >= self._per_team:
                raise OAuthChallengeError("OAuth Team challenge capacity reached")

            state = secrets.token_urlsafe(32)
            while state in self._pending:
                state = secrets.token_urlsafe(32)
            verifier = secrets.token_urlsafe(64)
            code_challenge = _base64url(hashlib.sha256(verifier.encode("ascii")).digest())
            public = OAuthAuthorizationChallenge(
                state=state,
                provider_id=intent.provider.id,
                scopes=intent.scopes,
                code_challenge=code_challenge,
                code_challenge_method=intent.provider.pkce_method,
                expires_in=self._ttl,
            )
            self._pending[state] = _PendingChallenge(
                public=public,
                session_digest=binding[0],
                team_id=binding[1],
                assistant_id=binding[2],
                connection_id=binding[3],
                code_verifier=verifier,
                expires_at=now + self._ttl,
            )
            self._by_binding[binding] = state
            return public

    def claim(
        self,
        *,
        state: object,
        session_binding: object,
        team_id: object,
        assistant_id: object,
        connection_id: object,
    ) -> OAuthExchange:
        identifier = _state(state)
        binding = self._binding(session_binding, team_id, assistant_id, connection_id)
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            challenge = self._pending.get(identifier)
            if challenge is None:
                raise OAuthChallengeNotFoundError("OAuth challenge is unavailable")
            if (
                not hmac.compare_digest(challenge.session_digest, binding[0])
                or challenge.team_id != binding[1]
                or challenge.assistant_id != binding[2]
                or challenge.connection_id != binding[3]
            ):
                raise OAuthChallengeNotFoundError("OAuth challenge is unavailable")
            self._remove(identifier)
            return OAuthExchange(
                provider_id=challenge.public.provider_id,
                scopes=challenge.public.scopes,
                code_verifier=challenge.code_verifier,
            )

    def cancel_session(self, session_binding: object) -> int:
        digest = _session_digest(session_binding)
        with self._lock:
            states = tuple(
                state
                for state, challenge in self._pending.items()
                if hmac.compare_digest(challenge.session_digest, digest)
            )
            for state in states:
                self._remove(state)
            return len(states)

    def cancel_team(self, team_id: object) -> int:
        team = _team_id(team_id)
        with self._lock:
            states = tuple(
                state for state, challenge in self._pending.items() if challenge.team_id == team
            )
            for state in states:
                self._remove(state)
            return len(states)

    def cancel_all(self) -> int:
        with self._lock:
            removed = len(self._pending)
            self._pending.clear()
            self._by_binding.clear()
            return removed
