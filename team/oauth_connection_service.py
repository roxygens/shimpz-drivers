"""Narrow controller-owned orchestration for Assistant OAuth connections.

This module composes the one-use PKCE challenge store, the fixed-endpoint OAuth
HTTP adapter, and the encrypted token store.  It deliberately owns no routes,
cookies, browser state, Assistant runtime calls, or Brain-visible data.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import assistant_connection_challenges
import oauth_connection_store
import oauth_http_client
import oauth_pkce_challenges
import oauth_providers

_CLIENT_ID = re.compile(r"[A-Za-z0-9._~-]{8,256}\Z")
_COMPONENT_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_TEAM_ID = re.compile(r"[a-z0-9_]{1,40}\Z")
_PENDING_ID = re.compile(r"[0-9a-f]{32}\Z")
_REDIRECT_URIS = frozenset(
    {
        oauth_http_client.LOCAL_REDIRECT_URI,
        oauth_http_client.HOSTED_REDIRECT_URI,
    }
)
MAX_REQUIREMENTS = 32
MAX_CONNECTIONS_PER_REQUIREMENT = 16


class OAuthConnectionServiceError(RuntimeError):
    """An OAuth connection could not be started or safely completed."""


class OAuthConnectionUnavailableError(OAuthConnectionServiceError):
    """No pending connection currently requires provider authorization."""


@dataclass(frozen=True, slots=True)
class OAuthConnectionCompletion:
    """Public completion identifiers; no authorization material is retained."""

    team_id: str
    assistant_id: str
    connection_id: str
    provider: str
    scopes: tuple[str, ...]
    generation: int


@dataclass(frozen=True, slots=True)
class _Candidate:
    team_id: str
    assistant_id: str
    connection_id: str
    provider: str
    scopes: tuple[str, ...]


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) > 64 or _COMPONENT_ID.fullmatch(value) is None:
        raise OAuthConnectionServiceError(f"pending OAuth {label} is unavailable")
    return value


def _declaration(value: object) -> tuple[str, tuple[str, ...]]:
    if isinstance(value, Mapping) and set(value) == {"provider", "scopes"}:
        provider = value.get("provider")
        scopes = value.get("scopes")
    else:
        try:
            provider = value.provider  # type: ignore[attr-defined]
            scopes = value.scopes  # type: ignore[attr-defined]
        except (AttributeError, TypeError) as exc:
            raise OAuthConnectionServiceError("OAuth connection declaration is unavailable") from exc
    try:
        intent = oauth_providers.connection_intent(provider, scopes)
    except oauth_providers.OAuthProviderError as exc:
        raise OAuthConnectionServiceError("OAuth connection declaration is unavailable") from exc
    return intent.provider.id, intent.scopes


def _candidates(
    pending: object,
) -> tuple[_Candidate, ...]:
    if not isinstance(pending, assistant_connection_challenges.PendingConnectionChallenge):
        raise OAuthConnectionServiceError("pending OAuth connection is unavailable")
    if (
        not isinstance(pending.requirements, tuple)
        or not 1 <= len(pending.requirements) <= MAX_REQUIREMENTS
        or not isinstance(pending.team_id, str)
        or _TEAM_ID.fullmatch(pending.team_id) is None
        or not isinstance(pending.id, str)
        or _PENDING_ID.fullmatch(pending.id) is None
        or not isinstance(pending.expires_at, int | float)
        or isinstance(pending.expires_at, bool)
        or pending.expires_at <= time.monotonic()
    ):
        raise OAuthConnectionServiceError("pending OAuth connection is unavailable")
    candidates: list[_Candidate] = []
    seen: set[tuple[str, str]] = set()
    for requirement in pending.requirements:
        if (
            not isinstance(requirement, assistant_connection_challenges.ConnectionRequirement)
            or not isinstance(requirement.connections, tuple)
            or not 1 <= len(requirement.connections) <= MAX_CONNECTIONS_PER_REQUIREMENT
        ):
            raise OAuthConnectionServiceError("pending OAuth connection is unavailable")
        assistant_id = _identifier(requirement.assistant_id, "Assistant")
        for raw_connection in requirement.connections:
            if not isinstance(raw_connection, tuple) or len(raw_connection) != 3:
                raise OAuthConnectionServiceError("pending OAuth connection is unavailable")
            connection_id, raw_provider, raw_scopes = raw_connection
            connection_id = _identifier(connection_id, "connection")
            provider, scopes = _declaration({"provider": raw_provider, "scopes": raw_scopes})
            binding = (assistant_id, connection_id)
            if binding in seen:
                raise OAuthConnectionServiceError("pending OAuth connection is unavailable")
            seen.add(binding)
            candidates.append(
                _Candidate(
                    team_id=pending.team_id,
                    assistant_id=assistant_id,
                    connection_id=connection_id,
                    provider=provider,
                    scopes=scopes,
                )
            )
    return tuple(sorted(candidates, key=lambda item: (item.assistant_id, item.connection_id)))


class OAuthConnectionService:
    """Start and complete only controller-reviewed OAuth Authorization Code flows."""

    def __init__(
        self,
        *,
        client_id: object,
        redirect_uri: object,
        challenge: oauth_pkce_challenges.OAuthPKCEChallengeStore,
        store: oauth_connection_store.OAuthConnectionStore,
        http: oauth_http_client.OAuthHTTPClient,
    ) -> None:
        if (
            not isinstance(challenge, oauth_pkce_challenges.OAuthPKCEChallengeStore)
            or not isinstance(store, oauth_connection_store.OAuthConnectionStore)
            or not isinstance(http, oauth_http_client.OAuthHTTPClient)
            or redirect_uri not in _REDIRECT_URIS
        ):
            raise OAuthConnectionServiceError("OAuth connection service configuration is invalid")
        # A self-hosted Admin may boot before its public X client id is configured.
        # Validation is deliberately lazy so only starting/completing OAuth fails.
        self._client_id = client_id
        self._redirect_uri = str(redirect_uri)
        self._challenge = challenge
        self._store = store
        self._http = http

    def __repr__(self) -> str:
        return "<OAuthConnectionService configured>"

    def _client_configuration(self) -> tuple[str, str]:
        if not isinstance(self._client_id, str) or _CLIENT_ID.fullmatch(self._client_id) is None:
            raise OAuthConnectionServiceError("OAuth connection client is not configured")
        return self._client_id, self._redirect_uri

    def authorization_url(
        self,
        pending: assistant_connection_challenges.PendingConnectionChallenge,
        session_binding: object,
    ) -> str:
        """Create one trusted URL for the first deterministic missing connection."""
        client_id, redirect_uri = self._client_configuration()
        try:
            candidates = _candidates(pending)
            metadata_by_binding: dict[
                tuple[str, str],
                oauth_connection_store.OAuthConnectionMetadata,
            ] = {}
            by_assistant: dict[str, dict[str, dict[str, object]]] = {}
            for candidate in candidates:
                by_assistant.setdefault(candidate.assistant_id, {})[candidate.connection_id] = {
                    "provider": candidate.provider,
                    "scopes": candidate.scopes,
                }
            for assistant_id, declarations in by_assistant.items():
                for item in self._store.metadata(pending.team_id, assistant_id, declarations):
                    metadata_by_binding[(assistant_id, item.id)] = item
            selected = next(
                (
                    candidate
                    for candidate in candidates
                    if metadata_by_binding[(candidate.assistant_id, candidate.connection_id)].status
                    in {"missing", "refresh-required", "reauthorization-required"}
                ),
                None,
            )
            if selected is None:
                raise OAuthConnectionUnavailableError("all pending OAuth connections are already configured")
            public = self._challenge.create(
                session_binding=session_binding,
                team_id=selected.team_id,
                assistant_id=selected.assistant_id,
                connection_id=selected.connection_id,
                provider_id=selected.provider,
                scopes=selected.scopes,
            )
            return oauth_http_client.authorization_url(
                provider_id=public.provider_id,
                client_id=client_id,
                redirect_uri=redirect_uri,
                state=public.state,
                code_challenge=public.code_challenge,
                scopes=public.scopes,
            )
        except OAuthConnectionUnavailableError:
            raise
        except (
            assistant_connection_challenges.ConnectionChallengeError,
            oauth_connection_store.OAuthConnectionStoreError,
            oauth_http_client.OAuthHTTPError,
            oauth_pkce_challenges.OAuthChallengeError,
            oauth_providers.OAuthProviderError,
            OAuthConnectionServiceError,
            KeyError,
            TypeError,
        ):
            raise OAuthConnectionServiceError("OAuth connection could not be started") from None

    def complete(
        self,
        state: object,
        code: object,
        session_binding: object,
        current_declaration_callback: Callable[[str, str, str], object],
    ) -> OAuthConnectionCompletion:
        """Claim once, revalidate the installed declaration, exchange, and seal tokens."""
        client_id, redirect_uri = self._client_configuration()
        if not callable(current_declaration_callback):
            raise OAuthConnectionServiceError("OAuth declaration resolver is unavailable")
        try:
            exchange = self._challenge.claim_callback(
                state=state,
                session_binding=session_binding,
            )
            try:
                current = current_declaration_callback(
                    exchange.team_id,
                    exchange.assistant_id,
                    exchange.connection_id,
                )
            except Exception:  # noqa: BLE001 -- redact every registry/parser failure at this boundary
                raise OAuthConnectionServiceError("OAuth connection declaration is unavailable") from None
            provider, scopes = _declaration(current)
            if provider != exchange.provider_id or scopes != exchange.scopes:
                raise OAuthConnectionServiceError("OAuth connection declaration changed")
            token_set = self._http.exchange_code(
                provider_id=provider,
                client_id=client_id,
                redirect_uri=redirect_uri,
                code=code,
                code_verifier=exchange.code_verifier,
                scopes=scopes,
            )
            metadata = self._store.put(
                exchange.team_id,
                exchange.assistant_id,
                exchange.connection_id,
                provider,
                scopes,
                token_set,
                None,
            )
            return OAuthConnectionCompletion(
                team_id=exchange.team_id,
                assistant_id=exchange.assistant_id,
                connection_id=exchange.connection_id,
                provider=metadata.provider,
                scopes=metadata.scopes,
                generation=metadata.generation,
            )
        except (
            oauth_connection_store.OAuthConnectionStoreError,
            oauth_http_client.OAuthHTTPError,
            oauth_pkce_challenges.OAuthChallengeError,
            oauth_providers.OAuthProviderError,
            OAuthConnectionServiceError,
        ):
            raise OAuthConnectionServiceError("OAuth connection could not be completed") from None

    def disconnect(self, team_id: object, assistant_id: object, connection_id: object) -> bool:
        """Delete local tokens; no network revoke or new decryption surface is introduced."""
        try:
            return self._store.delete_connection(team_id, assistant_id, connection_id)
        except oauth_connection_store.OAuthConnectionStoreError:
            raise OAuthConnectionServiceError("OAuth connection could not be disconnected") from None
