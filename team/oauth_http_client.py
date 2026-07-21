"""Strict OAuth 2.0 HTTP adapter for core-owned clients.

Only the trusted provider registry may choose endpoints. Assistant manifests,
browser input, and Power input cannot supply URLs, client credentials, or token
response shapes. Redirects are deliberately not followed.
"""

from __future__ import annotations

import http.client
import json
import re
from base64 import b64encode
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode, urlsplit

import oauth_providers

MAX_RESPONSE_BYTES = 32 * 1024
MAX_TOKEN_BYTES = 16 * 1024
HTTP_TIMEOUT_SECONDS = 10
LOCAL_REDIRECT_URI = "http://127.0.0.1:7777/api/oauth/cloudflare/callback"
HOSTED_REDIRECT_URI = "https://shimpz.com/api/oauth/cloudflare/callback"
_CLIENT_ID = re.compile(r"[A-Za-z0-9._~-]{8,256}\Z")
_STATE = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_PKCE = re.compile(r"[A-Za-z0-9_-]{43}\Z")
MAX_CLIENT_SECRET_BYTES = 1024


class OAuthHTTPError(RuntimeError):
    """The trusted OAuth exchange failed without reflecting provider data."""


@dataclass(frozen=True, slots=True)
class OAuthHTTPResponse:
    status: int
    content_type: str
    body: bytes


@dataclass(frozen=True, slots=True)
class OAuthTokenSet:
    access_token: str
    refresh_token: str | None
    scopes: tuple[str, ...]
    expires_in: int
    broker_lease: str | None = None


class OAuthTransport(Protocol):
    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> OAuthHTTPResponse: ...


class FixedHTTPSTransport:
    """Send one bounded HTTPS request without proxy or redirect behavior."""

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> OAuthHTTPResponse:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
            or parsed.query
            or parsed.fragment
        ):
            raise OAuthHTTPError("OAuth provider endpoint is invalid")
        connection = http.client.HTTPSConnection(parsed.hostname, timeout=HTTP_TIMEOUT_SECONDS)
        try:
            connection.request(method, parsed.path, body=body, headers=dict(headers))
            response = connection.getresponse()
            payload = response.read(MAX_RESPONSE_BYTES + 1)
            if len(payload) > MAX_RESPONSE_BYTES:
                raise OAuthHTTPError("OAuth provider response is invalid")
            return OAuthHTTPResponse(
                status=response.status,
                content_type=response.getheader("Content-Type", ""),
                body=payload,
            )
        except OAuthHTTPError:
            raise
        except (OSError, http.client.HTTPException) as exc:
            raise OAuthHTTPError("OAuth provider is unavailable") from exc
        finally:
            connection.close()


def _client_id(value: object) -> str:
    if not isinstance(value, str) or _CLIENT_ID.fullmatch(value) is None:
        raise OAuthHTTPError("OAuth client configuration is invalid")
    return value


def _redirect_uri(provider_id: str, value: object) -> str:
    if provider_id != "cloudflare" or value not in {LOCAL_REDIRECT_URI, HOSTED_REDIRECT_URI}:
        raise OAuthHTTPError("OAuth callback configuration is invalid")
    return str(value)


def _client_secret(value: object) -> str:
    if not isinstance(value, str):
        raise OAuthHTTPError("OAuth client configuration is invalid")
    try:
        encoded = value.encode("ascii")
    except UnicodeError as exc:
        raise OAuthHTTPError("OAuth client configuration is invalid") from exc
    if not 16 <= len(encoded) <= MAX_CLIENT_SECRET_BYTES or any(byte <= 32 or byte >= 127 for byte in encoded):
        raise OAuthHTTPError("OAuth client configuration is invalid")
    return value


def _authorization_code(value: object) -> str:
    if not isinstance(value, str):
        raise OAuthHTTPError("OAuth authorization response is invalid")
    try:
        encoded = value.encode("ascii")
    except UnicodeError as exc:
        raise OAuthHTTPError("OAuth authorization response is invalid") from exc
    if not 16 <= len(encoded) <= 4096 or any(byte <= 32 or byte >= 127 for byte in encoded):
        raise OAuthHTTPError("OAuth authorization response is invalid")
    return value


def _token(value: object) -> str:
    if not isinstance(value, str):
        raise OAuthHTTPError("OAuth provider response is invalid")
    try:
        encoded = value.encode("ascii")
    except UnicodeError as exc:
        raise OAuthHTTPError("OAuth provider response is invalid") from exc
    if not 16 <= len(encoded) <= MAX_TOKEN_BYTES or any(byte <= 32 or byte >= 127 for byte in encoded):
        raise OAuthHTTPError("OAuth provider response is invalid")
    return value


def _strict_object(payload: bytes) -> dict[str, object]:
    if not payload or len(payload) > MAX_RESPONSE_BYTES:
        raise OAuthHTTPError("OAuth provider response is invalid")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise OAuthHTTPError("OAuth provider response is invalid")
            result[key] = value
        return result

    try:
        decoded = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OAuthHTTPError("OAuth provider response is invalid") from exc
    if not isinstance(decoded, dict):
        raise OAuthHTTPError("OAuth provider response is invalid")
    return decoded


def _confidential_provider(provider_id: object) -> oauth_providers.OAuthProvider:
    try:
        provider = oauth_providers.resolve(provider_id)
    except oauth_providers.OAuthProviderError as exc:
        raise OAuthHTTPError("OAuth provider is unavailable") from exc
    if provider.client_auth_method != "client_secret_basic" or provider.pkce_method != "S256":
        raise OAuthHTTPError("OAuth provider configuration is invalid")
    return provider


def authorization_url(
    *,
    provider_id: object,
    client_id: object,
    redirect_uri: object,
    state: object,
    code_challenge: object,
    scopes: object,
) -> str:
    provider = _confidential_provider(provider_id)
    client = _client_id(client_id)
    redirect = _redirect_uri(provider.id, redirect_uri)
    if not isinstance(state, str) or _STATE.fullmatch(state) is None:
        raise OAuthHTTPError("OAuth challenge is invalid")
    if not isinstance(code_challenge, str) or _PKCE.fullmatch(code_challenge) is None:
        raise OAuthHTTPError("OAuth challenge is invalid")
    try:
        intent = oauth_providers.account_intent(provider.id, scopes)
    except oauth_providers.OAuthProviderError as exc:
        raise OAuthHTTPError("OAuth scopes are invalid") from exc
    query = urlencode(
        {
            "response_type": "code",
            "client_id": client,
            "redirect_uri": redirect,
            "scope": " ".join(intent.scopes),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
    )
    return f"{provider.authorization_endpoint}?{query}"


class OAuthHTTPClient:
    def __init__(self, transport: OAuthTransport | None = None) -> None:
        self._transport = transport or FixedHTTPSTransport()

    def _post(
        self,
        url: str,
        fields: Mapping[str, str],
        *,
        client_id: object,
        client_secret: object,
    ) -> OAuthHTTPResponse:
        payload = urlencode(fields).encode("ascii")
        client = _client_id(client_id)
        secret = _client_secret(client_secret)
        authorization = b64encode(f"{client}:{secret}".encode("ascii")).decode("ascii")
        response = self._transport.request(
            method="POST",
            url=url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Basic {authorization}",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "shimpz-team-controller/1",
            },
            body=payload,
        )
        if not 200 <= response.status < 300:
            raise OAuthHTTPError("OAuth provider request failed")
        if len(response.body) > MAX_RESPONSE_BYTES:
            raise OAuthHTTPError("OAuth provider response is invalid")
        return response

    @staticmethod
    def _tokens(
        response: OAuthHTTPResponse,
        *,
        expected_scopes: tuple[str, ...],
        previous_refresh_token: str | None = None,
    ) -> OAuthTokenSet:
        if not response.content_type.lower().split(";", 1)[0].strip() == "application/json":
            raise OAuthHTTPError("OAuth provider response is invalid")
        data = _strict_object(response.body)
        if not set(data) <= {"access_token", "refresh_token", "token_type", "expires_in", "scope"}:
            raise OAuthHTTPError("OAuth provider response is invalid")
        token_type = data.get("token_type")
        expires_in = data.get("expires_in")
        if (
            not isinstance(token_type, str)
            or token_type.lower() != "bearer"
            or type(expires_in) is not int
            or not 30 <= expires_in <= 31_536_000
        ):
            raise OAuthHTTPError("OAuth provider response is invalid")
        raw_scope = data.get("scope")
        if raw_scope is None:
            scopes = expected_scopes
        elif isinstance(raw_scope, str):
            scopes = tuple(sorted(raw_scope.split()))
        else:
            raise OAuthHTTPError("OAuth provider response is invalid")
        if not scopes or len(scopes) != len(set(scopes)) or scopes != tuple(sorted(expected_scopes)):
            raise OAuthHTTPError("OAuth provider response is invalid")
        refresh = data.get("refresh_token", previous_refresh_token)
        if refresh is not None:
            refresh = _token(refresh)
        if "offline_access" in scopes and refresh is None:
            raise OAuthHTTPError("OAuth provider response is invalid")
        return OAuthTokenSet(
            access_token=_token(data.get("access_token")),
            refresh_token=refresh,
            scopes=scopes,
            expires_in=expires_in,
        )

    def exchange_code(
        self,
        *,
        provider_id: object,
        client_id: object,
        client_secret: object,
        redirect_uri: object,
        code: object,
        code_verifier: object,
        scopes: object,
    ) -> OAuthTokenSet:
        provider = _confidential_provider(provider_id)
        client = _client_id(client_id)
        redirect = _redirect_uri(provider.id, redirect_uri)
        verifier = _authorization_code(code_verifier)
        if not 43 <= len(verifier) <= 128:
            raise OAuthHTTPError("OAuth challenge is invalid")
        try:
            expected_scopes = oauth_providers.account_intent(provider.id, scopes).scopes
        except oauth_providers.OAuthProviderError as exc:
            raise OAuthHTTPError("OAuth scopes are invalid") from exc
        response = self._post(
            provider.token_endpoint,
            {
                "code": _authorization_code(code),
                "grant_type": "authorization_code",
                "redirect_uri": redirect,
                "code_verifier": verifier,
            },
            client_id=client,
            client_secret=client_secret,
        )
        return self._tokens(response, expected_scopes=expected_scopes)

    def refresh(
        self,
        *,
        provider_id: object,
        client_id: object,
        client_secret: object,
        refresh_token: object,
        scopes: object,
    ) -> OAuthTokenSet:
        provider = _confidential_provider(provider_id)
        client = _client_id(client_id)
        previous = _token(refresh_token)
        try:
            expected_scopes = oauth_providers.account_intent(provider.id, scopes).scopes
        except oauth_providers.OAuthProviderError as exc:
            raise OAuthHTTPError("OAuth scopes are invalid") from exc
        response = self._post(
            provider.token_endpoint,
            {
                "refresh_token": previous,
                "grant_type": "refresh_token",
            },
            client_id=client,
            client_secret=client_secret,
        )
        return self._tokens(
            response,
            expected_scopes=expected_scopes,
            previous_refresh_token=previous,
        )

    def revoke(
        self,
        *,
        provider_id: object,
        client_id: object,
        client_secret: object,
        token: object,
    ) -> None:
        provider = _confidential_provider(provider_id)
        response = self._post(
            provider.revocation_endpoint,
            {"token": _token(token)},
            client_id=client_id,
            client_secret=client_secret,
        )
        if response.body and response.content_type.lower().split(";", 1)[0].strip() not in {
            "application/json",
            "text/plain",
        }:
            raise OAuthHTTPError("OAuth provider response is invalid")
