"""Closed HTTPS client for the Shimpz-hosted OAuth broker.

The local Controller can start, claim, refresh, and revoke Cloudflare grants without
ever receiving the OAuth Client Secret. Hosts, paths, scopes, response shapes, and
redirect behavior are fixed in reviewed source.
"""

from __future__ import annotations

import http.client
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode, urlsplit

import oauth_http_client
import oauth_providers

BROKER_ORIGIN = "https://shimpz.com"
MAX_RESPONSE_BYTES = 32 * 1024
MAX_TOKEN_BYTES = 16 * 1024
HTTP_TIMEOUT_SECONDS = 10
_BINDING = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_CLAIM = re.compile(r"[0-9a-f]{64}\Z")
_LEASE = re.compile(r"l1\.\d{10}\.[A-Za-z0-9_-]{43}\.[A-Za-z0-9_-]{43}\.[A-Za-z0-9_-]{43}\Z")


class OAuthBrokerClientError(RuntimeError):
    """A broker operation failed without reflecting private response data."""


@dataclass(frozen=True, slots=True)
class BrokerHTTPResponse:
    status: int
    content_type: str
    body: bytes


class BrokerTransport(Protocol):
    def request(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> BrokerHTTPResponse: ...


class FixedBrokerTransport:
    """POST only to one reviewed shimpz.com broker operation."""

    def request(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> BrokerHTTPResponse:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "shimpz.com"
            or parsed.port is not None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path
            not in {
                "/api/oauth/cloudflare/claim",
                "/api/oauth/cloudflare/refresh",
                "/api/oauth/cloudflare/revoke",
            }
        ):
            raise OAuthBrokerClientError("OAuth broker endpoint is invalid")
        connection = http.client.HTTPSConnection(parsed.hostname, timeout=HTTP_TIMEOUT_SECONDS)
        try:
            connection.request("POST", parsed.path, body=body, headers=dict(headers))
            response = connection.getresponse()
            payload = response.read(MAX_RESPONSE_BYTES + 1)
            if len(payload) > MAX_RESPONSE_BYTES:
                raise OAuthBrokerClientError("OAuth broker response is invalid")
            return BrokerHTTPResponse(
                response.status,
                response.getheader("Content-Type", ""),
                payload,
            )
        except OAuthBrokerClientError:
            raise
        except (OSError, http.client.HTTPException) as exc:
            raise OAuthBrokerClientError("OAuth broker is unavailable") from exc
        finally:
            connection.close()


def _binding(value: object, label: str = "binding") -> str:
    if not isinstance(value, str) or _BINDING.fullmatch(value) is None:
        raise OAuthBrokerClientError(f"OAuth {label} is invalid")
    return value


def _private_text(
    value: object,
    label: str,
    *,
    minimum: int = 16,
    maximum: int = MAX_TOKEN_BYTES,
) -> str:
    if not isinstance(value, str):
        raise OAuthBrokerClientError(f"OAuth {label} is invalid")
    try:
        encoded = value.encode("ascii")
    except UnicodeError as exc:
        raise OAuthBrokerClientError(f"OAuth {label} is invalid") from exc
    if not minimum <= len(encoded) <= maximum or any(byte <= 32 or byte >= 127 for byte in encoded):
        raise OAuthBrokerClientError(f"OAuth {label} is invalid")
    return value


def _intent(provider_id: object, scopes: object) -> tuple[str, tuple[str, ...]]:
    try:
        intent = oauth_providers.account_intent(provider_id, scopes)
    except oauth_providers.OAuthProviderError as exc:
        raise OAuthBrokerClientError("OAuth account declaration is invalid") from exc
    if intent.provider.id != "cloudflare":
        raise OAuthBrokerClientError("OAuth provider is unavailable")
    return intent.provider.id, intent.scopes


def _object(response: BrokerHTTPResponse) -> dict[str, object]:
    if (
        response.status != 200
        or response.content_type.lower().split(";", 1)[0].strip() != "application/json"
        or not response.body
        or len(response.body) > MAX_RESPONSE_BYTES
    ):
        raise OAuthBrokerClientError("OAuth broker operation failed")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise OAuthBrokerClientError("OAuth broker response is invalid")
            result[key] = value
        return result

    try:
        value = json.loads(response.body, object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OAuthBrokerClientError("OAuth broker response is invalid") from exc
    if not isinstance(value, dict):
        raise OAuthBrokerClientError("OAuth broker response is invalid")
    return value


class OAuthBrokerClient:
    def __init__(self, transport: BrokerTransport | None = None) -> None:
        self._transport = transport or FixedBrokerTransport()

    def __repr__(self) -> str:
        return "<OAuthBrokerClient shimpz.com>"

    def authorization_url(
        self,
        *,
        provider_id: object,
        state: object,
        code_challenge: object,
        scopes: object,
    ) -> str:
        _provider, canonical_scopes = _intent(provider_id, scopes)
        query = urlencode(
            {
                "state": _binding(state, "state"),
                "code_challenge": _binding(code_challenge, "challenge"),
                "scope": " ".join(canonical_scopes),
            }
        )
        return f"{BROKER_ORIGIN}/api/oauth/cloudflare/start?{query}"

    def _post(self, operation: str, payload: dict[str, object]) -> dict[str, object]:
        if operation not in {"claim", "refresh", "revoke"}:
            raise OAuthBrokerClientError("OAuth broker operation is invalid")
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        response = self._transport.request(
            url=f"{BROKER_ORIGIN}/api/oauth/cloudflare/{operation}",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "shimpz-local-controller/1",
            },
            body=body,
        )
        return _object(response)

    @staticmethod
    def _tokens(value: dict[str, object], scopes: tuple[str, ...]) -> oauth_http_client.OAuthTokenSet:
        if set(value) != {
            "access_token",
            "refresh_token",
            "expires_in",
            "scopes",
            "broker_lease",
        }:
            raise OAuthBrokerClientError("OAuth broker response is invalid")
        expires_in = value.get("expires_in")
        returned_scopes = value.get("scopes")
        lease = value.get("broker_lease")
        if (
            type(expires_in) is not int
            or not 30 <= expires_in <= 31_536_000
            or not isinstance(returned_scopes, list)
            or tuple(returned_scopes) != scopes
            or not isinstance(lease, str)
            or _LEASE.fullmatch(lease) is None
        ):
            raise OAuthBrokerClientError("OAuth broker response is invalid")
        return oauth_http_client.OAuthTokenSet(
            access_token=_private_text(value.get("access_token"), "access token"),
            refresh_token=_private_text(value.get("refresh_token"), "refresh token"),
            scopes=scopes,
            expires_in=expires_in,
            broker_lease=lease,
        )

    def claim(
        self,
        *,
        provider_id: object,
        claim: object,
        state: object,
        code_verifier: object,
        scopes: object,
    ) -> oauth_http_client.OAuthTokenSet:
        _provider, canonical_scopes = _intent(provider_id, scopes)
        if not isinstance(claim, str) or _CLAIM.fullmatch(claim) is None:
            raise OAuthBrokerClientError("OAuth claim is invalid")
        verifier = _private_text(
            code_verifier,
            "verifier",
            minimum=43,
            maximum=128,
        )
        return self._tokens(
            self._post(
                "claim",
                {
                    "claim": claim,
                    "state": _binding(state, "state"),
                    "code_verifier": verifier,
                },
            ),
            canonical_scopes,
        )

    def refresh(
        self,
        *,
        provider_id: object,
        refresh_token: object,
        broker_lease: object,
        scopes: object,
    ) -> oauth_http_client.OAuthTokenSet:
        _provider, canonical_scopes = _intent(provider_id, scopes)
        if not isinstance(broker_lease, str) or _LEASE.fullmatch(broker_lease) is None:
            raise OAuthBrokerClientError("OAuth broker lease is invalid")
        return self._tokens(
            self._post(
                "refresh",
                {
                    "refresh_token": _private_text(refresh_token, "refresh token"),
                    "broker_lease": broker_lease,
                    "scopes": list(canonical_scopes),
                },
            ),
            canonical_scopes,
        )

    def revoke(
        self,
        *,
        provider_id: object,
        token: object,
        broker_lease: object,
    ) -> None:
        _intent(provider_id, ("dns.read", "offline_access", "zone.read"))
        if not isinstance(broker_lease, str) or _LEASE.fullmatch(broker_lease) is None:
            raise OAuthBrokerClientError("OAuth broker lease is invalid")
        if self._post(
            "revoke",
            {
                "token": _private_text(token, "token"),
                "broker_lease": broker_lease,
            },
        ) != {"revoked": True}:
            raise OAuthBrokerClientError("OAuth broker response is invalid")
