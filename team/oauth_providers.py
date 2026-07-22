"""Core-owned OAuth provider metadata for reviewed Assistant accounts.

Assistant packages may name a provider and request reviewed scopes. They cannot
choose authorization endpoints, token endpoints, client authentication, or PKCE
methods. Adding or changing a provider therefore requires a controller release.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from urllib.parse import urlsplit, urlunsplit

MAX_REQUESTED_SCOPES = 32
_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_SCOPE = re.compile(r"[a-z][a-z0-9]*(?:[._:-][a-z0-9]+)*\Z")


class OAuthProviderError(RuntimeError):
    """A account referenced an unknown provider or disallowed OAuth intent."""


@dataclass(frozen=True, slots=True)
class OAuthProvider:
    id: str
    authorization_endpoint: str
    token_endpoint: str
    revocation_endpoint: str
    api_hosts: tuple[str, ...]
    allowed_scopes: frozenset[str]
    pkce_method: str
    client_auth_method: str


@dataclass(frozen=True, slots=True)
class OAuthAccountIntent:
    provider: OAuthProvider
    scopes: tuple[str, ...]


def _provider(
    *,
    provider_id: str,
    authorization_endpoint: str,
    token_endpoint: str,
    revocation_endpoint: str,
    api_hosts: tuple[str, ...],
    allowed_scopes: frozenset[str],
    client_auth_method: str,
) -> OAuthProvider:
    provider = OAuthProvider(
        id=provider_id,
        authorization_endpoint=authorization_endpoint,
        token_endpoint=token_endpoint,
        revocation_endpoint=revocation_endpoint,
        api_hosts=api_hosts,
        allowed_scopes=allowed_scopes,
        pkce_method="S256",
        client_auth_method=client_auth_method,
    )
    if _ID.fullmatch(provider.id) is None or not provider.api_hosts or not provider.allowed_scopes:
        raise RuntimeError("trusted OAuth provider registry is invalid")
    if provider.client_auth_method not in {"client_secret_basic", "none"}:
        raise RuntimeError("trusted OAuth provider registry is invalid")
    for endpoint in (
        provider.authorization_endpoint,
        provider.token_endpoint,
        provider.revocation_endpoint,
    ):
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is not None
            or parsed.query
            or parsed.fragment
        ):
            raise RuntimeError("trusted OAuth provider registry is invalid")
    if any(_SCOPE.fullmatch(scope) is None for scope in provider.allowed_scopes):
        raise RuntimeError("trusted OAuth provider registry is invalid")
    return provider


# Cloudflare's server-side Authorization Code client uses a secret. PKCE S256 is
# kept as an additional one-use binding even though it is optional for a
# confidential client. Only read scopes needed by the first Assistant release
# are admitted here.
_CLOUDFLARE = _provider(
    provider_id="cloudflare",
    authorization_endpoint="https://dash.cloudflare.com/oauth2/auth",
    token_endpoint=urlunsplit(("https", "dash.cloudflare.com", "/oauth2/token", "", "")),
    revocation_endpoint="https://dash.cloudflare.com/oauth2/revoke",
    api_hosts=("api.cloudflare.com",),
    allowed_scopes=frozenset({"dns.read", "offline_access", "zone.read"}),
    client_auth_method="client_secret_basic",
)

PROVIDERS = MappingProxyType({_CLOUDFLARE.id: _CLOUDFLARE})


def resolve(provider_id: object) -> OAuthProvider:
    """Resolve only a controller-reviewed provider identifier."""
    if not isinstance(provider_id, str) or _ID.fullmatch(provider_id) is None:
        raise OAuthProviderError("OAuth provider is unavailable")
    provider = PROVIDERS.get(provider_id)
    if provider is None:
        raise OAuthProviderError("OAuth provider is unavailable")
    return provider


def account_intent(provider_id: object, requested_scopes: object) -> OAuthAccountIntent:
    """Return one deterministic least-privilege scope set for a trusted provider."""
    provider = resolve(provider_id)
    if not isinstance(requested_scopes, list | tuple) or not 1 <= len(requested_scopes) <= MAX_REQUESTED_SCOPES:
        raise OAuthProviderError("OAuth scopes are invalid")
    scopes: list[str] = []
    for scope in requested_scopes:
        if not isinstance(scope, str) or _SCOPE.fullmatch(scope) is None:
            raise OAuthProviderError("OAuth scopes are invalid")
        scopes.append(scope)
    if len(scopes) != len(set(scopes)) or not set(scopes) <= provider.allowed_scopes:
        raise OAuthProviderError("OAuth scopes are invalid")
    return OAuthAccountIntent(provider, tuple(sorted(scopes)))
