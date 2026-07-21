"""Closed public and private contracts for Assistant OAuth accounts.

Public projections contain only reviewed intent and bounded account metadata.
OAuth tokens remain Controller-owned and are resolved into the exact Power RPC
envelope only at the last private boundary before an Assistant invocation.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Protocol

import assistant_account_challenges
import brain_runtime_client
import oauth_providers

MAX_BATCH_POWERS = 64
MAX_ACCOUNT_REQUIREMENTS = 64
MAX_INVENTORY_ASSISTANTS = 64
MAX_INVENTORY_ACCOUNTS = 256
MAX_ACCOUNTS_PER_POWER = 16
MAX_ACCESS_TOKEN_BYTES = 16 * 1024
MAX_PUBLIC_TEXT_BYTES = 512
_TEAM_ID = re.compile(r"[a-z0-9_]{1,40}\Z")
_COMPONENT_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_FORBIDDEN_PUBLIC_FIELDS = frozenset(
    {
        "access_token",
        "refresh_token",
        "client_id",
        "client_secret",
        "authorization_code",
        "code",
        "code_verifier",
        "token",
    }
)
_PROVIDER_PUBLIC_METADATA = {
    "cloudflare": (
        "Cloudflare",
        "Connect your Cloudflare account so this Assistant can use only its reviewed read permissions.",
    ),
    "x": (
        "X",
        "Connect your X account so this Assistant can use only its reviewed X permissions.",
    ),
}


class AccountFlowError(RuntimeError):
    """An Assistant account request violated its closed contract."""


class _AccountSpec(Protocol):
    provider: str
    scopes: tuple[str, ...]


class _PowerSpec(Protocol):
    summary: str
    accounts: tuple[str, ...]


class _AssistantSpec(Protocol):
    assistant_id: str
    name: str
    powers: Mapping[str, _PowerSpec]
    accounts: Mapping[str, _AccountSpec]


class _ActiveBinding(Protocol):
    spec: _AssistantSpec


class _AccountMetadata(Protocol):
    id: str
    provider: str
    scopes: tuple[str, ...]
    status: str
    account: object
    expires_at: int | None
    generation: int


class _AccountStore(Protocol):
    def metadata(
        self,
        team_id: object,
        assistant_id: object,
        declarations: object,
    ) -> tuple[_AccountMetadata, ...]: ...

    def resolve(
        self,
        team_id: object,
        assistant_id: object,
        account_id: object,
        provider: object,
        scopes: object,
        refresh_callback: Callable[[str, str | None], object],
    ) -> str: ...


RefreshCallback = Callable[[str, tuple[str, ...], str, str | None], object]


def _team_id(value: object) -> str:
    if not isinstance(value, str) or _TEAM_ID.fullmatch(value) is None:
        raise AccountFlowError("Team id is invalid")
    return value


def _component_id(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) > 64 or _COMPONENT_ID.fullmatch(value) is None:
        raise AccountFlowError(f"{label} is invalid")
    return value


def _public_text(value: object, label: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip() or not value.isprintable():
        raise AccountFlowError(f"{label} is invalid")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        raise AccountFlowError(f"{label} is invalid") from exc
    if len(encoded) > MAX_PUBLIC_TEXT_BYTES:
        raise AccountFlowError(f"{label} is invalid")
    return value


def _assistant(spec: object) -> _AssistantSpec:
    try:
        assistant_id = spec.assistant_id  # type: ignore[attr-defined]
        name = spec.name  # type: ignore[attr-defined]
        powers = spec.powers  # type: ignore[attr-defined]
        accounts = spec.accounts  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as exc:
        raise AccountFlowError("Assistant account contract is unavailable") from exc
    _component_id(assistant_id, "Assistant id")
    _public_text(name, "Assistant name")
    if not isinstance(powers, Mapping) or not isinstance(accounts, Mapping):
        raise AccountFlowError("Assistant account contract is unavailable")
    if len(accounts) > MAX_ACCOUNTS_PER_POWER:
        raise AccountFlowError("Assistant declares too many accounts")
    return spec  # type: ignore[return-value]


def _intent(account_id: object, declaration: object) -> tuple[str, str, tuple[str, ...]]:
    identifier = _component_id(account_id, "account id")
    try:
        provider = declaration.provider  # type: ignore[attr-defined]
        scopes = declaration.scopes  # type: ignore[attr-defined]
        resolved = oauth_providers.account_intent(provider, scopes)
    except (AttributeError, TypeError, oauth_providers.OAuthProviderError) as exc:
        raise AccountFlowError("Assistant account declaration is invalid") from exc
    return identifier, resolved.provider.id, resolved.scopes


def _provider_metadata(provider_id: str) -> tuple[str, str]:
    try:
        provider = oauth_providers.resolve(provider_id)
        name, summary = _PROVIDER_PUBLIC_METADATA[provider.id]
    except (KeyError, oauth_providers.OAuthProviderError) as exc:
        raise AccountFlowError("OAuth provider has no reviewed public metadata") from exc
    return name, summary


def _power(spec: _AssistantSpec, power_id: object) -> tuple[str, _PowerSpec]:
    identifier = _component_id(power_id, "Power id")
    power = spec.powers.get(identifier)
    if power is None:
        raise AccountFlowError("Power account contract is unavailable")
    try:
        summary = power.summary
        accounts = power.accounts
    except (AttributeError, TypeError) as exc:
        raise AccountFlowError("Power account contract is unavailable") from exc
    _public_text(summary, "Power summary")
    if not isinstance(accounts, tuple) or len(accounts) > MAX_ACCOUNTS_PER_POWER or len(accounts) != len(set(accounts)):
        raise AccountFlowError("Power account contract is invalid")
    return identifier, power


def _power_name(power_id: str) -> str:
    return " ".join(part.capitalize() for part in power_id.split("-"))


def _metadata_for(
    team_id: str,
    spec: _AssistantSpec,
    declarations: Mapping[str, _AccountSpec],
    store: _AccountStore,
) -> dict[str, _AccountMetadata]:
    try:
        metadata = store.metadata(team_id, spec.assistant_id, declarations)
    except Exception as exc:
        if isinstance(exc, AccountFlowError):
            raise
        raise AccountFlowError("Assistant account inventory is unavailable") from exc
    if not isinstance(metadata, tuple) or len(metadata) != len(declarations):
        raise AccountFlowError("Assistant account inventory is invalid")
    indexed: dict[str, _AccountMetadata] = {}
    for item in metadata:
        try:
            account_id = _component_id(item.id, "account id")
            status = item.status
            generation = item.generation
            declared = declarations[account_id]
            expected_id, expected_provider, expected_scopes = _intent(account_id, declared)
        except (AttributeError, KeyError, TypeError) as exc:
            raise AccountFlowError("Assistant account inventory is invalid") from exc
        if (
            account_id in indexed
            or item.provider != expected_provider
            or item.scopes != expected_scopes
            or status not in {"missing", "connected", "refresh-required", "reauthorization-required"}
            or type(generation) is not int
            or generation < 0
            or generation > 2**53 - 1
            or expected_id != account_id
        ):
            raise AccountFlowError("Assistant account inventory is invalid")
        if (status == "missing" and (item.account is not None or item.expires_at is not None or generation != 0)) or (
            status != "missing"
            and (type(item.expires_at) is not int or not 1 <= item.expires_at <= 2**53 - 1 or generation < 1)
        ):
            raise AccountFlowError("Assistant account inventory is invalid")
        indexed[account_id] = item
    if set(indexed) != set(declarations):
        raise AccountFlowError("Assistant account inventory is invalid")
    return indexed


def requirements_for_batch(
    team_id: str,
    bindings: Mapping[str, _ActiveBinding],
    requests: Sequence[brain_runtime_client.PowerRequest],
    store: _AccountStore,
) -> tuple[assistant_account_challenges.AccountRequirement, ...]:
    """Return every unusable account before the first Power may execute."""
    team = _team_id(team_id)
    if isinstance(requests, str | bytes) or len(requests) > MAX_BATCH_POWERS:
        raise AccountFlowError("Power batch has too many account requests")
    grouped: dict[str, dict[str, set[str]]] = {}
    specs: dict[str, _AssistantSpec] = {}
    for request in requests:
        if not isinstance(request, brain_runtime_client.PowerRequest):
            raise AccountFlowError("Power account request is invalid")
        active = bindings.get(request.assistant_id)
        if active is None:
            raise AccountFlowError("Power Assistant is unavailable")
        spec = _assistant(active.spec)
        if request.assistant_id != spec.assistant_id:
            raise AccountFlowError("Power Assistant binding is invalid")
        power_id, power = _power(spec, request.power)
        specs[spec.assistant_id] = spec
        for account_id in power.accounts:
            identifier = _component_id(account_id, "account id")
            if identifier not in spec.accounts:
                raise AccountFlowError("Power references an undeclared account")
            grouped.setdefault(spec.assistant_id, {}).setdefault(identifier, set()).add(power_id)

    requirements: list[assistant_account_challenges.AccountRequirement] = []
    for assistant_id in sorted(grouped):
        spec = specs[assistant_id]
        declarations = {identifier: spec.accounts[identifier] for identifier in sorted(grouped[assistant_id])}
        metadata = _metadata_for(team, spec, declarations, store)
        for account_id in sorted(declarations):
            item = metadata[account_id]
            if item.status == "connected":
                continue
            identifier, provider, scopes = _intent(account_id, declarations[account_id])
            requirements.append(
                assistant_account_challenges.AccountRequirement(
                    assistant_id=assistant_id,
                    assistant_name=spec.name,
                    power_ids=tuple(sorted(grouped[assistant_id][account_id])),
                    accounts=((identifier, provider, scopes),),
                )
            )
            if len(requirements) > MAX_ACCOUNT_REQUIREMENTS:
                raise AccountFlowError("Power batch requires too many Assistant accounts")
    return tuple(requirements)


def _expires_in(challenge: assistant_account_challenges.PendingAccountChallenge) -> int:
    try:
        remaining = math.ceil(challenge.expires_at - time.monotonic())
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise AccountFlowError("account challenge expiry is invalid") from exc
    if not 1 <= remaining <= 900:
        raise AccountFlowError("account challenge is expired")
    return remaining


def _assert_public_payload(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str) or key.lower() in _FORBIDDEN_PUBLIC_FIELDS:
                raise AccountFlowError("public account payload contains a sensitive field")
            _assert_public_payload(nested)
    elif isinstance(value, list | tuple):
        for nested in value:
            _assert_public_payload(nested)


def challenge_payload(
    challenge: assistant_account_challenges.PendingAccountChallenge,
    bindings: Mapping[str, _ActiveBinding],
) -> dict[str, object]:
    """Project one pending turn without Power input, OAuth state, or token material."""
    if (
        not isinstance(challenge, assistant_account_challenges.PendingAccountChallenge)
        or not 1 <= len(challenge.requirements) <= MAX_ACCOUNT_REQUIREMENTS
    ):
        raise AccountFlowError("account challenge is invalid")
    requirements: list[dict[str, object]] = []
    for requirement in challenge.requirements:
        if len(requirement.accounts) != 1:
            raise AccountFlowError("account challenge is invalid")
        active = bindings.get(requirement.assistant_id)
        if active is None:
            raise AccountFlowError("account challenge Assistant is unavailable")
        spec = _assistant(active.spec)
        if spec.assistant_id != requirement.assistant_id or spec.name != requirement.assistant_name:
            raise AccountFlowError("account challenge Assistant changed")
        account_id, provider, scopes = requirement.accounts[0]
        declaration = spec.accounts.get(account_id)
        if declaration is None or _intent(account_id, declaration) != (account_id, provider, scopes):
            raise AccountFlowError("account challenge declaration changed")
        powers: list[dict[str, str]] = []
        for power_id in requirement.power_ids:
            identifier, power = _power(spec, power_id)
            if account_id not in power.accounts:
                raise AccountFlowError("account challenge Power changed")
            powers.append(
                {
                    "id": identifier,
                    "name": _power_name(identifier),
                    "summary": str(_public_text(power.summary, "Power summary")),
                }
            )
        if not powers or len(powers) != len({item["id"] for item in powers}):
            raise AccountFlowError("account challenge Power list is invalid")
        name, summary = _provider_metadata(provider)
        requirements.append(
            {
                "assistant_id": spec.assistant_id,
                "assistant_name": spec.name,
                "account_id": account_id,
                "provider": provider,
                "name": name,
                "summary": summary,
                "scopes": list(scopes),
                "powers": powers,
            }
        )
    payload: dict[str, object] = {
        "team_id": challenge.team_id,
        "status": "accounts-required",
        "turn_id": challenge.id,
        "challenge_id": challenge.id,
        "expires_in": _expires_in(challenge),
        "requirements": requirements,
    }
    _assert_public_payload(payload)
    return payload


def _account_payload(value: object) -> dict[str, str | None] | None:
    if value is None:
        return None
    try:
        identifier = _public_text(value.id, "OAuth account id")  # type: ignore[attr-defined]
        name = _public_text(value.name, "OAuth account name", optional=True)  # type: ignore[attr-defined]
        username = _public_text(value.username, "OAuth account username", optional=True)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as exc:
        raise AccountFlowError("OAuth account metadata is invalid") from exc
    return {"id": identifier, "name": name, "username": username}


def _expiry_payload(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not int or not 1 <= value <= 2**53 - 1:
        raise AccountFlowError("OAuth account expiry is invalid")
    try:
        return datetime.fromtimestamp(value, UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    except (OSError, OverflowError, ValueError) as exc:
        raise AccountFlowError("OAuth account expiry is invalid") from exc


def inventory_payload(
    team_id: str,
    assistants: Sequence[_AssistantSpec],
    store: _AccountStore,
) -> dict[str, object]:
    """Return flat, status-only Admin rows for every declared account."""
    team = _team_id(team_id)
    if isinstance(assistants, str | bytes) or len(assistants) > MAX_INVENTORY_ASSISTANTS:
        raise AccountFlowError("Assistant account inventory is too large")
    listing: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_spec in sorted(assistants, key=lambda item: item.assistant_id):
        spec = _assistant(raw_spec)
        if spec.assistant_id in seen:
            raise AccountFlowError("Assistant account inventory is invalid")
        seen.add(spec.assistant_id)
        declarations = {identifier: spec.accounts[identifier] for identifier in sorted(spec.accounts)}
        metadata = _metadata_for(team, spec, declarations, store)
        for account_id in sorted(declarations):
            item = metadata[account_id]
            identifier, provider, scopes = _intent(account_id, declarations[account_id])
            name, summary = _provider_metadata(provider)
            status = "expired" if item.status == "refresh-required" else item.status
            listing.append(
                {
                    "assistant_id": spec.assistant_id,
                    "assistant_name": spec.name,
                    "id": identifier,
                    "provider": provider,
                    "name": name,
                    "summary": summary,
                    "scopes": list(scopes),
                    "status": status,
                    "account": _account_payload(item.account),
                    "expires_at": _expiry_payload(item.expires_at),
                }
            )
            if len(listing) > MAX_INVENTORY_ACCOUNTS:
                raise AccountFlowError("Assistant account inventory is too large")
    payload = {"accounts": listing}
    _assert_public_payload(payload)
    return payload


def resolve_power_accounts(
    team_id: str,
    spec: _AssistantSpec,
    power_id: str,
    store: _AccountStore,
    refresh_callback: RefreshCallback,
) -> dict[str, dict[str, str]]:
    """Resolve only one Power's declared access tokens into its private envelope."""
    team = _team_id(team_id)
    safe_spec = _assistant(spec)
    _, power = _power(safe_spec, power_id)
    if not callable(refresh_callback):
        raise AccountFlowError("OAuth refresh callback is invalid")
    resolved: dict[str, dict[str, str]] = {}
    for raw_account_id in power.accounts:
        account_id = _component_id(raw_account_id, "account id")
        declaration = safe_spec.accounts.get(account_id)
        if declaration is None:
            raise AccountFlowError("Power references an undeclared account")
        _, provider, scopes = _intent(account_id, declaration)
        try:
            access_token = store.resolve(
                team,
                safe_spec.assistant_id,
                account_id,
                provider,
                scopes,
                lambda token, lease, p=provider, s=scopes: refresh_callback(
                    p,
                    s,
                    token,
                    lease,
                ),
            )
        except Exception as exc:
            raise AccountFlowError("Assistant account could not be resolved") from exc
        if not isinstance(access_token, str):
            raise AccountFlowError("OAuth access token is invalid")
        try:
            encoded = access_token.encode("ascii")
        except UnicodeError as exc:
            raise AccountFlowError("OAuth access token is invalid") from exc
        if not 16 <= len(encoded) <= MAX_ACCESS_TOKEN_BYTES or any(byte <= 32 or byte >= 127 for byte in encoded):
            raise AccountFlowError("OAuth access token is invalid")
        resolved[account_id] = {"type": "oauth2-bearer", "access_token": access_token}
    return resolved
