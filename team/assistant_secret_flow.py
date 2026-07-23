"""Closed public contracts for Assistant secret inventory and JIT challenges.

This module never accepts a secret value while building metadata.  Submitted
values are validated against one exact, one-use challenge and projected into
per-Assistant mappings only at the controller boundary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Protocol

import assistant_secret_challenges
import assistant_secret_store
import brain_runtime_client
from local_registry import AssistantSpec

MAX_CHALLENGE_SECRETS = 64
MAX_PRIVATE_RPC_ENVELOPE_BYTES = 16 * 1024


class SecretFlowError(RuntimeError):
    """A secret request or submission violated its closed contract."""


def encode_private_rpc_envelope(payload: object) -> bytes:
    """Encode one bounded Controller-to-Assistant envelope exactly once."""
    try:
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        raise SecretFlowError("private Assistant RPC envelope is invalid") from exc
    if len(encoded) > MAX_PRIVATE_RPC_ENVELOPE_BYTES:
        raise SecretFlowError("private Assistant RPC envelope is too large")
    return encoded


def require_power_rpc_envelope(
    power_input: object,
    secret_values: Mapping[str, str],
    account_values: Mapping[str, object] | None = None,
) -> None:
    """Prove the complete private Power request fits before journaling it."""
    encode_private_rpc_envelope(
        {
            "input": power_input,
            "secrets": dict(secret_values),
            "accounts": dict(account_values or {}),
            "answers": [],
        }
    )


class _ActiveBinding(Protocol):
    spec: AssistantSpec


def requirements_for_batch(
    team_id: str,
    bindings: Mapping[str, _ActiveBinding],
    requests: Sequence[brain_runtime_client.PowerRequest],
    store: assistant_secret_store.AssistantSecretStore,
) -> tuple[assistant_secret_challenges.SecretRequirement, ...]:
    """Return every missing secret before the first Power in a batch may run."""
    grouped: dict[str, dict[str, set[str]]] = {}
    specs: dict[str, AssistantSpec] = {}
    for request in requests:
        active = bindings.get(request.assistant_id)
        if active is None:
            raise SecretFlowError("Power Assistant is unavailable")
        power = active.spec.powers.get(request.power)
        if power is None:
            raise SecretFlowError("Power secret contract is unavailable")
        if not power.secrets:
            continue
        specs[request.assistant_id] = active.spec
        group = grouped.setdefault(request.assistant_id, {})
        for secret_id in power.secrets:
            group.setdefault(secret_id, set()).add(request.power)

    requirements: list[assistant_secret_challenges.SecretRequirement] = []
    total = 0
    for assistant_id in sorted(grouped):
        spec = specs[assistant_id]
        group = grouped[assistant_id]
        declared_ids = tuple(sorted(group))
        metadata = store.metadata(team_id, assistant_id, declared_ids)
        missing = tuple(item.id for item in metadata if not item.configured)
        if not missing:
            continue
        total += len(missing)
        if total > MAX_CHALLENGE_SECRETS:
            raise SecretFlowError("Power batch requires too many Assistant secrets")
        requirements.append(
            assistant_secret_challenges.SecretRequirement(
                assistant_id=assistant_id,
                assistant_name=spec.name,
                power_ids=tuple(sorted({power_id for secret_id in missing for power_id in group[secret_id]})),
                secrets=tuple(
                    (secret_id, spec.secrets[secret_id].name, spec.secrets[secret_id].summary) for secret_id in missing
                ),
            )
        )
    return tuple(requirements)


def challenge_payload(
    challenge: assistant_secret_challenges.PendingSecretChallenge,
) -> dict[str, object]:
    """Project a challenge without pending Power inputs or secret values."""
    return {
        "team_id": challenge.team_id,
        "status": "secrets-required",
        "turn_id": challenge.id,
        "challenge_id": challenge.id,
        "requirements": [
            {
                "assistant_id": requirement.assistant_id,
                "assistant_name": requirement.assistant_name,
                "power_ids": list(requirement.power_ids),
                "secrets": [
                    {"id": secret_id, "name": name, "summary": summary}
                    for secret_id, name, summary in requirement.secrets
                ],
            }
            for requirement in challenge.requirements
        ],
    }


def inventory_payload(
    team_id: str,
    assistants: Sequence[AssistantSpec],
    store: assistant_secret_store.AssistantSecretStore,
) -> dict[str, object]:
    """List only declared public metadata for installed Assistants."""
    listing: list[dict[str, object]] = []
    for spec in sorted(assistants, key=lambda item: item.assistant_id):
        declared_ids = tuple(sorted(spec.secrets))
        metadata = {item.id: item for item in store.metadata(team_id, spec.assistant_id, declared_ids)}
        listing.append(
            {
                "id": spec.assistant_id,
                "name": spec.name,
                "secrets": [
                    {
                        "id": secret_id,
                        "name": spec.secrets[secret_id].name,
                        "summary": spec.secrets[secret_id].summary,
                        "configured": metadata[secret_id].configured,
                        "mask": metadata[secret_id].mask,
                    }
                    for secret_id in declared_ids
                ],
            }
        )
    return {"team_id": team_id, "assistants": listing}


def replacement_values(spec: AssistantSpec, body: object) -> dict[str, str]:
    """Validate one partial, atomic replacement batch against declared secret ids."""
    if not isinstance(body, dict) or set(body) != {"assistant_id", "values"}:
        raise SecretFlowError("secret replacement has an invalid shape")
    if body.get("assistant_id") != spec.assistant_id:
        raise SecretFlowError("secret replacement does not match its Assistant")
    values = body.get("values")
    if not isinstance(values, list) or not 1 <= len(values) <= len(spec.secrets):
        raise SecretFlowError("secret replacement has an invalid shape")
    replacements: dict[str, str] = {}
    for item in values:
        if not isinstance(item, dict) or set(item) != {"secret_id", "value"}:
            raise SecretFlowError("secret replacement has an invalid shape")
        secret_id = item.get("secret_id")
        value = item.get("value")
        if (
            not isinstance(secret_id, str)
            or secret_id not in spec.secrets
            or secret_id in replacements
            or not isinstance(value, str)
        ):
            raise SecretFlowError("secret replacement does not match its Assistant")
        replacements[secret_id] = value
    return replacements


def submission_values(
    challenge: assistant_secret_challenges.PendingSecretChallenge,
    body: object,
) -> dict[str, dict[str, str]]:
    """Require exactly one value for every missing secret and reject all extras."""
    if not isinstance(body, dict) or set(body) != {"challenge_id", "values"}:
        raise SecretFlowError("secret submission has an invalid shape")
    if body.get("challenge_id") != challenge.id:
        raise SecretFlowError("secret challenge is unavailable")
    values = body.get("values")
    if not isinstance(values, list) or not values or len(values) > MAX_CHALLENGE_SECRETS:
        raise SecretFlowError("secret submission has an invalid shape")

    expected = {
        (requirement.assistant_id, secret_id)
        for requirement in challenge.requirements
        for secret_id, _name, _summary in requirement.secrets
    }
    submitted: dict[tuple[str, str], str] = {}
    for item in values:
        if not isinstance(item, dict) or set(item) != {"assistant_id", "secret_id", "value"}:
            raise SecretFlowError("secret submission has an invalid shape")
        assistant_id = item.get("assistant_id")
        secret_id = item.get("secret_id")
        value = item.get("value")
        key = (assistant_id, secret_id)
        if (
            not isinstance(assistant_id, str)
            or not isinstance(secret_id, str)
            or not isinstance(value, str)
            or key not in expected
            or key in submitted
        ):
            raise SecretFlowError("secret submission does not match its challenge")
        submitted[key] = value
    if set(submitted) != expected:
        raise SecretFlowError("secret submission does not match its challenge")

    grouped: dict[str, dict[str, str]] = {}
    for (assistant_id, secret_id), value in submitted.items():
        grouped.setdefault(assistant_id, {})[secret_id] = value
    return grouped
