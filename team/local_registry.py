"""Immutable first-party Assistant registry for the single-owner local controller.

Only the image reference is release data.  The executable contract stays in reviewed
source so a registry document cannot turn the Docker socket into an arbitrary runner.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cloudflare_assistant_contract

REGISTRY_PATH = Path("/etc/shimpz/local-assistants.json")
_DIGEST_REF = re.compile(
    r"(?:[a-z0-9.-]+(?::[0-9]{1,5})?/)?"
    r"[a-z0-9]+(?:[._/-][a-z0-9]+)*@sha256:[0-9a-f]{64}"
)
_ZERO_DIGEST = "0" * 64


class RegistryError(RuntimeError):
    """The baked registry is missing or is not safe to execute."""


@dataclass(frozen=True, slots=True)
class PowerSpec:
    method: str
    path: str
    summary: str
    input_schema: dict[str, object]
    output_schema: dict[str, object]
    approval: Literal["none", "once", "each-run"]
    secrets: tuple[str, ...]
    accounts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SecretSpec:
    name: str
    summary: str


@dataclass(frozen=True, slots=True)
class AccountSpec:
    provider: str
    scopes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AssistantSpec:
    assistant_id: str
    name: str
    summary: str
    image: str
    rpc_command: str
    health_path: str
    powers: dict[str, PowerSpec]
    secrets: dict[str, SecretSpec]
    allowed_hosts: tuple[str, ...]
    accounts: dict[str, AccountSpec] = field(default_factory=dict)


def is_digest_ref(value: object) -> bool:
    return (
        isinstance(value, str)
        and _DIGEST_REF.fullmatch(value) is not None
        and not value.endswith(f"sha256:{_ZERO_DIGEST}")
    )


def _digest_ref(value: object) -> str:
    if not isinstance(value, str) or _DIGEST_REF.fullmatch(value) is None:
        raise RegistryError("an Assistant image must be an OCI sha256 digest reference")
    if not is_digest_ref(value):
        raise RegistryError("an Assistant release digest has not been bound")
    return value


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, AssistantSpec]:
    """Load the closed, build-baked local registry schema v1.

    There is deliberately no environment override: changing the allowlist requires a
    new controller image and therefore leaves a normal release/audit trail.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RegistryError("the baked Assistant registry is unreadable") from exc
    contracts = (cloudflare_assistant_contract,)
    expected_ids = {contract.ASSISTANT_ID for contract in contracts}
    if (
        not isinstance(raw, dict)
        or set(raw) != {"schema", "images"}
        or raw["schema"] != 2
        or not isinstance(raw["images"], dict)
        or set(raw["images"]) != expected_ids
    ):
        raise RegistryError("the baked Assistant registry has an unsupported shape")
    registry: dict[str, AssistantSpec] = {}
    for contract in contracts:
        spec = AssistantSpec(
            assistant_id=contract.ASSISTANT_ID,
            name=contract.ASSISTANT_NAME,
            summary=contract.ASSISTANT_SUMMARY,
            image=_digest_ref(raw["images"][contract.ASSISTANT_ID]),
            rpc_command=contract.ASSISTANT_RPC_COMMAND,
            health_path=getattr(contract, "ASSISTANT_HEALTH_PATH", "/healthz"),
            powers={power_id: PowerSpec(**power) for power_id, power in contract.power_contracts().items()},
            secrets={secret_id: SecretSpec(**secret) for secret_id, secret in contract.secret_contracts().items()},
            allowed_hosts=contract.ASSISTANT_ALLOWED_HOSTS,
            accounts={
                account_id: AccountSpec(**account) for account_id, account in contract.account_contracts().items()
            },
        )
        registry[spec.assistant_id] = spec
    return registry


def validate_power_input(assistant_id: str, power: str, payload: object) -> dict[str, object]:
    return cloudflare_assistant_contract.validate_power_input(assistant_id, power, payload)


def validate_power_output(assistant_id: str, power: str, payload: object) -> dict[str, object]:
    return cloudflare_assistant_contract.validate_power_output(assistant_id, power, payload)
