"""Immutable first-party Assistant registry for the single-owner local controller.

Only the image reference is release data.  The executable contract stays in reviewed
source so a registry document cannot turn the Docker socket into an arbitrary runner.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import assistant_manifest

REGISTRY_PATH = Path("/etc/shimpz/local-assistants.json")
_DIGEST_REF = re.compile(
    r"(?:[a-z0-9.-]+(?::[0-9]{1,5})?/)?"
    r"[a-z0-9]+(?:[._/-][a-z0-9]+)*@sha256:[0-9a-f]{64}"
)
_ZERO_DIGEST = "0" * 64
_REVIEWED_ASSISTANTS = assistant_manifest.load_reviewed_catalog()


class RegistryError(RuntimeError):
    """The baked registry is missing or is not safe to execute."""


@dataclass(frozen=True, slots=True)
class PowerSpec:
    method: str
    path: str
    summary: str
    input_schema: dict[str, object]
    output_schema: dict[str, object]
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
    machine_contract: dict[str, object] = field(default_factory=dict)


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
    expected_ids = set(_REVIEWED_ASSISTANTS)
    if (
        not isinstance(raw, dict)
        or set(raw) != {"schema", "images"}
        or raw["schema"] != 2
        or not isinstance(raw["images"], dict)
        or set(raw["images"]) != expected_ids
    ):
        raise RegistryError("the baked Assistant registry has an unsupported shape")
    registry: dict[str, AssistantSpec] = {}
    for assistant_id, contract in _REVIEWED_ASSISTANTS.items():
        spec = AssistantSpec(
            assistant_id=assistant_id,
            name=contract.name,
            summary=contract.summary,
            image=_digest_ref(raw["images"][assistant_id]),
            rpc_command=contract.rpc_command,
            health_path=contract.health_path,
            powers={
                power_id: PowerSpec(
                    method=power["method"],
                    path=power["path"],
                    summary=power_id.replace("-", " ").capitalize(),
                    input_schema=power["input_schema"],
                    output_schema=power["output_schema"],
                    secrets=(),
                    accounts=tuple(power["accounts"]),
                )
                for power_id, power in contract.powers.items()
            },
            secrets={},
            allowed_hosts=contract.allowed_hosts,
            accounts={
                account.id: AccountSpec(provider=account.provider, scopes=account.scopes)
                for account in contract.accounts
            },
            machine_contract=contract.machine_contract,
        )
        registry[spec.assistant_id] = spec
    return registry


def validate_power_input(assistant_id: str, power: str, payload: object) -> dict[str, object]:
    try:
        schema = _REVIEWED_ASSISTANTS[assistant_id].powers[power]["input_schema"]
    except KeyError as exc:
        raise ValueError("the Power has no declared input contract") from exc
    return assistant_manifest.validate_schema_payload(schema, payload)


def validate_power_output(assistant_id: str, power: str, payload: object) -> dict[str, object]:
    try:
        schema = _REVIEWED_ASSISTANTS[assistant_id].powers[power]["output_schema"]
    except KeyError as exc:
        raise ValueError("the Power has no declared output contract") from exc
    return assistant_manifest.validate_schema_payload(schema, payload)
