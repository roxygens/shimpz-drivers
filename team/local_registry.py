"""Immutable first-party Assistant registry for the single-owner local controller.

Only the image reference is release data.  The executable contract stays in reviewed
source so a registry document cannot turn the Docker socket into an arbitrary runner.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import assistant_contract

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


@dataclass(frozen=True, slots=True)
class AssistantSpec:
    assistant_id: str
    name: str
    summary: str
    image: str
    rpc_command: str
    health_path: str
    powers: dict[str, PowerSpec]
    allowed_hosts: tuple[str, ...]


def is_digest_ref(value: object) -> bool:
    return (
        isinstance(value, str)
        and _DIGEST_REF.fullmatch(value) is not None
        and not value.endswith(f"sha256:{_ZERO_DIGEST}")
    )


def _digest_ref(value: object) -> str:
    if not isinstance(value, str) or _DIGEST_REF.fullmatch(value) is None:
        raise RegistryError("the Shimpz Assistant image must be an OCI sha256 digest reference")
    if not is_digest_ref(value):
        raise RegistryError("the Shimpz Assistant release digest has not been bound")
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
    if not isinstance(raw, dict) or set(raw) != {"schema", "shimpz_assistant_image"} or raw["schema"] != 1:
        raise RegistryError("the baked Assistant registry has an unsupported shape")

    shimpz_assistant = AssistantSpec(
        assistant_id=assistant_contract.ASSISTANT_ID,
        name=assistant_contract.ASSISTANT_NAME,
        summary=assistant_contract.ASSISTANT_SUMMARY,
        image=_digest_ref(raw["shimpz_assistant_image"]),
        rpc_command=assistant_contract.ASSISTANT_RPC_COMMAND,
        health_path="/healthz",
        powers={power_id: PowerSpec(**contract) for power_id, contract in assistant_contract.power_contracts().items()},
        allowed_hosts=assistant_contract.ASSISTANT_ALLOWED_HOSTS,
    )
    return {shimpz_assistant.assistant_id: shimpz_assistant}


def validate_power_input(assistant_id: str, power: str, payload: object) -> dict[str, object]:
    return assistant_contract.validate_power_input(assistant_id, power, payload)


def validate_power_output(assistant_id: str, power: str, payload: object) -> dict[str, object]:
    return assistant_contract.validate_power_output(assistant_id, power, payload)
