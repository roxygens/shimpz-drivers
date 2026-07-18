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
    approval: Literal["none", "once", "each-run"]


@dataclass(frozen=True, slots=True)
class AssistantSpec:
    assistant_id: str
    image: str
    rpc_command: str
    health_path: str
    rules: str
    powers: dict[str, PowerSpec]


def _digest_ref(value: object) -> str:
    if not isinstance(value, str) or _DIGEST_REF.fullmatch(value) is None:
        raise RegistryError("the Hello Pulse image must be an OCI sha256 digest reference")
    if value.endswith(f"sha256:{_ZERO_DIGEST}"):
        raise RegistryError("the Hello Pulse release digest has not been bound")
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
    if not isinstance(raw, dict) or set(raw) != {"schema", "hello_pulse_image"} or raw["schema"] != 1:
        raise RegistryError("the baked Assistant registry has an unsupported shape")

    hello = AssistantSpec(
        assistant_id="hello-pulse",
        image=_digest_ref(raw["hello_pulse_image"]),
        rpc_command="/usr/local/bin/shimpz-assistant-rpc",
        health_path="/healthz",
        rules=(
            "Respond naturally to questions and conversation. Use the declared hello Power only when the Captain "
            "explicitly asks to run or demonstrate it. After a Power result, explain the outcome naturally."
        ),
        powers={
            "hello": PowerSpec(
                method="POST",
                path="/v1/powers/hello",
                summary="Return a friendly greeting for an optional name.",
                input_schema={
                    "type": "object",
                    "properties": {"name": {"type": "string", "minLength": 1, "maxLength": 80}},
                    "additionalProperties": False,
                },
                approval="none",
            )
        },
    )
    return {hello.assistant_id: hello}


def validate_hello_input(payload: object) -> dict[str, str]:
    """Validate the complete public input contract for Hello Pulse."""
    if not isinstance(payload, dict) or not set(payload).issubset({"name"}):
        raise ValueError("hello accepts only an optional name")
    name = payload.get("name", "Shimpz")
    if not isinstance(name, str) or not 1 <= len(name) <= 80 or name.strip() != name:
        raise ValueError("name must contain 1 to 80 trimmed characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise ValueError("name contains control characters")
    return {"name": name}


def validate_power_input(assistant_id: str, power: str, payload: object) -> dict[str, str]:
    if assistant_id == "hello-pulse" and power == "hello":
        return validate_hello_input(payload)
    raise ValueError("the Power has no declared input contract")


def validate_power_output(assistant_id: str, power: str, payload: object) -> dict[str, str]:
    if assistant_id != "hello-pulse" or power != "hello":
        raise ValueError("the Power has no declared output contract")
    if not isinstance(payload, dict) or set(payload) != {"message"}:
        raise ValueError("the Assistant returned an invalid result")
    message = payload["message"]
    if not isinstance(message, str) or not 1 <= len(message) <= 256:
        raise ValueError("the Assistant returned an invalid result")
    return {"message": message}
