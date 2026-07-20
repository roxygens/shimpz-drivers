"""Closed public metadata and exact submissions for local Power approval."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Protocol

import assistant_approval_challenges
import brain_runtime_client
from local_registry import AssistantSpec

MAX_APPROVAL_REQUESTS = 64
MAX_APPROVAL_INPUT_BYTES = 16 * 1024
MAX_APPROVAL_BATCH_INPUT_BYTES = 64 * 1024


class ApprovalFlowError(RuntimeError):
    """An approval request or submission violated its closed contract."""


class _ActiveBinding(Protocol):
    spec: AssistantSpec


def requirements_for_batch(
    bindings: Mapping[str, _ActiveBinding],
    requests: Sequence[brain_runtime_client.PowerRequest],
) -> tuple[assistant_approval_challenges.ApprovalRequirement, ...]:
    """Project only public metadata while retaining exact interrupt bindings internally."""
    if not requests or len(requests) > MAX_APPROVAL_REQUESTS:
        raise ApprovalFlowError("approval batch size is invalid")
    requirements: list[assistant_approval_challenges.ApprovalRequirement] = []
    seen: set[str] = set()
    total_input_bytes = 0
    for request in requests:
        active = bindings.get(request.assistant_id)
        power = active.spec.powers.get(request.power) if active is not None else None
        if power is None or request.approval == "none" or request.approval != power.approval:
            raise ApprovalFlowError("Power approval contract is unavailable")
        if request.interrupt_id in seen:
            raise ApprovalFlowError("approval batch repeats an interrupt")
        seen.add(request.interrupt_id)
        try:
            input_json = json.dumps(
                request.input,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            input_bytes = len(input_json.encode("utf-8"))
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ApprovalFlowError("Power approval input is not public JSON") from exc
        if input_bytes > MAX_APPROVAL_INPUT_BYTES:
            raise ApprovalFlowError("Power approval input exceeds its fixed limit")
        total_input_bytes += input_bytes
        if total_input_bytes > MAX_APPROVAL_BATCH_INPUT_BYTES:
            raise ApprovalFlowError("Power approval batch input exceeds its fixed limit")
        requirements.append(
            assistant_approval_challenges.ApprovalRequirement(
                interrupt_id=request.interrupt_id,
                assistant_id=request.assistant_id,
                assistant_name=active.spec.name,
                power_id=request.power,
                power_summary=power.summary,
                input_json=input_json,
            )
        )
    return tuple(requirements)


def challenge_payload(challenge: assistant_approval_challenges.PendingApprovalChallenge) -> dict[str, object]:
    """Expose bounded schema-validated input, but no interrupt id or provider credential."""
    return {
        "team_id": challenge.team_id,
        "status": "approval-required",
        "turn_id": challenge.id,
        "challenge_id": challenge.id,
        "requirements": [
            {
                "assistant_id": requirement.assistant_id,
                "assistant_name": requirement.assistant_name,
                "power_id": requirement.power_id,
                "power_summary": requirement.power_summary,
                "input": json.loads(requirement.input_json),
            }
            for requirement in challenge.requirements
        ],
    }


def approved_interrupts(
    challenge: assistant_approval_challenges.PendingApprovalChallenge,
    body: object,
) -> frozenset[str]:
    """Accept exactly one explicit affirmative decision for the complete paused batch."""
    if (
        not isinstance(body, dict)
        or set(body) != {"challenge_id", "approved"}
        or body.get("challenge_id") != challenge.id
        or body.get("approved") is not True
    ):
        raise ApprovalFlowError("approval submission does not match its challenge")
    identifiers = frozenset(requirement.interrupt_id for requirement in challenge.requirements)
    if len(identifiers) != len(challenge.requirements):
        raise ApprovalFlowError("approval challenge repeats an interrupt")
    return identifiers
