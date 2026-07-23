"""Closed metadata and exact submissions for Assistant human approval."""

from __future__ import annotations

import chat_orchestrator

from . import approval_challenges

MAX_ORDINAL = 63


class ApprovalFlowError(RuntimeError):
    """An approval request or submission violated its closed contract."""


def _text(value: object, name: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ApprovalFlowError(f"approval request {name} is invalid")
    return value


def requirement(
    interaction: chat_orchestrator.HumanInteraction,
    assistant_name: object,
    assistant_image: object,
    answer_count: int,
) -> approval_challenges.ApprovalRequirement:
    """Validate one SDK sentinel and bind its exact replay call site."""
    if not isinstance(interaction, chat_orchestrator.HumanInteraction):
        raise ApprovalFlowError("approval interaction is invalid")
    payload = interaction.payload
    if set(payload) != {
        "ordinal",
        "kind",
        "request_type",
        "title",
        "summary",
        "docs",
        "options",
        "runs",
    }:
        raise ApprovalFlowError("approval request shape is invalid")
    ordinal = payload["ordinal"]
    if (
        type(ordinal) is not int
        or not 0 <= ordinal <= MAX_ORDINAL
        or type(answer_count) is not int
        or ordinal != answer_count
        or payload["kind"] != "approval"
        or payload["request_type"] != "bool"
        or payload["options"] != []
        or payload["runs"] not in {"always", "once"}
        or not isinstance(assistant_name, str)
        or not assistant_name
        or not isinstance(assistant_image, str)
        or not assistant_image
    ):
        raise ApprovalFlowError("approval request binding is invalid")
    docs = payload["docs"]
    if docs is not None:
        docs = _text(docs, "docs", 2048)
    request = interaction.request
    return approval_challenges.ApprovalRequirement(
        interrupt_id=request.interrupt_id,
        assistant_id=request.assistant_id,
        assistant_name=assistant_name,
        power_id=request.power,
        assistant_image=assistant_image,
        ordinal=ordinal,
        title=_text(payload["title"], "title", 80),
        summary=_text(payload["summary"], "summary", 240),
        docs=docs,
        runs=payload["runs"],
    )


def challenge_payload(challenge: approval_challenges.PendingApprovalChallenge) -> dict[str, object]:
    """Expose only bounded public request metadata."""
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
                "title": requirement.title,
                "summary": requirement.summary,
                "docs": requirement.docs,
                "approval": requirement.runs,
            }
            for requirement in challenge.requirements
        ],
    }


def submitted_answer(
    challenge: approval_challenges.PendingApprovalChallenge,
    body: object,
) -> bool:
    """Accept one explicit affirmative decision for the challenged call site."""
    if (
        not isinstance(body, dict)
        or set(body) != {"challenge_id", "approved"}
        or body.get("challenge_id") != challenge.id
        or body.get("approved") is not True
    ):
        raise ApprovalFlowError("approval submission does not match its challenge")
    if len(challenge.requirements) != 1:
        raise ApprovalFlowError("approval challenge does not bind one call site")
    return True
