"""Closed metadata and typed submissions for Assistant human input."""

from __future__ import annotations

import math

import chat_orchestrator

from . import input_challenges

REQUEST_TYPES = frozenset({"str", "int", "float", "bool", "choice", "choices"})
MAX_ORDINAL = 63
MAX_OPTIONS = 64
MAX_OPTION_CHARS = 200
MAX_ANSWER_CHARS = 4096


class InputFlowError(RuntimeError):
    """A human-input request or submission violated its closed contract."""


def _text(value: object, name: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise InputFlowError(f"input request {name} is invalid")
    return value


def requirement(
    interaction: chat_orchestrator.HumanInteraction,
    assistant_image: object,
    answer_count: int,
) -> input_challenges.InputRequirement:
    """Validate one SDK sentinel and bind it to the exact replay position."""
    if not isinstance(interaction, chat_orchestrator.HumanInteraction):
        raise InputFlowError("input interaction is invalid")
    payload = interaction.payload
    if set(payload) != {"ordinal", "kind", "request_type", "title", "summary", "docs", "options"}:
        raise InputFlowError("input request shape is invalid")
    ordinal = payload["ordinal"]
    request_type = payload["request_type"]
    options = payload["options"]
    if (
        type(ordinal) is not int
        or not 0 <= ordinal <= MAX_ORDINAL
        or type(answer_count) is not int
        or ordinal != answer_count
        or not isinstance(request_type, str)
        or request_type not in REQUEST_TYPES
        or payload["kind"] != "request"
        or not isinstance(assistant_image, str)
        or not assistant_image
    ):
        raise InputFlowError("input request binding is invalid")
    if not isinstance(options, list) or len(options) > MAX_OPTIONS:
        raise InputFlowError("input request options are invalid")
    if request_type in {"choice", "choices"}:
        if (
            not options
            or any(
                not isinstance(option, str) or not option or len(option) > MAX_OPTION_CHARS or "\0" in option
                for option in options
            )
            or len(options) != len(set(options))
        ):
            raise InputFlowError("input request options are invalid")
    elif options:
        raise InputFlowError("primitive input request cannot declare options")
    docs = payload["docs"]
    if docs is not None:
        docs = _text(docs, "docs", 2048)
    request = interaction.request
    return input_challenges.InputRequirement(
        interrupt_id=request.interrupt_id,
        assistant_id=request.assistant_id,
        power_id=request.power,
        assistant_image=assistant_image,
        ordinal=ordinal,
        request_type=request_type,
        title=_text(payload["title"], "title", 80),
        summary=_text(payload["summary"], "summary", 240),
        docs=docs,
        options=tuple(options),
    )


def challenge_payload(challenge: input_challenges.PendingInputChallenge) -> dict[str, object]:
    """Expose only bounded public request metadata."""
    requirement = challenge.requirement
    return {
        "team_id": challenge.team_id,
        "status": "input-required",
        "turn_id": challenge.id,
        "challenge_id": challenge.id,
        "request": {
            "type": requirement.request_type,
            "title": requirement.title,
            "summary": requirement.summary,
            "docs": requirement.docs,
            "options": list(requirement.options),
        },
    }


def submitted_answer(challenge: input_challenges.PendingInputChallenge, body: object) -> object:
    """Accept exactly one answer matching the challenged SDK request type."""
    if (
        not isinstance(body, dict)
        or set(body) != {"challenge_id", "answer"}
        or body.get("challenge_id") != challenge.id
    ):
        raise InputFlowError("input submission does not match its challenge")
    answer = body["answer"]
    request = challenge.requirement
    if request.request_type == "str":
        valid = isinstance(answer, str) and len(answer) <= MAX_ANSWER_CHARS and "\0" not in answer
    elif request.request_type == "int":
        valid = type(answer) is int
    elif request.request_type == "float":
        valid = type(answer) is int or (type(answer) is float and math.isfinite(answer))
    elif request.request_type == "bool":
        valid = type(answer) is bool
    elif request.request_type == "choice":
        valid = isinstance(answer, str) and answer in request.options
    else:
        valid = (
            isinstance(answer, list)
            and len(answer) <= len(request.options)
            and all(isinstance(item, str) and item in request.options for item in answer)
            and len(answer) == len(set(answer))
        )
    if not valid:
        raise InputFlowError("input answer has the wrong type or value")
    return answer
