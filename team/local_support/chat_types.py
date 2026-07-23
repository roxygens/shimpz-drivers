"""Shared local chat bindings and continuation types."""

from dataclasses import dataclass
from http import HTTPStatus

import local_chat_continuations
from local_registry import AssistantSpec

from local_support.errors import ApiProblemError


@dataclass(frozen=True, slots=True)
class ActiveAssistant:
    spec: AssistantSpec
    container_id: str
    container: object | None = None


PendingLocalChat = local_chat_continuations.PendingLocalChat


def required_active_assistant(
    bindings: dict[str, ActiveAssistant],
    assistant_id: str,
) -> ActiveAssistant:
    active = bindings.get(assistant_id)
    if active is None:
        raise ApiProblemError(
            HTTPStatus.CONFLICT,
            "Brain requested an unavailable Assistant",
            code="assistant-unavailable",
        )
    return active
