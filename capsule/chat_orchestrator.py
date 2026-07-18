"""Deterministic Controller-owned loop between LangGraph suspensions and Assistant Powers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import brain_runtime_client

MAX_POWER_ROUNDS = 8


class ChatOrchestrationError(RuntimeError):
    """The Brain runtime violated the turn contract or could not finish safely."""


class ApprovalRequiredError(ChatOrchestrationError):
    """A declared Power requires a Captain approval that has not been granted."""

    def __init__(self, request: brain_runtime_client.PowerRequest) -> None:
        super().__init__(f"Power {request.power!r} requires {request.approval} approval")
        self.request = request


class ChatStoppedError(ChatOrchestrationError):
    """The Controller cancelled the active turn between two bounded operations."""


@dataclass(frozen=True, slots=True)
class InvokedPower:
    assistant_id: str
    power: str


@dataclass(frozen=True, slots=True)
class ChatOutcome:
    reply: str
    powers: tuple[InvokedPower, ...]


PowerInvoker = Callable[[str, str, Mapping[str, Any]], object]
CancellationCheck = Callable[[], bool]
ContextCheck = Callable[[], None]


def run(
    runtime: brain_runtime_client.BrainRuntimeClient,
    context: brain_runtime_client.RuntimeContext,
    message: str,
    invoke_power: PowerInvoker,
    *,
    cancelled: CancellationCheck = lambda: False,
    validate_context: ContextCheck = lambda: None,
) -> ChatOutcome:
    """Run a bounded turn; every model-requested Power returns through Controller validation."""
    if cancelled():
        raise ChatStoppedError("chat turn stopped")
    validate_context()
    turn = runtime.start(context, message)
    invoked: list[InvokedPower] = []
    declared = {
        (assistant.id, power.id): power
        for assistant in context.assistants
        for power in assistant.powers
    }

    for _round in range(MAX_POWER_ROUNDS + 1):
        if cancelled():
            raise ChatStoppedError("chat turn stopped")
        if turn.status == "completed":
            validate_context()
            return ChatOutcome(reply=turn.reply, powers=tuple(invoked))
        if _round == MAX_POWER_ROUNDS:
            raise ChatOrchestrationError("Brain exceeded the Power round limit")

        results: dict[str, object] = {}
        for request in turn.powers:
            if cancelled():
                raise ChatStoppedError("chat turn stopped")
            validate_context()
            power = declared.get((request.assistant_id, request.power))
            if power is None or request.approval != power.approval:
                raise ChatOrchestrationError("Brain requested an undeclared Power contract")
            if request.interrupt_id in results:
                raise ChatOrchestrationError("Brain repeated a Power interrupt id")
            if power.approval != "none":
                raise ApprovalRequiredError(request)
            results[request.interrupt_id] = invoke_power(request.assistant_id, power.id, request.input)
            invoked.append(InvokedPower(assistant_id=request.assistant_id, power=power.id))

        if not results:
            raise ChatOrchestrationError("Brain suspended without a Power request")
        validate_context()
        turn = runtime.resume(context, results)

    raise ChatOrchestrationError("Brain did not complete the chat turn")
