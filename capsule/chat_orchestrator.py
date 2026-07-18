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
PowerValidator = Callable[[str, str, Mapping[str, Any]], Mapping[str, Any]]
CancellationCheck = Callable[[], bool]
ContextCheck = Callable[[], None]


@dataclass(frozen=True, slots=True)
class _ValidatedPower:
    request: brain_runtime_client.PowerRequest
    power: brain_runtime_client.RuntimePower
    input: Mapping[str, Any]


def _validate_batch(
    requests: tuple[brain_runtime_client.PowerRequest, ...],
    declared: Mapping[tuple[str, str], brain_runtime_client.RuntimePower],
    validate_power: PowerValidator,
) -> tuple[_ValidatedPower, ...]:
    """Validate a complete suspension before allowing its first side effect."""
    if not requests:
        raise ChatOrchestrationError("Brain suspended without a Power request")

    seen_interrupts: set[str] = set()
    contracts: list[tuple[brain_runtime_client.PowerRequest, brain_runtime_client.RuntimePower]] = []
    for request in requests:
        power = declared.get((request.assistant_id, request.power))
        if power is None or request.approval != power.approval:
            raise ChatOrchestrationError("Brain requested an undeclared Power contract")
        if request.interrupt_id in seen_interrupts:
            raise ChatOrchestrationError("Brain repeated a Power interrupt id")
        seen_interrupts.add(request.interrupt_id)
        if power.approval != "none":
            raise ApprovalRequiredError(request)
        contracts.append((request, power))

    validated: list[_ValidatedPower] = []
    for request, power in contracts:
        safe_input = validate_power(request.assistant_id, power.id, request.input)
        if not isinstance(safe_input, Mapping):
            raise ChatOrchestrationError("Power validator returned an invalid input contract")
        validated.append(_ValidatedPower(request=request, power=power, input=dict(safe_input)))
    return tuple(validated)


def run(
    runtime: brain_runtime_client.BrainRuntimeClient,
    context: brain_runtime_client.RuntimeContext,
    message: str,
    validate_power: PowerValidator,
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
    seen_interrupts: set[str] = set()
    declared = {(assistant.id, power.id): power for assistant in context.assistants for power in assistant.powers}

    for _round in range(MAX_POWER_ROUNDS + 1):
        if cancelled():
            raise ChatStoppedError("chat turn stopped")
        if turn.status == "completed":
            validate_context()
            return ChatOutcome(reply=turn.reply, powers=tuple(invoked))
        if _round == MAX_POWER_ROUNDS:
            raise ChatOrchestrationError("Brain exceeded the Power round limit")

        validate_context()
        batch = _validate_batch(turn.powers, declared, validate_power)
        batch_interrupts = {item.request.interrupt_id for item in batch}
        if not seen_interrupts.isdisjoint(batch_interrupts):
            raise ChatOrchestrationError("Brain repeated a Power interrupt across rounds")
        seen_interrupts.update(batch_interrupts)
        results: dict[str, object] = {}
        for item in batch:
            if cancelled():
                raise ChatStoppedError("chat turn stopped")
            validate_context()
            results[item.request.interrupt_id] = invoke_power(
                item.request.assistant_id,
                item.power.id,
                item.input,
            )
            invoked.append(InvokedPower(assistant_id=item.request.assistant_id, power=item.power.id))

        validate_context()
        turn = runtime.resume(context, results)

    raise ChatOrchestrationError("Brain did not complete the chat turn")
