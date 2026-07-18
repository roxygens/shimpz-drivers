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


PowerInvoker = Callable[[brain_runtime_client.PowerRequest], object]
PowerValidator = Callable[[str, str, Mapping[str, Any]], Mapping[str, Any]]
BatchHook = Callable[[tuple[brain_runtime_client.PowerRequest, ...]], None]
CancellationCheck = Callable[[], bool]
ContextCheck = Callable[[], None]


def _validate_batch(
    requests: tuple[brain_runtime_client.PowerRequest, ...],
    declared: Mapping[tuple[str, str], brain_runtime_client.RuntimePower],
    validate_power: PowerValidator,
) -> tuple[brain_runtime_client.PowerRequest, ...]:
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

    validated: list[brain_runtime_client.PowerRequest] = []
    for request, power in contracts:
        safe_input = validate_power(request.assistant_id, power.id, request.input)
        if not isinstance(safe_input, Mapping):
            raise ChatOrchestrationError("Power validator returned an invalid input contract")
        validated.append(
            brain_runtime_client.PowerRequest(
                interrupt_id=request.interrupt_id,
                assistant_id=request.assistant_id,
                power=power.id,
                input=dict(safe_input),
                approval=request.approval,
            )
        )
    return tuple(validated)


def run(
    runtime: brain_runtime_client.BrainRuntimeClient,
    context: brain_runtime_client.RuntimeContext,
    message: str,
    validate_power: PowerValidator,
    invoke_power: PowerInvoker,
    *,
    prepare_batch: BatchHook = lambda _batch: None,
    batch_delivered: BatchHook = lambda _batch: None,
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
        batch_interrupts = {request.interrupt_id for request in batch}
        if not seen_interrupts.isdisjoint(batch_interrupts):
            raise ChatOrchestrationError("Brain repeated a Power interrupt across rounds")
        seen_interrupts.update(batch_interrupts)
        prepare_batch(batch)
        results: dict[str, object] = {}
        for request in batch:
            if cancelled():
                raise ChatStoppedError("chat turn stopped")
            validate_context()
            results[request.interrupt_id] = invoke_power(request)
            invoked.append(InvokedPower(assistant_id=request.assistant_id, power=request.power))

        validate_context()
        resumed = runtime.resume(context, results)
        if resumed.status == "power-required" and not seen_interrupts.isdisjoint(
            request.interrupt_id for request in resumed.powers
        ):
            raise ChatOrchestrationError("Brain repeated a Power interrupt across rounds")
        batch_delivered(batch)
        turn = resumed

    raise ChatOrchestrationError("Brain did not complete the chat turn")
