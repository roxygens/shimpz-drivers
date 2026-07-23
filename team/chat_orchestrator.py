"""Deterministic Controller-owned loop between LangGraph suspensions and Assistant Powers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import brain_runtime_client
import power_execution

MAX_POWER_ROUNDS = 8


class ChatOrchestrationError(RuntimeError):
    """The Brain runtime violated the turn contract or could not finish safely."""


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


@dataclass(frozen=True, slots=True)
class ChatContinuation:
    """In-memory, secret-free state needed to continue one LangGraph suspension."""

    turn: brain_runtime_client.RuntimeTurn
    seen_interrupts: tuple[str, ...]
    invoked: tuple[InvokedPower, ...]
    round_index: int


@dataclass(frozen=True, slots=True)
class ChatSuspension:
    continuation: ChatContinuation
    requests: tuple[brain_runtime_client.PowerRequest, ...]
    interaction: HumanInteraction | None = None


@dataclass(frozen=True, slots=True)
class HumanInteraction:
    """One exact Power call site that requested deterministic human replay."""

    request: brain_runtime_client.PowerRequest
    payload: dict[str, object]


PowerInvoker = Callable[[brain_runtime_client.PowerRequest], object]
PowerValidator = Callable[[str, str, Mapping[str, Any]], Mapping[str, Any]]
BatchHook = Callable[[tuple[brain_runtime_client.PowerRequest, ...]], None]
BatchPause = Callable[[tuple[brain_runtime_client.PowerRequest, ...]], bool]
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
        if power is None:
            raise ChatOrchestrationError("Brain requested an undeclared Power contract")
        if request.interrupt_id in seen_interrupts:
            raise ChatOrchestrationError("Brain repeated a Power interrupt id")
        seen_interrupts.add(request.interrupt_id)
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
            )
        )
    return tuple(validated)


def _drive(
    runtime: brain_runtime_client.BrainRuntimeClient,
    context: brain_runtime_client.RuntimeContext,
    continuation: ChatContinuation,
    validate_power: PowerValidator,
    invoke_power: PowerInvoker,
    *,
    prepare_batch: BatchHook = lambda _batch: None,
    batch_delivered: BatchHook = lambda _batch: None,
    pause_before_batch: BatchPause = lambda _batch: False,
    cancelled: CancellationCheck = lambda: False,
    validate_context: ContextCheck = lambda: None,
) -> ChatOutcome | ChatSuspension:
    turn = continuation.turn
    invoked = list(continuation.invoked)
    seen_interrupts = set(continuation.seen_interrupts)
    declared = {(assistant.id, power.id): power for assistant in context.assistants for power in assistant.powers}

    for _round in range(continuation.round_index, MAX_POWER_ROUNDS + 1):
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
        if pause_before_batch(batch):
            return ChatSuspension(
                continuation=ChatContinuation(
                    turn=turn,
                    seen_interrupts=tuple(sorted(seen_interrupts)),
                    invoked=tuple(invoked),
                    round_index=_round,
                ),
                requests=batch,
            )
        prepare_batch(batch)
        results: dict[str, object] = {}
        batch_invoked: list[InvokedPower] = []
        for request in batch:
            if cancelled():
                raise ChatStoppedError("chat turn stopped")
            validate_context()
            result = invoke_power(request)
            if isinstance(result, power_execution.RpcSuspension):
                if result.payload.get("kind") not in {"request", "approval"}:
                    raise ChatOrchestrationError("Power requested an invalid human interaction")
                return ChatSuspension(
                    continuation=ChatContinuation(
                        turn=turn,
                        seen_interrupts=tuple(sorted(seen_interrupts)),
                        invoked=tuple(invoked),
                        round_index=_round,
                    ),
                    requests=batch,
                    interaction=HumanInteraction(request, result.payload),
                )
            results[request.interrupt_id] = result
            batch_invoked.append(InvokedPower(assistant_id=request.assistant_id, power=request.power))

        validate_context()
        seen_interrupts.update(batch_interrupts)
        resumed = runtime.resume(context, results)
        if resumed.status == "power-required" and not seen_interrupts.isdisjoint(
            request.interrupt_id for request in resumed.powers
        ):
            raise ChatOrchestrationError("Brain repeated a Power interrupt across rounds")
        batch_delivered(batch)
        invoked.extend(batch_invoked)
        turn = resumed

    raise ChatOrchestrationError("Brain did not complete the chat turn")


def run_until_pause(
    runtime: brain_runtime_client.BrainRuntimeClient,
    context: brain_runtime_client.RuntimeContext,
    message: str,
    validate_power: PowerValidator,
    invoke_power: PowerInvoker,
    *,
    prepare_batch: BatchHook = lambda _batch: None,
    batch_delivered: BatchHook = lambda _batch: None,
    pause_before_batch: BatchPause = lambda _batch: False,
    cancelled: CancellationCheck = lambda: False,
    validate_context: ContextCheck = lambda: None,
) -> ChatOutcome | ChatSuspension:
    """Start a turn and optionally pause before an all-or-nothing Power batch."""
    if cancelled():
        raise ChatStoppedError("chat turn stopped")
    validate_context()
    turn = runtime.start(context, message)
    return _drive(
        runtime,
        context,
        ChatContinuation(turn=turn, seen_interrupts=(), invoked=(), round_index=0),
        validate_power,
        invoke_power,
        prepare_batch=prepare_batch,
        batch_delivered=batch_delivered,
        pause_before_batch=pause_before_batch,
        cancelled=cancelled,
        validate_context=validate_context,
    )


def continue_after_pause(
    runtime: brain_runtime_client.BrainRuntimeClient,
    context: brain_runtime_client.RuntimeContext,
    continuation: ChatContinuation,
    validate_power: PowerValidator,
    invoke_power: PowerInvoker,
    *,
    prepare_batch: BatchHook = lambda _batch: None,
    batch_delivered: BatchHook = lambda _batch: None,
    pause_before_batch: BatchPause = lambda _batch: False,
    cancelled: CancellationCheck = lambda: False,
    validate_context: ContextCheck = lambda: None,
) -> ChatOutcome | ChatSuspension:
    """Continue an admitted in-memory suspension without re-running the user turn."""
    return _drive(
        runtime,
        context,
        continuation,
        validate_power,
        invoke_power,
        prepare_batch=prepare_batch,
        batch_delivered=batch_delivered,
        pause_before_batch=pause_before_batch,
        cancelled=cancelled,
        validate_context=validate_context,
    )


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
    outcome = run_until_pause(
        runtime,
        context,
        message,
        validate_power,
        invoke_power,
        prepare_batch=prepare_batch,
        batch_delivered=batch_delivered,
        cancelled=cancelled,
        validate_context=validate_context,
    )
    if isinstance(outcome, ChatSuspension):
        raise ChatOrchestrationError("chat turn paused without a Controller continuation")
    return outcome
