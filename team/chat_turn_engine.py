"""Shared Controller-owned chat turn drive and suspension dispatch."""

from __future__ import annotations

from collections.abc import Callable

import assistant_chat
import chat_orchestrator


def drive(
    runtime: object,
    context: object,
    message: str | None,
    files: list[dict[str, object]],
    continuation: chat_orchestrator.ChatContinuation | None,
    validate_power: Callable,
    durable_batch: object,
    pause_before_batch: Callable,
    cancelled: Callable[[], bool],
    validate_context: Callable[[], None],
    *,
    pause_for_approval: Callable | None = None,
    approval_granted: Callable | None = None,
) -> chat_orchestrator.ChatOutcome | chat_orchestrator.ChatSuspension:
    """Run or resume one turn with the same durable Power hooks on both Controllers."""
    hooks = {
        "prepare_batch": durable_batch.prepare,
        "batch_delivered": durable_batch.delivered,
        "pause_before_batch": pause_before_batch,
        "cancelled": cancelled,
        "validate_context": validate_context,
    }
    if pause_for_approval is not None:
        hooks["pause_for_approval"] = pause_for_approval
    if approval_granted is not None:
        hooks["approval_granted"] = approval_granted
    if continuation is None:
        return chat_orchestrator.run_until_pause(
            runtime,
            context,
            assistant_chat.build_prompt(message, files),
            validate_power,
            durable_batch.invoke,
            **hooks,
        )
    return chat_orchestrator.continue_after_pause(
        runtime,
        context,
        continuation,
        validate_power,
        durable_batch.invoke,
        **hooks,
    )


def suspension_gate_count(*requirements: tuple[object, ...]) -> int:
    return sum(bool(group) for group in requirements)


def dispatch(
    outcome: chat_orchestrator.ChatOutcome | chat_orchestrator.ChatSuspension,
    requirements: tuple[tuple[object, ...], ...],
    pending: Callable[[chat_orchestrator.ChatSuspension], object],
    pause: tuple[Callable[[chat_orchestrator.ChatSuspension, tuple[object, ...], object], object], ...],
    complete: Callable[[chat_orchestrator.ChatOutcome], object],
) -> object:
    """Send exactly one suspension kind to its handler, or finish a terminal turn."""
    if not isinstance(outcome, chat_orchestrator.ChatSuspension):
        return complete(outcome)
    if len(requirements) != len(pause) or suspension_gate_count(*requirements) != 1:
        raise ValueError("invalid chat suspension")
    state = pending(outcome)
    for group, handler in zip(requirements, pause, strict=True):
        if group:
            return handler(outcome, group, state)
    raise AssertionError("unreachable")
