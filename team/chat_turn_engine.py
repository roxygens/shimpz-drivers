"""Shared Controller-owned chat turn drive and suspension dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import assistant_account_challenges
import assistant_account_flow
import assistant_chat
import brain_runtime_client
import chat_orchestrator
import oauth_account_store
import power_journal


@dataclass(slots=True)
class SegmentRequirements:
    """Mutable suspension gates populated while one shared segment is driven."""

    accounts: tuple[object, ...] = ()
    secrets: tuple[object, ...] = ()
    inputs: tuple[object, ...] = ()
    approvals: tuple[object, ...] = ()

    def groups(self, *, approvals: bool) -> tuple[tuple[object, ...], ...]:
        groups = (self.accounts, self.secrets, self.inputs)
        return (*groups, self.approvals) if approvals else groups


@dataclass(frozen=True, slots=True)
class PreparedSegment:
    """Controller-specific resources consumed by the shared segment state machine."""

    team_name: str
    identity: tuple[object, ...]
    context: object
    files: list[dict[str, object]]
    durable_batch: object


@dataclass(frozen=True, slots=True)
class SegmentStrategy:
    """Hosted/local adapters for state and errors that intentionally differ."""

    runtime: object
    prepare: Callable[[], PreparedSegment]
    validate_power: Callable
    pause_for_private_inputs: Callable[[tuple[object, ...], SegmentRequirements], bool]
    cancelled: Callable[[], bool]
    validate_context: Callable[[], None]
    raise_problem: Callable[[str, BaseException | None], None]
    finalize: Callable[[], None] = lambda: None
    pause_for_approval: Callable[[tuple[object, ...], SegmentRequirements], bool] | None = None
    approval_granted: Callable | None = None


@dataclass(frozen=True, slots=True)
class AccountResumeStrategy:
    """Controller adapters for admitting one account-gated continuation."""

    store: object
    team_id: str
    challenge_id: object
    pending_valid: Callable[[object], bool]
    pending_identity: Callable[[object], tuple[object, ...]]
    inspect: Callable[[object], AccountResumeContext]
    account_store: object
    challenge_response: Callable[[object], object]
    expired_error: Callable[[], BaseException]
    context_error: Callable[[], BaseException]
    contract_error: Callable[[], BaseException]
    cancel_extra: Callable[[], None] = lambda: None


@dataclass(frozen=True, slots=True)
class AccountResumeContext:
    identity: tuple[object, ...]
    bindings: object
    requests: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class AccountResumeAdmission:
    pending: object | None
    response: object | None


_DRIVE_ERRORS = (
    power_journal.PowerJournalError,
    chat_orchestrator.ChatStoppedError,
    chat_orchestrator.ApprovalRequiredError,
    chat_orchestrator.ChatOrchestrationError,
    brain_runtime_client.BrainRuntimeError,
)


def admit_account_resume(strategy: AccountResumeStrategy) -> AccountResumeAdmission:
    """Make account challenge, identity, requirement and one-use decisions once for both twins."""
    try:
        challenge = strategy.store.get(strategy.team_id, strategy.challenge_id)
    except assistant_account_challenges.AccountChallengeNotFoundError as exc:
        raise strategy.expired_error() from exc
    pending = challenge.payload
    if not strategy.pending_valid(pending):
        raise strategy.context_error()
    context = strategy.inspect(pending)
    if context.identity != strategy.pending_identity(pending):
        strategy.store.cancel_team(strategy.team_id)
        strategy.cancel_extra()
        raise strategy.context_error()
    try:
        missing = assistant_account_flow.requirements_for_batch(
            strategy.team_id,
            context.bindings,
            context.requests,
            strategy.account_store,
        )
    except (assistant_account_flow.AccountFlowError, oauth_account_store.OAuthAccountStoreError) as exc:
        raise strategy.contract_error() from exc
    if missing:
        return AccountResumeAdmission(None, strategy.challenge_response(challenge))
    try:
        claimed = strategy.store.claim(strategy.team_id, challenge.id)
    except assistant_account_challenges.AccountChallengeNotFoundError as exc:
        raise strategy.expired_error() from exc
    if claimed is not challenge:
        raise strategy.expired_error()
    return AccountResumeAdmission(pending, None)


def run_segment(
    strategy: SegmentStrategy,
    *,
    message: str | None,
    continuation: chat_orchestrator.ChatContinuation | None,
    expected_identity: tuple[object, ...] | None,
) -> tuple[
    str,
    tuple[object, ...],
    chat_orchestrator.ChatOutcome | chat_orchestrator.ChatSuspension,
    SegmentRequirements,
]:
    """Apply the same continuation, identity and suspension decisions on both Controllers."""
    if (message is None) == (continuation is None):
        strategy.raise_problem("invalid-continuation", None)
    segment = strategy.prepare()
    if expected_identity is not None and segment.identity != expected_identity:
        strategy.raise_problem("context-changed", None)
    requirements = SegmentRequirements()
    try:
        outcome = drive(
            strategy=strategy,
            segment=segment,
            message=message,
            continuation=continuation,
            requirements=requirements,
        )
    except _DRIVE_ERRORS as exc:
        strategy.raise_problem("drive-error", exc)
        raise AssertionError("chat error adapter returned") from exc
    strategy.finalize()
    groups = requirements.groups(approvals=strategy.pause_for_approval is not None)
    if isinstance(outcome, chat_orchestrator.ChatSuspension) and suspension_gate_count(*groups) != 1:
        strategy.raise_problem("invalid-suspension", None)
    return segment.team_name, segment.identity, outcome, requirements


def drive(
    *,
    strategy: SegmentStrategy,
    segment: PreparedSegment,
    message: str | None = None,
    continuation: chat_orchestrator.ChatContinuation | None = None,
    requirements: SegmentRequirements,
) -> chat_orchestrator.ChatOutcome | chat_orchestrator.ChatSuspension:
    """Run or resume one turn with the same durable Power hooks on both Controllers."""

    def pause_before_batch(requests: tuple[object, ...]) -> bool:
        return strategy.pause_for_private_inputs(requests, requirements)

    hooks = {
        "prepare_batch": segment.durable_batch.prepare,
        "batch_delivered": segment.durable_batch.delivered,
        "pause_before_batch": pause_before_batch,
        "cancelled": strategy.cancelled,
        "validate_context": strategy.validate_context,
    }
    approval_pause = strategy.pause_for_approval
    if approval_pause is not None:

        def pause_for_approval(requests: tuple[object, ...]) -> bool:
            return approval_pause(requests, requirements)

        hooks["pause_for_approval"] = pause_for_approval
    if strategy.approval_granted is not None:
        hooks["approval_granted"] = strategy.approval_granted
    if continuation is None:
        outcome = chat_orchestrator.run_until_pause(
            strategy.runtime,
            segment.context,
            assistant_chat.build_prompt(message, segment.files),
            strategy.validate_power,
            segment.durable_batch.invoke,
            **hooks,
        )
    else:
        outcome = chat_orchestrator.continue_after_pause(
            strategy.runtime,
            segment.context,
            continuation,
            strategy.validate_power,
            segment.durable_batch.invoke,
            **hooks,
        )
    if isinstance(outcome, chat_orchestrator.ChatSuspension) and outcome.interaction is not None:
        if outcome.interaction.payload["kind"] == "request":
            requirements.inputs = (outcome.interaction,)
        else:
            requirements.approvals = (outcome.interaction,)
    return outcome


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
