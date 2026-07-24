"""Hosted Team chat segment preparation, execution, and suspension dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import NoReturn

import assistant_account_challenges
import assistant_account_flow
import assistant_secret_challenges
import assistant_secret_flow
import assistant_secret_store
import brain_runtime_client
import chat_orchestrator
import chat_turn_engine
import docker.errors
import inference_config
import manifests
import marketplace
import oauth_account_store
import power_execution
import power_journal
import runtime_state
from container_policy import hosted_apps, hosted_resources
from container_policy import network as network_policy

from assistant_human import approval_challenges as assistant_approval_challenges
from assistant_human import approval_flow as assistant_approval_flow
from assistant_human import approval_grants as assistant_approval_grants
from assistant_human import hosted_assistants
from assistant_human import input_challenges as assistant_input_challenges
from assistant_human import input_flow as assistant_input_flow


def _current_team_anchor(
    team_id: str,
    container_id: str,
    owner: str,
    inspect_memo: dict[str, dict[str, dict]] | None = None,
):
    container = hosted_resources._get_container(manifests.team_container_name(team_id))
    if container is None:
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "Team identity changed during the chat turn")
    try:
        container.reload()
    except docker.errors.DockerException as exc:
        raise runtime_state.ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Team identity could not be inspected") from exc
    if (
        container.id != container_id
        or not network_policy.brain_identity_valid(container.attrs, team_id)
        or str(container.labels.get("team.owner", "")) != owner
    ):
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "Team identity changed during the chat turn")
    hosted_resources._require_running_team_isolation(container, inspect_memo)
    return container


def _hosted_chat_setup(
    team_id: str,
    file_ids: object,
    assistant_ids: tuple[str, ...],
    container,
    owner: str,
) -> tuple[
    str,
    tuple[hosted_assistants._ActiveAssistant, ...],
    list[dict[str, object]],
    inference_config.InferenceConfig,
    str,
    int,
    tuple[object, ...],
]:
    team_name = hosted_resources._team_name_from_anchor(container)
    assistants = hosted_assistants._select_team_assistants(
        hosted_assistants._active_team_assistants(team_id), assistant_ids
    )
    files = hosted_assistants._chat_file_metadata(team_id, file_ids)
    try:
        config = runtime_state._inference_store.load(team_id)
    except inference_config.InferenceConfigError as exc:
        raise runtime_state.ApiError(
            HTTPStatus.CONFLICT, "configure this Team's model provider before chatting"
        ) from exc
    api_key, generation = hosted_assistants._model_credential(owner, config.provider)
    hosted_assistants._require_model_credential_current(owner, config.provider, generation)
    identity = (
        container.id,
        owner,
        team_name,
        tuple((active.assistant_id, active.container.id) for active in assistants),
        files,
        config,
        generation,
    )
    return team_name, assistants, files, config, api_key, generation, identity


def _raise_hosted_chat_problem(reason: str, exc: BaseException | None) -> NoReturn:
    if reason == "invalid-continuation" or reason == "invalid-suspension":
        raise runtime_state.ApiError(
            HTTPStatus.INTERNAL_SERVER_ERROR, f"invalid chat {reason.removeprefix('invalid-')}"
        )
    if reason == "context-changed":
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry")
    if isinstance(exc, power_journal.PowerJournalError):
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE, "Team Power execution state is unavailable"
        ) from exc
    if isinstance(exc, chat_orchestrator.ChatStoppedError):
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "brain turn stopped") from exc
    if isinstance(exc, chat_orchestrator.ChatOrchestrationError):
        raise runtime_state.ApiError(HTTPStatus.BAD_GATEWAY, "Brain could not complete the Assistant turn") from exc
    if isinstance(exc, brain_runtime_client.BrainRuntimeError):
        raise runtime_state.ApiError(HTTPStatus.BAD_GATEWAY, "Brain runtime is unavailable") from exc
    raise AssertionError(f"unknown hosted chat failure: {reason}")


def _hosted_private_requirements(
    team_id: str,
    bindings: dict[str, hosted_assistants._ActiveAssistant],
    requests: tuple[brain_runtime_client.PowerRequest, ...],
) -> tuple[
    tuple[assistant_account_challenges.AccountRequirement, ...],
    tuple[assistant_secret_challenges.SecretRequirement, ...],
]:
    try:
        accounts = assistant_account_flow.requirements_for_batch(
            team_id,
            hosted_assistants._secret_bindings(bindings),
            requests,
            runtime_state._assistant_accounts,
        )
    except (
        assistant_account_flow.AccountFlowError,
        oauth_account_store.OAuthAccountStoreError,
    ) as exc:
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "Assistant account contract is unavailable") from exc
    if accounts:
        return accounts, ()
    try:
        secrets_required = assistant_secret_flow.requirements_for_batch(
            team_id,
            hosted_assistants._secret_bindings(bindings),
            requests,
            runtime_state._assistant_secrets,
        )
    except assistant_secret_store.AssistantSecretError as exc:
        runtime_state._raise_assistant_secret_error(exc)
    except assistant_secret_flow.SecretFlowError as exc:
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "Assistant secret contract is unavailable") from exc
    return (), secrets_required


def _hosted_approval_requirement(
    team_id: str,
    interactions: tuple[chat_orchestrator.HumanInteraction, ...],
    answers_by_interrupt: dict[str, tuple[object, ...]],
    bindings: dict[str, hosted_assistants._ActiveAssistant],
) -> tuple[assistant_approval_challenges.ApprovalRequirement | None, bool]:
    if not interactions:
        return None, False
    if len(interactions) != 1:
        raise runtime_state.ApiError(HTTPStatus.BAD_GATEWAY, "Assistant human approval request is invalid")
    interaction = interactions[0]
    answers = answers_by_interrupt.get(interaction.request.interrupt_id, ())
    active = bindings.get(interaction.request.assistant_id)
    if active is None:
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "Brain requested an unavailable Assistant")
    try:
        requirement = assistant_approval_flow.requirement(
            interaction,
            active.assistant_id.replace("-", " ").title(),
            hosted_assistants._hosted_power_identity(active)[1],
            len(answers),
        )
        granted = requirement.runs == "once" and runtime_state._assistant_approval_grants.is_granted(
            team_id,
            requirement.assistant_id,
            requirement.power_id,
            requirement.assistant_image,
            requirement.ordinal,
        )
    except assistant_approval_flow.ApprovalFlowError as exc:
        raise runtime_state.ApiError(HTTPStatus.BAD_GATEWAY, "Assistant human approval request is invalid") from exc
    except assistant_approval_grants.ApprovalGrantError as exc:
        raise runtime_state.ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Assistant approval state is unavailable") from exc
    return requirement, granted


def _hosted_answer_log(
    answer_logs: tuple[tuple[str, tuple[object, ...]], ...],
) -> dict[str, tuple[object, ...]]:
    answers_by_interrupt = dict(answer_logs)
    if len(answers_by_interrupt) != len(answer_logs):
        raise runtime_state.ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "invalid chat answer log")
    return answers_by_interrupt


@dataclass(frozen=True, slots=True)
class HostedChatSegmentRequest:
    team_id: str
    file_ids: object
    assistant_ids: tuple[str, ...]
    token: str
    container: object
    owner: str
    message: str | None = None
    continuation: chat_orchestrator.ChatContinuation | None = None
    expected_identity: tuple[object, ...] | None = None
    answer_logs: tuple[tuple[str, tuple[object, ...]], ...] = ()


def _hosted_chat_current_identity(
    request: HostedChatSegmentRequest,
    assistants: tuple[hosted_assistants._ActiveAssistant, ...],
    config: inference_config.InferenceConfig | None,
    generation: int,
    inspect_memo: dict[str, dict[str, dict]],
) -> tuple[object, ...]:
    if config is None:
        raise AssertionError("hosted chat segment was not prepared")
    current_anchor = _current_team_anchor(
        request.team_id,
        request.container.id,
        request.owner,
        inspect_memo,
    )
    team_name = hosted_resources._team_name_from_anchor(current_anchor)
    current_assistants = tuple(
        hosted_assistants._installed_assistant(request.team_id, active.assistant_id, inspect_memo)[2]
        for active in assistants
    )
    files = hosted_assistants._chat_file_metadata(request.team_id, request.file_ids)
    try:
        current_config = runtime_state._inference_store.load(request.team_id)
    except inference_config.InferenceConfigError as exc:
        raise runtime_state.ApiError(
            HTTPStatus.CONFLICT, "configure this Team's model provider before chatting"
        ) from exc
    hosted_assistants._require_model_credential_current(request.owner, config.provider, generation)
    return (
        current_anchor.id,
        request.owner,
        team_name,
        tuple(
            (active.assistant_id, container.id)
            for active, container in zip(assistants, current_assistants, strict=True)
        ),
        files,
        current_config,
        generation,
    )


def _execute_hosted_power(
    team_id: str,
    token: str,
    bindings: dict[str, hosted_assistants._ActiveAssistant],
    answers_by_interrupt: dict[str, tuple[object, ...]],
    inspect_memo: dict[str, dict[str, dict]],
    request: brain_runtime_client.PowerRequest,
) -> object:
    active = bindings.get(request.assistant_id)
    if active is None:
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "Brain requested an unavailable Assistant")
    invocation = hosted_assistants._invoke_assistant_power(
        hosted_assistants.PowerInvocationRequest(
            team_id=team_id,
            token=token,
            assistant_id=request.assistant_id,
            contract=active.contract,
            container=active.container,
            power=request.power,
            payload=request.input,
            answers=answers_by_interrupt.get(request.interrupt_id, ()),
            inspect_memo=inspect_memo,
        )
    )
    if "suspend" in invocation:
        return power_execution.RpcSuspension(invocation["suspend"])
    return invocation["result"]


def _run_hosted_chat_segment(request: HostedChatSegmentRequest) -> chat_turn_engine.SegmentResult:
    team_id, assistant_ids, token, container, owner = (
        request.team_id,
        request.assistant_ids,
        request.token,
        request.container,
        request.owner,
    )
    answers_by_interrupt = _hosted_answer_log(request.answer_logs)
    bindings: dict[str, hosted_assistants._ActiveAssistant] = {}
    initial_identity: tuple[object, ...] = ()
    config: inference_config.InferenceConfig | None = None
    generation = 0
    prepared_assistants: tuple[hosted_assistants._ActiveAssistant, ...] = ()
    inspect_memo: dict[str, dict[str, dict]] = {}

    def require_current_credential() -> None:
        if config is None:
            raise AssertionError("hosted chat segment was not prepared")
        hosted_assistants._require_model_credential_current(owner, config.provider, generation)

    def validate_power(assistant_id: str, power: str, power_input) -> object:
        return hosted_assistants._validate_assistant_power_input(bindings, assistant_id, power, power_input)

    def execute_power(request: brain_runtime_client.PowerRequest) -> object:
        require_current_credential()
        return _execute_hosted_power(team_id, token, bindings, answers_by_interrupt, inspect_memo, request)

    def prepare() -> chat_turn_engine.PreparedSegment:
        nonlocal bindings, config, generation, initial_identity, prepared_assistants
        team_name, prepared_assistants, files, config, api_key, generation, initial_identity = _hosted_chat_setup(
            team_id,
            request.file_ids,
            assistant_ids,
            container,
            owner,
        )
        genesis_by_id = {
            active.assistant_id: hosted_apps._require_assistant_genesis(active.container)
            for active in prepared_assistants
        }
        context = brain_runtime_client.RuntimeContext(
            thread_id=hosted_resources._brain_thread_id(team_id, container.id),
            team_name=team_name,
            assistants=tuple(
                brain_runtime_client.RuntimeAssistant(
                    id=active.assistant_id,
                    genesis=genesis_by_id[active.assistant_id],
                    powers=tuple(
                        brain_runtime_client.RuntimePower(
                            id=power_id,
                            summary=power.summary,
                            input_schema=power.input_schema,
                        )
                        for power_id, power in sorted(active.contract.powers.items())
                    ),
                )
                for active in prepared_assistants
            ),
            provider=config.provider,
            model=config.model,
            api_key=api_key,
        )
        bindings = {active.assistant_id: active for active in prepared_assistants}
        batch = power_execution.PowerBatch(
            runtime_state._power_execution_journal,
            container.id,
            context.thread_id,
            bindings,
            power_execution.PowerBatchStrategy(
                hosted_assistants._hosted_power_identity,
                execute_power,
                lambda request: hosted_assistants._require_hosted_power_rpc_envelope(
                    team_id,
                    bindings,
                    request,
                    answers_by_interrupt.get(request.interrupt_id, ()),
                ),
                lambda request: hosted_assistants._power_secret_generations(
                    team_id, bindings[request.assistant_id], request.power
                ),
                lambda request: hosted_assistants._power_account_generations(
                    team_id, bindings[request.assistant_id], request.power
                ),
            ),
        )
        return chat_turn_engine.PreparedSegment(team_name, initial_identity, context, files, batch)

    def pause_for_private_inputs(
        requests: tuple[object, ...],
        requirements: chat_turn_engine.SegmentRequirements,
    ) -> bool:
        requirements.accounts, requirements.secrets = _hosted_private_requirements(
            team_id,
            bindings,
            requests,
        )
        return bool(requirements.accounts or requirements.secrets)

    def validate_context() -> None:
        nonlocal inspect_memo
        inspect_memo = {}
        current_identity = _hosted_chat_current_identity(
            request,
            prepared_assistants,
            config,
            generation,
            inspect_memo,
        )
        if current_identity != initial_identity:
            raise runtime_state.ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry")

    current_message = request.message
    current_continuation = request.continuation
    current_identity = request.expected_identity
    while True:
        team_name, identity, outcome, requirements = chat_turn_engine.run_segment(
            chat_turn_engine.SegmentStrategy(
                runtime=runtime_state._brain_runtime,
                prepare=prepare,
                validate_power=validate_power,
                pause_for_private_inputs=pause_for_private_inputs,
                cancelled=lambda: runtime_state._token_cancelled(token),
                validate_context=validate_context,
                raise_problem=_raise_hosted_chat_problem,
                finalize=require_current_credential,
            ),
            message=current_message,
            continuation=current_continuation,
            expected_identity=current_identity,
        )
        approval_requirements: tuple[assistant_approval_challenges.ApprovalRequirement, ...] = ()
        requirement, granted = _hosted_approval_requirement(
            team_id,
            requirements.approvals,
            answers_by_interrupt,
            bindings,
        )
        if requirement is not None and granted:
            answers = answers_by_interrupt.get(requirement.interrupt_id, ())
            answers_by_interrupt[requirement.interrupt_id] = (*answers, True)
            if not isinstance(outcome, chat_orchestrator.ChatSuspension):
                raise AssertionError("approval requirement did not suspend")
            current_message = None
            current_continuation = outcome.continuation
            current_identity = identity
            continue
        if requirement is not None:
            approval_requirements = (requirement,)
        return chat_turn_engine.SegmentResult(
            team_name,
            identity,
            outcome,
            requirements.accounts,
            requirements.secrets,
            requirements.inputs,
            approval_requirements,
            tuple(sorted(answers_by_interrupt.items())),
        )


def _pause_hosted_chat(
    team_id: str,
    token: str,
    outcome: chat_orchestrator.ChatSuspension,
    requirements: tuple[assistant_secret_challenges.SecretRequirement, ...],
    pending: hosted_assistants._PendingHostedChat,
) -> dict[str, object]:
    try:
        challenge = runtime_state._assistant_secret_challenges.create(team_id, requirements, pending)
    except assistant_secret_challenges.SecretChallengeError as exc:
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "Assistant secret request is already pending") from exc
    _commit_hosted_suspension(team_id, token, outcome, pending, runtime_state._assistant_secret_challenges)
    return assistant_secret_flow.challenge_payload(challenge)


def _commit_hosted_suspension(
    team_id: str,
    token: str,
    outcome: chat_orchestrator.ChatSuspension,
    pending: hosted_assistants._PendingHostedChat,
    challenge_store: object,
) -> None:
    chat_turn_engine.commit_suspension(
        outcome.continuation,
        pending.continuation,
        lambda: runtime_state._commit_chat_terminal(team_id, token),
        lambda: challenge_store.cancel_team(team_id),
        lambda: runtime_state.ApiError(HTTPStatus.CONFLICT, "brain turn stopped"),
    )


def _hosted_account_challenge_payload(
    challenge: assistant_account_challenges.PendingAccountChallenge,
) -> dict[str, object]:
    bindings: dict[str, hosted_assistants._HostedAssistantSecretBinding] = {}
    try:
        for requirement in challenge.requirements:
            assistant_id, contract, container = hosted_assistants._installed_assistant(
                challenge.team_id,
                requirement.assistant_id,
            )
            active = hosted_assistants._ActiveAssistant(assistant_id, contract, container)
            bindings[assistant_id] = hosted_assistants._HostedAssistantSecretBinding(
                hosted_assistants._hosted_secret_spec(active)
            )
        return assistant_account_flow.challenge_payload(challenge, bindings)
    except (marketplace.MarketplaceError, assistant_account_flow.AccountFlowError) as exc:
        raise runtime_state.ApiError(
            HTTPStatus.CONFLICT, "Assistant account contract changed; retry the message"
        ) from exc


def _pause_hosted_connection(
    team_id: str,
    token: str,
    outcome: chat_orchestrator.ChatSuspension,
    requirements: tuple[assistant_account_challenges.AccountRequirement, ...],
    pending: hosted_assistants._PendingHostedChat,
) -> dict[str, object]:
    try:
        challenge = runtime_state._assistant_account_challenges.create(team_id, requirements, pending)
    except assistant_account_challenges.AccountChallengeError as exc:
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "Assistant account request is already pending") from exc
    _commit_hosted_suspension(team_id, token, outcome, pending, runtime_state._assistant_account_challenges)
    return _hosted_account_challenge_payload(challenge)


def _pause_hosted_input(
    team_id: str,
    token: str,
    outcome: chat_orchestrator.ChatSuspension,
    requirements: tuple[chat_orchestrator.HumanInteraction, ...],
    pending: hosted_assistants._PendingHostedChat,
) -> dict[str, object]:
    if len(requirements) != 1:
        raise runtime_state.ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "invalid human input suspension")
    interaction = requirements[0]
    answers = dict(pending.answer_logs).get(interaction.request.interrupt_id, ())
    try:
        assistant_id, contract, container = hosted_assistants._installed_assistant(
            team_id,
            interaction.request.assistant_id,
        )
        requirement = assistant_input_flow.requirement(
            interaction,
            hosted_assistants._hosted_power_identity(
                hosted_assistants._ActiveAssistant(assistant_id, contract, container)
            )[1],
            len(answers),
        )
        challenge = runtime_state._assistant_input_challenges.create(team_id, requirement, pending)
    except (
        marketplace.MarketplaceError,
        assistant_input_challenges.InputChallengeError,
        assistant_input_flow.InputFlowError,
    ) as exc:
        raise runtime_state.ApiError(HTTPStatus.BAD_GATEWAY, "Assistant human input request is invalid") from exc
    _commit_hosted_suspension(team_id, token, outcome, pending, runtime_state._assistant_input_challenges)
    return assistant_input_flow.challenge_payload(challenge)


def _pause_hosted_approval(
    team_id: str,
    token: str,
    outcome: chat_orchestrator.ChatSuspension,
    requirements: tuple[assistant_approval_challenges.ApprovalRequirement, ...],
    pending: hosted_assistants._PendingHostedChat,
) -> dict[str, object]:
    try:
        challenge = runtime_state._assistant_approval_challenges.create(team_id, requirements, pending)
    except assistant_approval_challenges.ApprovalChallengeError as exc:
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "Assistant approval is already pending") from exc
    _commit_hosted_suspension(team_id, token, outcome, pending, runtime_state._assistant_approval_challenges)
    return assistant_approval_flow.challenge_payload(challenge)


def _hosted_segment_response(
    team_id: str,
    token: str,
    segment: chat_turn_engine.SegmentResult,
    assistant_ids: tuple[str, ...],
    file_ids: tuple[str, ...],
    owner: str,
) -> dict[str, object]:
    def pending(suspension: chat_orchestrator.ChatSuspension) -> hosted_assistants._PendingHostedChat:
        return hosted_assistants._PendingHostedChat(
            continuation=suspension.continuation,
            assistant_ids=assistant_ids,
            file_ids=file_ids,
            owner=owner,
            identity=segment.identity,
            answer_logs=segment.answer_logs,
        )

    def complete(terminal: chat_orchestrator.ChatOutcome) -> dict[str, object]:
        if not runtime_state._commit_chat_terminal(team_id, token):
            raise runtime_state.ApiError(HTTPStatus.CONFLICT, "brain turn stopped")
        return {
            "team_id": team_id,
            "team_name": segment.team_name,
            "reply": terminal.reply[: hosted_assistants.CHAT_OUTPUT_CAP],
        }

    try:
        return chat_turn_engine.dispatch(
            segment.outcome,
            segment.requirement_groups(),
            pending,
            (
                lambda suspension, requirements, state: _pause_hosted_connection(
                    team_id, token, suspension, requirements, state
                ),
                lambda suspension, requirements, state: _pause_hosted_chat(
                    team_id, token, suspension, requirements, state
                ),
                lambda suspension, requirements, state: _pause_hosted_input(
                    team_id, token, suspension, requirements, state
                ),
                lambda suspension, requirements, state: _pause_hosted_approval(
                    team_id, token, suspension, requirements, state
                ),
            ),
            complete,
        )
    except ValueError as exc:
        raise runtime_state.ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc)) from exc


def _chat_in_turn(
    team_id: str,
    message: str,
    file_ids: object,
    assistant_ids: tuple[str, ...],
    token: str,
    container,
    owner: str,
) -> dict[str, object]:
    segment = _run_hosted_chat_segment(
        HostedChatSegmentRequest(
            team_id=team_id,
            file_ids=file_ids,
            assistant_ids=assistant_ids,
            token=token,
            container=container,
            owner=owner,
            message=message,
        )
    )
    return _hosted_segment_response(
        team_id,
        token,
        segment,
        assistant_ids,
        tuple(file_ids) if isinstance(file_ids, list) else (),
        owner,
    )
