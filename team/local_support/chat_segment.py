"""Local chat segment orchestration mixin."""

from dataclasses import dataclass
from http import HTTPStatus

import brain_runtime_client
import chat_orchestrator
import chat_turn_engine
import power_execution
from assistant_human import approval_challenges as assistant_approval_challenges
from assistant_human import approval_flow as assistant_approval_flow
from assistant_human import approval_grants as assistant_approval_grants

from local_support.chat_types import ActiveAssistant as _ActiveAssistant
from local_support.chat_types import required_active_assistant as _required_active_assistant
from local_support.errors import ApiProblemError as ApiProblem
from local_support.validation import brain_thread_id as _brain_thread_id


@dataclass(frozen=True, slots=True)
class SegmentRequest:
    team_id: str
    file_ids: list[str]
    assistant_ids: tuple[str, ...]
    provider: str
    api_key: str
    token: str
    message: str | None = None
    continuation: chat_orchestrator.ChatContinuation | None = None
    expected_identity: tuple[object, ...] | None = None
    answer_logs: tuple[tuple[str, tuple[object, ...]], ...] = ()


class LocalChatSegmentMixin:
    def _run_chat_segment(
        self,
        request: SegmentRequest,
    ) -> chat_turn_engine.SegmentResult:
        with self.storage.metadata_connection(request.team_id, request.file_ids) as metadata_connection:
            return self._run_chat_segment_with_metadata(request, metadata_connection)

    def _run_chat_segment_with_metadata(
        self,
        request: SegmentRequest,
        metadata_connection,
    ) -> chat_turn_engine.SegmentResult:
        answers_by_interrupt = dict(request.answer_logs)
        if len(answers_by_interrupt) != len(request.answer_logs):
            raise ApiProblem(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "invalid chat answer log",
                code="internal-error",
            )
        bindings: dict[str, _ActiveAssistant] = {}
        identity: tuple[object, ...] = ()
        network_id = ""

        def execute_power(power_request: brain_runtime_client.PowerRequest) -> object:
            active = _required_active_assistant(bindings, power_request.assistant_id)
            return self._invoke_chat_power(
                request.team_id,
                request.token,
                power_request.assistant_id,
                active.container_id,
                power_request.power,
                power_request.input,
                answers_by_interrupt.get(power_request.interrupt_id, ()),
            )

        def prepare() -> chat_turn_engine.PreparedSegment:
            nonlocal bindings, identity, network_id
            team_name, network_id, assistants, files, config = self._chat_setup(
                request.team_id,
                request.file_ids,
                request.provider,
                request.assistant_ids,
                metadata_connection,
            )
            identity = self._chat_identity(team_name, network_id, assistants, files, config)
            genesis_by_id = {active.spec.assistant_id: self._active_assistant_genesis(active) for active in assistants}
            context = brain_runtime_client.RuntimeContext(
                thread_id=_brain_thread_id(self.space_id, request.team_id, network_id),
                team_name=team_name,
                assistants=tuple(
                    brain_runtime_client.RuntimeAssistant(
                        id=active.spec.assistant_id,
                        genesis=genesis_by_id[active.spec.assistant_id],
                        powers=tuple(
                            brain_runtime_client.RuntimePower(
                                id=power_id,
                                summary=power.summary,
                                input_schema=power.input_schema,
                            )
                            for power_id, power in sorted(active.spec.powers.items())
                        ),
                    )
                    for active in assistants
                ),
                provider=config.provider,
                model=config.model,
                api_key=request.api_key,
            )
            bindings = {active.spec.assistant_id: active for active in assistants}
            batch = power_execution.PowerBatch(
                self.power_state,
                network_id,
                context.thread_id,
                bindings,
                power_execution.PowerBatchStrategy(
                    lambda active: (active.container_id, active.spec.image),
                    execute_power,
                    lambda power_request: self._require_power_rpc_envelope(
                        request.team_id,
                        bindings,
                        power_request,
                        answers_by_interrupt.get(power_request.interrupt_id, ()),
                    ),
                    lambda power_request: self._power_secret_generations(
                        request.team_id,
                        _required_active_assistant(bindings, power_request.assistant_id),
                        power_request.power,
                    ),
                    lambda power_request: self._power_account_generations(
                        request.team_id,
                        _required_active_assistant(bindings, power_request.assistant_id),
                        power_request.power,
                    ),
                ),
            )
            return chat_turn_engine.PreparedSegment(team_name, identity, context, files, batch)

        def private_inputs(
            requests: tuple[object, ...],
            requirements: chat_turn_engine.SegmentRequirements,
        ) -> bool:
            return self._require_chat_private_inputs(request.team_id, bindings, requests, requirements)

        def validate_current_context() -> None:
            self._validate_chat_context(
                request.team_id,
                request.file_ids,
                request.provider,
                request.assistant_ids,
                identity,
                metadata_connection,
            )

        current_message = request.message
        current_continuation = request.continuation
        current_identity = request.expected_identity
        while True:
            team_name, identity, outcome, requirements = chat_turn_engine.run_segment(
                chat_turn_engine.SegmentStrategy(
                    runtime=self.brain_runtime,
                    prepare=prepare,
                    validate_power=lambda assistant_id, power, payload: self._validate_chat_power(
                        bindings,
                        assistant_id,
                        power,
                        payload,
                    ),
                    pause_for_private_inputs=private_inputs,
                    cancelled=lambda: self._chat_cancelled(request.token),
                    validate_context=validate_current_context,
                    raise_problem=self._raise_chat_problem,
                ),
                message=current_message,
                continuation=current_continuation,
                expected_identity=current_identity,
            )
            approval_requirements: tuple[assistant_approval_challenges.ApprovalRequirement, ...] = ()
            if requirements.approvals:
                if len(requirements.approvals) != 1:
                    raise ApiProblem(
                        HTTPStatus.BAD_GATEWAY,
                        "Assistant human approval request is invalid",
                        code="invalid-assistant-approval-request",
                    )
                interaction = requirements.approvals[0]
                answers = answers_by_interrupt.get(interaction.request.interrupt_id, ())
                active = _required_active_assistant(bindings, interaction.request.assistant_id)
                try:
                    requirement = assistant_approval_flow.requirement(
                        interaction,
                        active.spec.name,
                        active.spec.image,
                        len(answers),
                    )
                    granted = requirement.runs == "once" and self.approval_grants.is_granted(
                        request.team_id,
                        requirement.assistant_id,
                        requirement.power_id,
                        requirement.assistant_image,
                        requirement.ordinal,
                    )
                except assistant_approval_flow.ApprovalFlowError as exc:
                    raise ApiProblem(
                        HTTPStatus.BAD_GATEWAY,
                        "Assistant human approval request is invalid",
                        code="invalid-assistant-approval-request",
                    ) from exc
                except assistant_approval_grants.ApprovalGrantError as exc:
                    self._raise_approval_grant_problem(exc)
                if granted:
                    answers_by_interrupt[requirement.interrupt_id] = (*answers, True)
                    if not isinstance(outcome, chat_orchestrator.ChatSuspension):
                        raise AssertionError("approval requirement did not suspend")
                    current_message = None
                    current_continuation = outcome.continuation
                    current_identity = identity
                    continue
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
