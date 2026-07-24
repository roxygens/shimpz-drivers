"""Local chat Power execution and context validation mixin."""

from http import HTTPStatus
from typing import NoReturn

import assistant_account_flow
import assistant_secret_flow
import assistant_secret_store
import brain_runtime_client
import chat_orchestrator
import chat_turn_engine
import inference_config
import oauth_account_store
import power_execution
import power_journal
from local_registry import validate_power_input

from local_support.chat_types import ActiveAssistant as _ActiveAssistant
from local_support.chat_types import required_active_assistant as _required_active_assistant
from local_support.errors import ApiProblemError as ApiProblem


class LocalChatExecutionMixin:
    def _invoke_chat_power(
        self,
        team_id: str,
        token: str,
        assistant_id: str,
        frozen_container_id: str,
        power: str,
        payload: object,
        answers: tuple[object, ...] = (),
    ) -> object:
        with self._lock(team_id):
            spec = self._resolve(assistant_id)
            network = self._network(team_id)
            container = self._assistant_container(team_id, assistant_id)
            self._validate_container(container, team_id, spec, network.name)
            if container.id != frozen_container_id:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "Team capabilities changed; retry",
                    code="team-context-changed",
                )
            with self._active_chat_guard:
                if (
                    self._active_chat_tokens.get(team_id) != token
                    or token in self._cancelled_chat_tokens
                    or team_id in self._active_power_containers
                ):
                    raise chat_orchestrator.ChatStoppedError("chat turn stopped")
                self._active_power_containers[team_id] = (token, container)
        try:
            invocation = (
                self.invoke(team_id, assistant_id, power, payload, answers=answers)
                if answers
                else self.invoke(team_id, assistant_id, power, payload)
            )
        except ApiProblem:
            if self._chat_cancelled(token):
                raise chat_orchestrator.ChatStoppedError("chat turn stopped") from None
            raise
        finally:
            with self._active_chat_guard:
                active = self._active_power_containers.get(team_id)
                if active is not None and active[0] == token:
                    self._active_power_containers.pop(team_id, None)
        if self._chat_cancelled(token):
            raise chat_orchestrator.ChatStoppedError("chat turn stopped")
        if "suspend" in invocation:
            return power_execution.RpcSuspension(invocation["suspend"])
        return invocation["result"]

    @staticmethod
    def _chat_identity(
        team_name: str,
        network_id: str,
        assistants: tuple[_ActiveAssistant, ...],
        files: list[dict[str, object]],
        config: inference_config.InferenceConfig,
    ) -> tuple[object, ...]:
        return (
            team_name,
            network_id,
            tuple((item.spec.assistant_id, item.spec.image, item.container_id) for item in assistants),
            files,
            config,
        )

    @staticmethod
    def _raise_chat_problem(reason: str, exc: BaseException | None) -> NoReturn:
        if reason == "invalid-continuation" or reason == "invalid-suspension":
            raise ApiProblem(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"invalid chat {reason.removeprefix('invalid-')}",
                code="internal-error",
            )
        if reason == "context-changed":
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Team capabilities changed; retry",
                code="team-context-changed",
            )
        if isinstance(exc, power_journal.PowerJournalError):
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Team Power execution state is unavailable",
                code="power-state-unavailable",
            ) from exc
        if isinstance(exc, chat_orchestrator.ChatStoppedError):
            raise ApiProblem(HTTPStatus.CONFLICT, "chat turn stopped", code="chat-stopped") from exc
        if isinstance(exc, chat_orchestrator.ChatOrchestrationError):
            raise ApiProblem(
                HTTPStatus.BAD_GATEWAY,
                "Brain could not complete the Team turn",
                code="brain-runtime-failed",
            ) from exc
        if isinstance(exc, brain_runtime_client.BrainRuntimeError):
            raise ApiProblem(
                HTTPStatus.BAD_GATEWAY,
                "Brain runtime is unavailable",
                code="brain-runtime-failed",
            ) from exc
        raise AssertionError(f"unknown local chat failure: {reason}")

    @staticmethod
    def _validate_chat_power(
        bindings: dict[str, _ActiveAssistant],
        assistant_id: str,
        power: str,
        payload: object,
    ) -> object:
        _required_active_assistant(bindings, assistant_id)
        try:
            return validate_power_input(assistant_id, power, payload)
        except ValueError as exc:
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                str(exc),
                code="invalid-power-input",
            ) from exc

    def _require_chat_private_inputs(
        self,
        team_id: str,
        bindings: dict[str, _ActiveAssistant],
        requests: tuple[brain_runtime_client.PowerRequest, ...],
        requirements: chat_turn_engine.SegmentRequirements,
    ) -> bool:
        try:
            requirements.accounts = assistant_account_flow.requirements_for_batch(
                team_id,
                bindings,
                requests,
                self.assistant_accounts,
            )
        except (
            assistant_account_flow.AccountFlowError,
            oauth_account_store.OAuthAccountStoreError,
        ) as exc:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant account contract is unavailable",
                code="assistant-account-contract-invalid",
            ) from exc
        if requirements.accounts:
            return True
        try:
            requirements.secrets = assistant_secret_flow.requirements_for_batch(
                team_id,
                bindings,
                requests,
                self.assistant_secrets,
            )
        except assistant_secret_store.AssistantSecretError as exc:
            self._raise_secret_problem(exc)
        except assistant_secret_flow.SecretFlowError as exc:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant secret contract is unavailable",
                code="assistant-secret-contract-invalid",
            ) from exc
        return bool(requirements.secrets)

    def _validate_chat_context(
        self,
        team_id: str,
        file_ids: list[str],
        provider: str,
        assistant_ids: tuple[str, ...],
        identity: tuple[object, ...],
        metadata_connection=None,
    ) -> None:
        current = self._chat_setup(team_id, file_ids, provider, assistant_ids, metadata_connection)
        if self._chat_identity(*current) != identity:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Team capabilities changed; retry",
                code="team-context-changed",
            )
