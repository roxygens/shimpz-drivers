"""Local chat suspension and challenge response mixin."""

from http import HTTPStatus

import assistant_account_challenges
import assistant_account_flow
import assistant_secret_challenges
import chat_orchestrator
import chat_turn_engine
from assistant_human import approval_challenges as assistant_approval_challenges
from assistant_human import input_challenges as assistant_input_challenges
from assistant_human import input_flow as assistant_input_flow

from local_support.chat_types import ActiveAssistant as _ActiveAssistant
from local_support.chat_types import PendingLocalChat as _PendingLocalChat
from local_support.errors import ApiProblemError as ApiProblem


class LocalChatPauseMixin:
    def _pause_chat(
        self,
        team_id: str,
        token: str,
        outcome: chat_orchestrator.ChatSuspension,
        requirements: tuple[assistant_secret_challenges.SecretRequirement, ...],
        payload: _PendingLocalChat,
    ) -> dict[str, object]:
        try:
            challenge = self.secret_challenges.create(team_id, requirements, payload)
        except assistant_secret_challenges.SecretChallengeError as exc:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant secret request is already pending",
                code="assistant-secret-challenge-conflict",
            ) from exc
        try:
            self._persist_chat_continuation("secrets", challenge, requirements, payload)
        except ApiProblem:
            self.secret_challenges.cancel_team(team_id)
            raise
        self._commit_suspension(team_id, token, outcome, payload, self.secret_challenges, challenge.id)
        return self._challenge_response(challenge)

    def _commit_suspension(
        self,
        team_id: str,
        token: str,
        outcome: chat_orchestrator.ChatSuspension,
        payload: _PendingLocalChat,
        challenge_store: object,
        challenge_id: str,
    ) -> None:
        chat_turn_engine.commit_suspension(
            outcome.continuation,
            payload.continuation,
            lambda: self._commit_chat_terminal(team_id, token),
            lambda: challenge_store.cancel_team(team_id),
            lambda: ApiProblem(HTTPStatus.CONFLICT, "chat turn stopped", code="chat-stopped"),
            lambda: self._delete_chat_continuation(team_id, challenge_id),
        )

    def _account_response(
        self,
        challenge: assistant_account_challenges.PendingAccountChallenge,
    ) -> dict[str, object]:
        bindings: dict[str, _ActiveAssistant] = {}
        for requirement in challenge.requirements:
            spec = self._resolve(requirement.assistant_id)
            bindings[spec.assistant_id] = _ActiveAssistant(spec, "")
        try:
            return assistant_account_flow.challenge_payload(challenge, bindings)
        except assistant_account_flow.AccountFlowError as exc:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant account contract changed; retry the message",
                code="assistant-account-contract-invalid",
            ) from exc

    def _pause_account(
        self,
        team_id: str,
        token: str,
        outcome: chat_orchestrator.ChatSuspension,
        requirements: tuple[assistant_account_challenges.AccountRequirement, ...],
        payload: _PendingLocalChat,
    ) -> dict[str, object]:
        try:
            challenge = self.account_challenges.create(team_id, requirements, payload)
        except assistant_account_challenges.AccountChallengeError as exc:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant account request is already pending",
                code="assistant-account-challenge-conflict",
            ) from exc
        try:
            self._persist_chat_continuation("accounts", challenge, requirements, payload)
        except ApiProblem:
            self.account_challenges.cancel_team(team_id)
            raise
        self._commit_suspension(team_id, token, outcome, payload, self.account_challenges, challenge.id)
        return self._account_response(challenge)

    def _pause_approval(
        self,
        team_id: str,
        token: str,
        outcome: chat_orchestrator.ChatSuspension,
        requirements: tuple[assistant_approval_challenges.ApprovalRequirement, ...],
        payload: _PendingLocalChat,
    ) -> dict[str, object]:
        try:
            challenge = self.approval_challenges.create(team_id, requirements, payload)
        except assistant_approval_challenges.ApprovalChallengeError as exc:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant approval is already pending",
                code="assistant-approval-challenge-conflict",
            ) from exc
        try:
            self._persist_chat_continuation("approval", challenge, requirements, payload)
        except ApiProblem:
            self.approval_challenges.cancel_team(team_id)
            raise
        self._commit_suspension(team_id, token, outcome, payload, self.approval_challenges, challenge.id)
        return self._approval_response(challenge)

    @staticmethod
    def _input_response(
        challenge: assistant_input_challenges.PendingInputChallenge,
    ) -> dict[str, object]:
        return assistant_input_flow.challenge_payload(challenge)

    def _pause_input(
        self,
        team_id: str,
        token: str,
        outcome: chat_orchestrator.ChatSuspension,
        requirements: tuple[chat_orchestrator.HumanInteraction, ...],
        payload: _PendingLocalChat,
    ) -> dict[str, object]:
        if len(requirements) != 1:
            raise ApiProblem(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "invalid human input suspension",
                code="internal-error",
            )
        interaction = requirements[0]
        answers = dict(payload.answer_logs).get(interaction.request.interrupt_id, ())
        try:
            spec = self._resolve(interaction.request.assistant_id)
            requirement = assistant_input_flow.requirement(interaction, spec.image, len(answers))
            challenge = self.input_challenges.create(team_id, requirement, payload)
        except (
            assistant_input_challenges.InputChallengeError,
            assistant_input_flow.InputFlowError,
        ) as exc:
            raise ApiProblem(
                HTTPStatus.BAD_GATEWAY,
                "Assistant human input request is invalid",
                code="invalid-assistant-input-request",
            ) from exc
        try:
            self._persist_chat_continuation("input", challenge, (challenge.requirement,), payload)
        except ApiProblem:
            self.input_challenges.cancel_team(team_id)
            raise
        self._commit_suspension(team_id, token, outcome, payload, self.input_challenges, challenge.id)
        return self._input_response(challenge)
