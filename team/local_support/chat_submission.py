"""Local chat human-submission transaction mixin."""

from dataclasses import replace
from http import HTTPStatus

import assistant_secret_challenges
import assistant_secret_flow
import assistant_secret_store
from assistant_human import approval_challenges as assistant_approval_challenges
from assistant_human import approval_flow as assistant_approval_flow
from assistant_human import approval_grants as assistant_approval_grants
from assistant_human import input_challenges as assistant_input_challenges
from assistant_human import input_flow as assistant_input_flow

from local_support.chat_types import PendingLocalChat as _PendingLocalChat
from local_support.errors import ApiProblemError as ApiProblem


class LocalChatSubmissionMixin:
    def _store_chat_input(
        self,
        team_id: str,
        challenge_id: object,
        provider: str,
        body: object,
    ) -> _PendingLocalChat:
        try:
            challenge = self.input_challenges.get(team_id, challenge_id)
            answer = assistant_input_flow.submitted_answer(challenge, body)
        except assistant_input_challenges.InputChallengeNotFoundError as exc:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant input request expired; retry the message",
                code="assistant-input-challenge-expired",
            ) from exc
        except (
            assistant_input_challenges.InputChallengeError,
            assistant_input_flow.InputFlowError,
        ) as exc:
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "Assistant input submission is invalid",
                code="invalid-assistant-input",
            ) from exc
        pending = challenge.payload
        if not isinstance(pending, _PendingLocalChat) or pending.provider != provider:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Team capabilities changed; retry",
                code="team-context-changed",
            )
        with self._lock(team_id):
            current = self._chat_setup(team_id, list(pending.file_ids), provider, pending.assistant_ids)
            if self._chat_identity(*current) != pending.identity:
                self.input_challenges.cancel_team(team_id)
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "Team capabilities changed; retry",
                    code="team-context-changed",
                )
            answer_logs = dict(pending.answer_logs)
            existing = answer_logs.get(challenge.requirement.interrupt_id, ())
            if len(existing) != challenge.requirement.ordinal:
                self.input_challenges.cancel_team(team_id)
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "Assistant input replay changed; retry the message",
                    code="assistant-input-replay-changed",
                )
            try:
                claimed = self.input_challenges.claim(team_id, challenge_id)
            except assistant_input_challenges.InputChallengeNotFoundError as exc:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "Assistant input request expired; retry the message",
                    code="assistant-input-challenge-expired",
                ) from exc
            if claimed is not challenge:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "Assistant input request expired; retry the message",
                    code="assistant-input-challenge-expired",
                )
            answer_logs[challenge.requirement.interrupt_id] = (*existing, answer)
        return replace(pending, answer_logs=tuple(sorted(answer_logs.items())))

    def _store_chat_approval(
        self,
        team_id: str,
        challenge_id: object,
        provider: str,
        body: object,
    ) -> _PendingLocalChat:
        try:
            challenge = self.approval_challenges.get(team_id, challenge_id)
            answer = assistant_approval_flow.submitted_answer(challenge, body)
        except assistant_approval_challenges.ApprovalChallengeNotFoundError as exc:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant approval expired; retry the message",
                code="assistant-approval-challenge-expired",
            ) from exc
        except (assistant_approval_challenges.ApprovalChallengeError, assistant_approval_flow.ApprovalFlowError) as exc:
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "Assistant approval submission is invalid",
                code="invalid-assistant-approval",
            ) from exc
        pending = challenge.payload
        if not isinstance(pending, _PendingLocalChat) or pending.provider != provider:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Team capabilities changed; retry",
                code="team-context-changed",
            )
        with self._lock(team_id):
            current = self._chat_setup(team_id, list(pending.file_ids), provider, pending.assistant_ids)
            if self._chat_identity(*current) != pending.identity:
                self.approval_challenges.cancel_team(team_id)
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "Team capabilities changed; retry",
                    code="team-context-changed",
                )
            requirement = challenge.requirements[0]
            answer_logs = dict(pending.answer_logs)
            existing = answer_logs.get(requirement.interrupt_id, ())
            if len(existing) != requirement.ordinal:
                self.approval_challenges.cancel_team(team_id)
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "Assistant approval replay changed; retry the message",
                    code="assistant-approval-replay-changed",
                )
            try:
                claimed = self.approval_challenges.claim(team_id, challenge_id)
                if claimed is not challenge:
                    raise assistant_approval_challenges.ApprovalChallengeNotFoundError(
                        "approval challenge is unavailable"
                    )
                if requirement.runs == "once":
                    self.approval_grants.grant_many(
                        (
                            assistant_approval_grants.Grant(
                                team_id=team_id,
                                assistant_id=requirement.assistant_id,
                                power_id=requirement.power_id,
                                image=requirement.assistant_image,
                                ordinal=requirement.ordinal,
                            ),
                        )
                    )
            except assistant_approval_challenges.ApprovalChallengeNotFoundError as exc:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "Assistant approval expired; retry the message",
                    code="assistant-approval-challenge-expired",
                ) from exc
            except assistant_approval_grants.ApprovalGrantError as exc:
                self._raise_approval_grant_problem(exc)
            answer_logs[requirement.interrupt_id] = (*existing, answer)
        return replace(pending, answer_logs=tuple(sorted(answer_logs.items())))

    def _store_chat_secrets(
        self,
        team_id: str,
        challenge_id: object,
        provider: str,
        body: dict[str, object],
    ) -> _PendingLocalChat:
        try:
            challenge = self.secret_challenges.get(team_id, challenge_id)
            values = assistant_secret_flow.submission_values(challenge, body)
        except assistant_secret_challenges.SecretChallengeNotFoundError as exc:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant secret request expired; retry the message",
                code="assistant-secret-challenge-expired",
            ) from exc
        except (assistant_secret_challenges.SecretChallengeError, assistant_secret_flow.SecretFlowError) as exc:
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "Assistant secret submission is invalid",
                code="invalid-assistant-secrets",
            ) from exc
        pending = challenge.payload
        if not isinstance(pending, _PendingLocalChat) or pending.provider != provider:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Team capabilities changed; retry",
                code="team-context-changed",
            )
        with self._lock(team_id):
            current = self._chat_setup(team_id, list(pending.file_ids), provider, pending.assistant_ids)
            if self._chat_identity(*current) != pending.identity:
                self.secret_challenges.cancel_team(team_id)
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "Team capabilities changed; retry",
                    code="team-context-changed",
                )

            def commit_secret_transaction(current) -> None:
                if current is not challenge:
                    raise assistant_secret_challenges.SecretChallengeNotFoundError("secret challenge is unavailable")
                self.assistant_secrets.put_for_assistants(team_id, values)

            try:
                claimed = self.secret_challenges.claim_after(
                    team_id,
                    challenge_id,
                    commit_secret_transaction,
                )
                if claimed is not challenge:
                    raise assistant_secret_challenges.SecretChallengeNotFoundError("secret challenge is unavailable")
            except assistant_secret_challenges.SecretChallengeNotFoundError as exc:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "Assistant secret request expired; retry the message",
                    code="assistant-secret-challenge-expired",
                ) from exc
            except assistant_secret_store.AssistantSecretError as exc:
                self._raise_secret_problem(exc)
        return pending
