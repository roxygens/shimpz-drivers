"""Local chat start and account-resume API mixin."""

from http import HTTPStatus

import chat_orchestrator
import chat_turn_engine

from local_support.chat_segment import SegmentRequest as _ChatSegmentRequest
from local_support.chat_types import PendingLocalChat as _PendingLocalChat
from local_support.errors import ApiProblemError as ApiProblem
from local_support.validation import validate_chat_assistant_ids, validate_team_id

MAX_CHAT_MESSAGE_CHARS = 16_000


class LocalChatApiMixin:
    def _pending_chat_continuation(self, team_id: str) -> dict[str, object] | None:
        existing_account = self.account_challenges.current(team_id)
        existing_secret = self.secret_challenges.current(team_id)
        existing_input = self.input_challenges.current(team_id)
        existing_approval = self.approval_challenges.current(team_id)
        if sum(item is not None for item in (existing_account, existing_secret, existing_input, existing_approval)) > 1:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Team chat continuation state is unavailable",
                code="chat-state-unavailable",
            )
        if existing_account is not None:
            return self._account_response(existing_account)
        if existing_secret is not None:
            return self._challenge_response(existing_secret)
        if existing_input is not None:
            return self._input_response(existing_input)
        if existing_approval is not None:
            return self._approval_response(existing_approval)
        return None

    def _segment_response(
        self,
        team_id: str,
        token: str,
        segment: chat_turn_engine.SegmentResult,
        assistant_ids: tuple[str, ...],
        file_ids: tuple[str, ...],
        provider: str,
    ) -> dict[str, object]:
        def pending(suspension: chat_orchestrator.ChatSuspension) -> _PendingLocalChat:
            return _PendingLocalChat(
                continuation=suspension.continuation,
                assistant_ids=assistant_ids,
                file_ids=file_ids,
                provider=provider,
                identity=segment.identity,
                answer_logs=segment.answer_logs,
            )

        def complete(terminal: chat_orchestrator.ChatOutcome) -> dict[str, object]:
            self._delete_chat_continuation(team_id)
            if not self._commit_chat_terminal(team_id, token):
                raise ApiProblem(HTTPStatus.CONFLICT, "chat turn stopped", code="chat-stopped")
            return {"team_id": team_id, "team_name": segment.team_name, "reply": terminal.reply}

        try:
            return chat_turn_engine.dispatch(
                segment.outcome,
                segment.requirement_groups(),
                pending,
                (
                    lambda suspension, requirements, state: self._pause_account(
                        team_id, token, suspension, requirements, state
                    ),
                    lambda suspension, requirements, state: self._pause_chat(
                        team_id, token, suspension, requirements, state
                    ),
                    lambda suspension, requirements, state: self._pause_input(
                        team_id, token, suspension, requirements, state
                    ),
                    lambda suspension, requirements, state: self._pause_approval(
                        team_id, token, suspension, requirements, state
                    ),
                ),
                complete,
            )
        except ValueError as exc:
            raise ApiProblem(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc), code="internal-error") from exc

    def chat(
        self,
        team_id: str,
        body: object,
        provider: str,
        api_key: str,
    ) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        if not isinstance(body, dict) or set(body) != {"message", "files", "assistant_ids"}:
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "Team chat requires only message, files, and assistant_ids",
                code="invalid-body",
            )
        message = body["message"]
        file_ids = body["files"]
        assistant_ids = validate_chat_assistant_ids(body["assistant_ids"])
        if (
            not isinstance(message, str)
            or not message.strip()
            or len(message) > MAX_CHAT_MESSAGE_CHARS
            or "\0" in message
        ):
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "message must be non-empty and within its size limit",
                code="invalid-message",
            )
        pending = self._pending_chat_continuation(team_id)
        if pending is not None:
            return pending
        with self._exclusive_chat_turn(team_id) as token:
            pending = self._pending_chat_continuation(team_id)
            if pending is not None:
                return pending
            segment = self._run_chat_segment(
                _ChatSegmentRequest(
                    team_id=team_id,
                    file_ids=file_ids,
                    assistant_ids=assistant_ids,
                    provider=provider,
                    api_key=api_key,
                    token=token,
                    message=message,
                )
            )
            return self._segment_response(
                team_id,
                token,
                segment,
                assistant_ids,
                tuple(file_ids),
                provider,
            )

    def resume_chat_accounts(
        self,
        team_id: str,
        body: object,
        provider: str,
        api_key: str,
    ) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        if not isinstance(body, dict) or set(body) != {"challenge_id"}:
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "Assistant account resume requires only challenge_id",
                code="invalid-body",
            )
        challenge_id = body["challenge_id"]

        with self._exclusive_chat_turn(team_id) as token:
            with self._lock(team_id):

                def inspect(pending: object) -> chat_turn_engine.AccountResumeContext:
                    if not isinstance(pending, _PendingLocalChat):
                        raise AssertionError("invalid local account continuation")
                    current = self._chat_setup(team_id, list(pending.file_ids), provider, pending.assistant_ids)
                    bindings = {active.spec.assistant_id: active for active in current[2]}
                    return chat_turn_engine.AccountResumeContext(
                        self._chat_identity(*current),
                        bindings,
                        pending.continuation.turn.powers,
                    )

                admission = chat_turn_engine.admit_account_resume(
                    chat_turn_engine.AccountResumeStrategy(
                        store=self.account_challenges,
                        team_id=team_id,
                        challenge_id=challenge_id,
                        pending_valid=lambda pending: (
                            isinstance(pending, _PendingLocalChat) and pending.provider == provider
                        ),
                        pending_identity=lambda pending: pending.identity,
                        inspect=inspect,
                        account_store=self.assistant_accounts,
                        challenge_response=self._account_response,
                        expired_error=lambda: ApiProblem(
                            HTTPStatus.CONFLICT,
                            "Assistant account request expired; retry the message",
                            code="assistant-account-challenge-expired",
                        ),
                        context_error=lambda: ApiProblem(
                            HTTPStatus.CONFLICT,
                            "Team capabilities changed; retry",
                            code="team-context-changed",
                        ),
                        contract_error=lambda: ApiProblem(
                            HTTPStatus.CONFLICT,
                            "Assistant account contract is unavailable",
                            code="assistant-account-contract-invalid",
                        ),
                        cancel_extra=lambda: self.oauth_pkce.cancel_team(team_id),
                    )
                )
                if admission.response is not None:
                    return admission.response
                pending = admission.pending
                if not isinstance(pending, _PendingLocalChat):
                    raise AssertionError("shared account resume returned invalid state")
            segment = self._run_chat_segment(
                _ChatSegmentRequest(
                    team_id=team_id,
                    file_ids=list(pending.file_ids),
                    assistant_ids=pending.assistant_ids,
                    provider=provider,
                    api_key=api_key,
                    token=token,
                    continuation=pending.continuation,
                    expected_identity=pending.identity,
                    answer_logs=pending.answer_logs,
                )
            )
            return self._segment_response(
                team_id,
                token,
                segment,
                pending.assistant_ids,
                pending.file_ids,
                provider,
            )
