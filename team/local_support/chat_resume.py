"""Local chat submission-resume and stop API mixin."""

from http import HTTPStatus

from local_support.chat_segment import SegmentRequest as _ChatSegmentRequest
from local_support.errors import ApiProblemError as ApiProblem
from local_support.validation import validate_team_id


class LocalChatResumeMixin:
    def submit_chat_secrets(
        self,
        team_id: str,
        body: object,
        provider: str,
        api_key: str,
    ) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        if not isinstance(body, dict):
            raise ApiProblem(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid secret submission", code="invalid-body")
        challenge_id = body.get("challenge_id")

        with self._exclusive_chat_turn(team_id) as token:
            # The active-turn token exists before the one-use secret challenge is consumed. Stop,
            # uninstall, and rotation therefore cannot observe an unowned persisted continuation.
            pending = self._store_chat_secrets(team_id, challenge_id, provider, body)
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

    def submit_chat_input(
        self,
        team_id: str,
        body: object,
        provider: str,
        api_key: str,
    ) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        challenge_id = body.get("challenge_id") if isinstance(body, dict) else None
        with self._exclusive_chat_turn(team_id) as token:
            pending = self._store_chat_input(team_id, challenge_id, provider, body)
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

    def submit_chat_approval(
        self,
        team_id: str,
        body: object,
        provider: str,
        api_key: str,
    ) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        challenge_id = body.get("challenge_id") if isinstance(body, dict) else None
        with self._exclusive_chat_turn(team_id) as token:
            # Install the active-turn token before consuming the challenge. Stop can now always
            # cancel either the pending challenge or this exact continuation; no unowned gap exists.
            pending = self._store_chat_approval(team_id, challenge_id, provider, body)
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

    def stop_chat(self, team_id: str) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        self._network(team_id)
        account_cancelled = self.account_challenges.cancel_team(team_id)
        self.oauth_pkce.cancel_team(team_id)
        challenge_cancelled = self.secret_challenges.cancel_team(team_id)
        approval_cancelled = self.approval_challenges.cancel_team(team_id)
        input_cancelled = self.input_challenges.cancel_team(team_id)
        continuation_cancelled = self._delete_chat_continuation(team_id)
        power_stopped = False
        with self._active_chat_guard:
            token = self._active_chat_tokens.get(team_id)
            if token is not None:
                self._cancelled_chat_tokens.add(token)
            active = self._active_power_containers.get(team_id)
            if token is not None and active is not None and active[0] == token:
                self._fail_stop_power(active[1])
                power_stopped = True
        accepted = (
            token is not None
            or account_cancelled
            or challenge_cancelled
            or input_cancelled
            or approval_cancelled
            or continuation_cancelled
        )
        return {
            "team_id": team_id,
            "requested": accepted,
            "accepted": accepted,
            "confirmed": power_stopped,
            "forced_restart": False,
        }
