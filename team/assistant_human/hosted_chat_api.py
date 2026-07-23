"""Hosted Team chat API, continuation, OAuth, and cancellation operations."""

from __future__ import annotations

import contextlib
from dataclasses import replace
from http import HTTPStatus

import assistant_account_challenges
import assistant_secret_challenges
import assistant_secret_flow
import assistant_secret_store
import audit
import chat_turn_engine
import docker.errors
import marketplace
import oauth_account_service
from http_boundary import controller_binding

from assistant_human import approval_challenges as assistant_approval_challenges
from assistant_human import approval_flow as assistant_approval_flow
from assistant_human import approval_grants as assistant_approval_grants
from assistant_human import input_challenges as assistant_input_challenges
from assistant_human import input_flow as assistant_input_flow

_controller = controller_binding.current()


def _chat(
    team_id: str,
    message: str,
    file_ids: object,
    assistant_ids: tuple[str, ...],
    lease: _controller._AuthorizationLease,
) -> dict:
    """Run one bounded Team turn across the explicit Controller-brokered Assistant scope."""
    pending = _controller._pending_hosted_chat(team_id)
    if pending is not None:
        return pending
    # The slot comes first. A losing concurrent request must not run even the local credential probe,
    # much less provider status or a second provider CLI.
    with _controller._exclusive_chat_turn(team_id, lease) as (token, container):
        pending = _controller._pending_hosted_chat(team_id)
        if pending is not None:
            return pending
        return _controller._chat_in_turn(team_id, message, file_ids, assistant_ids, token, container, lease.owner)


def _pending_hosted_chat(team_id: str) -> dict[str, object] | None:
    account = _controller._assistant_account_challenges.current(team_id)
    secret = _controller._assistant_secret_challenges.current(team_id)
    input_challenge = _controller._assistant_input_challenges.current(team_id)
    approval = _controller._assistant_approval_challenges.current(team_id)
    if sum(item is not None for item in (account, secret, input_challenge, approval)) > 1:
        raise _controller.ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Team chat continuation state is unavailable")
    if account is not None:
        return _controller._hosted_account_challenge_payload(account)
    if secret is not None:
        return assistant_secret_flow.challenge_payload(secret)
    if input_challenge is not None:
        return assistant_input_flow.challenge_payload(input_challenge)
    if approval is not None:
        return assistant_approval_flow.challenge_payload(approval)
    return None


def _current_account_declaration(team_id: str, assistant_id: str, account_id: str) -> object:
    try:
        installed_id, contract, _container = _controller._installed_assistant(team_id, assistant_id)
        declaration = contract.accounts.get(account_id)
        if installed_id != assistant_id or declaration is None:
            raise _controller.ApiError(HTTPStatus.CONFLICT, "Assistant account declaration changed")
    except _controller.ApiError, marketplace.MarketplaceError:
        # The OAuth service intentionally receives one opaque typed failure so
        # registry, Docker, and manifest details cannot reach the callback response.
        raise oauth_account_service.OAuthAccountDeclarationError(
            "installed Assistant account declaration is unavailable"
        ) from None
    else:
        return declaration


def _start_oauth_account(
    team_id: str,
    challenge_id: object,
    session_binding: object,
    lease: _controller._AuthorizationLease,
) -> dict[str, object]:
    _controller._require_current_authorization(team_id, lease, require_isolation=False)
    try:
        challenge = _controller._assistant_account_challenges.get(team_id, challenge_id)
    except assistant_account_challenges.AccountChallengeNotFoundError as exc:
        raise _controller.ApiError(HTTPStatus.CONFLICT, "Assistant account request expired; retry the message") from exc
    pending = challenge.payload
    if not isinstance(pending, _controller._PendingHostedChat) or pending.owner != lease.owner:
        raise _controller.ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry")
    try:
        authorization_url = _controller._oauth_accounts.authorization_url(challenge, session_binding)
    except oauth_account_service.OAuthAccountUnavailableError as exc:
        raise _controller.ApiError(HTTPStatus.CONFLICT, "Assistant accounts are already configured") from exc
    except oauth_account_service.OAuthAccountServiceError as exc:
        raise _controller.ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Assistant account could not be started") from exc
    return {"authorization_url": authorization_url}


def _complete_oauth_account(
    body: object,
    principal: tuple[str, str | None],
) -> dict[str, object]:
    if not isinstance(body, dict) or set(body) != {"state", "code", "session_binding"}:
        raise _controller.ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "OAuth callback is invalid")
    try:
        completion = _controller._oauth_accounts.complete(
            body["state"],
            body["code"],
            body["session_binding"],
            _controller._current_account_declaration,
        )
    except oauth_account_service.OAuthAccountServiceError as exc:
        raise _controller.ApiError(HTTPStatus.BAD_GATEWAY, "Assistant account could not be completed") from exc
    try:
        _controller._authorize(completion.team_id, principal)
    except Exception:
        with contextlib.suppress(oauth_account_service.OAuthAccountServiceError):
            _controller._oauth_accounts.disconnect(
                completion.team_id,
                completion.assistant_id,
                completion.account_id,
            )
        raise
    pending = _controller._assistant_account_challenges.current(completion.team_id)
    return {
        "connected": True,
        "team_id": completion.team_id,
        "assistant_id": completion.assistant_id,
        "account_id": completion.account_id,
        "provider": completion.provider,
        "scopes": list(completion.scopes),
        "challenge_id": pending.id if pending is not None else None,
    }


@_controller._serialize_against_team_chat
def _disconnect_oauth_account(
    team_id: str,
    assistant_id: str,
    account_id: str,
    lease: _controller._AuthorizationLease,
) -> dict[str, object]:
    with _controller._lock_for(team_id):
        _controller._require_current_authorization(team_id, lease, require_isolation=False)
        _controller._current_account_declaration(team_id, assistant_id, account_id)
        _controller._assistant_account_challenges.cancel_team(team_id)
        try:
            disconnected = _controller._oauth_accounts.disconnect(team_id, assistant_id, account_id)
        except oauth_account_service.OAuthAccountServiceError as exc:
            raise _controller.ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE, "Assistant account could not be disconnected"
            ) from exc
    return {"disconnected": disconnected}


def _resume_chat_accounts(
    team_id: str,
    challenge_id: object,
    lease: _controller._AuthorizationLease,
) -> dict[str, object]:
    with _controller._exclusive_chat_turn(team_id, lease) as (token, container):

        def inspect(pending: object) -> chat_turn_engine.AccountResumeContext:
            if not isinstance(pending, _controller._PendingHostedChat):
                raise AssertionError("invalid hosted account continuation")
            _, assistants, _files, _config, _key, _generation, current_identity = _controller._hosted_chat_setup(
                team_id,
                list(pending.file_ids),
                pending.assistant_ids,
                container,
                lease.owner,
            )
            bindings = {active.assistant_id: active for active in assistants}
            return chat_turn_engine.AccountResumeContext(
                current_identity,
                _controller._secret_bindings(bindings),
                pending.continuation.turn.powers,
            )

        admission = chat_turn_engine.admit_account_resume(
            chat_turn_engine.AccountResumeStrategy(
                store=_controller._assistant_account_challenges,
                team_id=team_id,
                challenge_id=challenge_id,
                pending_valid=lambda pending: (
                    isinstance(pending, _controller._PendingHostedChat) and pending.owner == lease.owner
                ),
                pending_identity=lambda pending: pending.identity,
                inspect=inspect,
                account_store=_controller._assistant_accounts,
                challenge_response=_controller._hosted_account_challenge_payload,
                expired_error=lambda: _controller.ApiError(
                    HTTPStatus.CONFLICT,
                    "Assistant account request expired; retry the message",
                ),
                context_error=lambda: _controller.ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry"),
                contract_error=lambda: _controller.ApiError(
                    HTTPStatus.CONFLICT, "Assistant account contract is unavailable"
                ),
            )
        )
        if admission.response is not None:
            return admission.response
        pending = admission.pending
        if not isinstance(pending, _controller._PendingHostedChat):
            raise AssertionError("shared account resume returned invalid state")

        segment = _controller._run_hosted_chat_segment(
            _controller.HostedChatSegmentRequest(
                team_id=team_id,
                file_ids=list(pending.file_ids),
                assistant_ids=pending.assistant_ids,
                token=token,
                container=container,
                owner=lease.owner,
                continuation=pending.continuation,
                expected_identity=pending.identity,
                answer_logs=pending.answer_logs,
            )
        )
        return _controller._hosted_segment_response(
            team_id,
            token,
            segment,
            pending.assistant_ids,
            pending.file_ids,
            pending.owner,
        )


def _submit_chat_secrets(
    team_id: str,
    body: object,
    lease: _controller._AuthorizationLease,
) -> dict[str, object]:
    try:
        challenge_id = body.get("challenge_id") if isinstance(body, dict) else None
        challenge = _controller._assistant_secret_challenges.get(team_id, challenge_id)
        values = assistant_secret_flow.submission_values(challenge, body)
    except assistant_secret_challenges.SecretChallengeNotFoundError as exc:
        raise _controller.ApiError(HTTPStatus.CONFLICT, "Assistant secret request expired; retry the message") from exc
    except (assistant_secret_challenges.SecretChallengeError, assistant_secret_flow.SecretFlowError) as exc:
        raise _controller.ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "Assistant secret submission is invalid") from exc
    pending = challenge.payload
    if not isinstance(pending, _controller._PendingHostedChat) or pending.owner != lease.owner:
        raise _controller.ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry")

    with _controller._exclusive_chat_turn(team_id, lease) as (token, container):
        *_unused, current_identity = _controller._hosted_chat_setup(
            team_id,
            list(pending.file_ids),
            pending.assistant_ids,
            container,
            lease.owner,
        )
        if current_identity != pending.identity:
            _controller._assistant_secret_challenges.cancel_team(team_id)
            raise _controller.ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry")

        def commit_secret_transaction(current) -> None:
            if current is not challenge:
                raise assistant_secret_challenges.SecretChallengeNotFoundError("secret challenge is unavailable")
            _controller._assistant_secrets.put_for_assistants(team_id, values)

        try:
            claimed = _controller._assistant_secret_challenges.claim_after(
                team_id,
                challenge.id,
                commit_secret_transaction,
            )
            if claimed is not challenge:
                raise assistant_secret_challenges.SecretChallengeNotFoundError("secret challenge is unavailable")
        except assistant_secret_challenges.SecretChallengeNotFoundError as exc:
            raise _controller.ApiError(
                HTTPStatus.CONFLICT, "Assistant secret request expired; retry the message"
            ) from exc
        except assistant_secret_store.AssistantSecretError as exc:
            _controller._raise_assistant_secret_error(exc)

        segment = _controller._run_hosted_chat_segment(
            _controller.HostedChatSegmentRequest(
                team_id=team_id,
                file_ids=list(pending.file_ids),
                assistant_ids=pending.assistant_ids,
                token=token,
                container=container,
                owner=lease.owner,
                continuation=pending.continuation,
                expected_identity=pending.identity,
                answer_logs=pending.answer_logs,
            )
        )
        return _controller._hosted_segment_response(
            team_id,
            token,
            segment,
            pending.assistant_ids,
            pending.file_ids,
            pending.owner,
        )


def _submit_chat_input(
    team_id: str,
    body: object,
    lease: _controller._AuthorizationLease,
) -> dict[str, object]:
    challenge_id = body.get("challenge_id") if isinstance(body, dict) else None
    try:
        challenge = _controller._assistant_input_challenges.get(team_id, challenge_id)
        answer = assistant_input_flow.submitted_answer(challenge, body)
    except assistant_input_challenges.InputChallengeNotFoundError as exc:
        raise _controller.ApiError(HTTPStatus.CONFLICT, "Assistant input request expired; retry the message") from exc
    except (assistant_input_challenges.InputChallengeError, assistant_input_flow.InputFlowError) as exc:
        raise _controller.ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "Assistant input submission is invalid") from exc
    pending = challenge.payload
    if not isinstance(pending, _controller._PendingHostedChat) or pending.owner != lease.owner:
        raise _controller.ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry")

    with _controller._exclusive_chat_turn(team_id, lease) as (token, container):
        *_unused, current_identity = _controller._hosted_chat_setup(
            team_id,
            list(pending.file_ids),
            pending.assistant_ids,
            container,
            lease.owner,
        )
        if current_identity != pending.identity:
            _controller._assistant_input_challenges.cancel_team(team_id)
            raise _controller.ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry")
        answer_logs = dict(pending.answer_logs)
        existing = answer_logs.get(challenge.requirement.interrupt_id, ())
        if len(existing) != challenge.requirement.ordinal:
            _controller._assistant_input_challenges.cancel_team(team_id)
            raise _controller.ApiError(HTTPStatus.CONFLICT, "Assistant input replay changed; retry the message")
        try:
            claimed = _controller._assistant_input_challenges.claim(team_id, challenge.id)
        except assistant_input_challenges.InputChallengeNotFoundError as exc:
            raise _controller.ApiError(
                HTTPStatus.CONFLICT, "Assistant input request expired; retry the message"
            ) from exc
        if claimed is not challenge:
            raise _controller.ApiError(HTTPStatus.CONFLICT, "Assistant input request expired; retry the message")
        answer_logs[challenge.requirement.interrupt_id] = (*existing, answer)
        resumed = replace(pending, answer_logs=tuple(sorted(answer_logs.items())))

        segment = _controller._run_hosted_chat_segment(
            _controller.HostedChatSegmentRequest(
                team_id=team_id,
                file_ids=list(resumed.file_ids),
                assistant_ids=resumed.assistant_ids,
                token=token,
                container=container,
                owner=lease.owner,
                continuation=resumed.continuation,
                expected_identity=resumed.identity,
                answer_logs=resumed.answer_logs,
            )
        )
        return _controller._hosted_segment_response(
            team_id,
            token,
            segment,
            resumed.assistant_ids,
            resumed.file_ids,
            resumed.owner,
        )


def _submit_chat_approval(
    team_id: str,
    body: object,
    lease: _controller._AuthorizationLease,
) -> dict[str, object]:
    challenge_id = body.get("challenge_id") if isinstance(body, dict) else None
    try:
        challenge = _controller._assistant_approval_challenges.get(team_id, challenge_id)
        answer = assistant_approval_flow.submitted_answer(challenge, body)
    except assistant_approval_challenges.ApprovalChallengeNotFoundError as exc:
        raise _controller.ApiError(HTTPStatus.CONFLICT, "Assistant approval expired; retry the message") from exc
    except (assistant_approval_challenges.ApprovalChallengeError, assistant_approval_flow.ApprovalFlowError) as exc:
        raise _controller.ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "Assistant approval submission is invalid") from exc
    pending = challenge.payload
    if not isinstance(pending, _controller._PendingHostedChat) or pending.owner != lease.owner:
        raise _controller.ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry")

    with _controller._exclusive_chat_turn(team_id, lease) as (token, container):
        *_unused, current_identity = _controller._hosted_chat_setup(
            team_id,
            list(pending.file_ids),
            pending.assistant_ids,
            container,
            lease.owner,
        )
        if current_identity != pending.identity:
            _controller._assistant_approval_challenges.cancel_team(team_id)
            raise _controller.ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry")
        requirement = challenge.requirements[0]
        answer_logs = dict(pending.answer_logs)
        existing = answer_logs.get(requirement.interrupt_id, ())
        if len(existing) != requirement.ordinal:
            _controller._assistant_approval_challenges.cancel_team(team_id)
            raise _controller.ApiError(HTTPStatus.CONFLICT, "Assistant approval replay changed; retry the message")
        try:
            claimed = _controller._assistant_approval_challenges.claim(team_id, challenge.id)
            if claimed is not challenge:
                raise assistant_approval_challenges.ApprovalChallengeNotFoundError("approval challenge is unavailable")
            if requirement.runs == "once":
                _controller._assistant_approval_grants.grant_many(
                    (
                        assistant_approval_grants.Grant(
                            team_id,
                            requirement.assistant_id,
                            requirement.power_id,
                            requirement.assistant_image,
                            requirement.ordinal,
                        ),
                    )
                )
        except assistant_approval_challenges.ApprovalChallengeNotFoundError as exc:
            raise _controller.ApiError(HTTPStatus.CONFLICT, "Assistant approval expired; retry the message") from exc
        except assistant_approval_grants.ApprovalGrantError as exc:
            raise _controller.ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE, "Assistant approval state is unavailable"
            ) from exc
        answer_logs[requirement.interrupt_id] = (*existing, answer)
        resumed = replace(pending, answer_logs=tuple(sorted(answer_logs.items())))

        segment = _controller._run_hosted_chat_segment(
            _controller.HostedChatSegmentRequest(
                team_id=team_id,
                file_ids=list(resumed.file_ids),
                assistant_ids=resumed.assistant_ids,
                token=token,
                container=container,
                owner=lease.owner,
                continuation=resumed.continuation,
                expected_identity=resumed.identity,
                answer_logs=resumed.answer_logs,
            )
        )
        return _controller._hosted_segment_response(
            team_id,
            token,
            segment,
            resumed.assistant_ids,
            resumed.file_ids,
            resumed.owner,
        )


def _stop_active_power(team_id: str, token: str | None) -> bool:
    if token is None:
        return False
    with _controller._active_chat_guard:
        active = _controller._active_power_container_ids.get(team_id)
    if active is None or active[0] != token:
        return False
    try:
        assistant_container = _controller._docker.containers.get(active[1])
    except docker.errors.NotFound:
        return True
    except docker.errors.DockerException as exc:
        raise _controller.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE, "active Assistant Power could not be inspected"
        ) from exc
    _controller._fail_stop_power(team_id, assistant_container)
    return True


def _stop_chat(team_id: str, lease: _controller._AuthorizationLease) -> dict:
    """Cancel one Controller-owned turn and fail-stop a Power already executing."""
    secret_cancelled = _controller._assistant_secret_challenges.cancel_team(team_id)
    account_cancelled = _controller._assistant_account_challenges.cancel_team(team_id)
    input_cancelled = _controller._assistant_input_challenges.cancel_team(team_id)
    approval_cancelled = _controller._assistant_approval_challenges.cancel_team(team_id)
    challenge_cancelled = secret_cancelled or account_cancelled or input_cancelled or approval_cancelled
    with _controller._lock_for(team_id):
        container = _controller._require_current_authorization(team_id, lease)
        container.reload()
        if container.status != "running":
            raise _controller.ApiError(
                HTTPStatus.CONFLICT, f"team {team_id!r} is not running (status={container.status})"
            )
        with _controller._active_chat_guard:
            token = _controller._active_chat_tokens.get(team_id)
            if token is not None and _controller._active_chat_container_ids.get(team_id) != container.id:
                raise _controller.ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
            if token is not None:
                _controller._cancelled_chat_tokens.add(token)
        power_stopped = _controller._stop_active_power(team_id, token)
    accepted = token is not None or challenge_cancelled
    audit.log("chat_stop", team_id, result="ok" if accepted else "denied")
    return {
        "team_id": team_id,
        "requested": accepted,
        "accepted": accepted,
        # An executing Power is synchronously terminated. A provider HTTP request is only marked
        # cancelled; its result is discarded before any subsequent Power or terminal reply.
        "confirmed": power_stopped,
        "forced_restart": False,
    }
