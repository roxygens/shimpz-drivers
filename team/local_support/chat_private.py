"""Local Assistant secrets, accounts, and approval configuration mixin."""

from http import HTTPStatus
from typing import NoReturn

from docker.errors import DockerException

import assistant_account_challenges
import assistant_account_flow
import assistant_secret_challenges
import assistant_secret_flow
import assistant_secret_store
import brain_runtime_client
import oauth_account_service
import oauth_account_store
import power_execution
import power_journal
from assistant_human import approval_challenges as assistant_approval_challenges
from assistant_human import approval_flow as assistant_approval_flow
from assistant_human import approval_grants as assistant_approval_grants
from local_registry import AssistantSpec
from local_support.chat_types import ActiveAssistant as _ActiveAssistant
from local_support.chat_types import required_active_assistant as _required_active_assistant
from local_support.errors import ApiProblemError as ApiProblem
from local_support.validation import validate_team_id


class LocalChatPrivateMixin:
    def _power_secret_generations(
        self,
        team_id: str,
        active: _ActiveAssistant,
        power_id: str,
    ) -> tuple[tuple[str, int], ...]:
        try:
            return power_execution.secret_generations(
                active.spec.powers,
                power_id,
                lambda secret_ids: self.assistant_secrets.metadata(
                    team_id,
                    active.spec.assistant_id,
                    secret_ids,
                ),
            )
        except assistant_secret_store.AssistantSecretError as exc:
            raise power_journal.PowerJournalConflictError("Power secret state is unavailable") from exc

    def _resolve_power_secrets(self, team_id: str, spec: AssistantSpec, power_id: str) -> dict[str, str]:
        power = spec.powers.get(power_id)
        if power is None:
            raise ApiProblem(
                power_execution.UNDECLARED_POWER_STATUS, "Power is not declared", code="power-not-declared"
            )
        try:
            return self.assistant_secrets.resolve_many(team_id, spec.assistant_id, power.secrets)
        except assistant_secret_store.AssistantSecretError as exc:
            self._raise_secret_problem(exc)

    def _power_account_generations(
        self,
        team_id: str,
        active: _ActiveAssistant,
        power_id: str,
    ) -> tuple[tuple[str, int], ...]:
        try:
            return power_execution.account_generations(
                active.spec.powers,
                active.spec.accounts,
                power_id,
                lambda declarations: self.assistant_accounts.metadata(
                    team_id,
                    active.spec.assistant_id,
                    declarations,
                ),
            )
        except oauth_account_store.OAuthAccountStoreError as exc:
            raise power_journal.PowerJournalConflictError("Power account state is unavailable") from exc

    def _refresh_oauth_account(
        self,
        provider: str,
        scopes: tuple[str, ...],
        refresh_token: str,
        broker_lease: str | None,
    ) -> object:
        return self.oauth_service.refresh(
            provider,
            scopes,
            refresh_token,
            broker_lease,
        )

    def _resolve_power_accounts(
        self,
        team_id: str,
        spec: AssistantSpec,
        power_id: str,
    ) -> dict[str, dict[str, str]]:
        try:
            return assistant_account_flow.resolve_power_accounts(
                team_id,
                spec,
                power_id,
                self.assistant_accounts,
                self._refresh_oauth_account,
            )
        except assistant_account_flow.AccountFlowError as exc:
            raise ApiProblem(
                power_execution.ACCOUNT_PRECONDITION_STATUS,
                "Assistant account is unavailable",
                code="assistant-account-unavailable",
            ) from exc

    def _require_power_rpc_envelope(
        self,
        team_id: str,
        bindings: dict[str, _ActiveAssistant],
        request: brain_runtime_client.PowerRequest,
        answers: tuple[object, ...] = (),
    ) -> None:
        active = _required_active_assistant(bindings, request.assistant_id)
        try:
            power_execution.require_rpc_envelope(
                active,
                request,
                lambda binding, power_id: self._resolve_power_secrets(team_id, binding.spec, power_id),
                lambda binding, power_id: self._resolve_power_accounts(team_id, binding.spec, power_id),
                answers,
            )
        except assistant_secret_flow.SecretFlowError as exc:
            raise ApiProblem(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "Assistant Power input is too large",
                code="assistant-power-input-too-large",
            ) from exc

    @staticmethod
    def _contains_secret(value: object, secrets_by_id: dict[str, str]) -> bool:
        return power_execution.contains_secret(value, secrets_by_id)

    def list_assistant_secrets(self, team_id: str) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        with self._lock(team_id):
            specs = [self._resolve(assistant_id) for assistant_id in self._assistant_ids(team_id)]
            try:
                return assistant_secret_flow.inventory_payload(team_id, specs, self.assistant_secrets)
            except assistant_secret_store.AssistantSecretError as exc:
                self._raise_secret_problem(exc)

    @staticmethod
    def _raise_account_problem(exc: oauth_account_store.OAuthAccountStoreError) -> NoReturn:
        raise ApiProblem(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Assistant account state is unavailable",
            code="assistant-account-state-unavailable",
        ) from exc

    def list_assistant_accounts(self, team_id: str) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        with self._lock(team_id):
            specs = [self._resolve(assistant_id) for assistant_id in self._assistant_ids(team_id)]
            try:
                payload = assistant_account_flow.inventory_payload(
                    team_id,
                    specs,
                    self.assistant_accounts,
                )
            except oauth_account_store.OAuthAccountStoreError as exc:
                self._raise_account_problem(exc)
            except assistant_account_flow.AccountFlowError as exc:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "Assistant account contract is unavailable",
                    code="assistant-account-contract-invalid",
                ) from exc
        return {"team_id": team_id, **payload}

    def start_assistant_account_authorization(
        self,
        team_id: object,
        challenge_id: object,
        session_binding: object,
    ) -> dict[str, object]:
        try:
            challenge = self.account_challenges.get(team_id, challenge_id)
            authorization_url = self.oauth_service.authorization_url(
                challenge,
                session_binding,
            )
        except assistant_account_challenges.AccountChallengeError as exc:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant account request expired; retry the message",
                code="assistant-account-challenge-expired",
            ) from exc
        except oauth_account_service.OAuthAccountServiceError as exc:
            raise ApiProblem(
                HTTPStatus.BAD_GATEWAY,
                "Assistant account authorization is unavailable",
                code="assistant-account-oauth-unavailable",
            ) from exc
        return {"authorization_url": authorization_url}

    def _current_account_declaration(
        self,
        team_id: str,
        assistant_id: str,
        account_id: str,
    ) -> object:
        with self._lock(team_id):
            spec = self._resolve(assistant_id)
            declaration = spec.accounts.get(account_id)
            if assistant_id not in self._assistant_ids(team_id, running_only=True) or declaration is None:
                raise oauth_account_service.OAuthAccountDeclarationError("OAuth account declaration is unavailable")
            try:
                container = self._assistant_container(team_id, assistant_id)
                container.reload()
            except (ApiProblem, DockerException) as exc:
                raise oauth_account_service.OAuthAccountDeclarationError(
                    "OAuth account declaration is unavailable"
                ) from exc
            attrs = container.attrs if isinstance(container.attrs, dict) else {}
            config = attrs.get("Config")
            if not isinstance(config, dict) or not self._has_current_assistant_artifact(config, spec):
                raise oauth_account_service.OAuthAccountDeclarationError("OAuth account declaration is unavailable")
            return declaration

    def complete_cloudflare_oauth_callback(
        self,
        *,
        state: object,
        claim: object,
        session_binding: object,
    ) -> dict[str, object]:
        try:
            completed = self.oauth_service.complete(
                state,
                claim,
                session_binding,
                self._current_account_declaration,
            )
        except oauth_account_service.OAuthAccountServiceError as exc:
            raise ApiProblem(
                HTTPStatus.BAD_GATEWAY,
                "Assistant account authorization could not be completed",
                code="assistant-account-oauth-unavailable",
            ) from exc
        return {
            "connected": True,
            "team_id": completed.team_id,
            "assistant_id": completed.assistant_id,
            "account_id": completed.account_id,
        }

    def disconnect_assistant_account(
        self,
        team_id: object,
        assistant_id: object,
        account_id: object,
    ) -> dict[str, object]:
        try:
            disconnected = self.oauth_service.disconnect(
                team_id,
                assistant_id,
                account_id,
            )
        except oauth_account_service.OAuthAccountServiceError as exc:
            raise ApiProblem(
                HTTPStatus.BAD_GATEWAY,
                "Assistant account could not be disconnected",
                code="assistant-account-oauth-unavailable",
            ) from exc
        return {"disconnected": disconnected}

    def replace_assistant_secrets(self, team_id: str, body: object) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        if not isinstance(body, dict):
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "Assistant secret replacement is invalid",
                code="invalid-assistant-secrets",
            )
        assistant_id = body.get("assistant_id")
        try:
            spec = self._resolve(assistant_id)
            replacements = assistant_secret_flow.replacement_values(spec, body)
        except (ApiProblem, assistant_secret_flow.SecretFlowError) as exc:
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "Assistant secret replacement is invalid",
                code="invalid-assistant-secrets",
            ) from exc
        chat_lock = self._chat_lock(team_id)
        if not chat_lock.acquire(blocking=False):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant secrets cannot change during an active chat turn",
                code="chat-active",
            )
        try:
            with self._lock(team_id):
                network = self._network(team_id)
                container = self._assistant_container(team_id, spec.assistant_id)
                self._validate_container(container, team_id, spec, network.name)
                # A paused continuation is bound to the generations it observed. Cancelling every
                # affected challenge before the atomic write prevents stale JIT input from winning later.
                self.secret_challenges.cancel_team(team_id)
                self.approval_challenges.cancel_team(team_id)
                self.input_challenges.cancel_team(team_id)
                self._delete_chat_continuation(team_id)
                try:
                    self.assistant_secrets.put_many(team_id, spec.assistant_id, replacements)
                    installed = self.list_assistants(team_id)["assistants"]
                    specs = [self._resolve(item["assistant"]) for item in installed]
                    return assistant_secret_flow.inventory_payload(team_id, specs, self.assistant_secrets)
                except assistant_secret_store.AssistantSecretError as exc:
                    self._raise_secret_problem(exc)
        finally:
            chat_lock.release()

    @staticmethod
    def _challenge_response(
        challenge: assistant_secret_challenges.PendingSecretChallenge,
    ) -> dict[str, object]:
        return assistant_secret_flow.challenge_payload(challenge)

    def pending_chat_secrets(self, team_id: str) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        self._network(team_id)
        challenge = self.secret_challenges.current(team_id)
        return self._challenge_response(challenge) if challenge is not None else {"team_id": team_id, "status": "none"}

    @staticmethod
    def _approval_response(
        challenge: assistant_approval_challenges.PendingApprovalChallenge,
    ) -> dict[str, object]:
        return assistant_approval_flow.challenge_payload(challenge)

    def pending_chat_approval(self, team_id: str) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        self._network(team_id)
        challenge = self.approval_challenges.current(team_id)
        return self._approval_response(challenge) if challenge is not None else {"team_id": team_id, "status": "none"}

    def pending_chat_input(self, team_id: str) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        self._network(team_id)
        challenge = self.input_challenges.current(team_id)
        return self._input_response(challenge) if challenge is not None else {"team_id": team_id, "status": "none"}

    def pending_chat_accounts(self, team_id: str) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        self._network(team_id)
        challenge = self.account_challenges.current(team_id)
        return self._account_response(challenge) if challenge is not None else {"team_id": team_id, "status": "none"}

    def list_assistant_approval_grants(self, team_id: str) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        self._network(team_id)
        try:
            grants = self.approval_grants.list_team(team_id)
        except assistant_approval_grants.ApprovalGrantError as exc:
            self._raise_approval_grant_problem(exc)
        identities = sorted({(item.assistant_id, item.power_id) for item in grants})
        return {
            "team_id": team_id,
            "grants": [{"assistant_id": assistant_id, "power_id": power_id} for assistant_id, power_id in identities],
        }

    def revoke_assistant_approval_grants(self, team_id: str) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        self._network(team_id)
        chat_lock = self._chat_lock(team_id)
        if not chat_lock.acquire(blocking=False):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant approvals cannot change during an active chat turn",
                code="chat-active",
            )
        try:
            with self._lock(team_id):
                revoked = self._revoke_team_approval_grants(team_id)
        finally:
            chat_lock.release()
        return {"team_id": team_id, "revoked": revoked}
