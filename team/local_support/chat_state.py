"""Local chat setup and persisted state mixin."""

import logging
import math
import time
from http import HTTPStatus
from typing import NoReturn

import assistant_account_challenges
import assistant_genesis
import assistant_manifest
import assistant_secret_challenges
import assistant_secret_store
import inference_config
import local_chat_continuation_store
import local_chat_continuations
import oauth_account_store
import team_storage
from assistant_human import approval_challenges as assistant_approval_challenges
from assistant_human import approval_grants as assistant_approval_grants
from assistant_human import input_challenges as assistant_input_challenges
from docker.errors import DockerException
from local_registry import AssistantSpec

from local_support.chat_types import ActiveAssistant as _ActiveAssistant
from local_support.chat_types import PendingLocalChat as _PendingLocalChat
from local_support.errors import ApiProblemError as ApiProblem
from local_support.labels import ASSISTANT_LABEL

log = logging.getLogger("shimpz-team-driver-local")
MAX_CHAT_FILES = 8


class LocalChatStateMixin:
    def _chat_file_metadata(self, team_id: str, file_ids: object) -> list[dict[str, object]]:
        if not isinstance(file_ids, list) or len(file_ids) > MAX_CHAT_FILES:
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                f"files must contain at most {MAX_CHAT_FILES} opaque ids",
                code="invalid-files",
            )
        try:
            return self.storage.metadata(team_id, file_ids)
        except team_storage.StorageNotFoundError as exc:
            raise ApiProblem(HTTPStatus.NOT_FOUND, "selected file not found", code="file-not-found") from exc
        except team_storage.StorageInputError as exc:
            raise ApiProblem(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc), code="invalid-files") from exc
        except team_storage.StorageError as exc:
            self._raise_storage_problem(exc)

    def _chat_setup(
        self,
        team_id: str,
        file_ids: object,
        provider: str,
        assistant_ids: tuple[str, ...],
    ) -> tuple[
        str,
        str,
        tuple[_ActiveAssistant, ...],
        list[dict[str, object]],
        inference_config.InferenceConfig,
    ]:
        with self._lock(team_id):
            network = self._network(team_id)
            team_name = self._validate_network(network, team_id)
            network_id = getattr(network, "id", None)
            if not isinstance(network_id, str) or not network_id:
                raise ApiProblem(HTTPStatus.CONFLICT, "Team resource ownership conflict", code="ownership-conflict")
            active_assistants = self._active_chat_assistants(team_id, network.name)
            active_by_id = {active.spec.assistant_id: active for active in active_assistants}
            try:
                assistants = tuple(active_by_id[assistant_id] for assistant_id in assistant_ids)
            except KeyError:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "a selected Assistant is unavailable",
                    code="assistant-unavailable",
                ) from None
            files = self._chat_file_metadata(team_id, file_ids)
            try:
                config = self.inference_store.load(team_id)
            except inference_config.InferenceConfigError as exc:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "Team model provider is not configured",
                    code="inference-not-configured",
                ) from exc
            if config.provider != provider:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "configured model provider changed; retry",
                    code="inference-provider-mismatch",
                )
        return team_name, network_id, assistants, files, config

    def _active_assistant_genesis(self, active: _ActiveAssistant) -> str:
        container = active.container
        if container is None:
            try:
                container = self.client.containers.get(active.container_id)
            except DockerException as exc:
                raise ApiProblem(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "installed Assistant Genesis could not be verified",
                    code="assistant-genesis-unavailable",
                ) from exc
        if getattr(container, "id", None) != active.container_id:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "installed Assistant Genesis failed its identity contract",
                code="assistant-genesis-drift",
            )
        try:
            return self._assistant_genesis_cache.get(container)
        except assistant_genesis.GenesisError as exc:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "installed Assistant Genesis failed its contract",
                code="assistant-genesis-invalid",
            ) from exc

    def _admit_assistant_allowed_hosts(self, container, spec: AssistantSpec) -> tuple[str, ...]:
        try:
            reviewed = assistant_manifest.reviewed_manifest_contract(
                allowed_hosts=spec.allowed_hosts,
                accounts=spec.accounts,
            )
            declared = self._assistant_allowed_hosts_cache.get(container, reviewed)
            self._assistant_machine_contract_cache.get(container, declared.accounts, spec.machine_contract)
        except assistant_manifest.ManifestError as exc:
            log.warning("Assistant manifest admission failed: %s", exc)
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "installed Assistant manifest failed its reviewed contract",
                code="assistant-manifest-invalid",
            ) from exc
        else:
            return declared.allowed_hosts

    def _active_chat_assistants(self, team_id: str, network_name: str) -> tuple[_ActiveAssistant, ...]:
        try:
            containers = self.client.containers.list(**self._assistant_filters(team_id))
        except DockerException as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Docker is unavailable",
                code="docker-unavailable",
            ) from exc
        active: list[_ActiveAssistant] = []
        for container in containers:
            assistant_id = (container.labels or {}).get(ASSISTANT_LABEL)
            spec = self.registry.get(assistant_id)
            if spec is None:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "an installed Assistant is no longer allowlisted",
                    code="assistant-registry-drift",
                )
            self._validate_container(container, team_id, spec, network_name)
            if container.id in self._blocked_power_workloads:
                raise ApiProblem(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Assistant Power execution is blocked until this Assistant is reinstalled",
                    code="assistant-power-blocked",
                )
            container.reload()
            if container.status == "running":
                active.append(_ActiveAssistant(spec=spec, container_id=container.id, container=container))
        active.sort(key=lambda item: item.spec.assistant_id)
        return tuple(active)

    @staticmethod
    def _raise_secret_problem(exc: assistant_secret_store.AssistantSecretError) -> NoReturn:
        if isinstance(exc, assistant_secret_store.AssistantSecretMissingError):
            raise ApiProblem(
                HTTPStatus.PRECONDITION_REQUIRED,
                "Assistant secrets are required",
                code="assistant-secrets-required",
            ) from exc
        if isinstance(exc, assistant_secret_store.AssistantSecretValidationError):
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "Assistant secret values are invalid",
                code="invalid-assistant-secrets",
            ) from exc
        raise ApiProblem(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Assistant secret state is unavailable",
            code="assistant-secret-state-unavailable",
        ) from exc

    def _delete_assistant_secret_state(self, team_id: str, assistant_id: str) -> None:
        try:
            self.assistant_secrets.delete_assistant(team_id, assistant_id)
        except assistant_secret_store.AssistantSecretError as exc:
            self._raise_secret_problem(exc)

    def _delete_team_secret_state(self, team_id: str) -> None:
        try:
            self.assistant_secrets.delete_team(team_id)
        except assistant_secret_store.AssistantSecretError as exc:
            self._raise_secret_problem(exc)

    def _delete_all_secret_state(self) -> None:
        try:
            self.assistant_secrets.delete_all()
        except assistant_secret_store.AssistantSecretError as exc:
            self._raise_secret_problem(exc)

    def _delete_assistant_account_state(self, team_id: str, assistant_id: str) -> None:
        try:
            self.assistant_accounts.delete_assistant(team_id, assistant_id)
        except oauth_account_store.OAuthAccountStoreError as exc:
            self._raise_account_problem(exc)

    def _delete_team_account_state(self, team_id: str) -> None:
        try:
            self.assistant_accounts.delete_team(team_id)
        except oauth_account_store.OAuthAccountStoreError as exc:
            self._raise_account_problem(exc)

    def _delete_all_account_state(self) -> None:
        try:
            self.assistant_accounts.delete_all()
        except oauth_account_store.OAuthAccountStoreError as exc:
            self._raise_account_problem(exc)

    def _retain_declared_assistant_account_state(self, team_id: str, spec: AssistantSpec) -> None:
        try:
            pruned = self.assistant_accounts.retain_declared(
                team_id,
                spec.assistant_id,
                tuple(sorted(spec.accounts)),
            )
        except oauth_account_store.OAuthAccountStoreError as exc:
            self._raise_account_problem(exc)
        if pruned:
            self.account_challenges.cancel_team(team_id)
            self._delete_chat_continuation(team_id)

    @staticmethod
    def _raise_approval_grant_problem(exc: assistant_approval_grants.ApprovalGrantError) -> NoReturn:
        raise ApiProblem(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Assistant approval state is unavailable",
            code="assistant-approval-state-unavailable",
        ) from exc

    def _revoke_assistant_approval_grants(self, team_id: str, assistant_id: str) -> None:
        try:
            self.approval_grants.revoke_assistant(team_id, assistant_id)
        except assistant_approval_grants.ApprovalGrantError as exc:
            self._raise_approval_grant_problem(exc)

    def _revoke_team_approval_grants(self, team_id: str) -> int:
        try:
            return self.approval_grants.revoke_team(team_id)
        except assistant_approval_grants.ApprovalGrantError as exc:
            self._raise_approval_grant_problem(exc)

    def _revoke_all_approval_grants(self) -> int:
        try:
            return self.approval_grants.revoke_all()
        except assistant_approval_grants.ApprovalGrantError as exc:
            self._raise_approval_grant_problem(exc)

    @staticmethod
    def _raise_chat_continuation_problem(exc: Exception) -> NoReturn:
        raise ApiProblem(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Team chat continuation state is unavailable",
            code="chat-state-unavailable",
        ) from exc

    def _persist_chat_continuation(
        self,
        kind: str,
        challenge: object,
        requirements: tuple[object, ...],
        pending: _PendingLocalChat,
    ) -> None:
        challenge_id = getattr(challenge, "id", None)
        expires_at = getattr(challenge, "expires_at", None)
        remaining_seconds = math.ceil(expires_at - time.monotonic()) if isinstance(expires_at, float) else 0
        if (
            not isinstance(challenge_id, str)
            or not 1 <= remaining_seconds <= local_chat_continuation_store.MAX_TTL_SECONDS
        ):
            self._raise_chat_continuation_problem(
                local_chat_continuation_store.ContinuationStoreError("challenge lifetime is invalid")
            )
        try:
            bindings, payload = local_chat_continuations.encode(kind, requirements, pending)
            self.chat_continuations.put(
                getattr(challenge, "team_id", None),
                kind,
                challenge_id,
                int(time.time()) + remaining_seconds,
                bindings,
                payload,
            )
        except (
            local_chat_continuation_store.ContinuationStoreError,
            local_chat_continuations.ContinuationCodecError,
        ) as exc:
            self._raise_chat_continuation_problem(exc)

    def _restore_chat_continuation(
        self,
        stored: local_chat_continuation_store.StoredContinuation,
    ) -> None:
        remaining_seconds = stored.expires_at - int(time.time())
        if remaining_seconds <= 0:
            return
        try:
            decoded = local_chat_continuations.decode(stored)
            if decoded.kind == "accounts":
                self.account_challenges.restore(
                    stored.team_id,
                    stored.challenge_id,
                    remaining_seconds,
                    decoded.requirements,
                    decoded.pending,
                )
            elif decoded.kind == "secrets":
                self.secret_challenges.restore(
                    stored.team_id,
                    stored.challenge_id,
                    remaining_seconds,
                    decoded.requirements,
                    decoded.pending,
                )
            elif decoded.kind == "input":
                if len(decoded.requirements) != 1:
                    raise local_chat_continuations.ContinuationCodecError(
                        "input continuation requirements are malformed"
                    )
                self.input_challenges.restore(
                    stored.team_id,
                    stored.challenge_id,
                    remaining_seconds,
                    decoded.requirements[0],
                    decoded.pending,
                )
            elif decoded.kind == "approval":
                self.approval_challenges.restore(
                    stored.team_id,
                    stored.challenge_id,
                    remaining_seconds,
                    decoded.requirements,
                    decoded.pending,
                )
            else:
                raise local_chat_continuations.ContinuationCodecError("continuation kind is malformed")
        except (
            local_chat_continuation_store.ContinuationStoreError,
            local_chat_continuations.ContinuationCodecError,
            assistant_account_challenges.AccountChallengeError,
            assistant_secret_challenges.SecretChallengeError,
            assistant_input_challenges.InputChallengeError,
            assistant_approval_challenges.ApprovalChallengeError,
        ) as exc:
            self._raise_chat_continuation_problem(exc)

    def _restore_all_chat_continuations(self) -> None:
        try:
            stored = self.chat_continuations.active()
        except local_chat_continuation_store.ContinuationStoreError as exc:
            self._raise_chat_continuation_problem(exc)
        for continuation in stored:
            self._restore_chat_continuation(continuation)

    def _delete_chat_continuation(
        self,
        team_id: str,
        challenge_id: str | None = None,
    ) -> bool:
        store = getattr(self, "chat_continuations", None)
        if store is None:
            return False
        try:
            return store.delete(team_id, challenge_id)
        except local_chat_continuation_store.ContinuationStoreError as exc:
            self._raise_chat_continuation_problem(exc)

    def _clear_chat_continuations(self) -> int:
        store = getattr(self, "chat_continuations", None)
        if store is None:
            return 0
        try:
            return store.clear()
        except local_chat_continuation_store.ContinuationStoreError as exc:
            self._raise_chat_continuation_problem(exc)
