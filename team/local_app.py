"""Minimal Docker controller for one locally owned Shimpz Space.

This is intentionally separate from the hosted Team controller.  An empty Team is
one labeled internal network; its only runnable resources are build-allowlisted,
digest-pinned first-party Assistants with a fixed Power contract.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import NoReturn

import assistant_account_challenges
import assistant_genesis
import assistant_help
import assistant_manifest
import assistant_secret_challenges
import assistant_secret_store
import brain_runtime_client
import brain_runtime_token_store
import docker
import inference_config
import local_chat_continuation_store
import local_token_store
import oauth_account_service
import oauth_account_store
import oauth_broker_client
import oauth_pkce_challenges
import power_execution
import power_journal
import team_storage
from assistant_human import approval_challenges as assistant_approval_challenges
from assistant_human import approval_grants as assistant_approval_grants
from assistant_human import input_challenges as assistant_input_challenges
from docker.errors import APIError, DockerException
from local_registry import (
    AssistantSpec,
    RegistryError,
    load_registry,
    validate_power_input,
    validate_power_output,
)
from local_support import audit as local_audit
from local_support.assistant_lifecycle import LocalAssistantLifecycleMixin
from local_support.assistant_resources import LocalAssistantResourcesMixin
from local_support.assistant_rpc import LocalAssistantRpcMixin
from local_support.assistant_rpc import UnsupportedAssistantRpcPathError as _UnsupportedAssistantRpcPathError
from local_support.chat_api import LocalChatApiMixin
from local_support.chat_execution import LocalChatExecutionMixin
from local_support.chat_pause import LocalChatPauseMixin
from local_support.chat_private import LocalChatPrivateMixin
from local_support.chat_resume import LocalChatResumeMixin
from local_support.chat_segment import LocalChatSegmentMixin
from local_support.chat_state import LocalChatStateMixin
from local_support.chat_submission import LocalChatSubmissionMixin
from local_support.egress import PROFILE, LocalEgressMixin
from local_support.errors import ApiProblemError as ApiProblem
from local_support.http import REQUEST_TIMEOUT_SECONDS, BoundedServer, Handler
from local_support.labels import (
    ASSISTANT_LABEL,
    KIND_LABEL,
    MANAGED_LABEL,
    PROFILE_LABEL,
    SPACE_LABEL,
    TEAM_LABEL,
    TEAM_NAME_LABEL,
)
from local_support.labels import IMAGE_LABEL as _LOCAL_IMAGE_LABEL
from local_support.team_lifecycle import LocalTeamLifecycleMixin
from local_support.validation import brain_thread_id as _local_brain_thread_id
from local_support.validation import (
    half_cpu_set,
    validate_space_id,
    validate_team_id,
    validate_team_name,
)

IMAGE_LABEL = _LOCAL_IMAGE_LABEL
_brain_thread_id = _local_brain_thread_id

log = logging.getLogger("shimpz-team-driver-local")

LISTEN_PORT = 7077
STORAGE_ROOT = Path("/var/lib/shimpz-local/storage")
INFERENCE_ROOT = Path("/var/lib/shimpz-local/inference")
LOCAL_POWER_JOURNAL_PATH = Path(
    os.environ.get(
        "SHIMPZ_LOCAL_POWER_JOURNAL_PATH",
        "/var/lib/shimpz-local/power-journal/journal.sqlite3",
    )
)
LOCAL_APPROVAL_GRANTS_PATH = Path(
    os.environ.get(
        "SHIMPZ_LOCAL_APPROVAL_GRANTS_PATH",
        "/var/lib/shimpz-local/assistant-approvals/grants.sqlite3",
    )
)
LOCAL_CHAT_CONTINUATIONS_STATE_PATH = Path(
    os.environ.get(
        "SHIMPZ_LOCAL_CHAT_CONTINUATIONS_STATE_PATH",
        str(local_chat_continuation_store.STATE_PATH),
    )
)
LOCAL_CHAT_CONTINUATIONS_KEY_PATH = Path(
    os.environ.get(
        "SHIMPZ_LOCAL_CHAT_CONTINUATIONS_KEY_PATH",
        str(local_chat_continuation_store.KEY_PATH),
    )
)


@dataclass(frozen=True, slots=True)
class LocalControllerDependencies:
    inference_store: inference_config.InferenceConfigStore | None = None
    brain_runtime: brain_runtime_client.BrainRuntimeClient | None = None
    power_state: power_journal.PowerJournal | None = None
    assistant_secrets: assistant_secret_store.AssistantSecretStore | None = None
    secret_challenges: assistant_secret_challenges.SecretChallengeStore | None = None
    assistant_accounts: oauth_account_store.OAuthAccountStore | None = None
    account_challenges: assistant_account_challenges.AccountChallengeStore | None = None
    oauth_pkce: oauth_pkce_challenges.OAuthPKCEChallengeStore | None = None
    oauth_broker: oauth_broker_client.OAuthBrokerClient | None = None
    oauth_service: oauth_account_service.BrokeredOAuthAccountService | None = None
    approval_challenges: assistant_approval_challenges.ApprovalChallengeStore | None = None
    approval_grants: assistant_approval_grants.ApprovalGrantStore | None = None
    input_challenges: assistant_input_challenges.InputChallengeStore | None = None
    chat_continuations: local_chat_continuation_store.EncryptedContinuationStore | None = None


class LocalController(
    LocalAssistantLifecycleMixin,
    LocalAssistantResourcesMixin,
    LocalAssistantRpcMixin,
    LocalChatApiMixin,
    LocalChatExecutionMixin,
    LocalChatPauseMixin,
    LocalChatPrivateMixin,
    LocalChatResumeMixin,
    LocalChatSegmentMixin,
    LocalChatStateMixin,
    LocalChatSubmissionMixin,
    LocalEgressMixin,
    LocalTeamLifecycleMixin,
):
    def __init__(
        self,
        client: docker.DockerClient,
        space_id: str,
        registry: dict[str, AssistantSpec],
        storage: team_storage.TeamStorage,
        dependencies: LocalControllerDependencies | None = None,
    ) -> None:
        dependencies = dependencies or LocalControllerDependencies()
        self.client = client
        self.space_id = validate_space_id(space_id)
        self.registry = registry
        self.storage = storage
        self.inference_store = dependencies.inference_store or inference_config.InferenceConfigStore(INFERENCE_ROOT)
        self.brain_runtime = dependencies.brain_runtime or brain_runtime_client.BrainRuntimeClient()
        self.power_state = (
            dependencies.power_state
            if dependencies.power_state is not None
            else power_journal.PowerJournal(LOCAL_POWER_JOURNAL_PATH)
        )
        self.assistant_secrets = dependencies.assistant_secrets or assistant_secret_store.AssistantSecretStore()
        self.secret_challenges = dependencies.secret_challenges or assistant_secret_challenges.SecretChallengeStore()
        self.assistant_accounts = dependencies.assistant_accounts or oauth_account_store.OAuthAccountStore()
        self.account_challenges = (
            dependencies.account_challenges or assistant_account_challenges.AccountChallengeStore()
        )
        self.oauth_pkce = dependencies.oauth_pkce or oauth_pkce_challenges.OAuthPKCEChallengeStore()
        self.oauth_broker = dependencies.oauth_broker or oauth_broker_client.OAuthBrokerClient(
            transport=oauth_broker_client.FixedBrokerTransport(
                proxy_host=os.environ.get("SHIMPZ_OAUTH_BROKER_PROXY_HOST"),
                proxy_token=os.environ.get("SHIMPZ_OAUTH_BROKER_PROXY_TOKEN"),
            ),
            callback_mode=os.environ.get("SHIMPZ_OAUTH_CALLBACK_MODE", "loopback"),
        )
        self.oauth_service = dependencies.oauth_service or oauth_account_service.BrokeredOAuthAccountService(
            challenge=self.oauth_pkce,
            store=self.assistant_accounts,
            broker=self.oauth_broker,
        )
        self.approval_challenges = (
            dependencies.approval_challenges or assistant_approval_challenges.ApprovalChallengeStore()
        )
        self.approval_grants = dependencies.approval_grants or assistant_approval_grants.ApprovalGrantStore(
            LOCAL_APPROVAL_GRANTS_PATH
        )
        self.input_challenges = dependencies.input_challenges or assistant_input_challenges.InputChallengeStore()
        self.chat_continuations = (
            dependencies.chat_continuations
            or local_chat_continuation_store.EncryptedContinuationStore(
                LOCAL_CHAT_CONTINUATIONS_STATE_PATH,
                LOCAL_CHAT_CONTINUATIONS_KEY_PATH,
            )
        )
        self._assistant_genesis_cache = assistant_genesis.GenesisCache()
        self._assistant_allowed_hosts_cache = assistant_manifest.ManifestContractCache()
        self._assistant_machine_contract_cache = assistant_manifest.MachineContractCache()
        self._blocked_power_workloads: set[str] = set()
        self._locks = tuple(threading.RLock() for _ in range(64))
        self._active_chat_guard = threading.Lock()
        self._chat_locks: dict[str, threading.Lock] = {}
        self._active_chat_tokens: dict[str, str] = {}
        self._active_power_containers: dict[str, tuple[str, object]] = {}
        self._cancelled_chat_tokens: set[str] = set()
        daemon_info = self._require_default_seccomp()
        self.cpuset_cpus = half_cpu_set(daemon_info.get("NCPU"))
        self._restore_all_chat_continuations()

    def _require_default_seccomp(self) -> dict:
        try:
            info = self.client.info()
            options = info.get("SecurityOptions", [])
        except DockerException as exc:
            raise RuntimeError("the Docker daemon is unavailable") from exc
        if not any(isinstance(option, str) and option.startswith("name=seccomp") for option in options):
            raise RuntimeError("the Docker daemon default seccomp profile is required")
        return info

    def _lock(self, team_id: str) -> threading.RLock:
        slot = hashlib.sha256(team_id.encode("ascii")).digest()[0] % len(self._locks)
        return self._locks[slot]

    def _chat_lock(self, team_id: str) -> threading.Lock:
        with self._active_chat_guard:
            return self._chat_locks.setdefault(team_id, threading.Lock())

    def _chat_cancelled(self, token: str) -> bool:
        with self._active_chat_guard:
            return token in self._cancelled_chat_tokens

    def _commit_chat_terminal(self, team_id: str, token: str) -> bool:
        """Commit a reply only when Stop did not win this Controller-owned turn."""
        with self._active_chat_guard:
            if token in self._cancelled_chat_tokens or self._active_chat_tokens.get(team_id) != token:
                return False
            self._active_chat_tokens.pop(team_id, None)
            return True

    def _cancel_chat_for_destroy(self, team_id: str) -> None:
        """Prevent another Power and synchronously stop one already executing."""
        with self._active_chat_guard:
            token = self._active_chat_tokens.get(team_id)
            if token is not None:
                self._cancelled_chat_tokens.add(token)
            active = self._active_power_containers.get(team_id)
            active_power = active[1] if token is not None and active is not None and active[0] == token else None
        if active_power is not None:
            self._fail_stop_power(active_power)

    @contextmanager
    def _exclusive_chat_turn(self, team_id: str):
        lock = self._chat_lock(team_id)
        if not lock.acquire(blocking=False):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Team already has an active chat turn",
                code="chat-active",
            )
        token = secrets.token_hex(16)
        with self._active_chat_guard:
            self._active_chat_tokens[team_id] = token
        try:
            yield token
        finally:
            with self._active_chat_guard:
                if self._active_chat_tokens.get(team_id) == token:
                    self._active_chat_tokens.pop(team_id, None)
                active = self._active_power_containers.get(team_id)
                if active is not None and active[0] == token:
                    self._active_power_containers.pop(team_id, None)
                self._cancelled_chat_tokens.discard(token)
            lock.release()

    def list_teams(self) -> dict[str, list[dict[str, str]]]:
        filters = {
            "label": [
                f"{MANAGED_LABEL}=1",
                f"{PROFILE_LABEL}={PROFILE}",
                f"{SPACE_LABEL}={self.space_id}",
                f"{KIND_LABEL}=team",
            ]
        }
        teams: list[dict[str, str]] = []
        try:
            networks = self.client.networks.list(filters=filters)
        except DockerException as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Docker is unavailable",
                code="docker-unavailable",
            ) from exc
        for network in networks:
            labels = network.attrs.get("Labels") or {}
            team_id = labels.get(TEAM_LABEL)
            if not isinstance(team_id, str):
                raise ApiProblem(HTTPStatus.CONFLICT, "Team resource ownership conflict", code="ownership-conflict")
            validate_team_id(team_id)
            team_name = self._validate_network(network, team_id)
            teams.append({"team_id": team_id, "team_name": team_name, "status": "running"})
        teams.sort(key=lambda item: item["team_id"])
        return {"teams": teams}

    def create_team(self, team_id: str, team_name: str) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        team_name = validate_team_name(team_name)
        with self._lock(team_id):
            existing = self._network(team_id, required=False)
            if existing is not None:
                existing_name = self._validate_network(existing, team_id)
                if existing_name != team_name:
                    raise ApiProblem(
                        HTTPStatus.CONFLICT,
                        "Team id already belongs to a different name",
                        code="team-name-conflict",
                    )
                return {"team_id": team_id, "team_name": team_name, "status": "running", "created": False}
            try:
                # A Team identity starts empty even after a daemon crash removed its network
                # before the previous lifecycle could clean the dedicated storage volume.
                self.storage.destroy(team_id)
            except team_storage.StorageError as exc:
                self._raise_storage_problem(exc)
            try:
                self.inference_store.delete(team_id)
            except inference_config.InferenceConfigError as exc:
                self._raise_inference_problem(exc)
            try:
                labels = self._base_labels(team_id, "team")
                labels[TEAM_NAME_LABEL] = team_name
                network = self.client.networks.create(
                    self._network_name(team_id),
                    driver="bridge",
                    internal=True,
                    attachable=False,
                    check_duplicate=True,
                    labels=labels,
                )
            except APIError as exc:
                # A concurrent idempotent creator is safe only when the resulting
                # resource proves the exact ownership/profile labels.
                network = self._network(team_id, required=False)
                if network is None:
                    raise ApiProblem(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "Docker could not create the Team",
                        code="docker-create-failed",
                    ) from exc
                existing_name = self._validate_network(network, team_id)
                if existing_name != team_name:
                    raise ApiProblem(
                        HTTPStatus.CONFLICT,
                        "Team id already belongs to a different name",
                        code="team-name-conflict",
                    ) from exc
                return {"team_id": team_id, "team_name": team_name, "status": "running", "created": False}
            self._validate_network(network, team_id)
            return {"team_id": team_id, "team_name": team_name, "status": "running", "created": True}

    @staticmethod
    def _raise_storage_problem(exc: team_storage.StorageError) -> NoReturn:
        if isinstance(exc, team_storage.StorageQuotaError):
            raise ApiProblem(
                HTTPStatus.INSUFFICIENT_STORAGE,
                str(exc),
                code="storage-quota-exceeded",
            ) from exc
        if isinstance(exc, team_storage.StorageNotFoundError):
            raise ApiProblem(HTTPStatus.NOT_FOUND, "file not found", code="file-not-found") from exc
        if isinstance(exc, team_storage.StorageInputError):
            raise ApiProblem(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc), code="invalid-file") from exc
        raise ApiProblem(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Team storage failed its safety checks",
            code="storage-safety-failed",
        ) from exc

    @staticmethod
    def _raise_inference_problem(exc: inference_config.InferenceConfigError) -> NoReturn:
        raise ApiProblem(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Team model provider metadata is unavailable",
            code="inference-store-failed",
        ) from exc

    def inference_status(self, team_id: str) -> dict[str, str]:
        team_id = validate_team_id(team_id)
        with self._lock(team_id):
            self._network(team_id)
            try:
                config = self.inference_store.load(team_id)
            except inference_config.InferenceConfigError as exc:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "Team model provider is not configured",
                    code="inference-not-configured",
                ) from exc
        return {"team_id": team_id, "provider": config.provider, "model": config.model}

    def configure_inference(self, team_id: str, body: object) -> dict[str, str]:
        team_id = validate_team_id(team_id)
        if not isinstance(body, dict) or set(body) != {"provider", "model"}:
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "inference requires only provider and model",
                code="invalid-body",
            )
        try:
            config = inference_config.normalize(body["provider"], body["model"])
        except inference_config.InferenceConfigError as exc:
            raise ApiProblem(HTTPStatus.BAD_REQUEST, str(exc), code="invalid-inference") from exc
        with self._lock(team_id):
            self._network(team_id)
            try:
                self.inference_store.save(team_id, config)
            except inference_config.InferenceConfigError as exc:
                self._raise_inference_problem(exc)
        return {"team_id": team_id, "provider": config.provider, "model": config.model}

    def put_file(
        self,
        team_id: str,
        filename: object,
        content: bytes,
        media_type: object,
    ) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        with self._lock(team_id):
            self._network(team_id)
            try:
                stored = self.storage.put(team_id, filename, content, media_type)
            except team_storage.StorageError as exc:
                self._raise_storage_problem(exc)
        return {"team_id": team_id, "file": stored}

    def list_files(self, team_id: str) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        with self._lock(team_id):
            self._network(team_id)
            try:
                listing = self.storage.list(team_id)
            except team_storage.StorageError as exc:
                self._raise_storage_problem(exc)
        return {"team_id": team_id, **listing}

    def delete_file(self, team_id: str, file_id: object) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        with self._lock(team_id):
            self._network(team_id)
            try:
                result = self.storage.delete(team_id, file_id)
            except team_storage.StorageError as exc:
                self._raise_storage_problem(exc)
        return {"team_id": team_id, **result}

    def list_registry(self) -> dict[str, list[dict[str, object]]]:
        return {
            "assistants": [
                {
                    "id": spec.assistant_id,
                    "title": spec.name,
                    "summary": spec.summary,
                    "powers": sorted(spec.powers),
                }
                for spec in sorted(self.registry.values(), key=lambda item: item.assistant_id)
            ]
        }

    def health(self) -> dict[str, str]:
        try:
            if self.client.ping() is not True:
                raise DockerException("unexpected Docker ping response")
        except DockerException as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Docker is unavailable",
                code="docker-unavailable",
            ) from exc
        return {"status": "ok"}

    def list_assistants(self, team_id: str) -> dict[str, list[dict[str, str]]]:
        team_id = validate_team_id(team_id)
        self._network(team_id)
        output: list[dict[str, str]] = []
        try:
            containers = self.client.containers.list(**self._assistant_filters(team_id))
        except DockerException as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Docker is unavailable",
                code="docker-unavailable",
            ) from exc
        for container in containers:
            labels = container.labels
            assistant_id = labels.get(ASSISTANT_LABEL)
            spec = self.registry.get(assistant_id)
            if spec is None:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "an installed Assistant is no longer allowlisted",
                    code="assistant-registry-drift",
                )
            config = self._validate_container_security(
                container,
                team_id,
                spec,
                self._network_name(team_id),
            )
            if self._has_current_assistant_artifact(config, spec):
                self._admit_assistant_allowed_hosts(container, spec)
                status = container.status
            else:
                status = "outdated"
            output.append({"assistant": assistant_id, "status": status})
        output.sort(key=lambda item: item["assistant"])
        return {"assistants": output}

    def assistant_help(self, team_id: str, assistant_id: str, locale: str = "en") -> dict[str, str]:
        """Read bounded Markdown only from one installed, running Assistant's fixed RPC."""
        team_id = validate_team_id(team_id)
        try:
            locale = assistant_help.validate_locale(locale)
        except ValueError as exc:
            raise ApiProblem(
                HTTPStatus.BAD_REQUEST,
                "Assistant Help locale is not supported",
                code="invalid-help-locale",
            ) from exc
        spec = self._resolve(assistant_id)
        with self._lock(team_id):
            network = self._network(team_id)
            container = self._assistant_container(team_id, assistant_id)
            self._validate_container(container, team_id, spec, network.name)
            container.reload()
            if container.status != "running":
                raise ApiProblem(HTTPStatus.CONFLICT, "Assistant is not running", code="assistant-not-running")
            try:
                raw_result = self._rpc(
                    container,
                    spec,
                    "GET",
                    f"/v1/help/{locale}",
                    {},
                    detect_unsupported_path=True,
                )
            except _UnsupportedAssistantRpcPathError:
                raw_result = self._rpc(container, spec, "GET", "/v1/help", {})
        try:
            help_payload = assistant_help.validate_payload(raw_result)
        except ValueError as exc:
            raise ApiProblem(
                HTTPStatus.BAD_GATEWAY,
                "Assistant Help returned an invalid result",
                code="invalid-assistant-help",
            ) from exc
        return {"assistant": spec.assistant_id, **help_payload}

    def invoke(
        self,
        team_id: str,
        assistant_id: str,
        power: str,
        payload: object,
        *,
        answers: tuple[object, ...] = (),
    ) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        spec = self._resolve(assistant_id)
        power_spec = spec.powers.get(power)
        if power_spec is None:
            raise ApiProblem(
                power_execution.UNDECLARED_POWER_STATUS, "Power is not declared", code="power-not-declared"
            )
        try:
            safe_payload = validate_power_input(assistant_id, power, payload)
        except ValueError as exc:
            raise ApiProblem(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc), code="invalid-power-input") from exc
        with self._lock(team_id):
            network = self._network(team_id)
            container = self._assistant_container(team_id, assistant_id)
            self._validate_container(container, team_id, spec, network.name)
            if container.id in self._blocked_power_workloads:
                raise ApiProblem(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Assistant Power execution is blocked until this Assistant is reinstalled",
                    code="assistant-power-blocked",
                )
            container.reload()
            if container.status != "running":
                raise ApiProblem(HTTPStatus.CONFLICT, "Assistant is not running", code="assistant-not-running")
            with self._active_chat_guard:
                active = self._active_power_containers.get(team_id)
                frozen_container = active[1] if active is not None else None
            if frozen_container is not None and frozen_container.id != container.id:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "Team capabilities changed; retry",
                    code="team-context-changed",
                )
            secret_values = self._resolve_power_secrets(team_id, spec, power)
            account_values = self._resolve_power_accounts(team_id, spec, power)
            local_audit.record(
                "assistant-power",
                result="ok",
                team_id=team_id,
                assistant=assistant_id,
                detail=f"started:{power}",
            )
            try:
                raw_result = self._rpc(
                    container,
                    spec,
                    power_spec.method,
                    power_spec.path,
                    {
                        "input": safe_payload,
                        "secrets": secret_values,
                        "accounts": account_values,
                        "answers": list(answers),
                    },
                )
            except ApiProblem:
                local_audit.record(
                    "assistant-power",
                    result="error",
                    team_id=team_id,
                    assistant=assistant_id,
                    detail=f"failed:{power}",
                )
                raise
        try:
            projected = power_execution.project_rpc_result(
                raw_result,
                secret_values,
                account_values,
                answers,
                lambda value: validate_power_output(assistant_id, power, value),
            )
        except power_execution.RpcSecretExposureError:
            local_audit.record(
                "assistant-power",
                result="error",
                team_id=team_id,
                assistant=assistant_id,
                detail=f"secret-exposure:{power}",
            )
            raise ApiProblem(
                HTTPStatus.BAD_GATEWAY,
                "the Assistant returned an unsafe result",
                code="assistant-secret-exposure",
            ) from None
        except power_execution.RpcInvalidResultError as exc:
            local_audit.record(
                "assistant-power",
                result="error",
                team_id=team_id,
                assistant=assistant_id,
                detail=f"invalid-output:{power}",
            )
            raise ApiProblem(
                HTTPStatus.BAD_GATEWAY,
                "the Assistant returned an invalid result",
                code="invalid-power-output",
            ) from exc
        if projected.suspended:
            local_audit.record(
                "assistant-power",
                result="ok",
                team_id=team_id,
                assistant=assistant_id,
                detail=f"suspended:{power}",
            )
            return {"assistant": assistant_id, "power": power, "suspend": projected.value}
        local_audit.record(
            "assistant-power",
            result="ok",
            team_id=team_id,
            assistant=assistant_id,
            detail=f"completed:{power}",
        )
        return {"assistant": assistant_id, "power": power, "result": projected.value}


def main() -> int:
    try:
        space_id = os.environ["SHIMPZ_SPACE_ID"]
        registry = load_registry()
        token = local_token_store.ensure_token()
        brain_runtime_token_store.ensure()
        client = docker.from_env(timeout=REQUEST_TIMEOUT_SECONDS)
        storage = team_storage.TeamStorage(STORAGE_ROOT)
        controller = LocalController(client, space_id, registry, storage)
        server = BoundedServer(("0.0.0.0", LISTEN_PORT), Handler, controller, token)
    except (KeyError, RegistryError, RuntimeError, DockerException) as exc:
        print(f"team-driver-local: startup failed: {exc}", file=sys.stderr, flush=True)
        return 1
    local_audit.record("startup", result="ok")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
