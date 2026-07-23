"""Minimal Docker controller for one locally owned Shimpz Space.

This is intentionally separate from the hosted Team controller.  An empty Team is
one labeled internal network; its only runnable resources are build-allowlisted,
digest-pinned first-party Assistants with a fixed Power contract.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
import socket
import sys
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import replace
from http import HTTPStatus
from pathlib import Path
from typing import NoReturn

import assistant_account_challenges
import assistant_genesis
import assistant_help
import assistant_manifest
import assistant_secret_challenges
import assistant_secret_flow
import assistant_secret_store
import brain_runtime_client
import brain_runtime_token_store
import chat_orchestrator
import chat_turn_engine
import docker
import egress_policy
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
from assistant_human import approval_flow as assistant_approval_flow
from assistant_human import approval_grants as assistant_approval_grants
from assistant_human import input_challenges as assistant_input_challenges
from assistant_human import input_flow as assistant_input_flow
from container_policy import local as local_container_policy
from docker.errors import APIError, DockerException, ImageNotFound, NotFound
from docker.types import LogConfig, Ulimit
from local_registry import (
    AssistantSpec,
    RegistryError,
    load_registry,
    validate_power_input,
    validate_power_output,
)
from local_support import audit as local_audit
from local_support.chat_execution import LocalChatExecutionMixin
from local_support.chat_pause import LocalChatPauseMixin
from local_support.chat_private import LocalChatPrivateMixin
from local_support.chat_segment import LocalChatSegmentMixin
from local_support.chat_segment import SegmentRequest as _ChatSegmentRequest
from local_support.chat_state import LocalChatStateMixin
from local_support.chat_types import ActiveAssistant as _ActiveAssistant
from local_support.chat_types import PendingLocalChat as _PendingLocalChat
from local_support.errors import ApiProblemError as ApiProblem
from local_support.http import REQUEST_TIMEOUT_SECONDS, BoundedServer, Handler
from local_support.labels import (
    ASSISTANT_LABEL,
    IMAGE_LABEL,
    KIND_LABEL,
    MANAGED_LABEL,
    PROFILE_LABEL,
    SPACE_LABEL,
    TEAM_LABEL,
    TEAM_NAME_LABEL,
)
from local_support.validation import (
    ASSISTANT_ID_RE as _ASSISTANT_ID,
)
from local_support.validation import (
    MAX_ASSISTANT_ID_LENGTH,
    MAX_TEAM_ID_LENGTH,
    half_cpu_set,
    validate_chat_assistant_ids,
    validate_space_id,
    validate_team_id,
    validate_team_name,
)
from local_support.validation import (
    TEAM_ID_RE as _TEAM_ID,
)
from local_support.validation import brain_thread_id as _brain_thread_id
from local_support.validation import space_prefix as _space_prefix

log = logging.getLogger("shimpz-team-driver-local")

LISTEN_PORT = 7077
PROFILE = "single-owner-local-v1"

MAX_RESPONSE_BYTES = assistant_help.MAX_HELP_BYTES * 6 + 1024
MAX_EGRESS_POLICY_BYTES = egress_policy.MAX_POLICY_BYTES
RPC_TIMEOUT_SECONDS = 8
HEALTH_TIMEOUT_SECONDS = 15
MAX_CHAT_MESSAGE_CHARS = 16_000
APP_EGRESS_PROXY_ALIAS = "app-egress-proxy"
APP_EGRESS_PROXY_PORT = 8889
APP_EGRESS_PROXY_KIND = "app-egress-proxy"
APP_EGRESS_POLICY_GID = 10017
APP_EGRESS_PROXY_CONTAINER = os.environ.get("SHIMPZ_APP_EGRESS_PROXY_CONTAINER", "").strip()
APP_EGRESS_POLICY_DIR = Path(
    os.environ.get(
        "SHIMPZ_APP_EGRESS_POLICY_DIR",
        "/var/lib/shimpz-local/app-egress",
    )
)

ASSISTANT_UID = local_container_policy.ASSISTANT_UID
ASSISTANT_WORKDIR = str(Path("/") / "tmp")
ASSISTANT_MEMORY = local_container_policy.ASSISTANT_MEMORY
ASSISTANT_NANO_CPUS = local_container_policy.ASSISTANT_NANO_CPUS
ASSISTANT_PIDS = local_container_policy.ASSISTANT_PIDS
READINESS_RECOVERY_ASSISTANTS = frozenset({"shimpz-cloudflare"})
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
_CONTAINER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class _UnsupportedAssistantRpcPathError(RuntimeError):
    """The fixed Assistant RPC adapter rejected a path it does not implement."""


def _is_replaceable_readiness_failure(assistant_id: str, problem: ApiProblem) -> bool:
    return assistant_id in READINESS_RECOVERY_ASSISTANTS and problem.code == "assistant-not-ready"


def _egress_store() -> egress_policy.EgressPolicyStore:
    return egress_policy.EgressPolicyStore(
        APP_EGRESS_POLICY_DIR,
        APP_EGRESS_POLICY_GID,
        "127.0.0.1,localhost",
        APP_EGRESS_PROXY_ALIAS,
        APP_EGRESS_PROXY_PORT,
    )


def _raise_egress_problem(exc: egress_policy.EgressPolicyError) -> NoReturn:
    if isinstance(exc, egress_policy.EgressPolicyDriftError):
        raise ApiProblem(
            HTTPStatus.CONFLICT,
            "Assistant egress policy failed its ownership contract",
            code="egress-policy-drift",
        ) from exc
    raise ApiProblem(
        HTTPStatus.SERVICE_UNAVAILABLE,
        "Assistant egress policy storage is unavailable",
        code="egress-policy-unavailable",
    ) from exc


def _serialize_against_local_team_chat(
    operation: Callable[..., dict[str, object]],
) -> Callable[..., dict[str, object]]:
    """Reject Assistant mutation before its first side effect while a Team turn owns the slot."""

    def guarded(controller, team_id: str, *args, **kwargs) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        lock = controller._chat_lock(team_id)
        if not lock.acquire(blocking=False):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant lifecycle cannot change during an active Team chat turn",
                code="chat-active",
            )
        try:
            return operation(controller, team_id, *args, **kwargs)
        finally:
            lock.release()

    return guarded


class LocalController(
    LocalChatExecutionMixin,
    LocalChatPauseMixin,
    LocalChatPrivateMixin,
    LocalChatSegmentMixin,
    LocalChatStateMixin,
):
    def __init__(
        self,
        client: docker.DockerClient,
        space_id: str,
        registry: dict[str, AssistantSpec],
        storage: team_storage.TeamStorage,
        inference_store: inference_config.InferenceConfigStore | None = None,
        brain_runtime: brain_runtime_client.BrainRuntimeClient | None = None,
        power_state: power_journal.PowerJournal | None = None,
        assistant_secrets: assistant_secret_store.AssistantSecretStore | None = None,
        secret_challenges: assistant_secret_challenges.SecretChallengeStore | None = None,
        assistant_accounts: oauth_account_store.OAuthAccountStore | None = None,
        account_challenges: assistant_account_challenges.AccountChallengeStore | None = None,
        oauth_pkce: oauth_pkce_challenges.OAuthPKCEChallengeStore | None = None,
        oauth_broker: oauth_broker_client.OAuthBrokerClient | None = None,
        oauth_service: oauth_account_service.BrokeredOAuthAccountService | None = None,
        approval_challenges: assistant_approval_challenges.ApprovalChallengeStore | None = None,
        approval_grants: assistant_approval_grants.ApprovalGrantStore | None = None,
        input_challenges: assistant_input_challenges.InputChallengeStore | None = None,
        chat_continuations: local_chat_continuation_store.EncryptedContinuationStore | None = None,
    ) -> None:
        self.client = client
        self.space_id = validate_space_id(space_id)
        self.registry = registry
        self.storage = storage
        self.inference_store = inference_store or inference_config.InferenceConfigStore(INFERENCE_ROOT)
        self.brain_runtime = brain_runtime or brain_runtime_client.BrainRuntimeClient()
        self.power_state = (
            power_state if power_state is not None else power_journal.PowerJournal(LOCAL_POWER_JOURNAL_PATH)
        )
        self.assistant_secrets = assistant_secrets or assistant_secret_store.AssistantSecretStore()
        self.secret_challenges = secret_challenges or assistant_secret_challenges.SecretChallengeStore()
        self.assistant_accounts = assistant_accounts or oauth_account_store.OAuthAccountStore()
        self.account_challenges = account_challenges or assistant_account_challenges.AccountChallengeStore()
        self.oauth_pkce = oauth_pkce or oauth_pkce_challenges.OAuthPKCEChallengeStore()
        self.oauth_broker = oauth_broker or oauth_broker_client.OAuthBrokerClient(
            transport=oauth_broker_client.FixedBrokerTransport(
                proxy_host=os.environ.get("SHIMPZ_OAUTH_BROKER_PROXY_HOST"),
                proxy_token=os.environ.get("SHIMPZ_OAUTH_BROKER_PROXY_TOKEN"),
            ),
            callback_mode=os.environ.get("SHIMPZ_OAUTH_CALLBACK_MODE", "loopback"),
        )
        self.oauth_service = oauth_service or oauth_account_service.BrokeredOAuthAccountService(
            challenge=self.oauth_pkce,
            store=self.assistant_accounts,
            broker=self.oauth_broker,
        )
        self.approval_challenges = approval_challenges or assistant_approval_challenges.ApprovalChallengeStore()
        self.approval_grants = approval_grants or assistant_approval_grants.ApprovalGrantStore(
            LOCAL_APPROVAL_GRANTS_PATH
        )
        self.input_challenges = input_challenges or assistant_input_challenges.InputChallengeStore()
        self.chat_continuations = chat_continuations or local_chat_continuation_store.EncryptedContinuationStore(
            LOCAL_CHAT_CONTINUATIONS_STATE_PATH,
            LOCAL_CHAT_CONTINUATIONS_KEY_PATH,
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

    def _base_labels(self, team_id: str, kind: str) -> dict[str, str]:
        return {
            MANAGED_LABEL: "1",
            PROFILE_LABEL: PROFILE,
            SPACE_LABEL: self.space_id,
            KIND_LABEL: kind,
            TEAM_LABEL: team_id,
        }

    def _network_name(self, team_id: str) -> str:
        return f"shimpz-local-{_space_prefix(self.space_id)}-team-{team_id}"

    def _container_name(self, team_id: str, assistant_id: str) -> str:
        return f"shimpz-local-{_space_prefix(self.space_id)}-{team_id}-assistant-{assistant_id}"

    def _egress_policy_identity(self, team_id: str, assistant_id: str) -> str:
        return f"{self.space_id}\0{team_id}\0{assistant_id}"

    def _egress_token(self, team_id: str, assistant_id: str, *, create: bool) -> str | None:
        try:
            return _egress_store().token(
                self._egress_policy_identity(team_id, assistant_id),
                create=create,
            )
        except egress_policy.EgressPolicyError as exc:
            _raise_egress_problem(exc)

    @staticmethod
    def _proxy_environment(token: str) -> dict[str, str]:
        try:
            return _egress_store().proxy_environment(token)
        except egress_policy.EgressPolicyError as exc:
            _raise_egress_problem(exc)

    def _write_egress_policy(
        self,
        team_id: str,
        spec: AssistantSpec,
        allowed_hosts: tuple[str, ...],
    ) -> dict[str, str]:
        try:
            store = _egress_store()
            token = store.token(
                self._egress_policy_identity(team_id, spec.assistant_id),
                create=True,
            )
            if token is None:
                raise egress_policy.EgressPolicyUnavailableError("egress token was not created")
            store.write(token, allowed_hosts)
            return store.proxy_environment(token)
        except egress_policy.EgressPolicyError as exc:
            _raise_egress_problem(exc)

    def _validate_egress_policy(
        self,
        team_id: str,
        spec: AssistantSpec,
        allowed_hosts: tuple[str, ...],
    ) -> dict[str, str]:
        try:
            store = _egress_store()
            token = store.validate_admitted(
                self._read_admitted_egress_policy(team_id, spec.assistant_id),
                allowed_hosts,
            )
            return store.proxy_environment(token)
        except egress_policy.EgressPolicyError as exc:
            _raise_egress_problem(exc)

    def _read_admitted_egress_policy(
        self,
        team_id: str,
        assistant_id: str,
    ) -> tuple[str, tuple[str, ...]] | None:
        """Read only a canonical policy previously admitted and owned by this controller."""
        try:
            return _egress_store().admitted(
                self._egress_policy_identity(team_id, assistant_id),
            )
        except egress_policy.EgressPolicyError as exc:
            _raise_egress_problem(exc)

    def _remove_egress_policy(self, team_id: str, assistant_id: str) -> None:
        try:
            _egress_store().remove(
                self._egress_policy_identity(team_id, assistant_id),
            )
        except egress_policy.EgressPolicyError as exc:
            _raise_egress_problem(exc)

    def _egress_proxy(self):
        if not APP_EGRESS_PROXY_CONTAINER or _CONTAINER_NAME.fullmatch(APP_EGRESS_PROXY_CONTAINER) is None:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Assistant egress proxy is unavailable",
                code="egress-proxy-unavailable",
            )
        try:
            proxy = self.client.containers.get(APP_EGRESS_PROXY_CONTAINER)
            proxy.reload()
        except (NotFound, DockerException) as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Assistant egress proxy is unavailable",
                code="egress-proxy-unavailable",
            ) from exc
        attrs = proxy.attrs
        config = attrs.get("Config") or {}
        host = attrs.get("HostConfig") or {}
        labels = config.get("Labels") or {}
        expected_labels = {
            MANAGED_LABEL: "1",
            PROFILE_LABEL: PROFILE,
            SPACE_LABEL: self.space_id,
            KIND_LABEL: APP_EGRESS_PROXY_KIND,
        }
        security_options = host.get("SecurityOpt") or []
        mounts = attrs.get("Mounts") or []
        policy_mounts = [mount for mount in mounts if mount.get("Destination") == "/policy"]
        if (
            proxy.name != APP_EGRESS_PROXY_CONTAINER
            or proxy.status != "running"
            or not self._labels_include(labels, expected_labels)
            or config.get("User") not in {"10005", "10005:10005"}
            or host.get("ReadonlyRootfs") is not True
            or "ALL" not in (host.get("CapDrop") or [])
            or not any(str(option).startswith("no-new-privileges") for option in security_options)
            or any("seccomp=unconfined" in str(option) for option in security_options)
            or host.get("Privileged") is not False
            or host.get("PortBindings") not in (None, {})
            or len(policy_mounts) != 1
            or policy_mounts[0].get("RW") is not False
        ):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant egress proxy failed its isolation profile",
                code="egress-proxy-drift",
            )
        return proxy

    def _connect_egress_proxy(self, network) -> None:
        proxy = self._egress_proxy()
        attached = ((proxy.attrs.get("NetworkSettings") or {}).get("Networks") or {}).get(network.name)
        if attached is None:
            try:
                network.connect(proxy, aliases=[APP_EGRESS_PROXY_ALIAS])
                proxy.reload()
            except DockerException as exc:
                raise ApiProblem(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Assistant egress proxy could not join the Team",
                    code="egress-proxy-unavailable",
                ) from exc
            attached = ((proxy.attrs.get("NetworkSettings") or {}).get("Networks") or {}).get(network.name)
        if not isinstance(attached, dict) or APP_EGRESS_PROXY_ALIAS not in (attached.get("Aliases") or []):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant egress proxy failed its Team attachment contract",
                code="egress-proxy-drift",
            )

    def _validate_egress_proxy_attachment(self, network_name: str) -> None:
        proxy = self._egress_proxy()
        attached = ((proxy.attrs.get("NetworkSettings") or {}).get("Networks") or {}).get(network_name)
        if not isinstance(attached, dict) or APP_EGRESS_PROXY_ALIAS not in (attached.get("Aliases") or []):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant egress proxy failed its Team attachment contract",
                code="egress-proxy-drift",
            )

    def _disconnect_egress_proxy(self, network) -> None:
        proxy = self._egress_proxy()
        attached = ((proxy.attrs.get("NetworkSettings") or {}).get("Networks") or {}).get(network.name)
        if attached is None:
            return
        try:
            network.disconnect(proxy)
            proxy.reload()
        except DockerException as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Assistant egress proxy could not leave the Team",
                code="egress-proxy-unavailable",
            ) from exc
        if network.name in ((proxy.attrs.get("NetworkSettings") or {}).get("Networks") or {}):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant egress proxy failed its Team attachment contract",
                code="egress-proxy-drift",
            )

    def _disconnect_egress_proxy_if_attached(self, network) -> None:
        try:
            network.reload()
        except DockerException as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Team network could not be inspected",
                code="docker-unavailable",
            ) from exc
        endpoints = network.attrs.get("Containers") or {}
        if not isinstance(endpoints, dict):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Team resource ownership conflict",
                code="ownership-conflict",
            )
        if any(endpoint.get("Name") == APP_EGRESS_PROXY_CONTAINER for endpoint in endpoints.values()):
            self._disconnect_egress_proxy(network)

    def _team_has_egress_assistant(self, team_id: str, *, excluding: str | None = None) -> bool:
        try:
            containers = self.client.containers.list(**self._assistant_filters(team_id))
        except DockerException as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Docker is unavailable",
                code="docker-unavailable",
            ) from exc
        for container in containers:
            assistant_id = (container.labels or {}).get(ASSISTANT_LABEL)
            if assistant_id == excluding:
                continue
            spec = self.registry.get(assistant_id)
            if spec is None:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "an installed Assistant is no longer allowlisted",
                    code="assistant-registry-drift",
                )
            self._validate_container_security(
                container,
                team_id,
                spec,
                self._network_name(team_id),
            )
            if spec.allowed_hosts:
                return True
        return False

    def _release_assistant_egress(
        self,
        team_id: str,
        assistant_id: str,
        network,
        *,
        remaining_egress: bool | None = None,
    ) -> None:
        self._remove_egress_policy(team_id, assistant_id)
        if remaining_egress is None:
            remaining_egress = self._team_has_egress_assistant(team_id)
        if not remaining_egress:
            self._disconnect_egress_proxy(network)

    def _remove_assistant_policy_if_needed(
        self,
        team_id: str,
        assistant_id: str,
        spec: AssistantSpec,
    ) -> None:
        if spec.allowed_hosts:
            self._remove_egress_policy(team_id, assistant_id)

    def _activate_assistant_egress(
        self,
        team_id: str,
        spec: AssistantSpec,
        network,
        allowed_hosts: tuple[str, ...],
    ) -> dict[str, str]:
        if not allowed_hosts:
            return {}
        environment = self._write_egress_policy(team_id, spec, allowed_hosts)
        try:
            self._connect_egress_proxy(network)
        except ApiProblem:
            self._remove_egress_policy(team_id, spec.assistant_id)
            raise
        return environment

    @staticmethod
    def _labels_include(actual: object, expected: dict[str, str]) -> bool:
        return isinstance(actual, dict) and all(actual.get(key) == value for key, value in expected.items())

    def _validate_network(self, network, team_id: str) -> str:
        network.reload()
        attrs = network.attrs
        expected = self._base_labels(team_id, "team")
        labels = attrs.get("Labels") or {}
        if (
            not self._labels_include(labels, expected)
            or attrs.get("Name") != self._network_name(team_id)
            or attrs.get("Driver") != "bridge"
            or attrs.get("Internal") is not True
            or attrs.get("Attachable") is not False
        ):
            raise ApiProblem(HTTPStatus.CONFLICT, "Team resource ownership conflict", code="ownership-conflict")
        try:
            return validate_team_name(labels.get(TEAM_NAME_LABEL))
        except ApiProblem as exc:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Team resource ownership conflict",
                code="ownership-conflict",
            ) from exc

    def _network(self, team_id: str, *, required: bool = True):
        try:
            network = self.client.networks.get(self._network_name(team_id))
        except NotFound:
            if required:
                raise ApiProblem(HTTPStatus.NOT_FOUND, "Team not found", code="team-not-found") from None
            return None
        self._validate_network(network, team_id)
        return network

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

    def _assistant_filters(self, team_id: str) -> dict[str, list[str] | bool]:
        return {
            "all": True,
            "filters": {
                "label": [
                    f"{MANAGED_LABEL}=1",
                    f"{PROFILE_LABEL}={PROFILE}",
                    f"{SPACE_LABEL}={self.space_id}",
                    f"{KIND_LABEL}=assistant",
                    f"{TEAM_LABEL}={team_id}",
                ]
            },
        }

    def _assistant_container(self, team_id: str, assistant_id: str, *, required: bool = True):
        name = self._container_name(team_id, assistant_id)
        try:
            container = self.client.containers.get(name)
        except NotFound:
            if required:
                raise ApiProblem(
                    HTTPStatus.NOT_FOUND,
                    "Assistant is not installed",
                    code="assistant-not-found",
                ) from None
            return None
        return container

    def _resolve(self, assistant_id: str) -> AssistantSpec:
        spec = self.registry.get(assistant_id)
        if spec is None:
            # Resolution is intentionally completed before any image lookup/pull.
            raise ApiProblem(HTTPStatus.NOT_FOUND, "Assistant is not allowlisted", code="assistant-not-allowlisted")
        return spec

    @staticmethod
    def _image_labels_valid(image, spec: AssistantSpec) -> bool:
        labels = (image.attrs.get("Config") or {}).get("Labels") or {}
        return (
            labels.get("org.shimpz.assistant.id") == spec.assistant_id and labels.get("org.shimpz.assistant.api") == "1"
        )

    def _trusted_image(self, spec: AssistantSpec):
        try:
            image = self.client.images.get(spec.image)
        except ImageNotFound:
            try:
                image = self.client.images.pull(spec.image)
            except DockerException as exc:
                raise ApiProblem(
                    HTTPStatus.BAD_GATEWAY,
                    "the trusted Assistant image could not be pulled",
                    code="image-pull-failed",
                ) from exc
        except DockerException as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Docker is unavailable",
                code="docker-unavailable",
            ) from exc
        image.reload()
        repo_digests = image.attrs.get("RepoDigests") or []
        if spec.image not in repo_digests or not self._image_labels_valid(image, spec):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "the Assistant image does not match its trusted contract",
                code="image-contract-mismatch",
            )
        return image

    def _assistant_labels(self, team_id: str, spec: AssistantSpec) -> dict[str, str]:
        labels = self._base_labels(team_id, "assistant")
        labels.update({ASSISTANT_LABEL: spec.assistant_id, IMAGE_LABEL: spec.image})
        return labels

    def _validate_container_profile(
        self,
        container,
        team_id: str,
        spec: AssistantSpec,
        network_name: str,
    ) -> tuple[dict, dict[str, str]]:
        container.reload()
        expected_labels = self._assistant_labels(team_id, spec)
        expected_labels.pop(IMAGE_LABEL)
        admitted = local_container_policy.inspect_profile(
            container.attrs,
            container.name,
            expected_labels,
            self._container_name(team_id, spec.assistant_id),
            spec.image,
            network_name,
            self.cpuset_cpus,
        )
        if admitted is None:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "the installed Assistant failed its isolation profile",
                code="assistant-isolation-drift",
            )
        return admitted

    def _validate_container_egress(
        self,
        team_id: str,
        spec: AssistantSpec,
        network_name: str,
        environment: dict[str, str],
    ) -> tuple[str, ...]:
        try:
            reviewed_hosts = assistant_manifest.canonical_allowed_hosts(spec.allowed_hosts)
        except assistant_manifest.ManifestError as exc:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "the reviewed Assistant allowed_hosts contract is invalid",
                code="assistant-registry-drift",
            ) from exc
        expected_proxy_environment = None
        if reviewed_hosts:
            expected_proxy_environment = self._validate_egress_policy(team_id, spec, reviewed_hosts)
            self._validate_egress_proxy_attachment(network_name)
        if not local_container_policy.egress_environment_valid(environment, expected_proxy_environment):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "the installed Assistant failed its isolation profile",
                code="assistant-isolation-drift",
            )
        return reviewed_hosts

    def _validate_container_isolation(
        self,
        container,
        team_id: str,
        spec: AssistantSpec,
        network_name: str,
    ) -> dict:
        config, environment = self._validate_container_profile(container, team_id, spec, network_name)
        self._validate_container_egress(
            team_id,
            spec,
            network_name,
            environment,
        )
        return config

    def _validate_container_security(
        self,
        container,
        team_id: str,
        spec: AssistantSpec,
        network_name: str,
    ) -> dict:
        config = self._validate_container_isolation(container, team_id, spec, network_name)
        self._admit_assistant_allowed_hosts(container, spec)
        return config

    @staticmethod
    def _has_current_assistant_artifact(config: dict, spec: AssistantSpec) -> bool:
        labels = config.get("Labels") or {}
        return config.get("Image") == spec.image and labels.get(IMAGE_LABEL) == spec.image

    def _validate_current_assistant_artifact(self, config: dict, spec: AssistantSpec) -> None:
        if not self._has_current_assistant_artifact(config, spec):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "the installed Assistant must be updated",
                code="assistant-update-required",
            )

    def _validate_container(self, container, team_id: str, spec: AssistantSpec, network_name: str) -> None:
        config = self._validate_container_security(container, team_id, spec, network_name)
        self._validate_current_assistant_artifact(config, spec)

    @staticmethod
    def _close_exec_stream(stream) -> None:
        power_execution.close_exec_stream(stream)

    def _fail_stop_power(self, container) -> None:
        """Stop, then kill if needed, and prove an ambiguous local Power cannot keep running."""
        try:
            container.stop(timeout=3)
        except NotFound:
            return
        except DockerException:
            pass
        if self._power_not_running(container):
            return
        try:
            container.kill()
        except NotFound:
            return
        except DockerException:
            pass
        if self._power_not_running(container):
            return
        self._blocked_power_workloads.add(container.id)
        raise ApiProblem(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Assistant Power termination could not be proved; reinstall the Assistant",
            code="assistant-power-blocked",
        )

    @staticmethod
    def _power_not_running(container) -> bool:
        try:
            container.reload()
        except NotFound:
            return True
        except DockerException:
            return False
        state = container.attrs.get("State")
        return isinstance(state, dict) and state.get("Running") is False

    def _read_rpc_frames(self, raw_socket: socket.socket, deadline: float) -> tuple[bytes, bytes]:
        return power_execution.read_rpc_frames(raw_socket, deadline, MAX_RESPONSE_BYTES)

    def _rpc(
        self,
        container,
        spec: AssistantSpec,
        method: str,
        path: str,
        payload: dict,
        *,
        detect_unsupported_path: bool = False,
    ) -> object:
        try:
            encoded = assistant_secret_flow.encode_private_rpc_envelope(payload)
        except assistant_secret_flow.SecretFlowError as exc:
            raise ApiProblem(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "request is too large",
                code="body-too-large",
            ) from exc

        def close_stream(stream: object) -> None:
            with suppress(Exception):
                self._close_exec_stream(stream)

        try:
            return power_execution.rpc_exchange(
                container.id,
                [spec.rpc_command, method, path],
                encoded,
                power_execution.RpcExchangeStrategy(
                    api=self.client.api,
                    user=ASSISTANT_UID,
                    workdir=ASSISTANT_WORKDIR,
                    timeout=RPC_TIMEOUT_SECONDS,
                    maximum=MAX_RESPONSE_BYTES,
                    transport_errors=(DockerException,),
                    fail_stop=lambda: self._fail_stop_power(container),
                    cancelled=lambda _exc: None,
                    close_stream=close_stream,
                ),
                detect_unsupported_path=detect_unsupported_path,
            )
        except power_execution.RpcExchangeError as exc:
            if exc.kind == "unsupported-path":
                raise _UnsupportedAssistantRpcPathError(path) from None
            message, code = {
                "timeout": ("Assistant Power timed out", "assistant-timeout"),
                "ambiguous": ("Assistant Power status is ambiguous", "assistant-rpc-failed"),
                "failed": ("Assistant Power failed", "assistant-rpc-failed"),
                "invalid-result": ("Assistant Power failed", "assistant-rpc-failed"),
            }.get(exc.kind, (None, None))
            status = power_execution.rpc_failure_status(exc.kind)
            raise ApiProblem(status, message, code=code) from exc

    def _wait_ready(self, container, spec: AssistantSpec) -> None:
        deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            container.reload()
            if container.status not in {"created", "running"}:
                break
            if container.status == "running":
                try:
                    result = self._rpc(container, spec, "GET", spec.health_path, {})
                except ApiProblem:
                    pass
                else:
                    if result == {"status": "ok"}:
                        return
            time.sleep(0.2)
        raise ApiProblem(HTTPStatus.BAD_GATEWAY, "Assistant did not become ready", code="assistant-not-ready")

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

    def _rollback_assistant_install(
        self,
        team_id: str,
        spec: AssistantSpec,
        network,
        container,
        *,
        egress_prepared: bool,
    ) -> ApiProblem | None:
        incomplete = False
        if container is not None:
            self._assistant_genesis_cache.discard(container.id)
            self._assistant_allowed_hosts_cache.discard(container.id)
            self._assistant_machine_contract_cache.discard(container.id)
            try:
                container.remove(force=True)
            except NotFound:
                pass
            except DockerException:
                incomplete = True
                with suppress(ApiProblem):
                    self._fail_stop_power(container)
        if egress_prepared:
            try:
                self._release_assistant_egress(team_id, spec.assistant_id, network)
            except ApiProblem:
                incomplete = True
        if incomplete:
            return ApiProblem(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Assistant install rollback is incomplete",
                code="assistant-install-rollback-incomplete",
            )
        return None

    def _create_assistant_container(self, team_id: str, spec: AssistantSpec, network, image) -> None:
        container = None
        egress_prepared = False
        try:
            proxy_environment: dict[str, str] = {}
            if spec.allowed_hosts:
                token = self._egress_token(team_id, spec.assistant_id, create=True)
                if token is None:
                    raise ApiProblem(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "Assistant egress token could not be saved",
                        code="egress-policy-unavailable",
                    )
                proxy_environment = self._proxy_environment(token)
                egress_prepared = True
            container = self.client.containers.create(
                image=spec.image,
                name=self._container_name(team_id, spec.assistant_id),
                command=None,
                detach=True,
                user=ASSISTANT_UID,
                network=network.name,
                labels=self._assistant_labels(team_id, spec),
                environment={
                    "SHIMPZ_ASSISTANT_ID": spec.assistant_id,
                    "SHIMPZ_TEAM_ID": team_id,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    **proxy_environment,
                },
                read_only=True,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                privileged=False,
                ipc_mode="private",
                cgroupns="private",
                mem_limit=ASSISTANT_MEMORY,
                memswap_limit=ASSISTANT_MEMORY,
                nano_cpus=ASSISTANT_NANO_CPUS,
                cpuset_cpus=self.cpuset_cpus,
                pids_limit=ASSISTANT_PIDS,
                ulimits=[Ulimit(name="nofile", soft=1024, hard=1024)],
                restart_policy={"Name": "no"},
                log_config=LogConfig(type=LogConfig.types.JSON, config={"max-size": "1m", "max-file": "2"}),
            )
            container.reload()
            if container.attrs.get("Image") != image.id:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "Docker resolved an unexpected Assistant image",
                    code="image-resolution-mismatch",
                )
            allowed_hosts = self._admit_assistant_allowed_hosts(container, spec)
            if allowed_hosts:
                self._activate_assistant_egress(team_id, spec, network, allowed_hosts)
            container.start()
            self._validate_container(container, team_id, spec, network.name)
            self._wait_ready(container, spec)
            self._active_assistant_genesis(_ActiveAssistant(spec, container.id, container))
        except ApiProblem as exc:
            cleanup_error = self._rollback_assistant_install(
                team_id,
                spec,
                network,
                container,
                egress_prepared=egress_prepared,
            )
            if cleanup_error is not None:
                raise cleanup_error from exc
            raise
        except DockerException as exc:
            cleanup_error = self._rollback_assistant_install(
                team_id,
                spec,
                network,
                container,
                egress_prepared=egress_prepared,
            )
            if cleanup_error is not None:
                raise cleanup_error from exc
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Docker could not install the Assistant",
                code="docker-install-failed",
            ) from exc

    def _replace_unready_assistant(self, team_id: str, spec: AssistantSpec, network, existing) -> None:
        # The reference Assistant is the only explicitly stateless recovery target. Resolve its trusted image before
        # removing anything, then revalidate ownership to close the pull/remove race.
        image = self._trusted_image(spec)
        self._validate_container(existing, team_id, spec, network.name)
        try:
            self._assistant_genesis_cache.discard(existing.id)
            self._assistant_allowed_hosts_cache.discard(existing.id)
            self._assistant_machine_contract_cache.discard(existing.id)
            existing.remove(force=True)
        except DockerException as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Docker could not replace the Assistant",
                code="docker-remove-failed",
            ) from exc
        self._create_assistant_container(team_id, spec, network, image)

    def _replace_outdated_assistant(self, team_id: str, spec: AssistantSpec, network, existing) -> None:
        image = self._trusted_image(spec)
        config = self._validate_container_security(existing, team_id, spec, network.name)
        if self._has_current_assistant_artifact(config, spec):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "the installed Assistant changed during update",
                code="assistant-update-conflict",
            )
        self._retain_declared_assistant_account_state(team_id, spec)
        remaining_egress = (
            self._team_has_egress_assistant(team_id, excluding=spec.assistant_id) if spec.allowed_hosts else None
        )
        try:
            self._assistant_genesis_cache.discard(existing.id)
            self._assistant_allowed_hosts_cache.discard(existing.id)
            self._assistant_machine_contract_cache.discard(existing.id)
            existing.remove(force=True)
        except DockerException as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Docker could not replace the Assistant",
                code="docker-remove-failed",
            ) from exc
        self._revoke_assistant_approval_grants(team_id, spec.assistant_id)
        if spec.allowed_hosts:
            self._release_assistant_egress(
                team_id,
                spec.assistant_id,
                network,
                remaining_egress=remaining_egress,
            )
        self._create_assistant_container(team_id, spec, network, image)

    @_serialize_against_local_team_chat
    def install_assistant(self, team_id: str, assistant_id: str) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        spec = self._resolve(assistant_id)
        with self._lock(team_id):
            network = self._network(team_id)
            existing = self._assistant_container(team_id, assistant_id, required=False)
            if existing is not None:
                config = self._validate_container_security(existing, team_id, spec, network.name)
                if not self._has_current_assistant_artifact(config, spec):
                    self._replace_outdated_assistant(team_id, spec, network, existing)
                    return {"assistant": assistant_id, "installed": False}
                self._validate_container_security(existing, team_id, spec, network.name)
                existing.reload()
                if existing.status != "running":
                    try:
                        existing.start()
                    except DockerException as exc:
                        raise ApiProblem(
                            HTTPStatus.SERVICE_UNAVAILABLE,
                            "Docker could not start the Assistant",
                            code="docker-start-failed",
                        ) from exc
                try:
                    self._wait_ready(existing, spec)
                except ApiProblem as exc:
                    if not _is_replaceable_readiness_failure(assistant_id, exc):
                        raise
                    self._replace_unready_assistant(team_id, spec, network, existing)
                else:
                    self._active_assistant_genesis(_ActiveAssistant(spec, existing.id, existing))
                return {"assistant": assistant_id, "installed": False}

            image = self._trusted_image(spec)
            self._revoke_assistant_approval_grants(team_id, spec.assistant_id)
            self._create_assistant_container(team_id, spec, network, image)
            return {"assistant": assistant_id, "installed": True}

    @_serialize_against_local_team_chat
    def uninstall_assistant(self, team_id: str, assistant_id: str) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        spec = self._resolve(assistant_id)
        self.secret_challenges.cancel_team(team_id)
        self.approval_challenges.cancel_team(team_id)
        self.input_challenges.cancel_team(team_id)
        self._delete_chat_continuation(team_id)
        with self._lock(team_id):
            network = self._network(team_id)
            self._revoke_assistant_approval_grants(team_id, assistant_id)
            container = self._assistant_container(team_id, assistant_id, required=False)
            if container is None:
                if self._egress_token(team_id, assistant_id, create=False) is not None:
                    remaining_egress = self._team_has_egress_assistant(team_id, excluding=assistant_id)
                    self._release_assistant_egress(
                        team_id,
                        assistant_id,
                        network,
                        remaining_egress=remaining_egress,
                    )
                self._delete_assistant_secret_state(team_id, assistant_id)
                self._delete_assistant_account_state(team_id, assistant_id)
                return {"assistant": assistant_id, "uninstalled": False}
            self._validate_container_security(container, team_id, spec, network.name)
            remaining_egress = (
                self._team_has_egress_assistant(team_id, excluding=assistant_id) if spec.allowed_hosts else None
            )
            try:
                container.remove(force=True)
            except DockerException as exc:
                raise ApiProblem(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Docker could not uninstall the Assistant",
                    code="docker-remove-failed",
                ) from exc
            self._blocked_power_workloads.discard(container.id)
            self._assistant_genesis_cache.discard(container.id)
            self._assistant_allowed_hosts_cache.discard(container.id)
            self._assistant_machine_contract_cache.discard(container.id)
            if spec.allowed_hosts:
                self._release_assistant_egress(
                    team_id,
                    assistant_id,
                    network,
                    remaining_egress=remaining_egress,
                )
            self._delete_assistant_secret_state(team_id, assistant_id)
            self._delete_assistant_account_state(team_id, assistant_id)
            return {"assistant": assistant_id, "uninstalled": True}

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

    def _purge_power_generation(self, generation: str) -> None:
        try:
            self.power_state.purge(generation)
        except power_journal.PowerJournalError as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Team Power execution state could not be deleted",
                code="power-state-unavailable",
            ) from exc

    def destroy_team(self, team_id: str) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        self.secret_challenges.cancel_team(team_id)
        self.approval_challenges.cancel_team(team_id)
        self.input_challenges.cancel_team(team_id)
        self._delete_chat_continuation(team_id)
        self._cancel_chat_for_destroy(team_id)

        chat_lock = self._chat_lock(team_id)
        if not chat_lock.acquire(timeout=30):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "active Team chat did not stop in time",
                code="chat-active",
            )
        try:
            with self._lock(team_id):
                self._revoke_team_approval_grants(team_id)
                network = self._network(team_id, required=False)
                try:
                    containers = self.client.containers.list(**self._assistant_filters(team_id))
                except DockerException as exc:
                    raise ApiProblem(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "Docker is unavailable",
                        code="docker-unavailable",
                    ) from exc

                for container in containers:
                    assistant_id = container.labels.get(ASSISTANT_LABEL)
                    spec = self.registry.get(assistant_id)
                    if spec is None or network is None:
                        raise ApiProblem(
                            HTTPStatus.CONFLICT,
                            "Team resources failed their ownership contract",
                            code="ownership-conflict",
                        )
                    self._validate_container_security(
                        container,
                        team_id,
                        spec,
                        network.name,
                    )

                if network is not None:
                    thread_id = _brain_thread_id(self.space_id, team_id, network.id)
                    try:
                        self.brain_runtime.delete_thread(thread_id)
                    except brain_runtime_client.BrainRuntimeError as exc:
                        raise ApiProblem(
                            HTTPStatus.SERVICE_UNAVAILABLE,
                            "Team conversation state could not be deleted",
                            code="brain-runtime-failed",
                        ) from exc
                    self._purge_power_generation(network.id)

                removed = 0
                for container in containers:
                    assistant_id = container.labels[ASSISTANT_LABEL]
                    spec = self.registry[assistant_id]
                    try:
                        container.remove(force=True)
                    except DockerException as exc:
                        raise ApiProblem(
                            HTTPStatus.SERVICE_UNAVAILABLE,
                            "Docker could not destroy the Team",
                            code="docker-remove-failed",
                        ) from exc
                    self._blocked_power_workloads.discard(container.id)
                    self._remove_assistant_policy_if_needed(team_id, assistant_id, spec)
                    removed += 1

                if network is None:
                    try:
                        storage_removed = self.storage.destroy(team_id)
                    except team_storage.StorageError as exc:
                        self._raise_storage_problem(exc)
                    try:
                        self.inference_store.delete(team_id)
                    except inference_config.InferenceConfigError as exc:
                        self._raise_inference_problem(exc)
                    self._delete_team_secret_state(team_id)
                    self._delete_team_account_state(team_id)
                    return {
                        "team_id": team_id,
                        "destroyed": False,
                        "assistants_removed": removed,
                        "storage_removed": storage_removed,
                    }
                self._disconnect_egress_proxy_if_attached(network)
                try:
                    storage_removed = self.storage.destroy(team_id)
                except team_storage.StorageError as exc:
                    self._raise_storage_problem(exc)
                try:
                    self.inference_store.delete(team_id)
                except inference_config.InferenceConfigError as exc:
                    self._raise_inference_problem(exc)
                try:
                    network.remove()
                except DockerException as exc:
                    raise ApiProblem(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "Docker could not destroy the Team",
                        code="docker-remove-failed",
                    ) from exc
                self._delete_team_secret_state(team_id)
                self._delete_team_account_state(team_id)
                return {
                    "team_id": team_id,
                    "destroyed": True,
                    "assistants_removed": removed,
                    "storage_removed": storage_removed,
                }
        finally:
            chat_lock.release()

    def _validate_reset_container(self, container) -> None:
        container.reload()
        labels = container.attrs.get("Config", {}).get("Labels") or {}
        team_id = labels.get(TEAM_LABEL)
        assistant_id = labels.get(ASSISTANT_LABEL)
        if (
            not isinstance(team_id, str)
            or len(team_id) > MAX_TEAM_ID_LENGTH
            or _TEAM_ID.fullmatch(team_id) is None
            or not isinstance(assistant_id, str)
            or len(assistant_id) > MAX_ASSISTANT_ID_LENGTH
            or _ASSISTANT_ID.fullmatch(assistant_id) is None
            or not isinstance(labels.get(IMAGE_LABEL), str)
            or not self._labels_include(labels, self._base_labels(team_id, "assistant"))
            or container.name != self._container_name(team_id, assistant_id)
        ):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "a labeled Space resource failed its ownership contract",
                code="ownership-conflict",
            )

    def reset_space(self) -> dict[str, object]:
        """Remove every exactly owned workload/network without accepting resource ids."""
        self.secret_challenges.cancel_all()
        self.approval_challenges.cancel_all()
        self.input_challenges.cancel_all()
        self._clear_chat_continuations()
        with ExitStack() as locks:
            for lock in self._locks:
                locks.enter_context(lock)
            assistant_filters = {
                "label": [
                    f"{MANAGED_LABEL}=1",
                    f"{PROFILE_LABEL}={PROFILE}",
                    f"{SPACE_LABEL}={self.space_id}",
                    f"{KIND_LABEL}=assistant",
                ]
            }
            network_filters = {
                "label": [
                    f"{MANAGED_LABEL}=1",
                    f"{PROFILE_LABEL}={PROFILE}",
                    f"{SPACE_LABEL}={self.space_id}",
                    f"{KIND_LABEL}=team",
                ]
            }
            try:
                containers = self.client.containers.list(all=True, filters=assistant_filters)
                networks = self.client.networks.list(filters=network_filters)
                for container in containers:
                    self._validate_reset_container(container)
                owned_assistants = {
                    (
                        container.attrs["Config"]["Labels"][TEAM_LABEL],
                        container.attrs["Config"]["Labels"][ASSISTANT_LABEL],
                    )
                    for container in containers
                }
                owned_team_ids: set[str] = set()
                for network in networks:
                    labels = network.attrs.get("Labels") or {}
                    team_id = labels.get(TEAM_LABEL)
                    if not isinstance(team_id, str):
                        raise ApiProblem(
                            HTTPStatus.CONFLICT,
                            "a labeled Space resource failed its ownership contract",
                            code="ownership-conflict",
                        )
                    validate_team_id(team_id)
                    self._validate_network(network, team_id)
                    owned_team_ids.add(team_id)
                owned_assistants.update(
                    (team_id, assistant_id) for team_id in owned_team_ids for assistant_id in self.registry
                )
                self._delete_all_secret_state()
                self._delete_all_account_state()
                self._revoke_all_approval_grants()
                for container in containers:
                    container.remove(force=True)
                    self._blocked_power_workloads.discard(container.id)
                for team_id, assistant_id in sorted(owned_assistants):
                    self._remove_egress_policy(team_id, assistant_id)
                for network in networks:
                    self._disconnect_egress_proxy_if_attached(network)
                storage_removed = self.storage.destroy_all()
                for network in networks:
                    team_id = network.attrs["Labels"][TEAM_LABEL]
                    self.inference_store.delete(team_id)
                for network in networks:
                    network.remove()
            except ApiProblem:
                raise
            except team_storage.StorageError as exc:
                self._raise_storage_problem(exc)
            except inference_config.InferenceConfigError as exc:
                self._raise_inference_problem(exc)
            except DockerException as exc:
                raise ApiProblem(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Docker could not reset the Space",
                    code="docker-reset-failed",
                ) from exc
            return {
                "reset": True,
                "assistants_removed": len(containers),
                "teams_removed": len(networks),
                "storage_removed": storage_removed,
            }


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
