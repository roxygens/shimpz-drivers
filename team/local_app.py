"""Minimal Docker controller for one locally owned Shimpz Space.

This is intentionally separate from the hosted Team controller.  An empty Team is
one labeled internal network; its only runnable resources are build-allowlisted,
digest-pinned first-party Assistants with a fixed Power contract.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import math
import os
import re
import secrets
import socket
import sys
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NoReturn

import assistant_account_challenges
import assistant_account_flow
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
import local_audit
import local_chat_continuation_store
import local_chat_continuations
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
from docker.errors import APIError, DockerException, ImageNotFound, NotFound
from docker.types import LogConfig, Ulimit
from http_boundary import local
from http_boundary import strict as strict_http
from local_registry import (
    AssistantSpec,
    RegistryError,
    is_digest_ref,
    load_registry,
    validate_power_input,
    validate_power_output,
)

log = logging.getLogger("shimpz-team-driver-local")

LISTEN_PORT = 7077
PROFILE = "single-owner-local-v1"
MANAGED_LABEL = "com.shimpz.local.managed"
PROFILE_LABEL = "com.shimpz.local.profile"
SPACE_LABEL = "com.shimpz.local.space-id"
KIND_LABEL = "com.shimpz.local.kind"
TEAM_LABEL = "com.shimpz.local.team-id"
TEAM_NAME_LABEL = "com.shimpz.local.team-name"
ASSISTANT_LABEL = "com.shimpz.local.assistant-id"
IMAGE_LABEL = "com.shimpz.local.image"

_TEAM_ID = re.compile(r"[a-z0-9_]{1,40}")
_ASSISTANT_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_SPACE_ID = re.compile(r"[a-z0-9][a-z0-9]*(?:-[a-z0-9]+)*")
_DOCKER_ID = re.compile(r"[0-9a-f]{12,64}")
MAX_TEAM_ID_LENGTH = 40
MAX_ASSISTANT_ID_LENGTH = 48
MAX_SPACE_ID_LENGTH = 48
MAX_BODY_BYTES = 16 * 1024
MAX_CHAT_BODY_BYTES = 24 * 1024
MAX_SECRET_BODY_BYTES = 512 * 1024
MAX_RESPONSE_BYTES = assistant_help.MAX_HELP_BYTES * 6 + 1024
MAX_API_RESPONSE_BYTES = 128 * 1024
MAX_EGRESS_POLICY_BYTES = egress_policy.MAX_POLICY_BYTES
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_FILE_BODY_BYTES = 4 * ((MAX_UPLOAD_BYTES + 2) // 3) + 8192
MAX_PATH_BYTES = 512
REQUEST_TIMEOUT_SECONDS = 10
RPC_TIMEOUT_SECONDS = 8
HEALTH_TIMEOUT_SECONDS = 15
MAX_CHAT_MESSAGE_CHARS = 16_000
MAX_CHAT_FILES = 8
MAX_CHAT_ASSISTANTS = 16
CHAT_PAUSED_STATUSES = frozenset({"accounts-required", "secrets-required", "input-required", "approval-required"})
MIN_API_KEY_BYTES = 16
MAX_API_KEY_BYTES = 8 * 1024
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

ASSISTANT_UID = "10001:10001"
ASSISTANT_WORKDIR = str(Path("/") / "tmp")
ASSISTANT_MEMORY = 128 * 1024 * 1024
ASSISTANT_NANO_CPUS = 250_000_000
ASSISTANT_PIDS = 64
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
_FILE_UPLOAD_SLOTS = threading.BoundedSemaphore(1)
_CONTAINER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class ApiProblem(RuntimeError):
    def __init__(self, status: HTTPStatus, message: str, *, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code


class _UnsupportedAssistantRpcPathError(RuntimeError):
    """The fixed Assistant RPC adapter rejected a path it does not implement."""


@dataclass(frozen=True, slots=True)
class _ActiveAssistant:
    spec: AssistantSpec
    container_id: str
    container: object | None = None


_PendingLocalChat = local_chat_continuations.PendingLocalChat


def _required_active_assistant(
    bindings: dict[str, _ActiveAssistant],
    assistant_id: str,
) -> _ActiveAssistant:
    active = bindings.get(assistant_id)
    if active is None:
        raise ApiProblem(
            HTTPStatus.CONFLICT,
            "Brain requested an unavailable Assistant",
            code="assistant-unavailable",
        )
    return active


def _is_replaceable_readiness_failure(assistant_id: str, problem: ApiProblem) -> bool:
    return assistant_id in READINESS_RECOVERY_ASSISTANTS and problem.code == "assistant-not-ready"


def validate_team_id(value: str) -> str:
    if len(value) > MAX_TEAM_ID_LENGTH or _TEAM_ID.fullmatch(value) is None:
        raise ApiProblem(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid Team id", code="invalid-team-id")
    return value


def validate_team_name(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 80
        or value.strip() != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ApiProblem(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            "Team name must contain 1 to 80 trimmed characters",
            code="invalid-team-name",
        )
    return value


def validate_assistant_id(value: object) -> str:
    if not isinstance(value, str) or len(value) > MAX_ASSISTANT_ID_LENGTH or _ASSISTANT_ID.fullmatch(value) is None:
        raise ApiProblem(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            "invalid Assistant id",
            code="invalid-assistant-id",
        )
    return value


def validate_chat_assistant_ids(value: object) -> tuple[str, ...]:
    """Return one explicit, bounded Assistant scope; empty means Brain-only."""
    if not isinstance(value, list) or len(value) > MAX_CHAT_ASSISTANTS:
        raise ApiProblem(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            f"assistant_ids must contain at most {MAX_CHAT_ASSISTANTS} ids",
            code="invalid-assistants",
        )
    try:
        assistant_ids = tuple(validate_assistant_id(item) for item in value)
    except ApiProblem:
        raise ApiProblem(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            "assistant_ids contains an invalid id",
            code="invalid-assistants",
        ) from None
    if len(set(assistant_ids)) != len(assistant_ids):
        raise ApiProblem(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            "assistant_ids must not contain duplicate ids",
            code="invalid-assistants",
        )
    return tuple(sorted(assistant_ids))


def validate_model_credential_headers(
    providers: list[str],
    api_keys: list[str],
) -> tuple[str, str]:
    """Validate the private Admin hand-off without copying a secret into an error."""
    if len(providers) != 1 or providers[0] not in inference_config.PROVIDERS or len(api_keys) != 1:
        raise ApiProblem(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            "one private model credential is required",
            code="invalid-model-credential",
        )
    api_key = api_keys[0]
    if not isinstance(api_key, str) or api_key.strip() != api_key or not api_key.isascii():
        raise ApiProblem(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            "private model credential is invalid",
            code="invalid-model-credential",
        )
    encoded = api_key.encode("ascii")
    if not MIN_API_KEY_BYTES <= len(encoded) <= MAX_API_KEY_BYTES or any(not 33 <= byte <= 126 for byte in encoded):
        raise ApiProblem(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            "private model credential is invalid",
            code="invalid-model-credential",
        )
    return providers[0], api_key


def validate_space_id(value: str) -> str:
    if len(value) > MAX_SPACE_ID_LENGTH or _SPACE_ID.fullmatch(value) is None:
        raise RuntimeError("SHIMPZ_SPACE_ID must be a lowercase, dash-separated identifier")
    return value


def _space_prefix(space_id: str) -> str:
    return hashlib.sha256(space_id.encode("ascii")).hexdigest()[:12]


def _brain_thread_id(space_id: str, team_id: str, network_id: str) -> str:
    """Bind local conversation state to one immutable Team network generation."""
    if (
        not isinstance(space_id, str)
        or len(space_id) > MAX_SPACE_ID_LENGTH
        or _SPACE_ID.fullmatch(space_id) is None
        or not isinstance(team_id, str)
        or len(team_id) > MAX_TEAM_ID_LENGTH
        or _TEAM_ID.fullmatch(team_id) is None
        or not isinstance(network_id, str)
        or _DOCKER_ID.fullmatch(network_id) is None
    ):
        raise ApiProblem(
            HTTPStatus.CONFLICT,
            "Team identity failed its persisted contract",
            code="ownership-conflict",
        )
    return f"local:{space_id}:{team_id}:{network_id}:default"


def half_cpu_set(processors: int) -> str:
    if isinstance(processors, bool) or not isinstance(processors, int) or processors < 1:
        raise RuntimeError("the Docker daemon reported an invalid CPU count")
    available = max(1, processors // 2)
    return "0" if available == 1 else f"0-{available - 1}"


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


_environment_map = egress_policy.environment_map


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


class LocalController:
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

    def _power_secret_generations(
        self,
        team_id: str,
        active: _ActiveAssistant,
        power_id: str,
    ) -> tuple[tuple[str, int], ...]:
        power = active.spec.powers.get(power_id)
        if power is None:
            raise power_journal.PowerJournalConflictError("Power secret contract is unavailable")
        try:
            metadata = self.assistant_secrets.metadata(
                team_id,
                active.spec.assistant_id,
                power.secrets,
            )
        except assistant_secret_store.AssistantSecretError as exc:
            raise power_journal.PowerJournalConflictError("Power secret state is unavailable") from exc
        return power_execution.private_generations(tuple(metadata), connected=False)

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
        power = active.spec.powers.get(power_id)
        if power is None:
            raise power_journal.PowerJournalConflictError("Power account contract is unavailable")
        declarations = {
            account_id: active.spec.accounts[account_id]
            for account_id in power.accounts
            if account_id in active.spec.accounts
        }
        if len(declarations) != len(power.accounts):
            raise power_journal.PowerJournalConflictError("Power account contract is unavailable")
        try:
            metadata = self.assistant_accounts.metadata(
                team_id,
                active.spec.assistant_id,
                declarations,
            )
        except oauth_account_store.OAuthAccountStoreError as exc:
            raise power_journal.PowerJournalConflictError("Power account state is unavailable") from exc
        return power_execution.private_generations(tuple(metadata), connected=True)

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
        secret_values = self._resolve_power_secrets(team_id, active.spec, request.power)
        account_values = self._resolve_power_accounts(team_id, active.spec, request.power)
        try:
            assistant_secret_flow.require_power_rpc_envelope(
                request.input,
                secret_values,
                account_values,
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
            installed = self.list_assistants(team_id)["assistants"]
            specs = [self._resolve(item["assistant"]) for item in installed]
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
            installed = self.list_assistants(team_id)["assistants"]
            specs = [self._resolve(item["assistant"]) for item in installed]
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
            installed = {item["assistant"]: item["status"] for item in self.list_assistants(team_id)["assistants"]}
            spec = self._resolve(assistant_id)
            declaration = spec.accounts.get(account_id)
            if installed.get(assistant_id) != "running" or declaration is None:
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
    ) -> None:
        current = self._chat_setup(team_id, file_ids, provider, assistant_ids)
        if self._chat_identity(*current) != identity:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Team capabilities changed; retry",
                code="team-context-changed",
            )

    def _run_chat_segment(
        self,
        team_id: str,
        file_ids: list[str],
        assistant_ids: tuple[str, ...],
        provider: str,
        api_key: str,
        token: str,
        *,
        message: str | None = None,
        continuation: chat_orchestrator.ChatContinuation | None = None,
        expected_identity: tuple[object, ...] | None = None,
        answer_logs: tuple[tuple[str, tuple[object, ...]], ...] = (),
    ) -> chat_turn_engine.SegmentResult:
        answers_by_interrupt = dict(answer_logs)
        if len(answers_by_interrupt) != len(answer_logs):
            raise ApiProblem(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "invalid chat answer log",
                code="internal-error",
            )
        bindings: dict[str, _ActiveAssistant] = {}
        identity: tuple[object, ...] = ()
        network_id = ""

        def execute_power(request: brain_runtime_client.PowerRequest) -> object:
            active = _required_active_assistant(bindings, request.assistant_id)
            return self._invoke_chat_power(
                team_id,
                token,
                request.assistant_id,
                active.container_id,
                request.power,
                request.input,
                answers_by_interrupt.get(request.interrupt_id, ()),
            )

        def prepare() -> chat_turn_engine.PreparedSegment:
            nonlocal bindings, identity, network_id
            team_name, network_id, assistants, files, config = self._chat_setup(
                team_id,
                file_ids,
                provider,
                assistant_ids,
            )
            identity = self._chat_identity(team_name, network_id, assistants, files, config)
            genesis_by_id = {active.spec.assistant_id: self._active_assistant_genesis(active) for active in assistants}
            context = brain_runtime_client.RuntimeContext(
                thread_id=_brain_thread_id(self.space_id, team_id, network_id),
                team_name=team_name,
                assistants=tuple(
                    brain_runtime_client.RuntimeAssistant(
                        id=active.spec.assistant_id,
                        genesis=genesis_by_id[active.spec.assistant_id],
                        powers=tuple(
                            brain_runtime_client.RuntimePower(
                                id=power_id,
                                summary=power.summary,
                                input_schema=power.input_schema,
                            )
                            for power_id, power in sorted(active.spec.powers.items())
                        ),
                    )
                    for active in assistants
                ),
                provider=config.provider,
                model=config.model,
                api_key=api_key,
            )
            bindings = {active.spec.assistant_id: active for active in assistants}
            batch = power_execution.PowerBatch(
                self.power_state,
                network_id,
                context.thread_id,
                bindings,
                lambda active: (active.container_id, active.spec.image),
                execute_power,
                lambda request: self._require_power_rpc_envelope(
                    team_id,
                    bindings,
                    request,
                    answers_by_interrupt.get(request.interrupt_id, ()),
                ),
                lambda request: self._power_secret_generations(
                    team_id,
                    _required_active_assistant(bindings, request.assistant_id),
                    request.power,
                ),
                lambda request: self._power_account_generations(
                    team_id,
                    _required_active_assistant(bindings, request.assistant_id),
                    request.power,
                ),
            )
            return chat_turn_engine.PreparedSegment(team_name, identity, context, files, batch)

        def private_inputs(
            requests: tuple[object, ...],
            requirements: chat_turn_engine.SegmentRequirements,
        ) -> bool:
            return self._require_chat_private_inputs(team_id, bindings, requests, requirements)

        def validate_current_context() -> None:
            self._validate_chat_context(
                team_id,
                file_ids,
                provider,
                assistant_ids,
                identity,
            )

        current_message = message
        current_continuation = continuation
        current_identity = expected_identity
        while True:
            team_name, identity, outcome, requirements = chat_turn_engine.run_segment(
                chat_turn_engine.SegmentStrategy(
                    runtime=self.brain_runtime,
                    prepare=prepare,
                    validate_power=lambda assistant_id, power, payload: self._validate_chat_power(
                        bindings,
                        assistant_id,
                        power,
                        payload,
                    ),
                    pause_for_private_inputs=private_inputs,
                    cancelled=lambda: self._chat_cancelled(token),
                    validate_context=validate_current_context,
                    raise_problem=self._raise_chat_problem,
                ),
                message=current_message,
                continuation=current_continuation,
                expected_identity=current_identity,
            )
            approval_requirements: tuple[assistant_approval_challenges.ApprovalRequirement, ...] = ()
            if requirements.approvals:
                if len(requirements.approvals) != 1:
                    raise ApiProblem(
                        HTTPStatus.BAD_GATEWAY,
                        "Assistant human approval request is invalid",
                        code="invalid-assistant-approval-request",
                    )
                interaction = requirements.approvals[0]
                answers = answers_by_interrupt.get(interaction.request.interrupt_id, ())
                active = _required_active_assistant(bindings, interaction.request.assistant_id)
                try:
                    requirement = assistant_approval_flow.requirement(
                        interaction,
                        active.spec.name,
                        active.spec.image,
                        len(answers),
                    )
                    granted = requirement.runs == "once" and self.approval_grants.is_granted(
                        team_id,
                        requirement.assistant_id,
                        requirement.power_id,
                        requirement.assistant_image,
                        requirement.ordinal,
                    )
                except assistant_approval_flow.ApprovalFlowError as exc:
                    raise ApiProblem(
                        HTTPStatus.BAD_GATEWAY,
                        "Assistant human approval request is invalid",
                        code="invalid-assistant-approval-request",
                    ) from exc
                except assistant_approval_grants.ApprovalGrantError as exc:
                    self._raise_approval_grant_problem(exc)
                if granted:
                    answers_by_interrupt[requirement.interrupt_id] = (*answers, True)
                    if not isinstance(outcome, chat_orchestrator.ChatSuspension):
                        raise AssertionError("approval requirement did not suspend")
                    current_message = None
                    current_continuation = outcome.continuation
                    current_identity = identity
                    continue
                approval_requirements = (requirement,)
            return chat_turn_engine.SegmentResult(
                team_name,
                identity,
                outcome,
                requirements.accounts,
                requirements.secrets,
                requirements.inputs,
                approval_requirements,
                tuple(sorted(answers_by_interrupt.items())),
            )

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
                team_id,
                file_ids,
                assistant_ids,
                provider,
                api_key,
                token,
                message=message,
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
                team_id,
                list(pending.file_ids),
                pending.assistant_ids,
                provider,
                api_key,
                token,
                continuation=pending.continuation,
                expected_identity=pending.identity,
                answer_logs=pending.answer_logs,
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
                team_id,
                list(pending.file_ids),
                pending.assistant_ids,
                provider,
                api_key,
                token,
                continuation=pending.continuation,
                expected_identity=pending.identity,
                answer_logs=pending.answer_logs,
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
                team_id,
                list(pending.file_ids),
                pending.assistant_ids,
                provider,
                api_key,
                token,
                continuation=pending.continuation,
                expected_identity=pending.identity,
                answer_logs=pending.answer_logs,
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
                team_id,
                list(pending.file_ids),
                pending.assistant_ids,
                provider,
                api_key,
                token,
                continuation=pending.continuation,
                expected_identity=pending.identity,
                answer_logs=pending.answer_logs,
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
        attrs = container.attrs
        config = attrs.get("Config") or {}
        host = attrs.get("HostConfig") or {}
        labels = config.get("Labels") or {}
        installed_image = labels.get(IMAGE_LABEL)
        expected_labels = self._assistant_labels(team_id, spec)
        expected_labels.pop(IMAGE_LABEL)
        security_options = host.get("SecurityOpt") or []
        networks = (attrs.get("NetworkSettings") or {}).get("Networks") or {}
        environment = _environment_map(config.get("Env"))
        if (
            environment is None
            or not self._labels_include(labels, expected_labels)
            or container.name != self._container_name(team_id, spec.assistant_id)
            or not is_digest_ref(installed_image)
            or config.get("Image") != installed_image
            or installed_image.rpartition("@sha256:")[0] != spec.image.rpartition("@sha256:")[0]
            or config.get("User") != ASSISTANT_UID
            or host.get("ReadonlyRootfs") is not True
            or "ALL" not in (host.get("CapDrop") or [])
            or not any(str(option).startswith("no-new-privileges") for option in security_options)
            or any("seccomp=unconfined" in str(option) for option in security_options)
            or host.get("Privileged") is not False
            or host.get("NetworkMode") != network_name
            or host.get("Memory") != ASSISTANT_MEMORY
            or host.get("MemorySwap") != ASSISTANT_MEMORY
            or host.get("NanoCpus") != ASSISTANT_NANO_CPUS
            or host.get("CpusetCpus") != self.cpuset_cpus
            or host.get("PidsLimit") != ASSISTANT_PIDS
            or host.get("IpcMode") != "private"
            or host.get("CgroupnsMode") != "private"
            or host.get("Tmpfs") not in (None, {})
            or host.get("AutoRemove") is not False
            or (host.get("RestartPolicy") or {}).get("Name") not in {"", "no"}
            or (host.get("LogConfig") or {}).get("Type") != "json-file"
            or (host.get("LogConfig") or {}).get("Config") != {"max-file": "2", "max-size": "1m"}
            or host.get("PortBindings") not in (None, {})
            or host.get("Binds") not in (None, [])
            or host.get("Devices") not in (None, [])
            or host.get("DeviceRequests") not in (None, [])
            or attrs.get("Mounts") not in (None, [])
            or set(networks) != {network_name}
        ):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "the installed Assistant failed its isolation profile",
                code="assistant-isolation-drift",
            )
        return config, environment

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
        if reviewed_hosts:
            expected_proxy_environment = self._validate_egress_policy(team_id, spec, reviewed_hosts)
            self._validate_egress_proxy_attachment(network_name)
            proxy_environment_valid = all(
                environment.get(key) == value for key, value in expected_proxy_environment.items()
            ) and not {"HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"}.intersection(environment)
        else:
            proxy_environment_valid = not {
                "HTTPS_PROXY",
                "https_proxy",
                "NO_PROXY",
                "no_proxy",
                "HTTP_PROXY",
                "http_proxy",
                "ALL_PROXY",
                "all_proxy",
            }.intersection(environment)
        if not proxy_environment_valid:
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
                self.client.api,
                container.id,
                [spec.rpc_command, method, path],
                user=ASSISTANT_UID,
                workdir=ASSISTANT_WORKDIR,
                encoded=encoded,
                timeout=RPC_TIMEOUT_SECONDS,
                maximum=MAX_RESPONSE_BYTES,
                transport_errors=(DockerException,),
                fail_stop=lambda: self._fail_stop_power(container),
                cancelled=lambda _exc: None,
                close_stream=close_stream,
                detect_unsupported_path=detect_unsupported_path,
            )
        except power_execution.RpcExchangeError as exc:
            if exc.kind == "unsupported-path":
                raise _UnsupportedAssistantRpcPathError(path) from None
            if exc.kind == "timeout":
                raise ApiProblem(
                    HTTPStatus.GATEWAY_TIMEOUT,
                    "Assistant Power timed out",
                    code="assistant-timeout",
                ) from exc
            if exc.kind == "ambiguous":
                raise ApiProblem(
                    HTTPStatus.BAD_GATEWAY,
                    "Assistant Power status is ambiguous",
                    code="assistant-rpc-failed",
                ) from exc
            if exc.kind in {"failed", "invalid-result"}:
                raise ApiProblem(
                    HTTPStatus.BAD_GATEWAY,
                    "Assistant Power failed",
                    code="assistant-rpc-failed",
                ) from exc
            raise AssertionError(f"unknown RPC failure: {exc.kind}") from exc

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
        private_values = power_execution.protected_rpc_values(secret_values, account_values, answers)
        inspected_result = raw_result.payload if isinstance(raw_result, power_execution.RpcSuspension) else raw_result
        if self._contains_secret(inspected_result, private_values):
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
            )
        if isinstance(raw_result, power_execution.RpcSuspension):
            local_audit.record(
                "assistant-power",
                result="ok",
                team_id=team_id,
                assistant=assistant_id,
                detail=f"suspended:{power}",
            )
            return {"assistant": assistant_id, "power": power, "suspend": raw_result.payload}
        try:
            result = validate_power_output(assistant_id, power, raw_result)
        except ValueError as exc:
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
        local_audit.record(
            "assistant-power",
            result="ok",
            team_id=team_id,
            assistant=assistant_id,
            detail=f"completed:{power}",
        )
        return {"assistant": assistant_id, "power": power, "result": result}

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


class BoundedServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 32

    def __init__(self, address, handler, controller: LocalController, token: str) -> None:
        super().__init__(address, handler)
        self.controller = controller
        self.token = token
        self._slots = threading.BoundedSemaphore(16)

    def process_request(self, request, client_address) -> None:
        if not self._slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


class Handler(BaseHTTPRequestHandler):
    server: BoundedServer
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args) -> None:
        return

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(REQUEST_TIMEOUT_SECONDS)

    def _authorized(self) -> bool:
        return strict_http.bearer_matches(self.headers, self.server.token)

    def _send(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
        if len(encoded) > MAX_API_RESPONSE_BYTES:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            encoded = b'{"error":"response exceeded its limit"}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        if status == HTTPStatus.UNAUTHORIZED:
            self.send_header("WWW-Authenticate", 'Bearer realm="shimpz-local"')
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(encoded)

    def _body(self, *, max_bytes: int = MAX_BODY_BYTES) -> dict[str, object]:
        try:
            return strict_http.read_json_object(
                self.headers,
                self.rfile,
                max_bytes=max_bytes,
            )
        except strict_http.HttpContractError as exc:
            raise ApiProblem(exc.status, exc.message, code=exc.code) from exc

    def _file_body(self) -> tuple[object, bytes, object]:
        body = self._body(max_bytes=MAX_FILE_BODY_BYTES)
        if set(body) not in ({"filename", "content_b64"}, {"filename", "content_b64", "media_type"}):
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "file upload requires filename, content_b64, and optional media_type",
                code="invalid-body",
            )
        encoded = body["content_b64"]
        if not isinstance(encoded, str):
            raise ApiProblem(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid file content", code="invalid-file")
        try:
            content = base64.b64decode(encoded, validate=True)
        except (binascii.Error, UnicodeError, ValueError) as exc:
            raise ApiProblem(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid file content", code="invalid-file") from exc
        if not content or len(content) > MAX_UPLOAD_BYTES:
            raise ApiProblem(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"file must contain 1 to {MAX_UPLOAD_BYTES} bytes",
                code="file-too-large",
            )
        return body["filename"], content, body.get("media_type")

    def _team_create_body(self) -> str:
        body = self._body()
        if set(body) != {"team_name"}:
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "Team creation requires only team_name",
                code="invalid-body",
            )
        return validate_team_name(body["team_name"])

    def _install_body(self) -> str:
        body = self._body()
        if set(body) != {"assistant"} or not isinstance(body["assistant"], str):
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "assistant must identify one allowlisted Assistant",
                code="invalid-body",
            )
        return body["assistant"]

    def _model_credential_headers(self) -> tuple[str, str]:
        return validate_model_credential_headers(
            self.headers.get_all("X-Shimpz-Model-Provider", failobj=[]),
            self.headers.get_all("X-Shimpz-Model-Api-Key", failobj=[]),
        )

    def _reject_body(self) -> None:
        try:
            strict_http.reject_body(self.headers)
        except strict_http.HttpContractError as exc:
            raise ApiProblem(exc.status, exc.message, code=exc.code) from exc

    def _path_parts(self) -> list[str]:
        try:
            return list(
                strict_http.parse_request_target(
                    self.path,
                    allow_query=False,
                    max_bytes=MAX_PATH_BYTES,
                ).parts
            )
        except strict_http.HttpContractError as exc:
            raise ApiProblem(exc.status, exc.message, code=exc.code) from exc

    def _fixed_route(
        self, parts: list[str]
    ) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        controller = self.server.controller
        if self.command == "GET" and parts == ["healthz"]:
            return HTTPStatus.OK, controller.health(), "health", None, None
        if self.command == "GET" and parts == ["v1", "assistants"]:
            return HTTPStatus.OK, controller.list_registry(), "registry-list", None, None
        if self.command == "GET" and parts == ["v1", "teams"]:
            return HTTPStatus.OK, controller.list_teams(), "team-list", None, None
        if self.command == "DELETE" and parts == ["v1", "space"]:
            return HTTPStatus.OK, controller.reset_space(), "space-reset", None, None
        if self.command == "POST" and parts == ["v1", "oauth", "cloudflare", "callback"]:
            body = self._body()
            if set(body) != {"state", "claim", "session_binding"}:
                raise ApiProblem(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "OAuth callback is invalid",
                    code="invalid-body",
                )
            result = controller.complete_cloudflare_oauth_callback(
                state=body["state"],
                claim=body["claim"],
                session_binding=body["session_binding"],
            )
            return HTTPStatus.OK, result, "assistant-account-complete", None, None
        return None

    def _file_route(self, parts: list[str]) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        if len(parts) not in {4, 5} or parts[:2] != ["v1", "teams"] or parts[3] != "files":
            return None
        controller = self.server.controller
        team_id = validate_team_id(parts[2])
        if len(parts) == 4 and self.command == "GET":
            return HTTPStatus.OK, controller.list_files(team_id), "file-list", team_id, None
        if len(parts) == 4 and self.command == "POST":
            if not _FILE_UPLOAD_SLOTS.acquire(blocking=False):
                raise ApiProblem(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    "another Team file upload is in progress",
                    code="file-upload-busy",
                )
            try:
                filename, content, media_type = self._file_body()
                return (
                    HTTPStatus.OK,
                    controller.put_file(team_id, filename, content, media_type),
                    "file-upload",
                    team_id,
                    None,
                )
            finally:
                _FILE_UPLOAD_SLOTS.release()
        if len(parts) == 5 and self.command == "DELETE":
            return (
                HTTPStatus.OK,
                controller.delete_file(team_id, parts[4]),
                "file-delete",
                team_id,
                None,
            )
        return None

    def _inference_route(
        self, parts: list[str]
    ) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        if len(parts) != 4 or parts[:2] != ["v1", "teams"] or parts[3] != "inference":
            return None
        team_id = validate_team_id(parts[2])
        if self.command == "GET":
            return (
                HTTPStatus.OK,
                self.server.controller.inference_status(team_id),
                "inference-status",
                team_id,
                None,
            )
        if self.command == "PUT":
            return (
                HTTPStatus.OK,
                self.server.controller.configure_inference(team_id, self._body()),
                "inference-configure",
                team_id,
                None,
            )
        return None

    def _chat_route(self, parts: list[str]) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        if len(parts) not in {4, 5} or parts[:2] != ["v1", "teams"] or parts[3] != "chat":
            return None
        team_id = validate_team_id(parts[2])
        if len(parts) == 4 and self.command == "POST":
            provider, api_key = self._model_credential_headers()
            body = self._body(max_bytes=MAX_CHAT_BODY_BYTES)
            payload = self.server.controller.chat(team_id, body, provider, api_key)
            return (
                HTTPStatus.PRECONDITION_REQUIRED if payload.get("status") in CHAT_PAUSED_STATUSES else HTTPStatus.OK,
                payload,
                "chat",
                team_id,
                None,
            )
        if len(parts) == 5 and parts[4] == "accounts" and self.command == "GET":
            return (
                HTTPStatus.OK,
                self.server.controller.pending_chat_accounts(team_id),
                "chat-account-pending",
                team_id,
                None,
            )
        if len(parts) == 5 and parts[4] == "accounts" and self.command == "POST":
            provider, api_key = self._model_credential_headers()
            payload = self.server.controller.resume_chat_accounts(
                team_id,
                self._body(max_bytes=MAX_BODY_BYTES),
                provider,
                api_key,
            )
            return (
                HTTPStatus.PRECONDITION_REQUIRED if payload.get("status") in CHAT_PAUSED_STATUSES else HTTPStatus.OK,
                payload,
                "chat-account-submit",
                team_id,
                None,
            )
        if len(parts) == 5 and parts[4] == "secrets" and self.command == "GET":
            return (
                HTTPStatus.OK,
                self.server.controller.pending_chat_secrets(team_id),
                "chat-secret-pending",
                team_id,
                None,
            )
        if len(parts) == 5 and parts[4] == "secrets" and self.command == "POST":
            provider, api_key = self._model_credential_headers()
            body = self._body(max_bytes=MAX_SECRET_BODY_BYTES)
            payload = self.server.controller.submit_chat_secrets(team_id, body, provider, api_key)
            return (
                HTTPStatus.PRECONDITION_REQUIRED if payload.get("status") in CHAT_PAUSED_STATUSES else HTTPStatus.OK,
                payload,
                "chat-secret-submit",
                team_id,
                None,
            )
        if len(parts) == 5 and parts[4] == "approval" and self.command == "GET":
            return (
                HTTPStatus.OK,
                self.server.controller.pending_chat_approval(team_id),
                "chat-approval-pending",
                team_id,
                None,
            )
        if len(parts) == 5 and parts[4] == "input" and self.command == "GET":
            return (
                HTTPStatus.OK,
                self.server.controller.pending_chat_input(team_id),
                "chat-input-pending",
                team_id,
                None,
            )
        if len(parts) == 5 and parts[4] == "input" and self.command == "POST":
            provider, api_key = self._model_credential_headers()
            body = self._body(max_bytes=MAX_SECRET_BODY_BYTES)
            payload = self.server.controller.submit_chat_input(team_id, body, provider, api_key)
            return (
                HTTPStatus.PRECONDITION_REQUIRED if payload.get("status") in CHAT_PAUSED_STATUSES else HTTPStatus.OK,
                payload,
                "chat-input-submit",
                team_id,
                None,
            )
        if len(parts) == 5 and parts[4] == "approval" and self.command == "POST":
            provider, api_key = self._model_credential_headers()
            body = self._body(max_bytes=MAX_SECRET_BODY_BYTES)
            payload = self.server.controller.submit_chat_approval(team_id, body, provider, api_key)
            return (
                HTTPStatus.PRECONDITION_REQUIRED if payload.get("status") in CHAT_PAUSED_STATUSES else HTTPStatus.OK,
                payload,
                "chat-approval-submit",
                team_id,
                None,
            )
        if len(parts) == 5 and parts[4] == "stop" and self.command == "POST":
            if self._body() != {}:
                raise ApiProblem(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "chat stop requires an empty object",
                    code="invalid-body",
                )
            return (
                HTTPStatus.OK,
                self.server.controller.stop_chat(team_id),
                "chat-stop",
                team_id,
                None,
            )
        return None

    def _assistant_secret_route(
        self,
        parts: list[str],
    ) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        if len(parts) != 4 or parts[:2] != ["v1", "teams"] or parts[3] != "assistant-secrets":
            return None
        team_id = validate_team_id(parts[2])
        if self.command == "GET":
            return (
                HTTPStatus.OK,
                self.server.controller.list_assistant_secrets(team_id),
                "assistant-secret-list",
                team_id,
                None,
            )
        if self.command == "PUT":
            return (
                HTTPStatus.OK,
                self.server.controller.replace_assistant_secrets(
                    team_id,
                    self._body(max_bytes=MAX_SECRET_BODY_BYTES),
                ),
                "assistant-secret-replace",
                team_id,
                None,
            )
        return None

    def _assistant_approval_route(
        self,
        parts: list[str],
    ) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        if len(parts) != 4 or parts[:2] != ["v1", "teams"] or parts[3] != "assistant-approvals":
            return None
        team_id = validate_team_id(parts[2])
        if self.command == "GET":
            return (
                HTTPStatus.OK,
                self.server.controller.list_assistant_approval_grants(team_id),
                "assistant-approval-list",
                team_id,
                None,
            )
        if self.command == "DELETE":
            return (
                HTTPStatus.OK,
                self.server.controller.revoke_assistant_approval_grants(team_id),
                "assistant-approval-revoke",
                team_id,
                None,
            )
        return None

    def _assistant_account_route(
        self,
        parts: list[str],
    ) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        if len(parts) < 4 or parts[:2] != ["v1", "teams"] or parts[3] != "assistant-accounts":
            return None
        team_id = validate_team_id(parts[2])
        if len(parts) == 4 and self.command == "GET":
            return (
                HTTPStatus.OK,
                self.server.controller.list_assistant_accounts(team_id),
                "assistant-account-list",
                team_id,
                None,
            )
        if len(parts) == 7 and parts[4] == "challenges" and parts[6] == "authorize" and self.command == "POST":
            body = self._body()
            if set(body) != {"session_binding"}:
                raise ApiProblem(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "OAuth authorization is invalid",
                    code="invalid-body",
                )
            return (
                HTTPStatus.OK,
                self.server.controller.start_assistant_account_authorization(
                    team_id,
                    parts[5],
                    body["session_binding"],
                ),
                "assistant-account-authorize",
                team_id,
                None,
            )
        if len(parts) == 6 and self.command == "DELETE":
            return (
                HTTPStatus.OK,
                self.server.controller.disconnect_assistant_account(
                    team_id,
                    parts[4],
                    parts[5],
                ),
                "assistant-account-disconnect",
                team_id,
                parts[4],
            )
        return None

    def _team_route(self, parts: list[str]) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        if len(parts) == 4 and parts[:2] == ["v1", "teams"] and parts[3] == "create":
            team_id = validate_team_id(parts[2])
            if self.command == "POST":
                return (
                    HTTPStatus.OK,
                    self.server.controller.create_team(team_id, self._team_create_body()),
                    "team-create",
                    team_id,
                    None,
                )
        if len(parts) == 3 and parts[:2] == ["v1", "teams"] and self.command == "DELETE":
            team_id = validate_team_id(parts[2])
            return (
                HTTPStatus.OK,
                self.server.controller.destroy_team(team_id),
                "team-destroy",
                team_id,
                None,
            )
        return None

    def _route(self) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None]:
        parts = self._path_parts()
        controller = self.server.controller
        if self.command not in {"POST", "PUT"}:
            self._reject_body()
        route = strict_http.resolve_controller_route(strict_http.LOCAL_CONTROLLER, self.command, tuple(parts))
        if route is None:
            raise ApiProblem(HTTPStatus.NOT_FOUND, "route not found", code="route-not-found")

        operation = route.operation
        grouped_resolver = {
            "fixed": self._fixed_route,
            "file": self._file_route,
            "inference": self._inference_route,
            "chat": self._chat_route,
            "assistant-secret": self._assistant_secret_route,
            "assistant-approval": self._assistant_approval_route,
            "assistant-account": self._assistant_account_route,
            "team": self._team_route,
        }.get(route.group)
        if grouped_resolver is not None:
            result = grouped_resolver(parts)
            if result is None:
                raise AssertionError("canonical local route group was not dispatched")
            return result

        team_id = validate_team_id(route.params["team_id"])
        if operation == "assistant-list":
            return HTTPStatus.OK, controller.list_assistants(team_id), operation, team_id, None
        if operation == "assistant-install":
            assistant_id = self._install_body()
            return (
                HTTPStatus.OK,
                controller.install_assistant(team_id, assistant_id),
                operation,
                team_id,
                assistant_id,
            )
        assistant_id = route.params["assistant_id"]
        if operation == "assistant-uninstall":
            return (
                HTTPStatus.OK,
                controller.uninstall_assistant(team_id, assistant_id),
                operation,
                team_id,
                assistant_id,
            )
        if operation == "assistant-help":
            return (
                HTTPStatus.OK,
                controller.assistant_help(team_id, assistant_id, route.params.get("locale", "en")),
                operation,
                team_id,
                assistant_id,
            )
        if operation == "assistant-invoke":
            return (
                HTTPStatus.OK,
                controller.invoke(team_id, assistant_id, route.params["power_id"], self._body()),
                operation,
                team_id,
                assistant_id,
            )
        raise AssertionError("canonical local route was not dispatched")

    def _handle(self) -> None:
        self.close_connection = True
        if not self._authorized():
            trace_id = local_audit.record("authentication", result="denied", detail="invalid-bearer")
            self._send(HTTPStatus.UNAUTHORIZED, {"error": "authentication required", "trace_id": trace_id})
            return

        local.dispatch_route(
            self._route,
            local_audit.record,
            self._send,
            ApiProblem,
            DockerException,
        )

    do_GET = _handle
    do_POST = _handle
    do_DELETE = _handle
    do_HEAD = _handle
    do_OPTIONS = _handle
    do_PATCH = _handle
    do_PUT = _handle


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
