"""Minimal Docker controller for one locally owned Shimpz Space.

This is intentionally separate from the hosted Capsule controller.  An empty Capsule is
one labeled internal network; its only runnable resources are build-allowlisted,
digest-pinned first-party Assistants with a fixed Power contract.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import secrets
import select
import socket
import struct
import sys
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import assistant_chat
import brain_runtime_client
import brain_runtime_token_store
import capsule_storage
import chat_orchestrator
import docker
import inference_config
import local_audit
import local_token_store
import power_journal
from docker.errors import APIError, DockerException, ImageNotFound, NotFound
from docker.types import LogConfig, Ulimit
from local_registry import (
    AssistantSpec,
    RegistryError,
    load_registry,
    validate_power_input,
    validate_power_output,
)

LISTEN_PORT = 7077
PROFILE = "single-owner-local-v1"
MANAGED_LABEL = "com.shimpz.local.managed"
PROFILE_LABEL = "com.shimpz.local.profile"
SPACE_LABEL = "com.shimpz.local.space-id"
KIND_LABEL = "com.shimpz.local.kind"
CAPSULE_LABEL = "com.shimpz.local.capsule-id"
CAPSULE_NAME_LABEL = "com.shimpz.local.capsule-name"
ASSISTANT_LABEL = "com.shimpz.local.assistant-id"
IMAGE_LABEL = "com.shimpz.local.image"

_CAPSULE_ID = re.compile(r"[a-z0-9_]{1,40}")
_ASSISTANT_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_SPACE_ID = re.compile(r"[a-z0-9][a-z0-9]*(?:-[a-z0-9]+)*")
_DOCKER_ID = re.compile(r"[0-9a-f]{12,64}")
MAX_CAPSULE_ID_LENGTH = 40
MAX_ASSISTANT_ID_LENGTH = 48
MAX_SPACE_ID_LENGTH = 48
MAX_BODY_BYTES = 16 * 1024
MAX_CHAT_BODY_BYTES = 24 * 1024
MAX_RESPONSE_BYTES = 32 * 1024
MAX_API_RESPONSE_BYTES = 128 * 1024
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_FILE_BODY_BYTES = 4 * ((MAX_UPLOAD_BYTES + 2) // 3) + 8192
MAX_PATH_BYTES = 512
REQUEST_TIMEOUT_SECONDS = 10
RPC_TIMEOUT_SECONDS = 8
HEALTH_TIMEOUT_SECONDS = 15
MAX_CHAT_MESSAGE_CHARS = 16_000
MAX_CHAT_FILES = 8
MIN_API_KEY_BYTES = 16
MAX_API_KEY_BYTES = 8 * 1024

ASSISTANT_UID = "10001:10001"
ASSISTANT_MEMORY = 128 * 1024 * 1024
ASSISTANT_NANO_CPUS = 250_000_000
ASSISTANT_PIDS = 64
STATELESS_RECOVERY_ASSISTANTS = frozenset({"hello-pulse"})
STORAGE_ROOT = Path("/var/lib/shimpz-local/storage")
INFERENCE_ROOT = Path("/var/lib/shimpz-local/inference")
LOCAL_POWER_JOURNAL_PATH = Path(
    os.environ.get(
        "SHIMPZ_LOCAL_POWER_JOURNAL_PATH",
        "/var/lib/shimpz-local/power-journal/journal.sqlite3",
    )
)
_FILE_UPLOAD_SLOTS = threading.BoundedSemaphore(1)


class ApiProblem(RuntimeError):
    def __init__(self, status: HTTPStatus, message: str, *, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code


@dataclass(frozen=True, slots=True)
class _ActiveAssistant:
    spec: AssistantSpec
    container_id: str


def _power_operation(
    request: brain_runtime_client.PowerRequest,
    active: _ActiveAssistant,
) -> power_journal.Operation:
    """Commit to a normalized request and immutable Assistant runtime, never raw journal input."""
    if not active.container_id or not active.spec.image:
        raise power_journal.PowerJournalConflictError("Assistant generation is invalid")
    try:
        encoded = json.dumps(
            {
                "approval": request.approval,
                "assistant_container_id": active.container_id,
                "assistant_id": request.assistant_id,
                "assistant_image": active.spec.image,
                "input": request.input,
                "power": request.power,
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        raise power_journal.PowerJournalConflictError("Power request cannot be fingerprinted") from exc
    return power_journal.Operation(request.interrupt_id, hashlib.sha256(encoded).hexdigest())


class _LocalPowerBatch:
    """Adapt one local Brain suspension to its network-generation journal."""

    def __init__(
        self,
        journal: power_journal.PowerJournal,
        generation: str,
        thread_id: str,
        bindings: dict[str, _ActiveAssistant],
        execute: Callable[[brain_runtime_client.PowerRequest], object],
    ) -> None:
        self._journal = journal
        self._generation = generation
        self._thread_id = thread_id
        self._bindings = bindings
        self._execute = execute
        self._batch: power_journal.Batch | None = None
        self._operations: dict[str, power_journal.Operation] = {}

    def prepare(self, requests: tuple[brain_runtime_client.PowerRequest, ...]) -> None:
        if self._batch is not None:
            raise power_journal.PowerJournalConflictError("Power batch is already prepared")
        operations: list[power_journal.Operation] = []
        for request in requests:
            active = self._bindings.get(request.assistant_id)
            if active is None:
                raise power_journal.PowerJournalConflictError("Power Assistant is unavailable")
            operations.append(_power_operation(request, active))
        self._batch = self._journal.prepare_batch(self._generation, self._thread_id, operations)
        self._operations = {operation.interrupt_id: operation for operation in operations}

    def invoke(self, request: brain_runtime_client.PowerRequest) -> object:
        if self._batch is None:
            raise power_journal.PowerJournalConflictError("Power batch is not prepared")
        operation = self._operations.get(request.interrupt_id)
        if operation is None:
            raise power_journal.PowerJournalConflictError("Power operation is not prepared")
        decision = self._journal.begin(self._batch, operation)
        if not decision.execute:
            return decision.result
        result = self._execute(request)
        self._journal.complete(self._batch, operation, result)
        return result

    def delivered(self, requests: tuple[brain_runtime_client.PowerRequest, ...]) -> None:
        if self._batch is None:
            raise power_journal.PowerJournalConflictError("Power batch is not prepared")
        expected = tuple(operation.interrupt_id for operation in self._batch.operations)
        if tuple(request.interrupt_id for request in requests) != expected:
            raise power_journal.PowerJournalConflictError("Power delivery batch changed")
        self._journal.delivered(self._batch)
        self._batch = None
        self._operations = {}


def _is_replaceable_readiness_failure(assistant_id: str, problem: ApiProblem) -> bool:
    return assistant_id in STATELESS_RECOVERY_ASSISTANTS and problem.code == "assistant-not-ready"


def validate_capsule_id(value: str) -> str:
    if len(value) > MAX_CAPSULE_ID_LENGTH or _CAPSULE_ID.fullmatch(value) is None:
        raise ApiProblem(HTTPStatus.UNPROCESSABLE_ENTITY, "invalid Capsule id", code="invalid-capsule-id")
    return value


def validate_capsule_name(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 80
        or value.strip() != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ApiProblem(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            "Capsule name must contain 1 to 80 trimmed characters",
            code="invalid-capsule-name",
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


def _brain_thread_id(space_id: str, capsule_id: str, network_id: str) -> str:
    """Bind local conversation state to one immutable Team network generation."""
    if (
        not isinstance(space_id, str)
        or len(space_id) > MAX_SPACE_ID_LENGTH
        or _SPACE_ID.fullmatch(space_id) is None
        or not isinstance(capsule_id, str)
        or len(capsule_id) > MAX_CAPSULE_ID_LENGTH
        or _CAPSULE_ID.fullmatch(capsule_id) is None
        or not isinstance(network_id, str)
        or _DOCKER_ID.fullmatch(network_id) is None
    ):
        raise ApiProblem(
            HTTPStatus.CONFLICT,
            "Team identity failed its persisted contract",
            code="ownership-conflict",
        )
    return f"local:{space_id}:{capsule_id}:{network_id}:default"


def half_cpu_set(processors: int) -> str:
    if isinstance(processors, bool) or not isinstance(processors, int) or processors < 1:
        raise RuntimeError("the Docker daemon reported an invalid CPU count")
    available = max(1, processors // 2)
    return "0" if available == 1 else f"0-{available - 1}"


class LocalController:
    def __init__(
        self,
        client: docker.DockerClient,
        space_id: str,
        registry: dict[str, AssistantSpec],
        storage: capsule_storage.CapsuleStorage,
        inference_store: inference_config.InferenceConfigStore | None = None,
        brain_runtime: brain_runtime_client.BrainRuntimeClient | None = None,
        power_state: power_journal.PowerJournal | None = None,
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
        self._blocked_power_workloads: set[str] = set()
        self._locks = tuple(threading.RLock() for _ in range(64))
        self._active_chat_guard = threading.Lock()
        self._chat_locks: dict[str, threading.Lock] = {}
        self._active_chat_tokens: dict[str, str] = {}
        self._active_power_containers: dict[str, tuple[str, object]] = {}
        self._cancelled_chat_tokens: set[str] = set()
        daemon_info = self._require_default_seccomp()
        self.cpuset_cpus = half_cpu_set(daemon_info.get("NCPU"))

    def _require_default_seccomp(self) -> dict:
        try:
            info = self.client.info()
            options = info.get("SecurityOptions", [])
        except DockerException as exc:
            raise RuntimeError("the Docker daemon is unavailable") from exc
        if not any(isinstance(option, str) and option.startswith("name=seccomp") for option in options):
            raise RuntimeError("the Docker daemon default seccomp profile is required")
        return info

    def _lock(self, capsule_id: str) -> threading.RLock:
        slot = hashlib.sha256(capsule_id.encode("ascii")).digest()[0] % len(self._locks)
        return self._locks[slot]

    def _chat_lock(self, capsule_id: str) -> threading.Lock:
        with self._active_chat_guard:
            return self._chat_locks.setdefault(capsule_id, threading.Lock())

    def _chat_cancelled(self, token: str) -> bool:
        with self._active_chat_guard:
            return token in self._cancelled_chat_tokens

    def _commit_chat_terminal(self, capsule_id: str, token: str) -> bool:
        """Commit a reply only when Stop did not win this Controller-owned turn."""
        with self._active_chat_guard:
            if token in self._cancelled_chat_tokens or self._active_chat_tokens.get(capsule_id) != token:
                return False
            self._active_chat_tokens.pop(capsule_id, None)
            return True

    def _cancel_chat_for_destroy(self, capsule_id: str) -> None:
        """Prevent another Power and synchronously stop one already executing."""
        with self._active_chat_guard:
            token = self._active_chat_tokens.get(capsule_id)
            if token is not None:
                self._cancelled_chat_tokens.add(token)
            active = self._active_power_containers.get(capsule_id)
            active_power = active[1] if token is not None and active is not None and active[0] == token else None
        if active_power is not None:
            self._fail_stop_power(active_power)

    @contextmanager
    def _exclusive_chat_turn(self, capsule_id: str):
        lock = self._chat_lock(capsule_id)
        if not lock.acquire(blocking=False):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Capsule already has an active chat turn",
                code="chat-active",
            )
        token = secrets.token_hex(16)
        with self._active_chat_guard:
            self._active_chat_tokens[capsule_id] = token
        try:
            yield token
        finally:
            with self._active_chat_guard:
                if self._active_chat_tokens.get(capsule_id) == token:
                    self._active_chat_tokens.pop(capsule_id, None)
                active = self._active_power_containers.get(capsule_id)
                if active is not None and active[0] == token:
                    self._active_power_containers.pop(capsule_id, None)
                self._cancelled_chat_tokens.discard(token)
            lock.release()

    def _base_labels(self, capsule_id: str, kind: str) -> dict[str, str]:
        return {
            MANAGED_LABEL: "1",
            PROFILE_LABEL: PROFILE,
            SPACE_LABEL: self.space_id,
            KIND_LABEL: kind,
            CAPSULE_LABEL: capsule_id,
        }

    def _network_name(self, capsule_id: str) -> str:
        return f"shimpz-local-{_space_prefix(self.space_id)}-capsule-{capsule_id}"

    def _container_name(self, capsule_id: str, assistant_id: str) -> str:
        return f"shimpz-local-{_space_prefix(self.space_id)}-{capsule_id}-assistant-{assistant_id}"

    @staticmethod
    def _labels_include(actual: object, expected: dict[str, str]) -> bool:
        return isinstance(actual, dict) and all(actual.get(key) == value for key, value in expected.items())

    def _validate_network(self, network, capsule_id: str) -> str:
        network.reload()
        attrs = network.attrs
        expected = self._base_labels(capsule_id, "capsule")
        labels = attrs.get("Labels") or {}
        if (
            not self._labels_include(labels, expected)
            or attrs.get("Name") != self._network_name(capsule_id)
            or attrs.get("Driver") != "bridge"
            or attrs.get("Internal") is not True
            or attrs.get("Attachable") is not False
        ):
            raise ApiProblem(HTTPStatus.CONFLICT, "Capsule resource ownership conflict", code="ownership-conflict")
        try:
            return validate_capsule_name(labels.get(CAPSULE_NAME_LABEL))
        except ApiProblem as exc:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Capsule resource ownership conflict",
                code="ownership-conflict",
            ) from exc

    def _network(self, capsule_id: str, *, required: bool = True):
        try:
            network = self.client.networks.get(self._network_name(capsule_id))
        except NotFound:
            if required:
                raise ApiProblem(HTTPStatus.NOT_FOUND, "Capsule not found", code="capsule-not-found") from None
            return None
        self._validate_network(network, capsule_id)
        return network

    def list_capsules(self) -> dict[str, list[dict[str, str]]]:
        filters = {
            "label": [
                f"{MANAGED_LABEL}=1",
                f"{PROFILE_LABEL}={PROFILE}",
                f"{SPACE_LABEL}={self.space_id}",
                f"{KIND_LABEL}=capsule",
            ]
        }
        capsules: list[dict[str, str]] = []
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
            capsule_id = labels.get(CAPSULE_LABEL)
            if not isinstance(capsule_id, str):
                raise ApiProblem(HTTPStatus.CONFLICT, "Capsule resource ownership conflict", code="ownership-conflict")
            validate_capsule_id(capsule_id)
            name = self._validate_network(network, capsule_id)
            capsules.append({"id": capsule_id, "name": name, "status": "running"})
        capsules.sort(key=lambda item: item["id"])
        return {"capsules": capsules}

    def create_capsule(self, capsule_id: str, name: str) -> dict[str, object]:
        capsule_id = validate_capsule_id(capsule_id)
        name = validate_capsule_name(name)
        with self._lock(capsule_id):
            existing = self._network(capsule_id, required=False)
            if existing is not None:
                existing_name = self._validate_network(existing, capsule_id)
                if existing_name != name:
                    raise ApiProblem(
                        HTTPStatus.CONFLICT,
                        "Capsule id already belongs to a different name",
                        code="capsule-name-conflict",
                    )
                return {"id": capsule_id, "name": name, "status": "running", "created": False}
            try:
                # A Capsule identity starts empty even after a daemon crash removed its network
                # before the previous lifecycle could clean the dedicated storage volume.
                self.storage.destroy(capsule_id)
            except capsule_storage.StorageError as exc:
                self._raise_storage_problem(exc)
            try:
                self.inference_store.delete(capsule_id)
            except inference_config.InferenceConfigError as exc:
                self._raise_inference_problem(exc)
            try:
                labels = self._base_labels(capsule_id, "capsule")
                labels[CAPSULE_NAME_LABEL] = name
                network = self.client.networks.create(
                    self._network_name(capsule_id),
                    driver="bridge",
                    internal=True,
                    attachable=False,
                    check_duplicate=True,
                    labels=labels,
                )
            except APIError as exc:
                # A concurrent idempotent creator is safe only when the resulting
                # resource proves the exact ownership/profile labels.
                network = self._network(capsule_id, required=False)
                if network is None:
                    raise ApiProblem(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "Docker could not create the Capsule",
                        code="docker-create-failed",
                    ) from exc
                existing_name = self._validate_network(network, capsule_id)
                if existing_name != name:
                    raise ApiProblem(
                        HTTPStatus.CONFLICT,
                        "Capsule id already belongs to a different name",
                        code="capsule-name-conflict",
                    ) from exc
                return {"id": capsule_id, "name": name, "status": "running", "created": False}
            self._validate_network(network, capsule_id)
            return {"id": capsule_id, "name": name, "status": "running", "created": True}

    @staticmethod
    def _raise_storage_problem(exc: capsule_storage.StorageError) -> None:
        if isinstance(exc, capsule_storage.StorageQuotaError):
            raise ApiProblem(
                HTTPStatus.INSUFFICIENT_STORAGE,
                str(exc),
                code="storage-quota-exceeded",
            ) from exc
        if isinstance(exc, capsule_storage.StorageNotFoundError):
            raise ApiProblem(HTTPStatus.NOT_FOUND, "file not found", code="file-not-found") from exc
        if isinstance(exc, capsule_storage.StorageInputError):
            raise ApiProblem(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc), code="invalid-file") from exc
        raise ApiProblem(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Capsule storage failed its safety checks",
            code="storage-safety-failed",
        ) from exc

    @staticmethod
    def _raise_inference_problem(exc: inference_config.InferenceConfigError) -> None:
        raise ApiProblem(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Capsule model provider metadata is unavailable",
            code="inference-store-failed",
        ) from exc

    def inference_status(self, capsule_id: str) -> dict[str, str]:
        capsule_id = validate_capsule_id(capsule_id)
        with self._lock(capsule_id):
            self._network(capsule_id)
            try:
                config = self.inference_store.load(capsule_id)
            except inference_config.InferenceConfigError as exc:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "Capsule model provider is not configured",
                    code="inference-not-configured",
                ) from exc
        return {"capsule": capsule_id, "provider": config.provider, "model": config.model}

    def configure_inference(self, capsule_id: str, body: object) -> dict[str, str]:
        capsule_id = validate_capsule_id(capsule_id)
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
        with self._lock(capsule_id):
            self._network(capsule_id)
            try:
                self.inference_store.save(capsule_id, config)
            except inference_config.InferenceConfigError as exc:
                self._raise_inference_problem(exc)
        return {"capsule": capsule_id, "provider": config.provider, "model": config.model}

    def _chat_file_metadata(self, capsule_id: str, file_ids: object) -> list[dict[str, object]]:
        if not isinstance(file_ids, list) or len(file_ids) > MAX_CHAT_FILES:
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                f"files must contain at most {MAX_CHAT_FILES} opaque ids",
                code="invalid-files",
            )
        try:
            return self.storage.metadata(capsule_id, file_ids)
        except capsule_storage.StorageNotFoundError as exc:
            raise ApiProblem(HTTPStatus.NOT_FOUND, "selected file not found", code="file-not-found") from exc
        except capsule_storage.StorageInputError as exc:
            raise ApiProblem(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc), code="invalid-files") from exc
        except capsule_storage.StorageError as exc:
            self._raise_storage_problem(exc)

    def _chat_setup(
        self,
        capsule_id: str,
        file_ids: object,
        provider: str,
    ) -> tuple[
        str,
        str,
        tuple[_ActiveAssistant, ...],
        list[dict[str, object]],
        inference_config.InferenceConfig,
    ]:
        with self._lock(capsule_id):
            network = self._network(capsule_id)
            team_name = self._validate_network(network, capsule_id)
            network_id = getattr(network, "id", None)
            if not isinstance(network_id, str) or not network_id:
                raise ApiProblem(HTTPStatus.CONFLICT, "Team resource ownership conflict", code="ownership-conflict")
            assistants = self._active_chat_assistants(capsule_id, network.name)
            files = self._chat_file_metadata(capsule_id, file_ids)
            try:
                config = self.inference_store.load(capsule_id)
            except inference_config.InferenceConfigError as exc:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "Capsule model provider is not configured",
                    code="inference-not-configured",
                ) from exc
            if config.provider != provider:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "configured model provider changed; retry",
                    code="inference-provider-mismatch",
                )
        return team_name, network_id, assistants, files, config

    def _active_chat_assistants(self, capsule_id: str, network_name: str) -> tuple[_ActiveAssistant, ...]:
        try:
            containers = self.client.containers.list(**self._assistant_filters(capsule_id))
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
            self._validate_container(container, capsule_id, spec, network_name)
            if container.id in self._blocked_power_workloads:
                raise ApiProblem(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Assistant Power execution is blocked until this Assistant is reinstalled",
                    code="assistant-power-blocked",
                )
            container.reload()
            if container.status == "running":
                active.append(_ActiveAssistant(spec=spec, container_id=container.id))
        active.sort(key=lambda item: item.spec.assistant_id)
        if not active:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "install and start at least one Assistant before chatting with this Team",
                code="team-has-no-active-assistants",
            )
        return tuple(active)

    def _invoke_chat_power(
        self,
        capsule_id: str,
        token: str,
        assistant_id: str,
        power: str,
        payload: object,
    ) -> object:
        with self._lock(capsule_id):
            spec = self._resolve(assistant_id)
            network = self._network(capsule_id)
            container = self._assistant_container(capsule_id, assistant_id)
            self._validate_container(container, capsule_id, spec, network.name)
            with self._active_chat_guard:
                if (
                    self._active_chat_tokens.get(capsule_id) != token
                    or token in self._cancelled_chat_tokens
                    or capsule_id in self._active_power_containers
                ):
                    raise chat_orchestrator.ChatStoppedError("chat turn stopped")
                self._active_power_containers[capsule_id] = (token, container)
            try:
                invocation = self.invoke(capsule_id, assistant_id, power, payload)
            except ApiProblem:
                if self._chat_cancelled(token):
                    raise chat_orchestrator.ChatStoppedError("chat turn stopped") from None
                raise
            finally:
                with self._active_chat_guard:
                    active = self._active_power_containers.get(capsule_id)
                    if active is not None and active[0] == token:
                        self._active_power_containers.pop(capsule_id, None)
            if self._chat_cancelled(token):
                raise chat_orchestrator.ChatStoppedError("chat turn stopped")
        return invocation["result"]

    def chat(
        self,
        capsule_id: str,
        body: object,
        provider: str,
        api_key: str,
    ) -> dict[str, object]:
        capsule_id = validate_capsule_id(capsule_id)
        if not isinstance(body, dict) or set(body) not in ({"message"}, {"message", "files"}):
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "Team chat requires only message and optional files",
                code="invalid-body",
            )
        message = body["message"]
        file_ids = body.get("files", [])
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
        with self._exclusive_chat_turn(capsule_id) as token:
            team_name, network_id, assistants, files, config = self._chat_setup(capsule_id, file_ids, provider)
            context = brain_runtime_client.RuntimeContext(
                thread_id=_brain_thread_id(self.space_id, capsule_id, network_id),
                team_name=team_name,
                assistants=tuple(
                    brain_runtime_client.RuntimeAssistant(
                        id=active.spec.assistant_id,
                        rules=active.spec.rules,
                        powers=tuple(
                            brain_runtime_client.RuntimePower(
                                id=power_id,
                                summary=power.summary,
                                input_schema=power.input_schema,
                                approval=power.approval,
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
            prompt = assistant_chat.build_prompt(message, files)
            bindings = {active.spec.assistant_id: active for active in assistants}

            def validate_power(assistant_id: str, power: str, payload) -> object:
                active = bindings.get(assistant_id)
                if active is None:
                    raise ApiProblem(
                        HTTPStatus.CONFLICT,
                        "Brain requested an unavailable Assistant",
                        code="assistant-unavailable",
                    )
                try:
                    return validate_power_input(assistant_id, power, payload)
                except ValueError as exc:
                    raise ApiProblem(
                        HTTPStatus.UNPROCESSABLE_ENTITY,
                        str(exc),
                        code="invalid-power-input",
                    ) from exc

            def execute_power(request: brain_runtime_client.PowerRequest) -> object:
                return self._invoke_chat_power(
                    capsule_id,
                    token,
                    request.assistant_id,
                    request.power,
                    request.input,
                )

            durable_batch = _LocalPowerBatch(
                self.power_state,
                network_id,
                context.thread_id,
                bindings,
                execute_power,
            )

            initial_identity = (
                team_name,
                network_id,
                tuple((item.spec.assistant_id, item.spec.image, item.container_id) for item in assistants),
                files,
                config,
            )

            def validate_context() -> None:
                current_team, current_network, current_assistants, current_files, current_config = self._chat_setup(
                    capsule_id,
                    file_ids,
                    provider,
                )
                current_identity = (
                    current_team,
                    current_network,
                    tuple((item.spec.assistant_id, item.spec.image, item.container_id) for item in current_assistants),
                    current_files,
                    current_config,
                )
                if current_identity != initial_identity:
                    raise ApiProblem(
                        HTTPStatus.CONFLICT,
                        "Team capabilities changed; retry",
                        code="team-context-changed",
                    )

            try:
                outcome = chat_orchestrator.run(
                    self.brain_runtime,
                    context,
                    prompt,
                    validate_power,
                    durable_batch.invoke,
                    prepare_batch=durable_batch.prepare,
                    batch_delivered=durable_batch.delivered,
                    cancelled=lambda: self._chat_cancelled(token),
                    validate_context=validate_context,
                )
            except power_journal.PowerJournalError as exc:
                raise ApiProblem(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Team Power execution state is unavailable",
                    code="power-state-unavailable",
                ) from exc
            except chat_orchestrator.ChatStoppedError as exc:
                raise ApiProblem(HTTPStatus.CONFLICT, "chat turn stopped", code="chat-stopped") from exc
            except chat_orchestrator.ApprovalRequiredError as exc:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "Assistant Power requires approval",
                    code="power-approval-required",
                ) from exc
            except chat_orchestrator.ChatOrchestrationError as exc:
                raise ApiProblem(
                    HTTPStatus.BAD_GATEWAY,
                    "Brain could not complete the Team turn",
                    code="brain-runtime-failed",
                ) from exc
            except brain_runtime_client.BrainRuntimeError as exc:
                raise ApiProblem(
                    HTTPStatus.BAD_GATEWAY,
                    "Brain runtime is unavailable",
                    code="brain-runtime-failed",
                ) from exc

            if not self._commit_chat_terminal(capsule_id, token):
                raise ApiProblem(HTTPStatus.CONFLICT, "chat turn stopped", code="chat-stopped")
            return {
                "capsule": capsule_id,
                "team": team_name,
                "reply": outcome.reply,
            }

    def stop_chat(self, capsule_id: str) -> dict[str, object]:
        capsule_id = validate_capsule_id(capsule_id)
        self._network(capsule_id)
        power_stopped = False
        with self._active_chat_guard:
            token = self._active_chat_tokens.get(capsule_id)
            if token is not None:
                self._cancelled_chat_tokens.add(token)
            active = self._active_power_containers.get(capsule_id)
            if token is not None and active is not None and active[0] == token:
                self._fail_stop_power(active[1])
                power_stopped = True
        accepted = token is not None
        return {
            "capsule": capsule_id,
            "requested": accepted,
            "accepted": accepted,
            "confirmed": power_stopped,
            "forced_restart": False,
        }

    def put_file(
        self,
        capsule_id: str,
        filename: object,
        content: bytes,
        media_type: object,
    ) -> dict[str, object]:
        capsule_id = validate_capsule_id(capsule_id)
        with self._lock(capsule_id):
            self._network(capsule_id)
            try:
                stored = self.storage.put(capsule_id, filename, content, media_type)
            except capsule_storage.StorageError as exc:
                self._raise_storage_problem(exc)
        return {"capsule": capsule_id, "file": stored}

    def list_files(self, capsule_id: str) -> dict[str, object]:
        capsule_id = validate_capsule_id(capsule_id)
        with self._lock(capsule_id):
            self._network(capsule_id)
            try:
                listing = self.storage.list(capsule_id)
            except capsule_storage.StorageError as exc:
                self._raise_storage_problem(exc)
        return {"capsule": capsule_id, **listing}

    def delete_file(self, capsule_id: str, file_id: object) -> dict[str, object]:
        capsule_id = validate_capsule_id(capsule_id)
        with self._lock(capsule_id):
            self._network(capsule_id)
            try:
                result = self.storage.delete(capsule_id, file_id)
            except capsule_storage.StorageError as exc:
                self._raise_storage_problem(exc)
        return {"capsule": capsule_id, **result}

    def _assistant_filters(self, capsule_id: str) -> dict[str, list[str] | bool]:
        return {
            "all": True,
            "filters": {
                "label": [
                    f"{MANAGED_LABEL}=1",
                    f"{PROFILE_LABEL}={PROFILE}",
                    f"{SPACE_LABEL}={self.space_id}",
                    f"{KIND_LABEL}=assistant",
                    f"{CAPSULE_LABEL}={capsule_id}",
                ]
            },
        }

    def _assistant_container(self, capsule_id: str, assistant_id: str, *, required: bool = True):
        name = self._container_name(capsule_id, assistant_id)
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

    def _assistant_labels(self, capsule_id: str, spec: AssistantSpec) -> dict[str, str]:
        labels = self._base_labels(capsule_id, "assistant")
        labels.update({ASSISTANT_LABEL: spec.assistant_id, IMAGE_LABEL: spec.image})
        return labels

    def _validate_container(self, container, capsule_id: str, spec: AssistantSpec, network_name: str) -> None:
        container.reload()
        attrs = container.attrs
        config = attrs.get("Config") or {}
        host = attrs.get("HostConfig") or {}
        expected_labels = self._assistant_labels(capsule_id, spec)
        security_options = host.get("SecurityOpt") or []
        networks = (attrs.get("NetworkSettings") or {}).get("Networks") or {}
        if (
            not self._labels_include(config.get("Labels"), expected_labels)
            or config.get("Image") != spec.image
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

    @staticmethod
    def _close_exec_stream(stream) -> None:
        response = getattr(stream, "_response", None)
        if response is not None:
            response.close()
        else:
            stream.close()

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

    @staticmethod
    def _read_exact(raw_socket: socket.socket, amount: int, deadline: float) -> bytes:
        output = bytearray()
        while len(output) < amount:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([raw_socket], [], [], remaining)[0]:
                raise TimeoutError
            chunk = raw_socket.recv(amount - len(output))
            if not chunk:
                raise EOFError
            output.extend(chunk)
        return bytes(output)

    def _read_rpc_frames(self, raw_socket: socket.socket, deadline: float) -> tuple[bytes, int]:
        stdout = bytearray()
        stderr_bytes = 0
        while True:
            try:
                header = self._read_exact(raw_socket, 8, deadline)
            except EOFError:
                break
            stream_id, length = struct.unpack(">BxxxL", header)
            if stream_id not in {1, 2}:
                raise ValueError("invalid Docker exec stream")
            if length > MAX_RESPONSE_BYTES + 1:
                raise ValueError("oversized Docker exec frame")
            chunk = self._read_exact(raw_socket, length, deadline)
            if stream_id == 1:
                stdout.extend(chunk)
                if len(stdout) > MAX_RESPONSE_BYTES:
                    raise ValueError("oversized Assistant response")
            else:
                stderr_bytes += len(chunk)
                if stderr_bytes > MAX_RESPONSE_BYTES:
                    raise ValueError("oversized Assistant error")
        return bytes(stdout), stderr_bytes

    def _rpc(self, container, spec: AssistantSpec, method: str, path: str, payload: dict) -> object:
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        if len(encoded) > MAX_BODY_BYTES:
            raise ApiProblem(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request is too large", code="body-too-large")
        try:
            created = self.client.api.exec_create(
                container.id,
                [spec.rpc_command, method, path],
                stdin=True,
                stdout=True,
                stderr=True,
                privileged=False,
                user=ASSISTANT_UID,
            )
            exec_id = created["Id"]
            stream = self.client.api.exec_start(exec_id, socket=True)
            raw_socket = getattr(stream, "_sock", None)
            if raw_socket is None:
                raise OSError("the Docker attach socket cannot half-close stdin")
            raw_socket.sendall(encoded)
            raw_socket.shutdown(socket.SHUT_WR)
            deadline = time.monotonic() + RPC_TIMEOUT_SECONDS
            stdout, stderr_bytes = self._read_rpc_frames(raw_socket, deadline)
        except TimeoutError as exc:
            self._fail_stop_power(container)
            raise ApiProblem(
                HTTPStatus.GATEWAY_TIMEOUT,
                "Assistant Power timed out",
                code="assistant-timeout",
            ) from exc
        except (DockerException, OSError, ValueError, KeyError) as exc:
            self._fail_stop_power(container)
            raise ApiProblem(HTTPStatus.BAD_GATEWAY, "Assistant Power failed", code="assistant-rpc-failed") from exc
        finally:
            if "stream" in locals():
                with suppress(Exception):
                    self._close_exec_stream(stream)

        try:
            details = self.client.api.exec_inspect(exec_id)
        except DockerException as exc:
            self._fail_stop_power(container)
            raise ApiProblem(
                HTTPStatus.BAD_GATEWAY,
                "Assistant Power status is ambiguous",
                code="assistant-rpc-failed",
            ) from exc
        exit_code = details.get("ExitCode")
        if not isinstance(exit_code, int):
            self._fail_stop_power(container)
            raise ApiProblem(
                HTTPStatus.BAD_GATEWAY,
                "Assistant Power status is ambiguous",
                code="assistant-rpc-failed",
            )
        try:
            decoded = json.loads(bytes(stdout))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ApiProblem(HTTPStatus.BAD_GATEWAY, "Assistant Power failed", code="assistant-rpc-failed") from exc
        if exit_code != 0 or stderr_bytes:
            raise ApiProblem(HTTPStatus.BAD_GATEWAY, "Assistant Power failed", code="assistant-rpc-failed")
        return decoded

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
                    "title": "Hello Pulse",
                    "summary": "A tiny first-party Assistant that proves the local install and Power flow.",
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

    def list_assistants(self, capsule_id: str) -> dict[str, list[dict[str, str]]]:
        capsule_id = validate_capsule_id(capsule_id)
        self._network(capsule_id)
        output: list[dict[str, str]] = []
        try:
            containers = self.client.containers.list(**self._assistant_filters(capsule_id))
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
            self._validate_container(container, capsule_id, spec, self._network_name(capsule_id))
            output.append({"assistant": assistant_id, "status": container.status})
        output.sort(key=lambda item: item["assistant"])
        return {"assistants": output}

    def _create_assistant_container(self, capsule_id: str, spec: AssistantSpec, network, image) -> None:
        container = None
        try:
            container = self.client.containers.create(
                image=spec.image,
                name=self._container_name(capsule_id, spec.assistant_id),
                command=None,
                detach=True,
                user=ASSISTANT_UID,
                network=network.name,
                labels=self._assistant_labels(capsule_id, spec),
                environment={
                    "SHIMPZ_ASSISTANT_ID": spec.assistant_id,
                    "SHIMPZ_CAPSULE_ID": capsule_id,
                    "PYTHONDONTWRITEBYTECODE": "1",
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
            container.start()
            self._validate_container(container, capsule_id, spec, network.name)
            self._wait_ready(container, spec)
        except ApiProblem:
            if container is not None:
                container.remove(force=True)
            raise
        except DockerException as exc:
            if container is not None:
                with suppress(DockerException):
                    container.remove(force=True)
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Docker could not install the Assistant",
                code="docker-install-failed",
            ) from exc

    def _replace_unready_assistant(self, capsule_id: str, spec: AssistantSpec, network, existing) -> None:
        # Hello Pulse is the only explicitly stateless recovery target. Resolve its trusted image before
        # removing anything, then revalidate ownership to close the pull/remove race.
        image = self._trusted_image(spec)
        self._validate_container(existing, capsule_id, spec, network.name)
        try:
            existing.remove(force=True)
        except DockerException as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Docker could not replace the Assistant",
                code="docker-remove-failed",
            ) from exc
        self._create_assistant_container(capsule_id, spec, network, image)

    def install_assistant(self, capsule_id: str, assistant_id: str) -> dict[str, object]:
        capsule_id = validate_capsule_id(capsule_id)
        spec = self._resolve(assistant_id)
        with self._lock(capsule_id):
            network = self._network(capsule_id)
            existing = self._assistant_container(capsule_id, assistant_id, required=False)
            if existing is not None:
                self._validate_container(existing, capsule_id, spec, network.name)
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
                    self._replace_unready_assistant(capsule_id, spec, network, existing)
                return {"assistant": assistant_id, "installed": False}

            image = self._trusted_image(spec)
            self._create_assistant_container(capsule_id, spec, network, image)
            return {"assistant": assistant_id, "installed": True}

    def uninstall_assistant(self, capsule_id: str, assistant_id: str) -> dict[str, object]:
        capsule_id = validate_capsule_id(capsule_id)
        spec = self._resolve(assistant_id)
        with self._lock(capsule_id):
            network = self._network(capsule_id)
            container = self._assistant_container(capsule_id, assistant_id, required=False)
            if container is None:
                return {"assistant": assistant_id, "uninstalled": False}
            self._validate_container(container, capsule_id, spec, network.name)
            try:
                container.remove(force=True)
            except DockerException as exc:
                raise ApiProblem(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Docker could not uninstall the Assistant",
                    code="docker-remove-failed",
                ) from exc
            self._blocked_power_workloads.discard(container.id)
            return {"assistant": assistant_id, "uninstalled": True}

    def invoke(self, capsule_id: str, assistant_id: str, power: str, payload: object) -> dict[str, object]:
        capsule_id = validate_capsule_id(capsule_id)
        spec = self._resolve(assistant_id)
        power_spec = spec.powers.get(power)
        if power_spec is None:
            raise ApiProblem(HTTPStatus.NOT_FOUND, "Power is not declared", code="power-not-declared")
        try:
            safe_payload = validate_power_input(assistant_id, power, payload)
        except ValueError as exc:
            raise ApiProblem(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc), code="invalid-power-input") from exc
        with self._lock(capsule_id):
            network = self._network(capsule_id)
            container = self._assistant_container(capsule_id, assistant_id)
            self._validate_container(container, capsule_id, spec, network.name)
            if container.id in self._blocked_power_workloads:
                raise ApiProblem(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Assistant Power execution is blocked until this Assistant is reinstalled",
                    code="assistant-power-blocked",
                )
            container.reload()
            if container.status != "running":
                raise ApiProblem(HTTPStatus.CONFLICT, "Assistant is not running", code="assistant-not-running")
            local_audit.record(
                "assistant-power",
                result="ok",
                capsule=capsule_id,
                assistant=assistant_id,
                detail=f"started:{power}",
            )
            try:
                raw_result = self._rpc(container, spec, power_spec.method, power_spec.path, safe_payload)
            except ApiProblem:
                local_audit.record(
                    "assistant-power",
                    result="error",
                    capsule=capsule_id,
                    assistant=assistant_id,
                    detail=f"failed:{power}",
                )
                raise
        try:
            result = validate_power_output(assistant_id, power, raw_result)
        except ValueError as exc:
            local_audit.record(
                "assistant-power",
                result="error",
                capsule=capsule_id,
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
            capsule=capsule_id,
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

    def destroy_capsule(self, capsule_id: str) -> dict[str, object]:
        capsule_id = validate_capsule_id(capsule_id)
        self._cancel_chat_for_destroy(capsule_id)

        chat_lock = self._chat_lock(capsule_id)
        if not chat_lock.acquire(timeout=30):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "active Team chat did not stop in time",
                code="chat-active",
            )
        try:
            with self._lock(capsule_id):
                network = self._network(capsule_id, required=False)
                try:
                    containers = self.client.containers.list(**self._assistant_filters(capsule_id))
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
                            "Capsule resources failed their ownership contract",
                            code="ownership-conflict",
                        )
                    self._validate_container(container, capsule_id, spec, network.name)

                if network is not None:
                    thread_id = _brain_thread_id(self.space_id, capsule_id, network.id)
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
                    try:
                        container.remove(force=True)
                    except DockerException as exc:
                        raise ApiProblem(
                            HTTPStatus.SERVICE_UNAVAILABLE,
                            "Docker could not destroy the Capsule",
                            code="docker-remove-failed",
                        ) from exc
                    self._blocked_power_workloads.discard(container.id)
                    removed += 1

                if network is None:
                    try:
                        storage_removed = self.storage.destroy(capsule_id)
                    except capsule_storage.StorageError as exc:
                        self._raise_storage_problem(exc)
                    try:
                        self.inference_store.delete(capsule_id)
                    except inference_config.InferenceConfigError as exc:
                        self._raise_inference_problem(exc)
                    return {
                        "id": capsule_id,
                        "destroyed": False,
                        "assistants_removed": removed,
                        "storage_removed": storage_removed,
                    }
                try:
                    storage_removed = self.storage.destroy(capsule_id)
                except capsule_storage.StorageError as exc:
                    self._raise_storage_problem(exc)
                try:
                    self.inference_store.delete(capsule_id)
                except inference_config.InferenceConfigError as exc:
                    self._raise_inference_problem(exc)
                try:
                    network.remove()
                except DockerException as exc:
                    raise ApiProblem(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "Docker could not destroy the Capsule",
                        code="docker-remove-failed",
                    ) from exc
                return {
                    "id": capsule_id,
                    "destroyed": True,
                    "assistants_removed": removed,
                    "storage_removed": storage_removed,
                }
        finally:
            chat_lock.release()

    def _validate_reset_container(self, container) -> None:
        container.reload()
        labels = container.attrs.get("Config", {}).get("Labels") or {}
        capsule_id = labels.get(CAPSULE_LABEL)
        assistant_id = labels.get(ASSISTANT_LABEL)
        if (
            not isinstance(capsule_id, str)
            or len(capsule_id) > MAX_CAPSULE_ID_LENGTH
            or _CAPSULE_ID.fullmatch(capsule_id) is None
            or not isinstance(assistant_id, str)
            or len(assistant_id) > MAX_ASSISTANT_ID_LENGTH
            or _ASSISTANT_ID.fullmatch(assistant_id) is None
            or not isinstance(labels.get(IMAGE_LABEL), str)
            or not self._labels_include(labels, self._base_labels(capsule_id, "assistant"))
            or container.name != self._container_name(capsule_id, assistant_id)
        ):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "a labeled Space resource failed its ownership contract",
                code="ownership-conflict",
            )

    def reset_space(self) -> dict[str, object]:
        """Remove every exactly owned workload/network without accepting resource ids."""
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
                    f"{KIND_LABEL}=capsule",
                ]
            }
            try:
                containers = self.client.containers.list(all=True, filters=assistant_filters)
                networks = self.client.networks.list(filters=network_filters)
                for container in containers:
                    self._validate_reset_container(container)
                for network in networks:
                    labels = network.attrs.get("Labels") or {}
                    capsule_id = labels.get(CAPSULE_LABEL)
                    if not isinstance(capsule_id, str):
                        raise ApiProblem(
                            HTTPStatus.CONFLICT,
                            "a labeled Space resource failed its ownership contract",
                            code="ownership-conflict",
                        )
                    validate_capsule_id(capsule_id)
                    self._validate_network(network, capsule_id)
                for container in containers:
                    container.remove(force=True)
                    self._blocked_power_workloads.discard(container.id)
                storage_removed = self.storage.destroy_all()
                for network in networks:
                    capsule_id = network.attrs["Labels"][CAPSULE_LABEL]
                    self.inference_store.delete(capsule_id)
                for network in networks:
                    network.remove()
            except ApiProblem:
                raise
            except capsule_storage.StorageError as exc:
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
                "capsules_removed": len(networks),
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
        values = self.headers.get_all("Authorization", failobj=[])
        expected = f"Bearer {self.server.token}"
        return len(values) == 1 and hmac.compare_digest(values[0], expected)

    def _send(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
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
        if self.headers.get_all("Transfer-Encoding", failobj=[]):
            raise ApiProblem(HTTPStatus.BAD_REQUEST, "chunked requests are not accepted", code="chunked-request")
        lengths = self.headers.get_all("Content-Length", failobj=[])
        if len(lengths) != 1:
            raise ApiProblem(HTTPStatus.LENGTH_REQUIRED, "one Content-Length is required", code="content-length")
        try:
            length = int(lengths[0])
        except ValueError as exc:
            raise ApiProblem(HTTPStatus.BAD_REQUEST, "invalid Content-Length", code="content-length") from exc
        if length < 2 or length > max_bytes:
            raise ApiProblem(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request is too large", code="body-too-large")
        content_types = self.headers.get_all("Content-Type", failobj=[])
        if len(content_types) != 1:
            raise ApiProblem(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "application/json is required", code="content-type")
        content_type = content_types[0].split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise ApiProblem(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "application/json is required", code="content-type")
        try:
            raw_body = self.rfile.read(length)
            if len(raw_body) != length:
                raise ValueError("short request body")
            body = json.loads(raw_body)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ApiProblem(HTTPStatus.BAD_REQUEST, "invalid JSON body", code="invalid-json") from exc
        if not isinstance(body, dict):
            raise ApiProblem(HTTPStatus.UNPROCESSABLE_ENTITY, "a JSON object is required", code="invalid-body")
        return body

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

    def _capsule_create_body(self) -> str:
        body = self._body()
        if set(body) != {"name"}:
            raise ApiProblem(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "Capsule creation requires only a name",
                code="invalid-body",
            )
        return validate_capsule_name(body["name"])

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
        if self.headers.get_all("Transfer-Encoding", failobj=[]):
            raise ApiProblem(HTTPStatus.BAD_REQUEST, "this request cannot have a body", code="unexpected-body")
        lengths = self.headers.get_all("Content-Length", failobj=[])
        if len(lengths) > 1:
            raise ApiProblem(HTTPStatus.BAD_REQUEST, "invalid Content-Length", code="content-length")
        if lengths:
            try:
                length = int(lengths[0])
            except ValueError as exc:
                raise ApiProblem(HTTPStatus.BAD_REQUEST, "invalid Content-Length", code="content-length") from exc
            if length != 0:
                raise ApiProblem(HTTPStatus.BAD_REQUEST, "this request cannot have a body", code="unexpected-body")

    def _path_parts(self) -> list[str]:
        if len(self.path.encode("utf-8", "replace")) > MAX_PATH_BYTES:
            raise ApiProblem(HTTPStatus.URI_TOO_LONG, "request path is too long", code="path-too-long")
        parsed = urlsplit(self.path)
        if parsed.query or parsed.fragment or "%" in parsed.path:
            raise ApiProblem(HTTPStatus.BAD_REQUEST, "query and encoded paths are not accepted", code="invalid-path")
        return [part for part in parsed.path.split("/") if part]

    def _fixed_route(
        self, parts: list[str]
    ) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        controller = self.server.controller
        if self.command == "GET" and parts == ["healthz"]:
            return HTTPStatus.OK, controller.health(), "health", None, None
        if self.command == "GET" and parts == ["v1", "assistants"]:
            return HTTPStatus.OK, controller.list_registry(), "registry-list", None, None
        if self.command == "GET" and parts == ["v1", "capsules"]:
            return HTTPStatus.OK, controller.list_capsules(), "capsule-list", None, None
        if self.command == "DELETE" and parts == ["v1", "space"]:
            return HTTPStatus.OK, controller.reset_space(), "space-reset", None, None
        return None

    def _file_route(self, parts: list[str]) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        if len(parts) not in {4, 5} or parts[:2] != ["v1", "capsules"] or parts[3] != "files":
            return None
        controller = self.server.controller
        capsule_id = validate_capsule_id(parts[2])
        if len(parts) == 4 and self.command == "GET":
            return HTTPStatus.OK, controller.list_files(capsule_id), "file-list", capsule_id, None
        if len(parts) == 4 and self.command == "POST":
            if not _FILE_UPLOAD_SLOTS.acquire(blocking=False):
                raise ApiProblem(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    "another Capsule file upload is in progress",
                    code="file-upload-busy",
                )
            try:
                filename, content, media_type = self._file_body()
                return (
                    HTTPStatus.OK,
                    controller.put_file(capsule_id, filename, content, media_type),
                    "file-upload",
                    capsule_id,
                    None,
                )
            finally:
                _FILE_UPLOAD_SLOTS.release()
        if len(parts) == 5 and self.command == "DELETE":
            return (
                HTTPStatus.OK,
                controller.delete_file(capsule_id, parts[4]),
                "file-delete",
                capsule_id,
                None,
            )
        return None

    def _inference_route(
        self, parts: list[str]
    ) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        if len(parts) != 4 or parts[:2] != ["v1", "capsules"] or parts[3] != "inference":
            return None
        capsule_id = validate_capsule_id(parts[2])
        if self.command == "GET":
            return (
                HTTPStatus.OK,
                self.server.controller.inference_status(capsule_id),
                "inference-status",
                capsule_id,
                None,
            )
        if self.command == "PUT":
            return (
                HTTPStatus.OK,
                self.server.controller.configure_inference(capsule_id, self._body()),
                "inference-configure",
                capsule_id,
                None,
            )
        return None

    def _chat_route(self, parts: list[str]) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        if len(parts) not in {4, 5} or parts[:2] != ["v1", "capsules"] or parts[3] != "chat":
            return None
        capsule_id = validate_capsule_id(parts[2])
        if len(parts) == 4 and self.command == "POST":
            provider, api_key = self._model_credential_headers()
            body = self._body(max_bytes=MAX_CHAT_BODY_BYTES)
            return (
                HTTPStatus.OK,
                self.server.controller.chat(capsule_id, body, provider, api_key),
                "chat",
                capsule_id,
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
                self.server.controller.stop_chat(capsule_id),
                "chat-stop",
                capsule_id,
                None,
            )
        return None

    def _capsule_route(
        self, parts: list[str]
    ) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None] | None:
        if len(parts) == 4 and parts[:2] == ["v1", "capsules"] and parts[3] == "create":
            capsule_id = validate_capsule_id(parts[2])
            if self.command == "POST":
                return (
                    HTTPStatus.OK,
                    self.server.controller.create_capsule(capsule_id, self._capsule_create_body()),
                    "capsule-create",
                    capsule_id,
                    None,
                )
        if len(parts) == 3 and parts[:2] == ["v1", "capsules"] and self.command == "DELETE":
            capsule_id = validate_capsule_id(parts[2])
            return (
                HTTPStatus.OK,
                self.server.controller.destroy_capsule(capsule_id),
                "capsule-destroy",
                capsule_id,
                None,
            )
        return None

    def _route(self) -> tuple[HTTPStatus, dict[str, object], str, str | None, str | None]:
        parts = self._path_parts()
        controller = self.server.controller
        if self.command not in {"POST", "PUT"}:
            self._reject_body()

        fixed_route = self._fixed_route(parts)
        if fixed_route is not None:
            return fixed_route
        file_route = self._file_route(parts)
        if file_route is not None:
            return file_route
        inference_route = self._inference_route(parts)
        if inference_route is not None:
            return inference_route
        chat_route = self._chat_route(parts)
        if chat_route is not None:
            return chat_route
        capsule_route = self._capsule_route(parts)
        if capsule_route is not None:
            return capsule_route
        if len(parts) == 4 and parts[:2] == ["v1", "capsules"] and parts[3] == "assistants":
            capsule_id = validate_capsule_id(parts[2])
            if self.command == "GET":
                return HTTPStatus.OK, controller.list_assistants(capsule_id), "assistant-list", capsule_id, None
            if self.command == "POST":
                assistant_id = self._install_body()
                return (
                    HTTPStatus.OK,
                    controller.install_assistant(capsule_id, assistant_id),
                    "assistant-install",
                    capsule_id,
                    assistant_id,
                )
        if len(parts) == 5 and parts[:2] == ["v1", "capsules"] and parts[3] == "assistants":
            capsule_id = validate_capsule_id(parts[2])
            assistant_id = parts[4]
            if self.command == "DELETE":
                return (
                    HTTPStatus.OK,
                    controller.uninstall_assistant(capsule_id, assistant_id),
                    "assistant-uninstall",
                    capsule_id,
                    assistant_id,
                )
        if (
            len(parts) == 7
            and parts[:2] == ["v1", "capsules"]
            and parts[3] == "assistants"
            and parts[5] == "powers"
            and self.command == "POST"
        ):
            capsule_id = validate_capsule_id(parts[2])
            assistant_id = parts[4]
            power = parts[6]
            payload = self._body()
            return (
                HTTPStatus.OK,
                controller.invoke(capsule_id, assistant_id, power, payload),
                "assistant-invoke",
                capsule_id,
                assistant_id,
            )
        raise ApiProblem(HTTPStatus.NOT_FOUND, "route not found", code="route-not-found")

    def _handle(self) -> None:
        self.close_connection = True
        operation = "request"
        capsule_id = None
        assistant_id = None
        if not self._authorized():
            trace_id = local_audit.record("authentication", result="denied", detail="invalid-bearer")
            self._send(HTTPStatus.UNAUTHORIZED, {"error": "authentication required", "trace_id": trace_id})
            return
        try:
            status, payload, operation, capsule_id, assistant_id = self._route()
            trace_id = local_audit.record(
                operation,
                result="ok",
                capsule=capsule_id,
                assistant=assistant_id,
            )
            payload["trace_id"] = trace_id
            self._send(status, payload)
        except ApiProblem as exc:
            trace_id = local_audit.record(
                operation,
                result="denied" if exc.status < 500 else "error",
                capsule=capsule_id,
                assistant=assistant_id,
                detail=exc.code,
            )
            self._send(exc.status, {"error": exc.message, "trace_id": trace_id})
        except DockerException:
            trace_id = local_audit.record(operation, result="error", detail="docker-error")
            self._send(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Docker is unavailable", "trace_id": trace_id})
        except Exception:
            trace_id = local_audit.record(operation, result="error", detail="internal-error")
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal error", "trace_id": trace_id})

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
        storage = capsule_storage.CapsuleStorage(STORAGE_ROOT)
        controller = LocalController(client, space_id, registry, storage)
        server = BoundedServer(("0.0.0.0", LISTEN_PORT), Handler, controller, token)
    except (KeyError, RegistryError, RuntimeError, DockerException) as exc:
        print(f"capsule-driver-local: startup failed: {exc}", file=sys.stderr, flush=True)
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
