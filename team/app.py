"""team-driver — a socket-holding sidecar dedicated to Team lifecycle.

Besides shimpz-driver, this is the ONLY container holding /var/run/docker.sock — and it exposes ONLY
named operations (create/list/status/logs/stop/start/restart/destroy), never a generic Docker
passthrough. A Team is one isolated `shimpz-brain`: its OWN internal network, its OWN config+workspace
volumes, and a SCOPED Postgres database (provisioned via pg-driver — this driver never holds the
superuser). Every mutating call is bearer-gated → validated → mutated → audited (trace_id returned).
A compromised caller can only ever request what validate.py permits.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import http.client
import ipaddress
import json
import math
import os
import secrets
import select
import socket
import struct
import threading
import time
import weakref
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import accounts_client
import assistant_chat
import assistant_contract
import audit
import brain_credentials_client
import brain_runtime_client
import brain_runtime_token_store
import chat_orchestrator
import cleanup_state
import docker
import docker.errors
import docker.utils.socket as docker_socket
import inference_config
import manifests
import marketplace
import marketplace_image
import network_policy
import pgdriver_client
import power_journal
import r2driver_client
import team_storage
import token_store
import validate

ALL_INTERFACES = str(ipaddress.IPv4Address(0))


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


LISTEN_PORT = int(os.environ.get("SHIMPZ_TEAMDRIVER_PORT", "7077"))
# The host has 125 GiB and each Team has a 2 GiB hard ceiling: 32 leaves roughly half the host for
# the platform, installed apps and Docker overhead. Operators may lower it, but public callers never
# choose either quota.
MAX_TEAMS = _positive_int_env("SHIMPZ_MAX_TEAMS", 32)
MAX_TEAMS_PER_OWNER = _positive_int_env("SHIMPZ_MAX_TEAMS_PER_OWNER", 1)
# Per-team app allowance — an owner can't exhaust the host by installing without bound either.
MAX_APPS_PER_TEAM = _positive_int_env("SHIMPZ_MAX_APPS_PER_TEAM", 20)
GLOBAL_MEMORY_BUDGET_BYTES = manifests.hard_memory_bytes(
    os.environ.get("SHIMPZ_TEAM_GLOBAL_MEM_BUDGET", "64g"),
    setting="SHIMPZ_TEAM_GLOBAL_MEM_BUDGET",
)
OWNER_MEMORY_BUDGET_BYTES = manifests.hard_memory_bytes(
    os.environ.get("SHIMPZ_TEAM_OWNER_MEM_BUDGET", "8g"),
    setting="SHIMPZ_TEAM_OWNER_MEM_BUDGET",
)
_LARGEST_RESOURCE_LIMIT = max(manifests.MEM_LIMIT_BYTES, manifests.APP_MEM_LIMIT_BYTES)
if GLOBAL_MEMORY_BUDGET_BYTES < _LARGEST_RESOURCE_LIMIT:
    raise ValueError("SHIMPZ_TEAM_GLOBAL_MEM_BUDGET is smaller than one Team resource")
if not _LARGEST_RESOURCE_LIMIT <= OWNER_MEMORY_BUDGET_BYTES <= GLOBAL_MEMORY_BUDGET_BYTES:
    raise ValueError("SHIMPZ_TEAM_OWNER_MEM_BUDGET must fit one resource and the global memory budget")
MAX_JSON_BODY_BYTES = max(1024, int(os.environ.get("SHIMPZ_TEAM_MAX_JSON_BODY_BYTES", str(128 * 1024))))
MAX_DRIVER_JSON_BODY_BYTES = 64 * 1024
CREATE_RATE_LIMIT = _positive_int_env("SHIMPZ_TEAM_CREATE_RATE_LIMIT", 5)
CREATE_RATE_WINDOW_SECONDS = _positive_int_env("SHIMPZ_TEAM_CREATE_RATE_WINDOW_SECONDS", 3600)
INSTALL_RATE_LIMIT = _positive_int_env("SHIMPZ_TEAM_INSTALL_RATE_LIMIT", 20)
INSTALL_RATE_WINDOW_SECONDS = _positive_int_env("SHIMPZ_TEAM_INSTALL_RATE_WINDOW_SECONDS", 3600)
CHAT_RATE_LIMIT = _positive_int_env("SHIMPZ_TEAM_CHAT_RATE_LIMIT", 30)
CHAT_RATE_WINDOW_SECONDS = _positive_int_env("SHIMPZ_TEAM_CHAT_RATE_WINDOW_SECONDS", 60)
FILE_UPLOAD_RATE_LIMIT = _positive_int_env("SHIMPZ_TEAM_FILE_UPLOAD_RATE_LIMIT", 60)
FILE_UPLOAD_RATE_WINDOW_SECONDS = _positive_int_env("SHIMPZ_TEAM_FILE_UPLOAD_RATE_WINDOW_SECONDS", 3600)
MAX_HTTP_CONCURRENCY = _positive_int_env("SHIMPZ_TEAM_MAX_HTTP_CONCURRENCY", 64)
HTTP_CONNECTION_TIMEOUT_SECONDS = _positive_int_env("SHIMPZ_TEAM_HTTP_CONNECTION_TIMEOUT_SECONDS", 30)
# Same volume app-egress-proxy reads (<token>.json allowlists) — shared with shimpz-driver by design:
# ONE proxy serves every token-gated app, team-scoped or not, each confined to its own hosts.
APP_EGRESS_POLICY_DIR = Path(os.environ.get("SHIMPZ_APP_EGRESS_POLICY_DIR", "/app-egress-policy"))
TEAM_STORAGE_ROOT = Path("/var/lib/team-driver/storage")
POWER_JOURNAL_PATH = Path(
    os.environ.get(
        "SHIMPZ_TEAM_POWER_JOURNAL_PATH",
        "/var/lib/team-driver/power-journal/journal.sqlite3",
    )
)
HEALTH_RETRIES = int(os.environ.get("SHIMPZ_HEALTH_RETRIES", "40"))
HEALTH_DELAY_SECONDS = float(os.environ.get("SHIMPZ_HEALTH_DELAY_SECONDS", "1.5"))

_docker = docker.from_env()
_token = token_store.ensure_token()

# Per-team lock: create/destroy of the SAME team must serialize; different teams run parallel.
# Weak maps retain one lock exactly while a holder or waiter has a strong reference. After destroy (or
# any other terminal operation), the final holder releases its reference and the TEAM_ID disappears without
# ever allowing an old locked object and a new unlocked object to coexist.
_locks_guard = threading.Lock()
_locks: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
_chat_locks_guard = threading.Lock()
_chat_locks: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
_active_chat_guard = threading.Lock()
_active_chat_tokens: dict[str, str] = {}
_active_chat_container_ids: dict[str, str] = {}
_active_power_container_ids: dict[str, tuple[str, str]] = {}
_blocked_power_workloads: set[tuple[str, str]] = set()
_cancelled_chat_tokens: set[str] = set()
# The capacity lock protects only inventory + reservation mutations. Slow provisioning is represented
# by `_capacity_reservations` and runs after this lock is released.
_capacity_lock = threading.Lock()
_storage_lock = threading.Lock()
_storage_instance: team_storage.TeamStorage | None = None
_power_journal_lock = threading.Lock()
_power_journal_instance: power_journal.PowerJournal | None = None
_brain_runtime = brain_runtime_client.BrainRuntimeClient()


def _validated_team_name(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 80
        or value.strip() != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("Team name must contain 1 to 80 trimmed characters")
    return value


def _team_name_from_anchor(container) -> str:
    try:
        return _validated_team_name((container.labels or {}).get("team.name"))
    except ValueError as exc:
        raise ApiError(HTTPStatus.CONFLICT, "Team identity failed its persisted contract") from exc


_inference_store = inference_config.InferenceConfigStore()


def _storage() -> team_storage.TeamStorage:
    global _storage_instance
    with _storage_lock:
        if _storage_instance is None:
            _storage_instance = team_storage.TeamStorage(TEAM_STORAGE_ROOT)
        return _storage_instance


def _power_execution_journal() -> power_journal.PowerJournal:
    """Open the private journal only when a Power batch or generation needs it."""
    global _power_journal_instance
    with _power_journal_lock:
        if _power_journal_instance is None:
            _power_journal_instance = power_journal.PowerJournal(POWER_JOURNAL_PATH)
        return _power_journal_instance


def _lock_for(team_id: str) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(team_id)
        if lock is None:
            lock = threading.Lock()
            _locks[team_id] = lock
        return lock


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class _UnsupportedAssistantRpcPathError(RuntimeError):
    """The fixed Assistant RPC adapter rejected a path it does not implement."""


def _brain_thread_id(team_id: str, anchor_id: str) -> str:
    """Bind hosted conversation state to one immutable Team lifecycle."""
    if (
        not isinstance(team_id, str)
        or validate.TEAM_ID_RE.fullmatch(team_id) is None
        or not isinstance(anchor_id, str)
        or not 12 <= len(anchor_id) <= 64
        or any(character not in "0123456789abcdef" for character in anchor_id)
    ):
        raise ApiError(HTTPStatus.CONFLICT, "Team identity failed its persisted contract")
    return f"hosted:{team_id}:{anchor_id}:default"


class _FixedWindowRateLimiter:
    """Thread-safe fixed-window admission with deterministic time injection for contract tests."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        if limit < 1 or window_seconds < 1:
            raise ValueError("rate limit and window must be positive")
        self.limit = limit
        self.window_seconds = window_seconds
        self._guard = threading.Lock()
        self._counts: dict[str, tuple[int, int]] = {}
        self._last_bucket: int | None = None

    def consume(self, key: str, *, now: float | None = None) -> int:
        """Consume one event; return zero when allowed or whole retry-after seconds when denied."""
        current = time.monotonic() if now is None else now
        bucket = math.floor(current / self.window_seconds)
        with self._guard:
            if bucket != self._last_bucket:
                self._counts = {stored_key: value for stored_key, value in self._counts.items() if value[0] == bucket}
                self._last_bucket = bucket
            stored_bucket, count = self._counts.get(key, (bucket, 0))
            if stored_bucket != bucket:
                count = 0
            if count >= self.limit:
                boundary = (bucket + 1) * self.window_seconds
                return max(1, math.ceil(boundary - current))
            self._counts[key] = (bucket, count + 1)
        return 0


_rate_limiters = {
    "create": _FixedWindowRateLimiter(CREATE_RATE_LIMIT, CREATE_RATE_WINDOW_SECONDS),
    "install": _FixedWindowRateLimiter(INSTALL_RATE_LIMIT, INSTALL_RATE_WINDOW_SECONDS),
    "chat": _FixedWindowRateLimiter(CHAT_RATE_LIMIT, CHAT_RATE_WINDOW_SECONDS),
    "stream": _FixedWindowRateLimiter(CHAT_RATE_LIMIT, CHAT_RATE_WINDOW_SECONDS),
    "stop": _FixedWindowRateLimiter(CHAT_RATE_LIMIT, CHAT_RATE_WINDOW_SECONDS),
    "file_upload": _FixedWindowRateLimiter(FILE_UPLOAD_RATE_LIMIT, FILE_UPLOAD_RATE_WINDOW_SECONDS),
}

_file_upload_slots = threading.BoundedSemaphore(2)


def _rate_key(principal: tuple[str, str | None]) -> str:
    kind, account_id = principal
    return f"{kind}:{account_id or 'operator'}"


def _enforce_rate(operation: str, principal: tuple[str, str | None]) -> None:
    retry_after = _rate_limiters[operation].consume(_rate_key(principal))
    if retry_after:
        raise ApiError(
            HTTPStatus.TOO_MANY_REQUESTS,
            f"{operation} rate limit exceeded; retry in {retry_after}s",
        )


def _chat_lock_for(team_id: str) -> threading.Lock:
    with _chat_locks_guard:
        lock = _chat_locks.get(team_id)
        if lock is None:
            lock = threading.Lock()
            _chat_locks[team_id] = lock
        return lock


def _clear_team_id_runtime_state(team_id: str) -> None:
    """Forget terminal in-memory state without deleting a lock that another request references."""
    with _active_chat_guard:
        token = _active_chat_tokens.pop(team_id, None)
        _active_chat_container_ids.pop(team_id, None)
        _active_power_container_ids.pop(team_id, None)
        for blocked in tuple(_blocked_power_workloads):
            if blocked[0] == team_id:
                _blocked_power_workloads.discard(blocked)
        if token is not None:
            _cancelled_chat_tokens.discard(token)


def _token_cancelled(token: str) -> bool:
    with _active_chat_guard:
        return token in _cancelled_chat_tokens


def _commit_chat_terminal(team_id: str, token: str) -> bool:
    """Linearization point: False means a user Stop acquired the token first."""
    with _active_chat_guard:
        if token in _cancelled_chat_tokens:
            return False
        if _active_chat_tokens.get(team_id) == token:
            _active_chat_tokens.pop(team_id, None)
            _active_chat_container_ids.pop(team_id, None)
        return True


@contextlib.contextmanager
def _exclusive_chat_turn(team_id: str, lease: _AuthorizationLease):
    """Hold one Controller-owned agent turn without creating a process in the Team."""
    lock = _chat_lock_for(team_id)
    if not lock.acquire(blocking=False):
        raise ApiError(HTTPStatus.CONFLICT, f"team {team_id!r} already has an active chat turn")
    try:
        container = _require_current_authorization(team_id, lease)
        container.reload()
        if container.status != "running":
            raise ApiError(HTTPStatus.CONFLICT, f"team {team_id!r} is not running (status={container.status})")
    except BaseException:
        lock.release()
        raise
    token = secrets.token_hex(16)
    with _active_chat_guard:
        _active_chat_tokens[team_id] = token
        _active_chat_container_ids[team_id] = container.id
    try:
        yield token, container
    finally:
        with _active_chat_guard:
            _active_chat_tokens.pop(team_id, None)
            _active_chat_container_ids.pop(team_id, None)
            _active_power_container_ids.pop(team_id, None)
            _cancelled_chat_tokens.discard(token)
        lock.release()


# ── docker helpers ───────────────────────────────────────────────────────────
def _get_container(name: str):
    try:
        return _docker.containers.get(name)
    except docker.errors.NotFound:
        return None


def _require_team_runtime() -> None:
    """Fail closed unless Docker preserves the complete hostile-tenant daemon posture."""
    try:
        info = _docker.info()
    except docker.errors.DockerException as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "cannot verify Docker isolation posture") from exc
    if not isinstance(info, dict):
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "cannot verify Docker isolation posture")
    if not network_policy.daemon_runtime_registration_valid(info, manifests.RUNTIME, manifests.RUNTIME_PATH):
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            f"required Team runtime {manifests.RUNTIME!r} is not loaded from {manifests.RUNTIME_PATH!r}",
        )
    if not network_policy.daemon_security_options_valid(info):
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "required Docker built-in seccomp and AppArmor defaults are unavailable",
        )


def _team_runtime(container) -> str:
    """Return Docker's immutable runtime selection for one existing Team."""
    try:
        container.reload()
    except docker.errors.DockerException as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "cannot verify Team runtime isolation") from exc
    runtime = container.attrs.get("HostConfig", {}).get("Runtime")
    return str(runtime or "runc")


def _trusted_image_id(image_ref: str) -> str:
    """Resolve one release-owned local reference to the immutable ID Engine will execute."""
    if not isinstance(image_ref, str) or not image_ref:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Team isolation is blocked: untrusted workload image role")
    try:
        image = _docker.images.get(image_ref)
        image_id = image.id
    except (AttributeError, docker.errors.DockerException) as exc:
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Team isolation is blocked: trusted workload image is unavailable",
        ) from exc
    if not isinstance(image_id, str) or not image_id:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Team isolation is blocked: invalid workload image identity")
    return image_id


def _prepare_marketplace_image(spec: marketplace.AppSpec) -> None:
    """Materialize and prove only registry-owned digest artifacts before a new App can run."""
    if not marketplace.is_digest_image(spec.image):
        return
    try:
        marketplace_image.ensure_digest_artifact(_docker.images, spec)
    except marketplace_image.ImageTrustError as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, str(exc)) from exc


def _trusted_workload_image(container, team_id: str) -> tuple[str, str]:
    """Resolve this exact workload role's configured ref to the currently trusted immutable ID."""
    labels = container.attrs.get("Config", {}).get("Labels", {})
    if network_policy.brain_identity_valid(container.attrs, team_id):
        provider = labels.get("team.brain")
        provider_spec = manifests.BRAINS.get(provider) if isinstance(provider, str) else None
        image_ref = provider_spec.get("image") if provider_spec is not None else None
    else:
        app_id = labels.get("team.app")
        app_spec = marketplace.APPS.get(app_id) if isinstance(app_id, str) else None
        image_ref = app_spec.image if app_spec is not None else None
    if not isinstance(image_ref, str) or not image_ref:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Team isolation is blocked: untrusted workload image role")
    return image_ref, _trusted_image_id(image_ref)


def _require_team_isolation_mode(container, *, require_running: bool) -> None:
    """Validate exact static posture, plus live network membership whenever the workload is running."""
    runtime = _team_runtime(container)
    if runtime != manifests.RUNTIME:
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            f"Team isolation is blocked: required runtime {manifests.RUNTIME!r}, found {runtime!r}; "
            "destroy and recreate the Team",
        )
    state = container.attrs.get("State")
    running = state.get("Running") if isinstance(state, dict) else None
    if not isinstance(running, bool) or (require_running and not running):
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Team isolation is blocked: workload running state cannot be proved",
        )
    labels = container.attrs.get("Config", {}).get("Labels", {})
    team_id = labels.get("team.id") if isinstance(labels, dict) else None
    if not isinstance(team_id, str) or not team_id:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Team isolation is blocked: invalid workload identity")
    image_ref, image_id = _trusted_workload_image(container, team_id)
    if not network_policy.workload_security_valid(
        container.attrs,
        team_id,
        manifests.RUNTIME,
        expected_image_ref=image_ref,
        expected_image_id=image_id,
    ):
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Team isolation is blocked: workload security or network attachment drifted; destroy and recreate the Team",
        )
    kind = network_policy.CORE_KIND
    try:
        network = _docker.networks.get(network_policy.network_name(team_id, kind))
    except docker.errors.DockerException as exc:
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Team isolation is blocked: required network is missing",
        ) from exc
    _require_network_policy(
        network,
        team_id,
        kind,
        # Engine 29 omits created/stopped endpoints from network inspect while preserving their
        # exact attachments in container inspect. Static workload posture above proves those
        # endpoints; a running anchor must additionally be visible as the live Brain role.
        require_brain=running and network_policy.brain_identity_valid(container.attrs, team_id),
        require_dependencies=True,
    )
    if not network_policy.workload_endpoint_valid(
        network.attrs,
        container.attrs,
        team_id,
        kind,
    ):
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Team isolation is blocked: workload endpoint identity or aliases drifted",
        )
    if running and not network_policy.workload_live_membership_valid(
        network.attrs,
        container.attrs,
        team_id,
        kind,
    ):
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Team isolation is blocked: running workload is missing from its network inventory",
        )


def _require_team_isolation(container) -> None:
    """State-aware admission: stopped is exact/static; running additionally proves live membership."""
    _require_team_isolation_mode(container, require_running=False)


def _require_running_team_isolation(container) -> None:
    """Require a running workload and its complete live core-network membership."""
    _require_team_isolation_mode(container, require_running=True)


def _team_not_running(container) -> bool:
    """Return true only for an exact stopped state or an absent container identity."""
    try:
        container.reload()
    except docker.errors.NotFound:
        return True
    except docker.errors.DockerException:
        return False
    state = container.attrs.get("State")
    return isinstance(state, dict) and state.get("Running") is False


def _fail_stop_team(container, *, timeout: int = 10) -> None:
    """Actively stop/kill and prove a workload is no longer running."""
    try:
        container.stop(timeout=timeout)
    except docker.errors.NotFound:
        return
    except docker.errors.DockerException:
        pass
    if _team_not_running(container):
        return
    try:
        container.kill()
    except docker.errors.NotFound:
        return
    except docker.errors.DockerException:
        pass
    if not _team_not_running(container):
        raise ApiError(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Team isolation failed and the workload could not be proved stopped",
        )


def _remove_team_container(container, *, timeout: int = 10) -> bool:
    """Remove one workload by immutable ID, fail-stopping any survivor before returning false."""
    container_id = container.id
    try:
        _fail_stop_team(container, timeout=timeout)
    except ApiError:
        return False
    with contextlib.suppress(docker.errors.DockerException):
        container.remove(force=True)
    try:
        survivor = _docker.containers.get(container_id)
    except docker.errors.NotFound:
        return True
    except docker.errors.DockerException:
        return False
    try:
        _fail_stop_team(survivor, timeout=timeout)
    except ApiError:
        return False
    with contextlib.suppress(docker.errors.DockerException):
        survivor.remove(force=True)
    try:
        _docker.containers.get(container_id)
    except docker.errors.NotFound:
        return True
    except docker.errors.DockerException:
        return False
    return False


def _start_team_with_isolation(container) -> None:
    """Prove a stopped workload, start it, then fail-stop unless live membership also proves."""
    _require_team_isolation(container)
    # Re-read Docker's daemon posture at the final mutation boundary. A registration or default-profile
    # drift after create/preflight must leave the hostile workload stopped.
    _require_team_runtime()
    try:
        container.start()
    except docker.errors.DockerException as exc:
        # Engine may have committed a start before the client observed its response. Treat every
        # start error as ambiguous and actively fail-stop instead of assuming the workload stayed down.
        _fail_stop_team(container)
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Team start could not be proved; workload was stopped",
        ) from exc
    try:
        _require_running_team_isolation(container)
    except ApiError:
        _fail_stop_team(container)
        raise


@dataclass(frozen=True)
class _MemoryUsage:
    total: int
    by_owner: dict[str, int]


@dataclass(frozen=True)
class _CapacityReservation:
    key: str
    owner: str
    memory_bytes: int
    team_slot: bool


@dataclass(frozen=True)
class _CleanupResult:
    """Proof that both runtime artifacts and scoped database state were removed."""

    artifacts_removed: bool
    db_dropped: bool

    @property
    def complete(self) -> bool:
        return self.artifacts_removed and self.db_dropped


_capacity_reservations: dict[str, _CapacityReservation] = {}


def _capacity_key(container) -> str:
    team_id = str(container.labels.get("team.id", ""))
    if container.labels.get("team.app.driver"):
        return f"app:{team_id}:{container.labels.get('team.app', '')}"
    return f"team:{team_id}"


def _admitted_resource_containers() -> list:
    """Return every hard-limited Team resource once, or fail closed on Docker inventory errors."""
    try:
        resources = {
            container.id: container
            for label in ("team.driver", "team.app.driver")
            for container in _docker.containers.list(all=True, filters={"label": label})
        }
    except docker.errors.DockerException as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "cannot verify Team memory inventory") from exc
    return list(resources.values())


def _memory_usage(*, exclude_keys: frozenset[str] = frozenset()) -> _MemoryUsage:
    """Count inspected Docker hard limits; zero/missing limits are unsafe and reject admission."""
    total = 0
    by_owner: dict[str, int] = defaultdict(int)
    for container in _admitted_resource_containers():
        if _capacity_key(container) in exclude_keys:
            # The corresponding in-flight reservation already accounts for this resource. Docker
            # exposes a newly-created container before provisioning/health commits, so counting both
            # here would spuriously halve capacity during slow creates.
            continue
        try:
            container.reload()
            raw_limit = container.attrs.get("HostConfig", {}).get("Memory")
        except (AttributeError, docker.errors.DockerException) as exc:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "cannot verify Team memory hard limits") from exc
        if isinstance(raw_limit, bool) or not isinstance(raw_limit, (int, float)) or raw_limit <= 0:
            raise ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                f"Team resource {container.name!r} has no verifiable memory hard limit",
            )
        limit = int(raw_limit)
        owner = str(container.labels.get("team.owner", ""))
        total += limit
        by_owner[owner] += limit
    return _MemoryUsage(total=total, by_owner=dict(by_owner))


def _physical_teams(*, exclude_keys: frozenset[str]) -> list:
    try:
        teams = _docker.containers.list(all=True, filters={"label": "team.driver"})
    except docker.errors.DockerException as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "cannot verify Team count inventory") from exc
    return [container for container in teams if _capacity_key(container) not in exclude_keys]


@contextlib.contextmanager
def _reserve_capacity(
    key: str,
    owner: str,
    requested: int,
    *,
    team_slot: bool,
):
    """Atomically reserve quota, then release the global lock before any slow provisioning.

    A resource can become visible in Docker while its PG/image/health transaction is still running.
    Inventory therefore excludes keys with live reservations and adds those reservations exactly once.
    Rollback completes before the `finally` drops the reservation, so another caller never observes a
    phantom free slot between the failed transaction and cleanup.
    """
    reservation = _CapacityReservation(
        key=key,
        owner=owner,
        memory_bytes=requested,
        team_slot=team_slot,
    )
    with _capacity_lock:
        if key in _capacity_reservations:
            raise ApiError(HTTPStatus.CONFLICT, "Team resource admission is already in progress")
        reserved_keys = frozenset(_capacity_reservations)
        if team_slot:
            physical = _physical_teams(exclude_keys=reserved_keys)
            team_reservations = [item for item in _capacity_reservations.values() if item.team_slot]
            current = len(physical) + len(team_reservations)
            if current >= MAX_TEAMS:
                raise ApiError(HTTPStatus.TOO_MANY_REQUESTS, f"team limit reached ({current}/{MAX_TEAMS})")
            owner_count = sum(container.labels.get("team.owner", "") == owner for container in physical) + sum(
                item.owner == owner for item in team_reservations
            )
            if owner_count >= MAX_TEAMS_PER_OWNER:
                raise ApiError(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    f"team limit reached for this owner ({owner_count}/{MAX_TEAMS_PER_OWNER})",
                )
        usage = _memory_usage(exclude_keys=reserved_keys)
        reserved_total = sum(item.memory_bytes for item in _capacity_reservations.values())
        committed_total = usage.total + reserved_total
        if committed_total + requested > GLOBAL_MEMORY_BUDGET_BYTES:
            detail = (
                f"global Team memory budget reached ({committed_total}/{GLOBAL_MEMORY_BUDGET_BYTES} bytes committed)"
            )
            raise ApiError(
                HTTPStatus.TOO_MANY_REQUESTS,
                detail,
            )
        owner_used = usage.by_owner.get(owner, 0) + sum(
            item.memory_bytes for item in _capacity_reservations.values() if item.owner == owner
        )
        if owner_used + requested > OWNER_MEMORY_BUDGET_BYTES:
            detail = (
                f"Team memory budget reached for this owner ({owner_used}/{OWNER_MEMORY_BUDGET_BYTES} bytes committed)"
            )
            raise ApiError(
                HTTPStatus.TOO_MANY_REQUESTS,
                detail,
            )
        _capacity_reservations[key] = reservation
    try:
        yield
    finally:
        with _capacity_lock:
            if _capacity_reservations.get(key) == reservation:
                _capacity_reservations.pop(key, None)


def _network_container_metadata(network) -> dict[str, dict]:
    try:
        network.reload()
        member_ids = network.attrs.get("Containers", {})
        if not isinstance(member_ids, dict):
            raise TypeError("invalid network member inventory")
        containers: dict[str, dict] = {}
        for container_id in member_ids:
            container = _docker.containers.get(container_id)
            container.reload()
            metadata = dict(container.attrs)
            metadata.setdefault("Id", container.id)
            metadata.setdefault("Name", f"/{container.name}")
            containers[container_id] = metadata
    except (AttributeError, TypeError, docker.errors.DockerException) as exc:
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "cannot verify Team network isolation",
        ) from exc
    else:
        return containers


def _require_network_policy(
    network,
    team_id: str,
    kind: str,
    *,
    require_brain: bool,
    require_dependencies: bool,
) -> None:
    containers = _network_container_metadata(network)
    if not network_policy.network_members_valid(
        network.attrs,
        containers,
        team_id,
        kind,
        require_brain=require_brain,
        require_dependencies=require_dependencies,
    ):
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            f"Team isolation is blocked: invalid or contaminated {kind} network",
        )


def _ensure_team_network_kind(team_id: str, kind: str):
    net_name = network_policy.network_name(team_id, kind)
    try:
        network = _docker.networks.get(net_name)
    except docker.errors.NotFound:
        try:
            network = _docker.networks.create(
                net_name,
                driver="bridge",
                internal=True,
                attachable=False,
                labels=network_policy.network_labels(team_id, kind),
            )
        except docker.errors.DockerException as exc:
            raise ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                f"could not create the Team {kind} network",
            ) from exc
    except docker.errors.DockerException as exc:
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            f"could not inspect the Team {kind} network",
        ) from exc
    _require_network_policy(
        network,
        team_id,
        kind,
        require_brain=False,
        require_dependencies=False,
    )
    return network


def _ensure_team_network(team_id: str):
    return _ensure_team_network_kind(team_id, network_policy.CORE_KIND)


def _already_connected(exc: docker.errors.APIError) -> bool:
    """True only for the ONE idempotent case: this container is already on this network (403)."""
    resp = exc.response
    return (
        resp is not None
        and resp.status_code == HTTPStatus.FORBIDDEN
        and "already exists in network" in (exc.explanation or "")
    )


def _safe_connect(network, container_name: str, *, aliases: list[str] | None = None, required: bool) -> None:
    try:
        container = _docker.containers.get(container_name)
    except docker.errors.NotFound as exc:
        if required:
            raise ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR, f"required shared-plane container {container_name!r} not found"
            ) from exc
        return
    expected_shared_role = network_policy.shared_service_role_for_name(container_name)
    if expected_shared_role is not None:
        try:
            container.reload()
        except docker.errors.DockerException as exc:
            raise ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"failed to inspect required shared service {container_name!r}",
            ) from exc
        if not network_policy.shared_service_identity_valid(container.attrs, expected_shared_role):
            raise ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"required shared-plane container {container_name!r} has invalid role metadata",
            )
    try:
        network.connect(container, aliases=aliases)
    except docker.errors.APIError as exc:
        if _already_connected(exc):
            return
        if required:
            raise ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"failed to connect required service {container_name!r} to its Team network",
            ) from exc


def _wire_network_deps(network, dependencies: list[tuple[str, list[str]]]) -> None:
    for container_name, aliases in dependencies:
        _safe_connect(network, container_name, aliases=aliases, required=True)


def _teardown_team_network_kind(team_id: str, kind: str) -> bool:
    """Remove one owned plane, or report residue without mutating any foreign identity."""
    try:
        network = _docker.networks.get(network_policy.network_name(team_id, kind))
    except docker.errors.NotFound:
        return True
    except docker.errors.DockerException:
        return False
    try:
        network.reload()
    except docker.errors.DockerException:
        return False
    # A same-name foreign network is not ours to mutate. Reuse rejects it, and teardown leaves it for
    # an operator instead of disconnecting unrelated containers based on a name alone.
    if not network_policy.network_identity_valid(network.attrs, team_id, kind):
        return False
    cleanup_complete = True
    for container_id in dict(network.attrs.get("Containers", {})):
        try:
            container = _docker.containers.get(container_id)
            container.reload()
        except docker.errors.DockerException:
            cleanup_complete = False
            continue
        if not network_policy.network_member_managed(container.attrs, team_id, kind):
            cleanup_complete = False
            continue
        try:
            network.disconnect(container_id, force=True)
        except docker.errors.APIError:
            cleanup_complete = False
    try:
        network.reload()
        if network.attrs.get("Containers"):
            return False
        network.remove()
    except docker.errors.DockerException:
        return False
    return cleanup_complete


def _teardown_team_networks(team_id: str) -> bool:
    return _teardown_team_network_kind(team_id, network_policy.CORE_KIND)


def _describe(container) -> dict:
    team_id = str(container.labels.get("team.id", ""))
    try:
        inference = _inference_store.load(team_id)
    except inference_config.InferenceConfigError:
        inference = None
    return {
        "team_id": team_id,
        "team_name": container.labels.get("team.name"),
        "owner": container.labels.get("team.owner", ""),
        "provider": inference.provider if inference is not None else None,
        "model": inference.model if inference is not None else None,
        "status": container.status,
        "container": container.name,
    }


@dataclass(frozen=True)
class _AuthorizationLease:
    team_id: str
    container_id: str
    owner: str
    principal: tuple[str, str | None]
    cleanup_nonce: str = ""


def _cleanup_record(team_id: str) -> cleanup_state.Record | None:
    try:
        return cleanup_state.load(team_id)
    except cleanup_state.CleanupStateError as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Team cleanup state is unavailable") from exc


def _authorize_container(team_id: str, principal: tuple[str, str | None], container) -> _AuthorizationLease:
    if not network_policy.brain_identity_valid(container.attrs, team_id):
        raise ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
    owner = str(container.labels.get("team.owner", ""))
    kind, account_id = principal
    if kind != "operator" and owner != account_id:
        raise ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
    return _AuthorizationLease(
        team_id=team_id,
        container_id=container.id,
        owner=owner,
        principal=principal,
    )


def _authorize(team_id: str, principal: tuple[str, str | None]) -> _AuthorizationLease:
    """Operator may touch any team; an account may only touch a team it owns.

    This first pass returns an identity lease. Every sensitive operation must revalidate it only after
    acquiring its lifecycle/chat lock: authorization that waited behind destroy/recreate is never
    transferable to the new container that happens to reuse the same TEAM_ID.
    """
    container = _get_container(manifests.team_container_name(team_id))
    if container is None:
        raise ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
    return _authorize_container(team_id, principal, container)


def _authorize_destroy(team_id: str, principal: tuple[str, str | None]) -> _AuthorizationLease:
    """Authorize against the Brain, or its durable non-runnable cleanup successor."""
    container = _get_container(manifests.team_container_name(team_id))
    if container is not None:
        return _authorize_container(team_id, principal, container)
    record = _cleanup_record(team_id)
    if record is None or not cleanup_state.principal_authorized(record, principal):
        raise ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
    return _AuthorizationLease(
        team_id=team_id,
        container_id=record.brain_id,
        owner=record.owner,
        principal=principal,
        cleanup_nonce=record.nonce,
    )


def _require_cleanup_authorization(team_id: str, lease: _AuthorizationLease) -> cleanup_state.Record:
    """Revalidate the exact durable ownership record after acquiring the lifecycle lock."""
    record = _cleanup_record(team_id)
    if (
        not lease.cleanup_nonce
        or lease.team_id != team_id
        or record is None
        or record.nonce != lease.cleanup_nonce
        or record.owner != lease.owner
        or record.brain_id != lease.container_id
        or not cleanup_state.principal_authorized(record, lease.principal)
        or _get_container(manifests.team_container_name(team_id)) is not None
    ):
        raise ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
    return record


def _require_current_authorization(
    team_id: str,
    lease: _AuthorizationLease,
    *,
    require_isolation: bool = True,
    allow_pending_cleanup: bool = False,
):
    """Revalidate owner + immutable Docker identity; caller already holds the operation lock."""
    container = _get_container(manifests.team_container_name(team_id))
    if (
        lease.cleanup_nonce
        or lease.team_id != team_id
        or container is None
        or not network_policy.brain_identity_valid(container.attrs, team_id)
        or container.id != lease.container_id
        or str(container.labels.get("team.owner", "")) != lease.owner
    ):
        # Accounts must not learn that a different tenant recreated this name. Operators receive the
        # same retry-safe 404 contract instead of accidentally mutating an object they never selected.
        raise ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
    kind, account_id = lease.principal
    if kind != "operator" and lease.owner != account_id:
        raise ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
    if not allow_pending_cleanup and _cleanup_record(team_id) is not None:
        raise ApiError(HTTPStatus.CONFLICT, f"team {team_id!r} has an incomplete teardown; retry destroy")
    if require_isolation:
        _require_team_isolation(container)
    return container


# ── installed apps (the P4 deploy arm) ───────────────────────────────────────
def _team_app_containers(team_id: str) -> list:
    """Every installed-app container of team `team_id` (its OWN label set — never `team.driver`)."""
    return _docker.containers.list(all=True, filters={"label": ["team.app.driver", f"team.id={team_id}"]})


def _app_egress_token(team_id: str, app_id: str) -> str:
    """The app instance's stable egress token (its Proxy-Authorization to app-egress-proxy).

    Kept in the policy volume (drivers + proxy only) and reused across reinstalls, exactly like
    shimpz-driver's per-app tokens — the proxy maps token → this instance's own allowlist.
    """
    tdir = APP_EGRESS_POLICY_DIR / ".tokens"
    tdir.mkdir(parents=True, exist_ok=True)
    tf = tdir / f"{manifests.team_app_container_name(team_id, app_id)}.token"
    with contextlib.suppress(OSError):
        tok = tf.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    tok = secrets.token_hex(16)
    tf.write_text(tok, encoding="utf-8")
    return tok


def _write_egress_policy(token: str, egress: tuple[str, ...]) -> None:
    APP_EGRESS_POLICY_DIR.mkdir(parents=True, exist_ok=True)
    (APP_EGRESS_POLICY_DIR / f"{token}.json").write_text(json.dumps(sorted(egress)), encoding="utf-8")


def _remove_egress_policy(team_id: str, app_id: str) -> bool:
    """Remove an App's policy and token without losing the token needed for a retry."""
    tf = APP_EGRESS_POLICY_DIR / ".tokens" / f"{manifests.team_app_container_name(team_id, app_id)}.token"
    try:
        token = tf.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if token:
        try:
            (APP_EGRESS_POLICY_DIR / f"{token}.json").unlink(missing_ok=True)
        except OSError:
            # Keep the token file: it is the durable pointer needed to retry policy cleanup.
            return False
    try:
        tf.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def _probe_app_health(container, port: int, health_path: str) -> bool:
    """Probe the registry-declared endpoint; only an exact HTTP 200 proves the App ready."""
    url = f"http://127.0.0.1:{port}{health_path}"
    script = (
        "import http.client,sys\n"
        "connection=http.client.HTTPConnection('127.0.0.1', int(sys.argv[1]), timeout=3)\n"
        "try:\n"
        "    connection.request('GET', sys.argv[2])\n"
        "    print(connection.getresponse().status)\n"
        "finally:\n"
        "    connection.close()\n"
    )
    probes = (
        [
            "curl",
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "--max-time",
            "3",
            url,
        ],
        ["python3", "-c", script, str(port), health_path],
    )
    for probe in probes:
        try:
            rc, out = container.exec_run(probe)
        except docker.errors.APIError:  # the binary isn't in this image — try the other one
            continue
        answer = out.decode(errors="replace").strip() if rc == 0 else ""
        if answer.isdigit():
            return marketplace.health_response_ok(int(answer))
    return False


def _wait_app_healthy(container, port: int, health_path: str) -> tuple[bool, str]:
    for attempt in range(HEALTH_RETRIES):
        container.reload()
        if container.status in ("exited", "dead"):
            return False, f"container not running (status={container.status})"
        if container.status == "running" and _probe_app_health(container, port, health_path):
            return True, "ok"
        if attempt < HEALTH_RETRIES - 1:
            time.sleep(HEALTH_DELAY_SECONDS)
    return False, "health probe never answered"


def _app_ready_now(container, port: int, health_path: str) -> tuple[bool, str]:
    """Re-prove running + exact endpoint health at the install response commit seam."""
    try:
        container.reload()
        if container.status != "running":
            return False, f"container not running (status={container.status})"
        if not _probe_app_health(container, port, health_path):
            return False, "declared health endpoint did not answer 200"
        # The endpoint may have answered while the process was exiting. Reload once more so a
        # container that died during or immediately after the probe cannot be reported as running.
        container.reload()
    except docker.errors.DockerException:
        return False, "container readiness could not be verified"
    if container.status != "running":
        return False, f"container exited during its health probe (status={container.status})"
    return True, container.status


def _teardown_app(
    team_id: str,
    app_id: str,
    *,
    container=None,
    drop_db: bool = True,
) -> _CleanupResult:
    """Remove one exact managed App, retaining retry state whenever cleanup is incomplete."""
    if container is None:
        try:
            container = _get_container(manifests.team_app_container_name(team_id, app_id))
        except docker.errors.DockerException:
            return _CleanupResult(False, not drop_db)

    if container is not None:
        try:
            container.reload()
        except docker.errors.DockerException:
            return _CleanupResult(False, not drop_db)
        if not network_policy.app_identity_valid(container.attrs, team_id, app_id):
            # A deterministic-name collision or drifted ownership label is not ours to delete.
            return _CleanupResult(False, not drop_db)
        db_label = container.labels.get("team.app.db")
        if db_label not in (None, "0", "1"):
            return _CleanupResult(False, not drop_db)
        # Missing means a legacy App from before the label existed; conservatively assume it has a DB.
        drop_db = drop_db and db_label != "0"

    policy_removed = _remove_egress_policy(team_id, app_id)
    container_removed = container is None
    if container is not None and policy_removed:
        container_removed = _remove_team_container(container)
    elif container is not None:
        # Preserve the labeled retry anchor, but do not leave tenant code running after a failed removal.
        with contextlib.suppress(ApiError):
            _fail_stop_team(container)

    artifacts_removed = policy_removed and container_removed
    if not drop_db:
        return _CleanupResult(artifacts_removed, True)
    if not artifacts_removed:
        # Keep the DB registration intact until the retryable container/policy phase has completed.
        return _CleanupResult(False, False)
    try:
        pgdriver_client.drop_app_db(team_id, app_id)
    except pgdriver_client.PgDriverError, http.client.HTTPException, OSError, ValueError:
        return _CleanupResult(True, False)
    return _CleanupResult(True, True)


def _install_app(
    team_id: str,
    app_id: str,
    spec: marketplace.AppSpec,
    owner: str,
    lease: _AuthorizationLease,
) -> dict:
    with _lock_for(team_id):
        team = _require_current_authorization(team_id, lease)
        if owner != lease.owner:
            raise ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
        _prepare_marketplace_image(spec)
        team_name = team.labels.get("team.name", "")
        existing = _get_container(manifests.team_app_container_name(team_id, app_id))
        if existing is not None:  # idempotent only for this exact, still-isolated installed App
            try:
                existing.reload()
            except docker.errors.DockerException as exc:
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    f"cannot verify installed app {app_id!r}",
                ) from exc
            expected_labels = {
                "team.app.driver": "1",
                "team.id": team_id,
                "team.app": app_id,
                "team.owner": owner,
            }
            if any(str(existing.labels.get(key, "")) != value for key, value in expected_labels.items()):
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    f"existing container for app {app_id!r} has invalid ownership metadata; uninstall it first",
                )
            _require_team_isolation(existing)
            configured_image = str(existing.attrs.get("Config", {}).get("Image", ""))
            if configured_image != spec.image:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    f"installed app {app_id!r} uses a different image; uninstall it before reinstalling",
                )
            ready, status = _app_ready_now(existing, spec.port, spec.health_path)
            if not ready:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    f"installed app {app_id!r} is not ready ({status}); uninstall it before reinstalling",
                )
            return {"team_id": team_id, "app": app_id, "status": status, "installed": False}
        if len(_team_app_containers(team_id)) >= MAX_APPS_PER_TEAM:
            raise ApiError(HTTPStatus.TOO_MANY_REQUESTS, f"app limit reached for {team_id!r} ({MAX_APPS_PER_TEAM})")
        key = f"app:{team_id}:{app_id}"
        with _reserve_capacity(key, owner, manifests.APP_MEM_LIMIT_BYTES, team_slot=False):
            # Return the same explicit admission error as Team create instead of relying on a
            # lower-level Docker create failure when the hostile-tenant runtime is unavailable.
            _require_team_runtime()
            # Transactional like _create: on ANY failure the app's own artifacts are rolled back (the
            # team itself is never touched) — no orphan DB, policy file, or half-started container.
            # Only the reservation remains global while these external calls run; unrelated Teams
            # can reserve/provision concurrently.
            try:
                database_url = ""
                if spec.db:
                    database_url = pgdriver_client.create_app_db(team_id, app_id)["database_url"]
                network = _ensure_team_network(team_id)
                proxy_env: dict[str, str] = {}
                if spec.egress:
                    token = _app_egress_token(team_id, app_id)
                    _write_egress_policy(token, spec.egress)
                    # Only the authenticated app proxy may join the core network. The broad Brain proxy
                    # is confined to the separate Brain-egress network and is unreachable from this App.
                    _safe_connect(
                        network,
                        manifests.APP_EGRESS_CONTAINER,
                        aliases=["app-egress-proxy"],
                        required=True,
                    )
                    proxy_env = {
                        "HTTPS_PROXY": f"http://{token}@app-egress-proxy:8889",
                        "https_proxy": f"http://{token}@app-egress-proxy:8889",
                    }
                kwargs = manifests.build_team_app_kwargs(
                    team_id,
                    app_id,
                    spec,
                    database_url=database_url,
                    proxy_env=proxy_env,
                    owner=owner,
                    team_name=team_name,
                )
                _require_team_runtime()
                container = _docker.containers.create(**kwargs)
                # Re-attach with the app aliases (create can't set them): the team brain and sibling
                # apps reach it as http://<app-id>:<port>, and as http://<app-id>.team:<port> — the
                # `.team` form tail-matches the NO_PROXY suffix baked into every team container, so
                # proxied clients skip egress. The App remains attached only to the internal core plane.
                network.disconnect(container)
                network.connect(container, aliases=[app_id, f"{app_id}.team"])
                _start_team_with_isolation(container)
                healthy, reason = _wait_app_healthy(container, spec.port, spec.health_path)
                if not healthy:
                    raise ApiError(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        f"app {app_id!r} failed its health probe ({reason}; rolled back)",
                    )
                _require_team_isolation(container)
                ready, committed_status = _app_ready_now(container, spec.port, spec.health_path)
                if not ready:
                    raise ApiError(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        f"app {app_id!r} lost readiness before install commit ({committed_status}; rolled back)",
                    )
            except Exception as exc:
                cleanup = _teardown_app(team_id, app_id, drop_db=spec.db)
                if not cleanup.complete:
                    raise ApiError(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "app install failed and rollback is incomplete; retry uninstall or contact the operator",
                    ) from exc
                if isinstance(exc, ApiError):
                    raise
                raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "app install failed and was rolled back") from exc
        return {
            "team_id": team_id,
            "app": app_id,
            "status": committed_status,
            "installed": True,
            **({"database": manifests.team_app_db_project(team_id, app_id)} if spec.db else {}),
        }


def _uninstall_app(team_id: str, app_id: str, lease: _AuthorizationLease) -> dict:
    with _lock_for(team_id):
        # Removal is a remediation operation and must remain available for a legacy blocked Team.
        _require_current_authorization(team_id, lease, require_isolation=False)
        cleanup = _teardown_app(team_id, app_id)
        if not cleanup.complete:
            raise ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "app teardown is incomplete; retry uninstall or contact the operator",
            )
        return {"team_id": team_id, "app": app_id, "uninstalled": True, "db_dropped": cleanup.db_dropped}


def _list_apps(team_id: str, lease: _AuthorizationLease) -> dict:
    with _lock_for(team_id):
        # Read-only inventory lets the owner see and remove residual Apps without executing tenant code.
        _require_current_authorization(team_id, lease, require_isolation=False)
        apps = [
            {
                "app": app_id,
                "status": c.status,
                "container": c.name,
                "powers": sorted(spec.assistant.powers) if spec is not None and spec.assistant is not None else [],
            }
            for c in _team_app_containers(team_id)
            for app_id in [c.labels.get("team.app")]
            for spec in [marketplace.APPS.get(app_id) if isinstance(app_id, str) else None]
        ]
        return {"team_id": team_id, "apps": apps}


# ── Controller-owned Assistant chat ─────────────────────────────────────────────────────────────
CHAT_OUTPUT_CAP = 60000
MAX_INBOX_FILE_BYTES = 25 * 1024 * 1024
# Base64 expands by 4/3; leave a small fixed envelope for the JSON keys and filename.
MAX_FILE_BODY_BYTES = 4 * ((MAX_INBOX_FILE_BYTES + 2) // 3) + 8192
MAX_ASSISTANT_RPC_INPUT_BYTES = 16 * 1024
MAX_ASSISTANT_RPC_OUTPUT_BYTES = assistant_contract.MAX_HELP_BYTES * 6 + 1024
ASSISTANT_RPC_TIMEOUT_SECONDS = 8
MAX_CHAT_FILES = 8


@dataclass(frozen=True, slots=True)
class _ActiveAssistant:
    assistant_id: str
    contract: marketplace.AssistantContract
    container: object


def _power_operation(
    request: brain_runtime_client.PowerRequest,
    assistant_container_id: object,
) -> power_journal.Operation:
    """Commit to one normalized request without persisting its raw input."""
    if not isinstance(assistant_container_id, str) or not assistant_container_id:
        raise power_journal.PowerJournalConflictError("Assistant generation is invalid")
    try:
        encoded = json.dumps(
            {
                "approval": request.approval,
                "assistant_container_id": assistant_container_id,
                "assistant_id": request.assistant_id,
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


class _HostedPowerBatch:
    """Adapt one orchestrator suspension to the generation-scoped durable journal."""

    def __init__(
        self,
        generation: str,
        thread_id: str,
        bindings: dict[str, _ActiveAssistant],
        execute: Callable[[brain_runtime_client.PowerRequest], object],
    ) -> None:
        self._generation = generation
        self._thread_id = thread_id
        self._bindings = bindings
        self._execute = execute
        self._journal: power_journal.PowerJournal | None = None
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
            operations.append(_power_operation(request, active.container.id))
        journal = _power_execution_journal()
        batch = journal.prepare_batch(self._generation, self._thread_id, operations)
        self._journal = journal
        self._batch = batch
        self._operations = {operation.interrupt_id: operation for operation in operations}

    def invoke(self, request: brain_runtime_client.PowerRequest) -> object:
        if self._journal is None or self._batch is None:
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
        if self._journal is None or self._batch is None:
            raise power_journal.PowerJournalConflictError("Power batch is not prepared")
        expected = tuple(operation.interrupt_id for operation in self._batch.operations)
        if tuple(request.interrupt_id for request in requests) != expected:
            raise power_journal.PowerJournalConflictError("Power delivery batch changed")
        self._journal.delivered(self._batch)
        self._journal = None
        self._batch = None
        self._operations = {}


def _close_exec_stream(stream) -> None:
    """Close docker-py's owning HTTP response before its raw socket (Python 3.14 safe)."""
    response = getattr(stream, "_response", None)
    if response is not None:
        response.close()
    else:
        stream.close()


def _installed_assistant(team_id: str, assistant_id: object):
    assistant_id, spec = marketplace.resolve(assistant_id)
    contract = spec.assistant
    if contract is None:
        raise ApiError(HTTPStatus.NOT_FOUND, f"{assistant_id!r} is not an Assistant")
    container = _get_container(manifests.team_app_container_name(team_id, assistant_id))
    if container is None:
        raise ApiError(HTTPStatus.CONFLICT, f"Assistant {assistant_id!r} is not installed in this Team")
    with _active_chat_guard:
        if (team_id, container.id) in _blocked_power_workloads:
            raise ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Assistant Power execution is blocked until this Assistant is reinstalled",
            )
    try:
        container.reload()
    except docker.errors.DockerException as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "installed Assistant could not be verified") from exc
    if (
        not network_policy.app_identity_valid(container.attrs, team_id, assistant_id)
        or str(container.attrs.get("Config", {}).get("Image", "")) != spec.image
    ):
        raise ApiError(HTTPStatus.CONFLICT, "installed Assistant failed its identity contract")
    _require_running_team_isolation(container)
    return assistant_id, contract, container


def _active_team_assistants(team_id: str) -> tuple[_ActiveAssistant, ...]:
    active: list[_ActiveAssistant] = []
    seen: set[str] = set()
    try:
        installed = _team_app_containers(team_id)
    except docker.errors.DockerException as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "installed Assistants could not be listed") from exc
    for candidate in installed:
        assistant_id = (candidate.labels or {}).get("team.app")
        spec = marketplace.APPS.get(assistant_id) if isinstance(assistant_id, str) else None
        if spec is None or spec.assistant is None:
            continue
        try:
            candidate.reload()
        except docker.errors.DockerException as exc:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "installed Assistant could not be inspected") from exc
        if candidate.status != "running":
            continue
        if assistant_id in seen:
            raise ApiError(HTTPStatus.CONFLICT, "duplicate installed Assistant identity")
        current_id, contract, container = _installed_assistant(team_id, assistant_id)
        seen.add(current_id)
        active.append(_ActiveAssistant(current_id, contract, container))
    active.sort(key=lambda item: item.assistant_id)
    if not active:
        raise ApiError(HTTPStatus.CONFLICT, "install and start at least one Assistant before chatting with this Team")
    return tuple(active)


def _read_rpc_exact(raw_socket: socket.socket, amount: int, deadline: float) -> bytes:
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


def _read_rpc_frames(raw_socket: socket.socket, deadline: float) -> tuple[bytes, bytes]:
    stdout = bytearray()
    stderr = bytearray()
    while True:
        try:
            header = _read_rpc_exact(raw_socket, 8, deadline)
        except EOFError:
            break
        stream_id, length = struct.unpack(">BxxxL", header)
        if stream_id not in {docker_socket.STDOUT, docker_socket.STDERR}:
            raise ValueError("invalid Assistant RPC stream")
        if length > MAX_ASSISTANT_RPC_OUTPUT_BYTES + 1:
            raise ValueError("oversized Assistant RPC frame")
        chunk = _read_rpc_exact(raw_socket, length, deadline)
        target = stdout if stream_id == docker_socket.STDOUT else stderr
        target.extend(chunk)
        if len(stdout) + len(stderr) > MAX_ASSISTANT_RPC_OUTPUT_BYTES:
            raise ValueError("oversized Assistant RPC response")
    return bytes(stdout), bytes(stderr)


def _register_active_power(team_id: str, token: str, container) -> None:
    with _active_chat_guard:
        if _active_chat_tokens.get(team_id) != token or token in _cancelled_chat_tokens:
            raise ApiError(HTTPStatus.CONFLICT, "brain turn stopped")
        if team_id in _active_power_container_ids:
            raise ApiError(HTTPStatus.CONFLICT, "Team already has an active Assistant Power")
        _active_power_container_ids[team_id] = (token, container.id)


def _release_active_power(team_id: str, token: str, container_id: str) -> None:
    with _active_chat_guard:
        if _active_power_container_ids.get(team_id) == (token, container_id):
            _active_power_container_ids.pop(team_id, None)


def _register_optional_power(team_id: str, token: str | None, container) -> None:
    if token is not None:
        _register_active_power(team_id, token, container)


def _release_optional_power(team_id: str, token: str | None, container_id: str) -> None:
    if token is not None:
        _release_active_power(team_id, token, container_id)


def _raise_if_rpc_cancelled(token: str | None, exc: BaseException | None = None) -> None:
    if token is not None and _token_cancelled(token):
        raise ApiError(HTTPStatus.CONFLICT, "brain turn stopped") from exc


def _fail_stop_power(team_id: str, container) -> None:
    """Prove an ambiguous Assistant RPC can no longer execute before returning an error."""
    try:
        _fail_stop_team(container, timeout=3)
    except ApiError as exc:
        with _active_chat_guard:
            _blocked_power_workloads.add((team_id, container.id))
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Assistant Power termination could not be proved; reinstall the Assistant",
        ) from exc


def _assistant_rpc_exchange(
    team_id: str,
    container,
    command: str,
    method: str,
    path: str,
    payload: dict,
    *,
    token: str | None,
    operation: str,
    detect_unsupported_path: bool = False,
) -> object:
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    if len(encoded) > MAX_ASSISTANT_RPC_INPUT_BYTES:
        raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Power input is too large")
    _register_optional_power(team_id, token, container)
    stream = None
    try:
        try:
            created = _docker.api.exec_create(
                container.id,
                [command, method, path],
                stdin=True,
                stdout=True,
                stderr=True,
                privileged=False,
                user="10001:10001",
                workdir=manifests.CONTAINER_TMP,
                environment={},
            )
            exec_id = created["Id"]
            stream = _docker.api.exec_start(exec_id, socket=True)
            raw_socket = getattr(stream, "_sock", None)
            if raw_socket is None:
                raise OSError("Docker attach socket cannot half-close stdin")
            raw_socket.sendall(encoded)
            raw_socket.shutdown(socket.SHUT_WR)
            deadline = time.monotonic() + ASSISTANT_RPC_TIMEOUT_SECONDS
            stdout, stderr = _read_rpc_frames(raw_socket, deadline)
        except TimeoutError as exc:
            _fail_stop_power(team_id, container)
            _raise_if_rpc_cancelled(token, exc)
            raise ApiError(HTTPStatus.GATEWAY_TIMEOUT, f"{operation} timed out") from exc
        except (docker.errors.DockerException, OSError, ValueError, KeyError) as exc:
            _fail_stop_power(team_id, container)
            _raise_if_rpc_cancelled(token, exc)
            raise ApiError(HTTPStatus.BAD_GATEWAY, f"{operation} failed") from exc
        finally:
            if stream is not None:
                with contextlib.suppress(Exception):
                    _close_exec_stream(stream)

        try:
            details = _docker.api.exec_inspect(exec_id)
        except docker.errors.DockerException as exc:
            _fail_stop_power(team_id, container)
            _raise_if_rpc_cancelled(token, exc)
            raise ApiError(HTTPStatus.BAD_GATEWAY, f"{operation} status is ambiguous") from exc
        if not isinstance(details.get("ExitCode"), int):
            _fail_stop_power(team_id, container)
            _raise_if_rpc_cancelled(token)
            raise ApiError(HTTPStatus.BAD_GATEWAY, f"{operation} status is ambiguous")
        if details["ExitCode"] != 0 or stderr:
            if detect_unsupported_path and details["ExitCode"] == 2 and not stdout and not stderr:
                raise _UnsupportedAssistantRpcPathError(path)
            _raise_if_rpc_cancelled(token)
            raise ApiError(HTTPStatus.BAD_GATEWAY, f"{operation} failed")
        try:
            return json.loads(bytes(stdout))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ApiError(HTTPStatus.BAD_GATEWAY, f"{operation} returned an invalid result") from exc
    finally:
        _release_optional_power(team_id, token, container.id)


def _assistant_rpc(
    team_id: str,
    token: str,
    container,
    command: str,
    method: str,
    path: str,
    payload: dict,
) -> object:
    return _assistant_rpc_exchange(
        team_id,
        container,
        command,
        method,
        path,
        payload,
        token=token,
        operation="Assistant Power",
    )


def _assistant_help(
    team_id: str,
    assistant_id: str,
    lease: _AuthorizationLease,
    locale: str = "en",
) -> dict[str, str]:
    """Read bounded Markdown through one fixed RPC from an installed running Assistant."""
    try:
        locale = assistant_contract.validate_help_locale(locale)
    except ValueError as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, "Assistant Help locale is not supported") from exc
    with _lock_for(team_id):
        _require_current_authorization(team_id, lease)
        current_id, contract, container = _installed_assistant(team_id, assistant_id)
        try:
            raw_result = _assistant_rpc_exchange(
                team_id,
                container,
                contract.rpc_command,
                "GET",
                f"/v1/help/{locale}",
                {},
                token=None,
                operation="Assistant Help",
                detect_unsupported_path=True,
            )
        except _UnsupportedAssistantRpcPathError:
            raw_result = _assistant_rpc_exchange(
                team_id,
                container,
                contract.rpc_command,
                "GET",
                "/v1/help",
                {},
                token=None,
                operation="Assistant Help",
            )
    try:
        help_payload = assistant_contract.validate_help_payload(raw_result)
    except ValueError as exc:
        raise ApiError(HTTPStatus.BAD_GATEWAY, f"Assistant Help from {current_id!r} is invalid") from exc
    return {"assistant": current_id, **help_payload}


def _invoke_assistant_power(
    team_id: str,
    token: str,
    assistant_id: str,
    contract: marketplace.AssistantContract,
    container,
    power: object,
    payload: object,
) -> dict[str, object]:
    if (
        not isinstance(power, str)
        or assistant_chat.POWER_ID_RE.fullmatch(power) is None
        or power not in contract.powers
    ):
        raise ApiError(HTTPStatus.BAD_REQUEST, "Assistant requested an undeclared Power")
    try:
        safe_input = marketplace.validate_power_input(assistant_id, power, payload)
    except ValueError as exc:
        raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc)) from exc
    _current_id, _current_contract, current_container = _installed_assistant(team_id, assistant_id)
    if current_container.id != container.id:
        raise ApiError(HTTPStatus.CONFLICT, "installed Assistant changed during the chat turn")
    power_spec = contract.powers[power]
    audit.log(
        "assistant_power",
        team_id,
        result="ok",
        phase="started",
        assistant=assistant_id,
        power=power,
    )
    try:
        raw_result = _assistant_rpc(
            team_id,
            token,
            container,
            contract.rpc_command,
            power_spec.method,
            power_spec.path,
            safe_input,
        )
    except ApiError as exc:
        audit.log(
            "assistant_power",
            team_id,
            result="error",
            assistant=assistant_id,
            power=power,
            status=int(exc.status),
        )
        raise
    try:
        result = marketplace.validate_power_output(assistant_id, power, raw_result)
    except ValueError as exc:
        audit.log(
            "assistant_power",
            team_id,
            result="error",
            assistant=assistant_id,
            power=power,
            reason="invalid-output",
        )
        raise ApiError(HTTPStatus.BAD_GATEWAY, "Assistant Power returned an invalid result") from exc
    audit.log(
        "assistant_power",
        team_id,
        result="ok",
        phase="completed",
        assistant=assistant_id,
        power=power,
    )
    return {"assistant": assistant_id, "power": power, "result": result}


def _validate_assistant_power_input(bindings, assistant_id: str, power: str, power_input) -> object:
    """Normalize one hosted Power input without touching Docker or another external system."""
    if assistant_id not in bindings:
        raise ApiError(HTTPStatus.CONFLICT, "Brain requested an unavailable Assistant")
    try:
        return marketplace.validate_power_input(assistant_id, power, power_input)
    except ValueError as exc:
        raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc)) from exc


def _chat_file_metadata(team_id: str, file_ids: object) -> list[dict[str, object]]:
    if file_ids is None:
        return []
    if not isinstance(file_ids, list) or len(file_ids) > MAX_CHAT_FILES:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"files must contain at most {MAX_CHAT_FILES} opaque ids")
    try:
        return _storage().metadata(team_id, file_ids)
    except team_storage.StorageNotFoundError as exc:
        raise ApiError(HTTPStatus.NOT_FOUND, "selected file not found in this Team") from exc
    except team_storage.StorageInputError as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    except team_storage.StorageError as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Team storage failed its safety checks") from exc


def _model_credential(owner: str, provider: str) -> tuple[str, int]:
    if not owner:
        raise ApiError(HTTPStatus.CONFLICT, "this Team has no account owner for model credentials")
    try:
        credential = brain_credentials_client.resolve(owner, provider)
    except brain_credentials_client.BrainCredentialError as exc:
        raise ApiError(HTTPStatus.BAD_GATEWAY, "model credential service is unavailable") from exc
    if credential is None:
        raise ApiError(HTTPStatus.CONFLICT, f"configure the {provider!r} API key before chatting")
    auth_type, api_key, generation = credential
    if auth_type != "api_key":
        raise ApiError(HTTPStatus.CONFLICT, "the selected model provider requires an API key")
    return api_key, generation


def _require_model_credential_current(owner: str, provider: str, generation: int) -> None:
    try:
        current = brain_credentials_client.generation_is_current(owner, provider, generation)
    except brain_credentials_client.BrainCredentialError as exc:
        raise ApiError(HTTPStatus.BAD_GATEWAY, "model credential could not be verified") from exc
    if not current:
        raise ApiError(HTTPStatus.CONFLICT, "model credential changed or was revoked; retry")


def _current_team_anchor(team_id: str, container_id: str, owner: str):
    container = _get_container(manifests.team_container_name(team_id))
    if container is None:
        raise ApiError(HTTPStatus.CONFLICT, "Team identity changed during the chat turn")
    try:
        container.reload()
    except docker.errors.DockerException as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Team identity could not be inspected") from exc
    if (
        container.id != container_id
        or not network_policy.brain_identity_valid(container.attrs, team_id)
        or str(container.labels.get("team.owner", "")) != owner
    ):
        raise ApiError(HTTPStatus.CONFLICT, "Team identity changed during the chat turn")
    _require_running_team_isolation(container)
    return container


def _chat_in_turn(
    team_id: str,
    message: str,
    file_ids: object,
    token: str,
    container,
    owner: str,
) -> dict:
    team_name = _team_name_from_anchor(container)
    thread_id = _brain_thread_id(team_id, container.id)
    assistants = _active_team_assistants(team_id)
    files = _chat_file_metadata(team_id, file_ids)
    try:
        config = _inference_store.load(team_id)
    except inference_config.InferenceConfigError as exc:
        raise ApiError(HTTPStatus.CONFLICT, "configure this Team's model provider before chatting") from exc
    api_key, generation = _model_credential(owner, config.provider)

    def require_current_credential() -> None:
        _require_model_credential_current(owner, config.provider, generation)

    require_current_credential()
    runtime_context = brain_runtime_client.RuntimeContext(
        thread_id=thread_id,
        team_name=team_name,
        assistants=tuple(
            brain_runtime_client.RuntimeAssistant(
                id=active.assistant_id,
                rules=active.contract.rules,
                powers=tuple(
                    brain_runtime_client.RuntimePower(
                        id=power_id,
                        summary=power.summary,
                        input_schema=power.input_schema,
                        approval=power.approval,
                    )
                    for power_id, power in sorted(active.contract.powers.items())
                ),
            )
            for active in assistants
        ),
        provider=config.provider,
        model=config.model,
        api_key=api_key,
    )
    prompt = assistant_chat.build_prompt(message, files)
    bindings = {active.assistant_id: active for active in assistants}

    def validate_power(assistant_id: str, power: str, power_input) -> object:
        return _validate_assistant_power_input(bindings, assistant_id, power, power_input)

    def execute_power(request: brain_runtime_client.PowerRequest) -> object:
        require_current_credential()
        active = bindings.get(request.assistant_id)
        if active is None:
            raise ApiError(HTTPStatus.CONFLICT, "Brain requested an unavailable Assistant")
        invocation = _invoke_assistant_power(
            team_id,
            token,
            request.assistant_id,
            active.contract,
            active.container,
            request.power,
            request.input,
        )
        return invocation["result"]

    durable_batch = _HostedPowerBatch(
        container.id,
        runtime_context.thread_id,
        bindings,
        execute_power,
    )

    initial_identity = (
        container.id,
        team_name,
        tuple((active.assistant_id, active.container.id) for active in assistants),
        files,
        config,
    )

    def validate_context() -> None:
        require_current_credential()
        current_anchor = _current_team_anchor(team_id, container.id, owner)
        current_assistants = _active_team_assistants(team_id)
        current_files = _chat_file_metadata(team_id, file_ids)
        try:
            current_config = _inference_store.load(team_id)
        except inference_config.InferenceConfigError as exc:
            raise ApiError(HTTPStatus.CONFLICT, "Team model configuration changed; retry") from exc
        current_identity = (
            current_anchor.id,
            _team_name_from_anchor(current_anchor),
            tuple((active.assistant_id, active.container.id) for active in current_assistants),
            current_files,
            current_config,
        )
        if current_identity != initial_identity:
            raise ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry")

    try:
        outcome = chat_orchestrator.run(
            _brain_runtime,
            runtime_context,
            prompt,
            validate_power,
            durable_batch.invoke,
            prepare_batch=durable_batch.prepare,
            batch_delivered=durable_batch.delivered,
            cancelled=lambda: _token_cancelled(token),
            validate_context=validate_context,
        )
    except power_journal.PowerJournalError as exc:
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Team Power execution state is unavailable",
        ) from exc
    except chat_orchestrator.ChatStoppedError as exc:
        raise ApiError(HTTPStatus.CONFLICT, "brain turn stopped") from exc
    except chat_orchestrator.ApprovalRequiredError as exc:
        raise ApiError(
            HTTPStatus.CONFLICT,
            "Assistant Power requires Captain approval",
        ) from exc
    except chat_orchestrator.ChatOrchestrationError as exc:
        raise ApiError(HTTPStatus.BAD_GATEWAY, "Brain could not complete the Assistant turn") from exc
    except brain_runtime_client.BrainRuntimeError as exc:
        raise ApiError(HTTPStatus.BAD_GATEWAY, "Brain runtime is unavailable") from exc
    require_current_credential()
    if not _commit_chat_terminal(team_id, token):
        raise ApiError(HTTPStatus.CONFLICT, "brain turn stopped")
    return {
        "team_id": team_id,
        "team_name": team_name,
        "reply": outcome.reply[:CHAT_OUTPUT_CAP],
    }


def _chat(
    team_id: str,
    message: str,
    file_ids: object,
    lease: _AuthorizationLease,
) -> dict:
    """Run one bounded Team turn across every active, Controller-brokered Assistant Power."""
    # The slot comes first. A losing concurrent request must not run even the local credential probe,
    # much less provider status or a second provider CLI.
    with _exclusive_chat_turn(team_id, lease) as (token, container):
        return _chat_in_turn(team_id, message, file_ids, token, container, lease.owner)


def _stop_active_power(team_id: str, token: str | None) -> bool:
    if token is None:
        return False
    with _active_chat_guard:
        active = _active_power_container_ids.get(team_id)
    if active is None or active[0] != token:
        return False
    try:
        assistant_container = _docker.containers.get(active[1])
    except docker.errors.NotFound:
        return True
    except docker.errors.DockerException as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "active Assistant Power could not be inspected") from exc
    _fail_stop_power(team_id, assistant_container)
    return True


def _stop_chat(team_id: str, lease: _AuthorizationLease) -> dict:
    """Cancel one Controller-owned turn and fail-stop a Power already executing."""
    with _lock_for(team_id):
        container = _require_current_authorization(team_id, lease)
        container.reload()
        if container.status != "running":
            raise ApiError(HTTPStatus.CONFLICT, f"team {team_id!r} is not running (status={container.status})")
        with _active_chat_guard:
            token = _active_chat_tokens.get(team_id)
            if token is not None and _active_chat_container_ids.get(team_id) != container.id:
                raise ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
            if token is not None:
                _cancelled_chat_tokens.add(token)
        power_stopped = _stop_active_power(team_id, token)
    accepted = token is not None
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


def _put_inbox_file(
    team_id: str,
    filename: object,
    content_b64: object,
    media_type: object,
    lease: _AuthorizationLease,
) -> dict:
    """Store an opaque object outside every Brain and Assistant filesystem."""
    if not isinstance(content_b64, str):
        raise ApiError(HTTPStatus.BAD_REQUEST, "invalid base64 content")
    try:
        data = base64.b64decode(content_b64 or "", validate=True)
    except (binascii.Error, UnicodeError, ValueError) as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, "invalid base64 content") from exc
    if not data or len(data) > MAX_INBOX_FILE_BYTES:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"file must be 1..{MAX_INBOX_FILE_BYTES} bytes")
    with _lock_for(team_id):
        _require_current_authorization(team_id, lease, require_isolation=False)
        try:
            stored = _storage().put(team_id, filename, data, media_type)
        except team_storage.StorageQuotaError as exc:
            raise ApiError(HTTPStatus.INSUFFICIENT_STORAGE, str(exc)) from exc
        except team_storage.StorageInputError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        except team_storage.StorageError as exc:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Team storage failed its safety checks") from exc
        return {"team_id": team_id, "file": stored}


def _list_team_files(team_id: str, lease: _AuthorizationLease) -> dict:
    with _lock_for(team_id):
        _require_current_authorization(team_id, lease, require_isolation=False)
        try:
            listing = _storage().list(team_id)
        except team_storage.StorageError as exc:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Team storage failed its safety checks") from exc
        return {"team_id": team_id, **listing}


def _delete_team_file(team_id: str, file_id: object, lease: _AuthorizationLease) -> dict:
    with _lock_for(team_id):
        _require_current_authorization(team_id, lease, require_isolation=False)
        try:
            result = _storage().delete(team_id, file_id)
        except team_storage.StorageNotFoundError as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, "file not found") from exc
        except team_storage.StorageError as exc:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Team storage failed its safety checks") from exc
        return {"team_id": team_id, **result}


# ── operations ───────────────────────────────────────────────────────────────
def _remove_volume(team_id: str, kind: str) -> bool:
    name = network_policy.volume_name(team_id, kind)
    try:
        volume = _docker.volumes.get(name)
    except docker.errors.NotFound:
        return True
    except docker.errors.DockerException:
        return False
    try:
        volume.reload()
    except docker.errors.NotFound:
        return True
    except docker.errors.DockerException:
        return False
    if not network_policy.volume_identity_valid(volume.attrs, team_id, kind):
        return False
    try:
        volume.remove(force=True)
    except docker.errors.NotFound:
        return True
    except docker.errors.DockerException:
        return False
    return True


def _owned_teardown_brain(team_id: str, owner: str, brain_id: str):
    try:
        brain = _get_container(manifests.team_container_name(team_id))
    except docker.errors.DockerException:
        return False, None
    if brain is None:
        return True, None
    try:
        brain.reload()
    except docker.errors.DockerException:
        return False, None
    valid = (
        network_policy.brain_identity_valid(brain.attrs, team_id)
        and brain.id == brain_id
        and str(brain.labels.get("team.owner", "")) == owner
    )
    return valid, brain


def _stop_teardown_brain(brain) -> bool:
    if brain is None:
        return True
    try:
        _fail_stop_team(brain, timeout=30)
    except ApiError:
        return False
    return True


def _teardown_apps(team_id: str) -> bool:
    try:
        app_containers = _team_app_containers(team_id)
    except docker.errors.DockerException:
        return False
    cleanup_complete = True
    for app_container in app_containers:
        app_id = app_container.labels.get("team.app", "")
        if not isinstance(app_id, str) or marketplace.APP_ID_RE.fullmatch(app_id) is None:
            cleanup_complete = False
            continue
        # The Team-level database drop removes every registered App database in one scoped call.
        result = _teardown_app(team_id, app_id, container=app_container, drop_db=False)
        cleanup_complete = result.artifacts_removed and cleanup_complete
    return cleanup_complete


def _teardown_network_planes(team_id: str) -> bool:
    return _teardown_team_networks(team_id)


def _remove_teardown_brain(brain) -> bool:
    if brain is None:
        return True
    return _remove_team_container(brain, timeout=30)


def _teardown_volumes(team_id: str) -> bool:
    results = [
        _remove_volume(team_id, kind)
        for kind in (network_policy.CONFIG_VOLUME_KIND, network_policy.WORKSPACE_VOLUME_KIND)
    ]
    return all(results)


def _teardown_storage(team_id: str) -> bool:
    if _storage_instance is None and not TEAM_STORAGE_ROOT.exists():
        return True
    try:
        _storage().destroy(team_id)
    except team_storage.StorageError:
        return False
    return True


def _teardown_inference(team_id: str) -> bool:
    try:
        _inference_store.delete(team_id)
    except inference_config.InferenceConfigError:
        return False
    return True


def _retire_teardown_r2(team_id: str) -> bool:
    """Cut off every new Team R2 operation before deleting any tenant artifact."""
    try:
        r2driver_client.retire_team(team_id)
    except r2driver_client.R2DriverError:
        return False
    return True


def _drop_teardown_database(team_id: str, record: cleanup_state.Record) -> cleanup_state.Record | None:
    if record.db_dropped:
        return record
    try:
        pgdriver_client.drop_team(team_id)
        return cleanup_state.mark_db_dropped(record)
    except (
        pgdriver_client.PgDriverError,
        cleanup_state.CleanupStateError,
        http.client.HTTPException,
        OSError,
        ValueError,
    ):
        return None


def _finalize_teardown(team_id: str, record: cleanup_state.Record) -> bool:
    try:
        # R2 destroys its encrypted bundles and hashed principal first. Its local cleartext principal
        # is removed only after that authenticated 200; both finalizers remain safe to replay.
        r2driver_client.finalize_team_drop(team_id)
        pgdriver_client.finalize_team_drop(team_id)
        cleanup_state.finish(record)
    except (
        r2driver_client.R2DriverError,
        pgdriver_client.PgDriverError,
        cleanup_state.CleanupStateError,
        http.client.HTTPException,
        OSError,
        ValueError,
    ):
        return False
    return True


def _teardown(team_id: str, *, owner: str, brain_id: str) -> _CleanupResult:
    """Remove every Team artifact, preserving a durable owner-bound retry anchor throughout."""
    brain_valid, brain = _owned_teardown_brain(team_id, owner, brain_id)
    if not brain_valid:
        return _CleanupResult(False, False)

    # Persist the immutable tenant/Brain identity before the first mutation. Once Docker releases the
    # Brain's volume references this record—not a runnable workload—authorizes only a retrying DELETE.
    try:
        record = cleanup_state.begin(team_id, owner, brain_id)
    except cleanup_state.CleanupStateError:
        return _CleanupResult(False, False)
    if not _stop_teardown_brain(brain):
        return _CleanupResult(False, record.db_dropped)
    # Fail closed while the durable cleanup record and stopped Brain still exist. No credential,
    # volume, network or database artifact is removed until R2 has revoked this Team principal.
    if not _retire_teardown_r2(team_id):
        return _CleanupResult(False, record.db_dropped)
    if (
        not _teardown_apps(team_id)
        or not _teardown_storage(team_id)
        or not _teardown_inference(team_id)
        or not _teardown_network_planes(team_id)
    ):
        return _CleanupResult(False, record.db_dropped)
    if not _remove_teardown_brain(brain) or not _teardown_volumes(team_id):
        return _CleanupResult(False, record.db_dropped)
    record = _drop_teardown_database(team_id, record)
    if record is None:
        return _CleanupResult(False, False)
    # pg-driver keeps a retired, idempotent principal until this provisioner-authorized finalizer;
    # only then is the controller's cleartext principal removed. Both operations are retry-safe.
    if not _finalize_teardown(team_id, record):
        return _CleanupResult(False, True)
    return _CleanupResult(True, True)


def _create(team_id: str, body: dict, owner: str = "") -> dict:
    try:
        team_name = _validated_team_name(body.get("team_name", team_id))
    except ValueError as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    try:
        inference = inference_config.normalize(body.get("provider"), body.get("model"))
    except inference_config.InferenceConfigError as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    # The current hosted Team identity remains a sandboxed lifecycle anchor. Model inference is
    # now a separate service, so changing provider/model never replaces this container.
    anchor_brain = manifests.DEFAULT_BRAIN
    anchor_model = manifests.model_for_brain(anchor_brain)
    with _lock_for(team_id):
        pending_cleanup = _cleanup_record(team_id)
        if pending_cleanup is not None:
            if owner and pending_cleanup.owner != owner:
                raise ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
            raise ApiError(
                HTTPStatus.CONFLICT,
                f"team {team_id!r} has an incomplete teardown; retry destroy before creating it",
            )
        existing = _get_container(manifests.team_container_name(team_id))
        if existing is not None:
            # An account may only "re-create" (get) its OWN team; a name collision with a different
            # owner is invisible (404), never a hijack of someone else's team.
            existing_owner = existing.labels.get("team.owner", "")
            if owner and existing_owner != owner:
                raise ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
            # Upgrade fail-close: idempotent create must not bless a legacy runc container. Test
            # data can be destroyed/recreated; production migration must be an explicit release step.
            _require_team_runtime()
            _require_team_isolation(existing)
            existing_name = _team_name_from_anchor(existing)
            if "team_name" in body and team_name != existing_name:
                raise ApiError(HTTPStatus.CONFLICT, "Team name differs from the persisted identity")
            _inference_store.save(team_id, inference)
            return {
                "team_id": team_id,
                "team_name": existing_name,
                "provider": inference.provider,
                "model": inference.model,
                "status": existing.status,
                "created": False,
            }
        if not _teardown_storage(team_id):
            raise ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "stale Team storage could not be cleared before creation",
            )
        # Reserve count + memory atomically, then let unrelated Teams enter admission while the
        # runtime check, credential service, Postgres, Docker start and health work proceed.
        with _reserve_capacity(f"team:{team_id}", owner, manifests.MEM_LIMIT_BYTES, team_slot=True):
            # Quotas are an admission decision of their own: an owner already at the limit must receive
            # 429 even while the hostile-tenant runtime is unavailable. A different owner reaches this
            # independent fail-closed host gate and still cannot provision without the required runtime.
            _require_team_runtime()
            # Transactional: on ANY failure, roll back everything partially created before surfacing —
            # never leak an orphan DB/role, network, or volume for an operator to hunt down later.
            container = None
            try:
                db = pgdriver_client.provision_team(team_id)
                try:
                    # Principal registration precedes every runnable/public Brain artifact. A retry
                    # reuses the same local principal and the R2 lifecycle endpoint is idempotent.
                    r2driver_client.provision_team(team_id)
                except r2driver_client.R2DriverError as exc:
                    raise ApiError(exc.status, exc.message) from exc
                network = _ensure_team_network(team_id)
                _wire_network_deps(network, manifests.core_deps())
                _require_network_policy(
                    network,
                    team_id,
                    network_policy.CORE_KIND,
                    require_brain=False,
                    require_dependencies=True,
                )
                kwargs = manifests.build_team_kwargs(
                    team_id,
                    team_name,
                    database_url=db["database_url"],
                    owner=owner,
                    brain=anchor_brain,
                    model=anchor_model,
                )
                _require_team_runtime()
                container = _docker.containers.create(**kwargs)
                _start_team_with_isolation(container)
                _inference_store.save(team_id, inference)
            except Exception as exc:
                cleanup = _teardown(
                    team_id,
                    owner=owner,
                    brain_id=container.id if container is not None else "",
                )
                if not cleanup.complete:
                    raise ApiError(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "Team create failed and rollback is incomplete; contact the operator",
                    ) from exc
                if isinstance(exc, ApiError):
                    raise
                raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "Team create failed and was rolled back") from exc
        return {
            "team_id": team_id,
            "team_name": team_name,
            "provider": inference.provider,
            "model": inference.model,
            "status": "running",
            "created": True,
            "database": manifests.team_db_project(team_id),
        }


def _destroy(team_id: str, lease: _AuthorizationLease) -> dict:
    with _lock_for(team_id):
        # Destruction is the supported remediation for a legacy or drifted runtime.
        if lease.cleanup_nonce:
            _require_cleanup_authorization(team_id, lease)
            container = None
        else:
            container = _require_current_authorization(
                team_id,
                lease,
                require_isolation=False,
                allow_pending_cleanup=True,
            )
            # A running chat is terminated by stopping the Brain before its lock can drain. Commit
            # the retry authorization first so even a timeout or ambiguous Docker stop leaves the
            # owner with a durable path back into DELETE.
            try:
                cleanup_state.begin(team_id, lease.owner, lease.container_id)
            except cleanup_state.CleanupStateError as exc:
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Team cleanup state is unavailable",
                ) from exc
        chat_lock = _chat_lock_for(team_id)
        if container is not None:
            container.reload()
            if container.status == "running":
                _fail_stop_team(container, timeout=30)
        if not chat_lock.acquire(timeout=30):
            raise ApiError(HTTPStatus.CONFLICT, "the active chat turn did not stop in time")
        try:
            try:
                _brain_runtime.delete_thread(_brain_thread_id(team_id, lease.container_id))
            except brain_runtime_client.BrainRuntimeError as exc:
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Team conversation state could not be deleted",
                ) from exc
            try:
                _power_execution_journal().purge(lease.container_id)
            except power_journal.PowerJournalError as exc:
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Team Power execution state could not be deleted",
                ) from exc
            cleanup = _teardown(team_id, owner=lease.owner, brain_id=lease.container_id)
            _clear_team_id_runtime_state(team_id)
            if not cleanup.complete:
                raise ApiError(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "Team teardown is incomplete; retry destroy or contact the operator",
                )
            return {"team_id": team_id, "destroyed": True, "db_dropped": cleanup.db_dropped}
        finally:
            chat_lock.release()


def _list(owner: str | None = None) -> dict:
    """All teams for the operator; only the account's own when `owner` is set."""
    teams = _docker.containers.list(all=True, filters={"label": "team.driver"})
    if owner is not None:
        teams = [container for container in teams if container.labels.get("team.owner", "") == owner]
    return {"teams": [_describe(container) for container in teams]}


def _status(team_id: str, lease: _AuthorizationLease) -> dict:
    with _lock_for(team_id):
        # Status remains readable so the UI can offer Stop/Destroy remediation.
        return _describe(_require_current_authorization(team_id, lease, require_isolation=False))


def _inference_status(team_id: str, lease: _AuthorizationLease) -> dict:
    with _lock_for(team_id):
        _require_current_authorization(team_id, lease)
        try:
            config = _inference_store.load(team_id)
        except inference_config.InferenceConfigError as exc:
            raise ApiError(HTTPStatus.CONFLICT, "Team model provider is not configured") from exc
    return {"team_id": team_id, "provider": config.provider, "model": config.model}


def _configure_inference(team_id: str, body: object, lease: _AuthorizationLease) -> dict:
    if not isinstance(body, dict) or set(body) != {"provider", "model"}:
        raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "inference requires provider and model")
    try:
        config = inference_config.normalize(body["provider"], body["model"])
    except inference_config.InferenceConfigError as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    with _lock_for(team_id):
        _require_current_authorization(team_id, lease)
        try:
            _inference_store.save(team_id, config)
        except inference_config.InferenceConfigError as exc:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Team model provider could not be saved") from exc
    audit.log("inference_configure", team_id, result="ok", provider=config.provider, model=config.model)
    return {"team_id": team_id, "provider": config.provider, "model": config.model}


def _logs(team_id: str, lines: int, lease: _AuthorizationLease) -> dict:
    with _lock_for(team_id):
        container = _require_current_authorization(team_id, lease, require_isolation=False)
        return {"team_id": team_id, "logs": container.logs(tail=lines).decode("utf-8", "replace")}


def _lifecycle(team_id: str, op: str, lease: _AuthorizationLease) -> dict:
    with _lock_for(team_id):
        # Stop is always available as remediation. Start/restart require both an exact per-container
        # runtime and a currently registered daemon runtime; Docker may never fall back to runc.
        container = _require_current_authorization(team_id, lease, require_isolation=op != "stop")
        if op in {"start", "restart"}:
            _require_team_runtime()
        container.reload()
        # Stop first so the provider tree cannot keep mutating the volume, then wait for the turn's
        # cleanup/finally to relinquish its slot. Start takes the same slot before making the container
        # runnable. The lifecycle lock serializes all of this with configure/deconfigure/destroy.
        if op in {"stop", "restart"} and container.status == "running":
            _fail_stop_team(container, timeout=30)
        chat_lock = _chat_lock_for(team_id)
        if not chat_lock.acquire(timeout=30):
            raise ApiError(HTTPStatus.CONFLICT, "the active chat turn did not stop in time")
        try:
            container.reload()
            if op in {"start", "restart"} and container.status != "running":
                _start_team_with_isolation(container)
        finally:
            chat_lock.release()
    return {"team_id": team_id, "op": op, "status": "ok"}


def _r2_driver_operation(
    team_id: str,
    lease: _AuthorizationLease,
    operation: Callable[[], dict[str, object]],
) -> dict[str, object]:
    """Revalidate the exact Team, lazily provision its principal, then make one fixed R2 call."""
    with _lock_for(team_id):
        _require_current_authorization(team_id, lease)
        try:
            # Existing Teams acquire R2 only here, after their owner and immutable container id
            # have both been rechecked inside the lifecycle lock. Provision is deliberately replayable.
            r2driver_client.ensure_provisioned(team_id)
            return operation()
        except r2driver_client.R2DriverError as exc:
            raise ApiError(exc.status, exc.message) from exc


# ── HTTP ─────────────────────────────────────────────────────────────────────
class _BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Thread-per-request server with hard admission and slow-client expiry."""

    daemon_threads = True

    def __init__(self, *args, max_concurrency: int = MAX_HTTP_CONCURRENCY, **kwargs) -> None:
        self._request_slots = threading.BoundedSemaphore(max_concurrency)
        super().__init__(*args, **kwargs)

    def get_request(self):
        request, client_address = super().get_request()
        request.settimeout(HTTP_CONNECTION_TIMEOUT_SECONDS)
        return request, client_address

    def process_request(self, request, client_address) -> None:
        # Backpressure happens before a thread exists. At the ceiling, at most the kernel's bounded
        # listen backlog plus this accepted socket waits; Python thread count cannot grow unbounded.
        self._request_slots.acquire()
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._request_slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()


class Handler(BaseHTTPRequestHandler):
    server_version = "team-driver/1.0"

    def log_message(self, *_args) -> None:  # audit.log is the ONLY log source
        pass

    def _principal(self) -> tuple[str, str | None] | None:
        """('operator', None) for the admin bearer; ('account', <id>) for a valid account token; else None.

        The operator token (the admin panel) has full access. A store-forwarded account token is verified
        against the accounts service and scopes every op to that account's OWN teams — the store holds
        no privileged secret, this driver is the enforcer.
        """
        if self.headers.get("Authorization", "") == f"Bearer {_token}":
            return ("operator", None)
        account_token = self.headers.get("X-Shimpz-Account", "")
        if account_token:
            account_id = accounts_client.verify(account_token)
            if account_id:
                return ("account", account_id)
        return None

    def _send_json(self, status: HTTPStatus, payload: dict, *, no_store: bool = False) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if no_store:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream_chat(
        self,
        team_id: str,
        message: str,
        file_ids: object,
        lease: _AuthorizationLease,
    ) -> None:
        """Preserve the NDJSON transport while exposing only the validated terminal reply."""
        terminal: dict[str, object]
        stream_error = None
        with _exclusive_chat_turn(team_id, lease) as (token, container):
            # The durable token is claimed before a 200 or any response byte reaches the client.
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def emit(obj: dict) -> None:
                line = (json.dumps(obj, ensure_ascii=False) + "\n").encode()
                self.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n")
                self.wfile.flush()

            try:
                result = _chat_in_turn(team_id, message, file_ids, token, container, lease.owner)
                terminal = {
                    "type": "done",
                    "reply": result["reply"],
                    "team_id": result["team_id"],
                }
                emit(terminal)
            except ApiError as exc:
                terminal = (
                    {"type": "stopped"}
                    if exc.status == HTTPStatus.CONFLICT and exc.message == "brain turn stopped"
                    else {"type": "error", "status": int(exc.status), "detail": exc.message}
                )
                with contextlib.suppress(OSError):
                    emit(terminal)
            except (docker.errors.DockerException, OSError) as exc:
                stream_error = type(exc).__name__
                terminal = {"type": "error", "status": 500, "detail": "brain stream failed"}
                with contextlib.suppress(OSError):
                    emit(terminal)
            finally:
                with contextlib.suppress(OSError):
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
        audit.log(
            "chat",
            team_id,
            result="ok" if terminal["type"] == "done" else "error",
            streamed=True,
            status=terminal.get("status"),
            reason=stream_error,
        )

    def _read_body(self, *, max_bytes: int = MAX_JSON_BODY_BYTES) -> dict:
        raw_length = self.headers.get("Content-Length", "0") or "0"
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "invalid Content-Length") from exc
        if length < 0:
            raise ApiError(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
        if length > max_bytes:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"request body too large (max {max_bytes} bytes)")
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid JSON body: {exc}") from exc

    def _read_driver_body(self, keys: set[str]) -> dict[str, object]:
        """Read one closed Driver mutation document; arbitrary scripts/shapes never cross the bridge."""
        if self.headers.get("Transfer-Encoding") is not None:
            raise ApiError(HTTPStatus.BAD_REQUEST, "chunked Driver requests are not supported")
        content_type = self.headers.get("Content-Type", "").partition(";")[0].strip().lower()
        if content_type != "application/json":
            raise ApiError(HTTPStatus.BAD_REQUEST, "Content-Type must be application/json")
        body = self._read_body(max_bytes=MAX_DRIVER_JSON_BODY_BYTES)
        if not isinstance(body, dict) or set(body) != keys:
            raise ApiError(HTTPStatus.BAD_REQUEST, "request body does not match the Driver operation")
        return body

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        principal = self._principal()
        if principal is None:
            if self.client_address[0] == "127.0.0.1":
                audit.log("auth", self.path, result="denied", level="info", source="loopback-probe")
            else:
                audit.log("auth", self.path, result="denied")
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "invalid or missing credentials"})
            return
        try:
            self._route(method, principal)
        except ApiError as exc:
            audit.log(method.lower(), self.path, result="denied", reason=exc.message)
            self._send_json(exc.status, {"error": exc.message})
        except validate.ValidationError as exc:
            audit.log(method.lower(), self.path, result="denied", reason=str(exc))
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except marketplace.MarketplaceError as exc:
            audit.log(method.lower(), self.path, result="denied", reason=str(exc))
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except Exception as exc:
            audit.log(method.lower(), self.path, result="error", reason=type(exc).__name__)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal driver error"})

    def _route(self, method: str, principal: tuple[str, str | None]) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        parts = [p for p in path.split("/") if p]
        kind, account_id = principal

        if method == "GET" and path == "/v1/teams":
            self._send_json(HTTPStatus.OK, _list(owner=account_id if kind == "account" else None))
            return

        if len(parts) >= 3 and parts[0] == "v1" and parts[1] == "teams":
            team_id = validate.validate_team_id(parts[2])
            sub = parts[3] if len(parts) > 3 else ""
            if method == "POST" and sub == "create":
                _enforce_rate("create", principal)
                body = self._read_body()
                # an account owns what it creates; an operator may create-on-behalf via an explicit owner
                owner = account_id or str(body.get("owner", "")).strip()
                result = _create(team_id, body, owner)
                trace = audit.log("create", team_id, result="ok", created=result.get("created"), owner=owner)
                self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
                return
            if method == "DELETE" and sub == "":
                # A completed Brain removal may leave a bounded durable cleanup record while volume
                # deletion is retried. Only Destroy may authorize against that non-runnable successor.
                lease = _authorize_destroy(team_id, principal)
                result = _destroy(team_id, lease)
                trace = audit.log("destroy", team_id, result="ok", db_dropped=result["db_dropped"])
                self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
                return
            # every other op acts on an EXISTING team → gate on ownership first (404 if not yours)
            lease = _authorize(team_id, principal)
            if sub == "drivers":
                self._route_driver(method, parts, team_id, lease)
                return
            if sub == "apps":
                self._route_apps(method, parts, team_id, principal, lease)
                return
            if sub == "assistants":
                if len(parts) >= 6 and parts[5] == "help" and (parsed.query or parsed.fragment or "%" in parsed.path):
                    raise ApiError(HTTPStatus.BAD_REQUEST, "query and encoded paths are not accepted")
                self._route_assistants(method, parts, team_id, lease)
                return
            if sub == "inference":
                self._route_inference(method, parts, team_id, lease)
                return
            if sub == "chat":
                self._route_chat(method, parts, team_id, principal, lease)
                return
            if sub == "files":
                self._route_files(method, parts, team_id, lease, principal)
                return
            if method == "GET" and sub == "status":
                self._send_json(HTTPStatus.OK, _status(team_id, lease))
                return
            if method == "GET" and sub == "logs":
                self._send_json(HTTPStatus.OK, _logs(team_id, int(query.get("lines", "200")), lease))
                return
            if method == "POST" and sub in ("stop", "start", "restart"):
                result = _lifecycle(team_id, sub, lease)
                audit.log(sub, team_id, result="ok")
                self._send_json(HTTPStatus.OK, result)
                return

        raise ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} {path}")

    def _route_files(
        self,
        method: str,
        parts: list[str],
        team_id: str,
        lease: _AuthorizationLease,
        principal: tuple[str, str | None],
    ) -> None:
        if method == "GET" and len(parts) == 4:
            self._send_json(HTTPStatus.OK, _list_team_files(team_id, lease))
            return
        if method == "POST" and len(parts) == 4:
            _enforce_rate("file_upload", principal)
            if not _file_upload_slots.acquire(blocking=False):
                raise ApiError(HTTPStatus.TOO_MANY_REQUESTS, "another Team file upload is in progress")
            try:
                body = self._read_body(max_bytes=MAX_FILE_BODY_BYTES)
                if not isinstance(body, dict) or set(body) not in (
                    {"filename", "content_b64"},
                    {"filename", "content_b64", "media_type"},
                ):
                    raise ApiError(
                        HTTPStatus.UNPROCESSABLE_ENTITY,
                        "file upload requires filename, content_b64, and optional media_type",
                    )
                result = _put_inbox_file(
                    team_id,
                    body["filename"],
                    body["content_b64"],
                    body.get("media_type"),
                    lease,
                )
            finally:
                _file_upload_slots.release()
            trace = audit.log(
                "team_file_upload",
                team_id,
                result="ok",
                file_id=result["file"]["id"],
                bytes=result["file"]["size"],
            )
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        if method == "DELETE" and len(parts) == 5:
            result = _delete_team_file(team_id, parts[4], lease)
            trace = audit.log(
                "team_file_delete",
                team_id,
                result="ok",
                file_id=result["id"],
                deleted=result["deleted"],
            )
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        raise ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} /{'/'.join(parts)}")

    def _route_inference(
        self,
        method: str,
        parts: list[str],
        team_id: str,
        lease: _AuthorizationLease,
    ) -> None:
        if len(parts) == 4 and method == "GET":
            self._send_json(HTTPStatus.OK, _inference_status(team_id, lease))
            return
        if len(parts) == 4 and method == "PUT":
            self._send_json(HTTPStatus.OK, _configure_inference(team_id, self._read_body(), lease))
            return
        raise ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} /{'/'.join(parts)}")

    def _route_driver(
        self,
        method: str,
        parts: list[str],
        team_id: str,
        lease: _AuthorizationLease,
    ) -> None:
        """Closed Admin surface for the single proven Driver implementation: Cloudflare R2."""
        if len(parts) < 5 or parts[4] != "r2":
            raise ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} /{'/'.join(parts)}")
        if method == "GET" and len(parts) == 5:
            result = _r2_driver_operation(team_id, lease, lambda: r2driver_client.driver_document(team_id))
            self._send_json(HTTPStatus.OK, result)
            return
        if method == "POST" and len(parts) == 6 and parts[5] == "credentials":
            body = self._read_driver_body({"profile_id", "label", "values", "idempotency_key"})
            result = _r2_driver_operation(team_id, lease, lambda: r2driver_client.create_credential(team_id, body))
            audit.log("driver_credential_create", team_id, result="ok", driver="r2", credential=result.get("id"))
            self._send_json(HTTPStatus.OK, result)
            return
        if len(parts) == 7 and parts[5] == "credentials":
            credential_id = parts[6]
            if method == "PUT":
                body = self._read_driver_body(
                    {"profile_id", "label", "values", "expected_generation"},
                )
                result = _r2_driver_operation(
                    team_id,
                    lease,
                    lambda: r2driver_client.rotate_credential(team_id, credential_id, body),
                )
                audit.log(
                    "driver_credential_rotate",
                    team_id,
                    result="ok",
                    driver="r2",
                    credential=result.get("id"),
                )
                self._send_json(HTTPStatus.OK, result)
                return
            if method == "DELETE":
                body = self._read_driver_body({"expected_generation"})
                result = _r2_driver_operation(
                    team_id,
                    lease,
                    lambda: r2driver_client.remove_credential(team_id, credential_id, body),
                )
                audit.log(
                    "driver_credential_remove",
                    team_id,
                    result="ok",
                    driver="r2",
                    credential=result.get("id"),
                )
                self._send_json(HTTPStatus.OK, result)
                return
        if method == "POST" and len(parts) == 8 and parts[5] == "credentials" and parts[7] == "verify":
            credential_id = parts[6]
            self._read_driver_body(set())
            result = _r2_driver_operation(
                team_id,
                lease,
                lambda: r2driver_client.verify_credential(team_id, credential_id),
            )
            audit.log(
                "driver_credential_verify",
                team_id,
                result="ok",
                driver="r2",
                credential=result.get("id"),
            )
            self._send_json(HTTPStatus.OK, result)
            return
        raise ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} /{'/'.join(parts)}")

    def _route_chat(
        self,
        method: str,
        parts: list[str],
        team_id: str,
        principal: tuple[str, str | None],
        lease: _AuthorizationLease,
    ) -> None:
        """/v1/teams/{team_id}/chat[/stream|/stop|/asks|/answer] — the Captain's brain conversation.

        Ownership was already enforced by _authorize. `chat` (bare) is the non-streaming fallback;
        `chat/stream` is the live NDJSON turn; the rest are the shimpz-ask surface + the Stop control.
        """
        sub2 = parts[4] if len(parts) > 4 else ""
        if method == "POST" and sub2 in {"", "stream"}:
            body = self._read_body()
            if not isinstance(body, dict) or set(body) not in ({"message"}, {"message", "files"}):
                raise ApiError(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "Team chat requires message and optional files",
                )
            message = validate.validate_chat_message(body["message"])
            file_ids = body.get("files")
            if sub2 == "stream":
                _enforce_rate("stream", principal)
                self._stream_chat(team_id, message, file_ids, lease)
                return
            _enforce_rate("chat", principal)
            result = _chat(team_id, message, file_ids, lease)
            audit.log(
                "chat",
                team_id,
                result="ok",
                chars_in=len(message),
                chars_out=len(result["reply"]),
            )
            self._send_json(HTTPStatus.OK, result)
            return
        if method == "POST" and sub2 == "stop":
            _enforce_rate("stop", principal)
            self._send_json(HTTPStatus.OK, _stop_chat(team_id, lease))
            return
        raise ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} /{'/'.join(parts)}")

    def _route_apps(
        self,
        method: str,
        parts: list[str],
        team_id: str,
        principal: tuple[str, str | None],
        lease: _AuthorizationLease,
    ) -> None:
        """/v1/teams/{team_id}/apps[/{app}] — the P4 deploy arm. Ownership was already enforced."""
        kind, account_id = principal
        if method == "POST" and len(parts) == 4:
            _enforce_rate("install", principal)
            app_id, spec = marketplace.resolve(self._read_body().get("app"))
            # The marketplace gate, enforced where the socket lives: a NON-first-party app needs a
            # VERIFIED Shimpz account — on a self-hosted Space the verify call IS the phone-home
            # (SHIMPZ_ACCOUNTS_URL → shimpz.com), so not even the Space operator bypasses it.
            if not spec.first_party and kind != "account":
                raise ApiError(HTTPStatus.UNAUTHORIZED, f"installing {app_id!r} requires a valid Shimpz account")
            owner = account_id or lease.owner
            result = _install_app(team_id, app_id, spec, owner, lease)
            trace = audit.log("install", team_id, result="ok", app=app_id, installed=result["installed"])
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        if method == "GET" and len(parts) == 4:
            self._send_json(HTTPStatus.OK, _list_apps(team_id, lease))
            return
        if method == "DELETE" and len(parts) == 5:
            # Shape-validated only — NOT resolved: an app later pulled from the registry must still
            # be uninstallable from every team that has it.
            app_id = marketplace.validate_app_id(parts[4])
            result = _uninstall_app(team_id, app_id, lease)
            trace = audit.log("uninstall", team_id, result="ok", app=app_id, db_dropped=result["db_dropped"])
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        raise ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} /{'/'.join(parts)}")

    def _route_assistants(
        self,
        method: str,
        parts: list[str],
        team_id: str,
        lease: _AuthorizationLease,
    ) -> None:
        """Expose only fixed read contracts; install lifecycle remains on the canonical Apps route."""
        if method == "GET" and len(parts) in {6, 7} and parts[5] == "help":
            assistant_id = marketplace.validate_app_id(parts[4])
            locale = parts[6] if len(parts) == 7 else "en"
            help_payload = _assistant_help(team_id, assistant_id, lease, locale)
            trace = audit.log(
                "assistant_help",
                team_id,
                result="ok",
                assistant=help_payload["assistant"],
            )
            self._send_json(
                HTTPStatus.OK,
                {**help_payload, "trace_id": trace},
                no_store=True,
            )
            return
        raise ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} /{'/'.join(parts)}")


def main() -> None:
    # The Controller owns this bearer. The runtime receives the same named volume read-only and
    # cannot rotate or replace its authority.
    brain_runtime_token_store.ensure()
    _BoundedThreadingHTTPServer((ALL_INTERFACES, LISTEN_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
