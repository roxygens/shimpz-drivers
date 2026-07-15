"""capsule-driver — a socket-holding sidecar dedicated to Capsule lifecycle.

Besides shimpz-driver, this is the ONLY container holding /var/run/docker.sock — and it exposes ONLY
named operations (create/list/status/logs/stop/start/restart/destroy), never a generic Docker
passthrough. A Capsule is one isolated `shimpz-brain`: its OWN internal network, its OWN config+workspace
volumes, and a SCOPED Postgres database (provisioned via pg-driver — this driver never holds the
superuser). Every mutating call is bearer-gated → validated → mutated → audited (trace_id returned).
A compromised caller can only ever request what validate.py permits.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import http.client
import ipaddress
import json
import math
import os
import secrets
import socket
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
import audit
import brain_credentials_client
import cleanup_state
import docker
import docker.errors
import docker.utils.socket as docker_socket
import manifests
import marketplace
import network_policy
import pgdriver_client
import r2driver_client
import token_store
import validate

ALL_INTERFACES = str(ipaddress.IPv4Address(0))


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


LISTEN_PORT = int(os.environ.get("SHIMPZ_CAPSULEDRIVER_PORT", "7077"))
# The host has 125 GiB and each Capsule has a 2 GiB hard ceiling: 32 leaves roughly half the host for
# the platform, installed apps and Docker overhead. Operators may lower it, but public callers never
# choose either quota.
MAX_CAPSULES = _positive_int_env("SHIMPZ_MAX_CAPSULES", 32)
MAX_CAPSULES_PER_OWNER = _positive_int_env("SHIMPZ_MAX_CAPSULES_PER_OWNER", 1)
# Per-capsule app allowance — an owner can't exhaust the host by installing without bound either.
MAX_APPS_PER_CAPSULE = _positive_int_env("SHIMPZ_MAX_APPS_PER_CAPSULE", 20)
GLOBAL_MEMORY_BUDGET_BYTES = manifests.hard_memory_bytes(
    os.environ.get("SHIMPZ_CAPSULE_GLOBAL_MEM_BUDGET", "64g"),
    setting="SHIMPZ_CAPSULE_GLOBAL_MEM_BUDGET",
)
OWNER_MEMORY_BUDGET_BYTES = manifests.hard_memory_bytes(
    os.environ.get("SHIMPZ_CAPSULE_OWNER_MEM_BUDGET", "8g"),
    setting="SHIMPZ_CAPSULE_OWNER_MEM_BUDGET",
)
_LARGEST_RESOURCE_LIMIT = max(manifests.MEM_LIMIT_BYTES, manifests.APP_MEM_LIMIT_BYTES)
if GLOBAL_MEMORY_BUDGET_BYTES < _LARGEST_RESOURCE_LIMIT:
    raise ValueError("SHIMPZ_CAPSULE_GLOBAL_MEM_BUDGET is smaller than one Capsule resource")
if not _LARGEST_RESOURCE_LIMIT <= OWNER_MEMORY_BUDGET_BYTES <= GLOBAL_MEMORY_BUDGET_BYTES:
    raise ValueError("SHIMPZ_CAPSULE_OWNER_MEM_BUDGET must fit one resource and the global memory budget")
MAX_JSON_BODY_BYTES = max(1024, int(os.environ.get("SHIMPZ_CAPSULE_MAX_JSON_BODY_BYTES", str(128 * 1024))))
MAX_DRIVER_JSON_BODY_BYTES = 64 * 1024
CREATE_RATE_LIMIT = _positive_int_env("SHIMPZ_CAPSULE_CREATE_RATE_LIMIT", 5)
CREATE_RATE_WINDOW_SECONDS = _positive_int_env("SHIMPZ_CAPSULE_CREATE_RATE_WINDOW_SECONDS", 3600)
INSTALL_RATE_LIMIT = _positive_int_env("SHIMPZ_CAPSULE_INSTALL_RATE_LIMIT", 20)
INSTALL_RATE_WINDOW_SECONDS = _positive_int_env("SHIMPZ_CAPSULE_INSTALL_RATE_WINDOW_SECONDS", 3600)
CHAT_RATE_LIMIT = _positive_int_env("SHIMPZ_CAPSULE_CHAT_RATE_LIMIT", 30)
CHAT_RATE_WINDOW_SECONDS = _positive_int_env("SHIMPZ_CAPSULE_CHAT_RATE_WINDOW_SECONDS", 60)
MAX_HTTP_CONCURRENCY = _positive_int_env("SHIMPZ_CAPSULE_MAX_HTTP_CONCURRENCY", 64)
HTTP_CONNECTION_TIMEOUT_SECONDS = _positive_int_env("SHIMPZ_CAPSULE_HTTP_CONNECTION_TIMEOUT_SECONDS", 30)
# Same volume app-egress-proxy reads (<token>.json allowlists) — shared with shimpz-driver by design:
# ONE proxy serves every token-gated app, capsule-scoped or not, each confined to its own hosts.
APP_EGRESS_POLICY_DIR = Path(os.environ.get("SHIMPZ_APP_EGRESS_POLICY_DIR", "/app-egress-policy"))
HEALTH_RETRIES = int(os.environ.get("SHIMPZ_HEALTH_RETRIES", "40"))
HEALTH_DELAY_SECONDS = float(os.environ.get("SHIMPZ_HEALTH_DELAY_SECONDS", "1.5"))

_docker = docker.from_env()
_token = token_store.ensure_token()

# Per-capsule lock: create/destroy of the SAME capsule must serialize; different capsules run parallel.
# Weak maps retain one lock exactly while a holder or waiter has a strong reference. After destroy (or
# any other terminal operation), the final holder releases its reference and the CID disappears without
# ever allowing an old locked object and a new unlocked object to coexist.
_locks_guard = threading.Lock()
_locks: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
_chat_locks_guard = threading.Lock()
_chat_locks: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
_active_chat_guard = threading.Lock()
_active_chat_tokens: dict[str, str] = {}
_active_chat_container_ids: dict[str, str] = {}
_cancelled_chat_tokens: set[str] = set()
_blocked_chat_capsules: set[str] = set()
_credential_mutations: set[str] = set()
_credential_mutation_execs: dict[str, tuple[str, str]] = {}
_CHAT_BUSY_EXIT = 75
_CHAT_BUSY_MARKER = "shimpz-chat durable slot is already active"
# `authenticated` is a statement about an observed successful provider turn, not about a credential
# file existing. Bind that observation to the Docker container identity + provider so a destroy/recreate
# or provider replacement can never inherit it. Driver restarts conservatively return false until the
# next successful real turn.
_verified_brains: dict[str, tuple[str, str]] = {}
# The capacity lock protects only inventory + reservation mutations. Slow provisioning is represented
# by `_capacity_reservations` and runs after this lock is released.
_capacity_lock = threading.Lock()


def _lock_for(cid: str) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(cid)
        if lock is None:
            lock = threading.Lock()
            _locks[cid] = lock
        return lock


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


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
    "asks": _FixedWindowRateLimiter(CHAT_RATE_LIMIT, CHAT_RATE_WINDOW_SECONDS),
    "answer": _FixedWindowRateLimiter(CHAT_RATE_LIMIT, CHAT_RATE_WINDOW_SECONDS),
}


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


def _chat_lock_for(cid: str) -> threading.Lock:
    with _chat_locks_guard:
        lock = _chat_locks.get(cid)
        if lock is None:
            lock = threading.Lock()
            _chat_locks[cid] = lock
        return lock


def _clear_brain_authentication(cid: str) -> None:
    with _active_chat_guard:
        _verified_brains.pop(cid, None)


def _require_no_credential_mutation(cid: str) -> None:
    with _active_chat_guard:
        if cid in _credential_mutations:
            raise ApiError(HTTPStatus.CONFLICT, "Brain credential change is still in progress")


def _mark_brain_authenticated(cid: str, container) -> None:
    with _active_chat_guard:
        _verified_brains[cid] = (container.id, _brain_id(container))


def _brain_authenticated(cid: str, container) -> bool:
    with _active_chat_guard:
        return _verified_brains.get(cid) == (container.id, _brain_id(container))


def _clear_cid_runtime_state(cid: str) -> None:
    """Forget terminal in-memory state without deleting a lock that another request references."""
    with _active_chat_guard:
        token = _active_chat_tokens.pop(cid, None)
        _active_chat_container_ids.pop(cid, None)
        if token is not None:
            _cancelled_chat_tokens.discard(token)
        _blocked_chat_capsules.discard(cid)
        _credential_mutations.discard(cid)
        _credential_mutation_execs.pop(cid, None)
        _verified_brains.pop(cid, None)


def _token_cancelled(token: str) -> bool:
    with _active_chat_guard:
        return token in _cancelled_chat_tokens


def _commit_chat_terminal(cid: str, token: str) -> bool:
    """Linearization point: False means a user Stop acquired the token first."""
    with _active_chat_guard:
        if token in _cancelled_chat_tokens:
            return False
        if _active_chat_tokens.get(cid) == token:
            _active_chat_tokens.pop(cid, None)
            _active_chat_container_ids.pop(cid, None)
        return True


def _chat_control_output(result) -> str:
    output = result.output or b""
    return output.decode("utf-8", "replace") if isinstance(output, bytes) else str(output)


def _durable_active_chat_token(container) -> str | None:
    """Recover this exact container's root-owned active turn after driver state loss."""
    try:
        result = container.exec_run(
            ["shimpz-chat-stop", "current", container.id],
            user=_CHAT_CONTROL_USER,
        )
    except docker.errors.DockerException as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Capsule chat admission state is unavailable") from exc
    if result.exit_code == 3:
        return None
    output = _chat_control_output(result).strip()
    if result.exit_code != 0 or len(output) != 32 or any(character not in "0123456789abcdef" for character in output):
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Capsule chat admission state is invalid")
    return output


def _claim_durable_chat_turn(container, token: str) -> None:
    try:
        result = container.exec_run(
            ["shimpz-chat-stop", "claim", token, container.id],
            user=_CHAT_CONTROL_USER,
        )
    except docker.errors.DockerException as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Capsule chat admission state is unavailable") from exc
    output = _chat_control_output(result).lower()
    if result.exit_code == 0:
        return
    if result.exit_code == _CHAT_BUSY_EXIT and _CHAT_BUSY_MARKER in output:
        raise ApiError(HTTPStatus.CONFLICT, "capsule already has an active chat turn")
    raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Capsule chat admission claim failed")


def _require_no_durable_chat_turn(container) -> None:
    if _durable_active_chat_token(container) is not None:
        raise ApiError(HTTPStatus.CONFLICT, "stop the active chat turn before changing its Brain credential")


def _request_chat_stop(cid: str, container) -> tuple[str | None, bool]:
    """Linearization point for the user Stop side of a turn."""
    with _active_chat_guard:
        token = _active_chat_tokens.get(cid)
        if token is not None and _active_chat_container_ids.get(cid) != container.id:
            raise ApiError(HTTPStatus.NOT_FOUND, f"capsule {cid!r} not found")
        if token is not None:
            _cancelled_chat_tokens.add(token)
            return token, False
    recovered = _durable_active_chat_token(container)
    with _active_chat_guard:
        # A same-driver turn may have won admission while the Docker recovery exec was in flight.
        token = _active_chat_tokens.get(cid)
        if token is not None:
            if _active_chat_container_ids.get(cid) != container.id:
                raise ApiError(HTTPStatus.NOT_FOUND, f"capsule {cid!r} not found")
            _cancelled_chat_tokens.add(token)
            return token, False
        if recovered is not None:
            _cancelled_chat_tokens.add(recovered)
        return recovered, recovered is not None


@contextlib.contextmanager
def _exclusive_chat_turn(cid: str, lease: _AuthorizationLease):
    """Hold the one active brain-turn slot for `cid`, or fail immediately with 409.

    This lock is deliberately separate from lifecycle/app mutation locks. In particular, Stop never
    takes it: a Captain must always be able to terminate the turn that currently owns the slot.
    """
    lock = _chat_lock_for(cid)
    if not lock.acquire(blocking=False):
        raise ApiError(HTTPStatus.CONFLICT, f"capsule {cid!r} already has an active chat turn")
    try:
        container = _require_current_authorization(cid, lease)
        container.reload()
        if container.status != "running":
            raise ApiError(HTTPStatus.CONFLICT, f"capsule {cid!r} is not running (status={container.status})")
    except BaseException:
        lock.release()
        raise
    token = secrets.token_hex(16)
    with _active_chat_guard:
        if cid in _blocked_chat_capsules:
            lock.release()
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Capsule chat cleanup requires a lifecycle restart")
        if cid in _credential_mutations:
            lock.release()
            raise ApiError(HTTPStatus.CONFLICT, "Brain credential change is still in progress")
    try:
        _claim_durable_chat_turn(container, token)
    except BaseException:
        lock.release()
        raise
    with _active_chat_guard:
        _active_chat_tokens[cid] = token
        _active_chat_container_ids[cid] = container.id
    try:
        yield token, container
    finally:
        cleanup_ok = True
        try:
            try:
                container.reload()
            except docker.errors.DockerException:
                cleanup_ok = False
            else:
                if container.status == "running":
                    try:
                        result = container.exec_run(["shimpz-chat-stop", "clear", token], user=_CHAT_CONTROL_USER)
                        cleanup_ok = result.exit_code == 0
                    except docker.errors.DockerException:
                        cleanup_ok = False
                    if not cleanup_ok:
                        try:
                            _fail_stop_capsule(container)
                            cleanup_ok = True
                        except ApiError:
                            cleanup_ok = False
        finally:
            with _active_chat_guard:
                _active_chat_tokens.pop(cid, None)
                _active_chat_container_ids.pop(cid, None)
                _cancelled_chat_tokens.discard(token)
                if cleanup_ok:
                    _blocked_chat_capsules.discard(cid)
                else:
                    _blocked_chat_capsules.add(cid)
            lock.release()


# ── docker helpers ───────────────────────────────────────────────────────────
def _get_container(name: str):
    try:
        return _docker.containers.get(name)
    except docker.errors.NotFound:
        return None


def _require_capsule_runtime() -> None:
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
            f"required Capsule runtime {manifests.RUNTIME!r} is not loaded from {manifests.RUNTIME_PATH!r}",
        )
    if not network_policy.daemon_security_options_valid(info):
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "required Docker built-in seccomp and AppArmor defaults are unavailable",
        )


def _capsule_runtime(container) -> str:
    """Return Docker's immutable runtime selection for one existing Capsule."""
    try:
        container.reload()
    except docker.errors.DockerException as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "cannot verify Capsule runtime isolation") from exc
    runtime = container.attrs.get("HostConfig", {}).get("Runtime")
    return str(runtime or "runc")


def _trusted_image_id(image_ref: str) -> str:
    """Resolve one release-owned local reference to the immutable ID Engine will execute."""
    if not isinstance(image_ref, str) or not image_ref:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Capsule isolation is blocked: untrusted workload image role")
    try:
        image = _docker.images.get(image_ref)
        image_id = image.id
    except (AttributeError, docker.errors.DockerException) as exc:
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Capsule isolation is blocked: trusted workload image is unavailable",
        ) from exc
    if not isinstance(image_id, str) or not image_id:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Capsule isolation is blocked: invalid workload image identity")
    return image_id


def _trusted_workload_image(container, cid: str) -> tuple[str, str]:
    """Resolve this exact workload role's configured ref to the currently trusted immutable ID."""
    labels = container.attrs.get("Config", {}).get("Labels", {})
    if network_policy.brain_identity_valid(container.attrs, cid):
        provider = labels.get("capsule.brain")
        provider_spec = manifests.BRAINS.get(provider) if isinstance(provider, str) else None
        image_ref = provider_spec.get("image") if provider_spec is not None else None
    else:
        app_id = labels.get("capsule.app")
        app_spec = marketplace.APPS.get(app_id) if isinstance(app_id, str) else None
        image_ref = app_spec.image if app_spec is not None else None
    if not isinstance(image_ref, str) or not image_ref:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Capsule isolation is blocked: untrusted workload image role")
    return image_ref, _trusted_image_id(image_ref)


def _require_capsule_isolation_mode(container, *, require_running: bool) -> None:
    """Validate exact static posture, plus live network membership whenever the workload is running."""
    runtime = _capsule_runtime(container)
    if runtime != manifests.RUNTIME:
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            f"Capsule isolation is blocked: required runtime {manifests.RUNTIME!r}, found {runtime!r}; "
            "destroy and recreate the Capsule",
        )
    state = container.attrs.get("State")
    running = state.get("Running") if isinstance(state, dict) else None
    if not isinstance(running, bool) or (require_running and not running):
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Capsule isolation is blocked: workload running state cannot be proved",
        )
    labels = container.attrs.get("Config", {}).get("Labels", {})
    cid = labels.get("capsule.id") if isinstance(labels, dict) else None
    if not isinstance(cid, str) or not cid:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Capsule isolation is blocked: invalid workload identity")
    image_ref, image_id = _trusted_workload_image(container, cid)
    if not network_policy.workload_security_valid(
        container.attrs,
        cid,
        manifests.RUNTIME,
        expected_image_ref=image_ref,
        expected_image_id=image_id,
    ):
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Capsule isolation is blocked: workload security or network attachment drifted; "
            "destroy and recreate the Capsule",
        )
    expected_kinds = network_policy.workload_network_kinds(container.attrs, cid)
    if expected_kinds is None:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Capsule isolation is blocked: invalid workload role")
    is_brain = network_policy.BRAIN_EGRESS_KIND in expected_kinds
    _require_capsule_volumes(cid)
    for kind in (network_policy.CORE_KIND, network_policy.BRAIN_EGRESS_KIND):
        try:
            network = _docker.networks.get(network_policy.network_name(cid, kind))
        except docker.errors.DockerException as exc:
            raise ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Capsule isolation is blocked: required network is missing",
            ) from exc
        _require_network_policy(
            network,
            cid,
            kind,
            # Engine 29 omits created/stopped endpoints from network inspect while preserving their
            # exact attachments in container inspect. Static workload posture above proves those
            # endpoints; a running Brain must additionally be visible as the plane's live Brain role.
            require_brain=running and is_brain,
            require_dependencies=True,
        )
        if kind in expected_kinds and not network_policy.workload_endpoint_valid(
            network.attrs,
            container.attrs,
            cid,
            kind,
        ):
            raise ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Capsule isolation is blocked: workload endpoint identity or aliases drifted",
            )
        if (
            running
            and kind in expected_kinds
            and not network_policy.workload_live_membership_valid(
                network.attrs,
                container.attrs,
                cid,
                kind,
            )
        ):
            raise ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Capsule isolation is blocked: running workload is missing from its network inventory",
            )


def _require_capsule_isolation(container) -> None:
    """State-aware admission: stopped is exact/static; running additionally proves live membership."""
    _require_capsule_isolation_mode(container, require_running=False)


def _require_running_capsule_isolation(container) -> None:
    """Require a running workload and its complete live two-plane membership."""
    _require_capsule_isolation_mode(container, require_running=True)


def _capsule_not_running(container) -> bool:
    """Return true only for an exact stopped state or an absent container identity."""
    try:
        container.reload()
    except docker.errors.NotFound:
        return True
    except docker.errors.DockerException:
        return False
    state = container.attrs.get("State")
    return isinstance(state, dict) and state.get("Running") is False


def _fail_stop_capsule(container, *, timeout: int = 10) -> None:
    """Actively stop/kill and prove a workload is no longer running."""
    try:
        container.stop(timeout=timeout)
    except docker.errors.NotFound:
        return
    except docker.errors.DockerException:
        pass
    if _capsule_not_running(container):
        return
    try:
        container.kill()
    except docker.errors.NotFound:
        return
    except docker.errors.DockerException:
        pass
    if not _capsule_not_running(container):
        raise ApiError(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Capsule isolation failed and the workload could not be proved stopped",
        )


def _remove_capsule_container(container, *, timeout: int = 10) -> bool:
    """Remove one workload by immutable ID, fail-stopping any survivor before returning false."""
    container_id = container.id
    try:
        _fail_stop_capsule(container, timeout=timeout)
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
        _fail_stop_capsule(survivor, timeout=timeout)
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


def _start_capsule_with_isolation(container) -> None:
    """Prove a stopped workload, start it, then fail-stop unless live membership also proves."""
    _require_capsule_isolation(container)
    # Re-read Docker's daemon posture at the final mutation boundary. A registration or default-profile
    # drift after create/preflight must leave the hostile workload stopped.
    _require_capsule_runtime()
    try:
        container.start()
    except docker.errors.DockerException as exc:
        # Engine may have committed a start before the client observed its response. Treat every
        # start error as ambiguous and actively fail-stop instead of assuming the workload stayed down.
        _fail_stop_capsule(container)
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Capsule start could not be proved; workload was stopped",
        ) from exc
    try:
        _require_running_capsule_isolation(container)
    except ApiError:
        _fail_stop_capsule(container)
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
    capsule_slot: bool


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
    cid = str(container.labels.get("capsule.id", ""))
    if container.labels.get("capsule.app.driver"):
        return f"app:{cid}:{container.labels.get('capsule.app', '')}"
    return f"capsule:{cid}"


def _admitted_resource_containers() -> list:
    """Return every hard-limited Capsule resource once, or fail closed on Docker inventory errors."""
    try:
        resources = {
            container.id: container
            for label in ("capsule.driver", "capsule.app.driver")
            for container in _docker.containers.list(all=True, filters={"label": label})
        }
    except docker.errors.DockerException as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "cannot verify Capsule memory inventory") from exc
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
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "cannot verify Capsule memory hard limits") from exc
        if isinstance(raw_limit, bool) or not isinstance(raw_limit, (int, float)) or raw_limit <= 0:
            raise ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                f"Capsule resource {container.name!r} has no verifiable memory hard limit",
            )
        limit = int(raw_limit)
        owner = str(container.labels.get("capsule.owner", ""))
        total += limit
        by_owner[owner] += limit
    return _MemoryUsage(total=total, by_owner=dict(by_owner))


def _physical_capsules(*, exclude_keys: frozenset[str]) -> list:
    try:
        capsules = _docker.containers.list(all=True, filters={"label": "capsule.driver"})
    except docker.errors.DockerException as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "cannot verify Capsule count inventory") from exc
    return [container for container in capsules if _capacity_key(container) not in exclude_keys]


@contextlib.contextmanager
def _reserve_capacity(
    key: str,
    owner: str,
    requested: int,
    *,
    capsule_slot: bool,
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
        capsule_slot=capsule_slot,
    )
    with _capacity_lock:
        if key in _capacity_reservations:
            raise ApiError(HTTPStatus.CONFLICT, "Capsule resource admission is already in progress")
        reserved_keys = frozenset(_capacity_reservations)
        if capsule_slot:
            physical = _physical_capsules(exclude_keys=reserved_keys)
            capsule_reservations = [item for item in _capacity_reservations.values() if item.capsule_slot]
            current = len(physical) + len(capsule_reservations)
            if current >= MAX_CAPSULES:
                raise ApiError(HTTPStatus.TOO_MANY_REQUESTS, f"capsule limit reached ({current}/{MAX_CAPSULES})")
            owner_count = sum(container.labels.get("capsule.owner", "") == owner for container in physical) + sum(
                item.owner == owner for item in capsule_reservations
            )
            if owner_count >= MAX_CAPSULES_PER_OWNER:
                raise ApiError(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    f"capsule limit reached for this owner ({owner_count}/{MAX_CAPSULES_PER_OWNER})",
                )
        usage = _memory_usage(exclude_keys=reserved_keys)
        reserved_total = sum(item.memory_bytes for item in _capacity_reservations.values())
        committed_total = usage.total + reserved_total
        if committed_total + requested > GLOBAL_MEMORY_BUDGET_BYTES:
            detail = (
                f"global Capsule memory budget reached ({committed_total}/{GLOBAL_MEMORY_BUDGET_BYTES} bytes committed)"
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
                f"Capsule memory budget reached for this owner "
                f"({owner_used}/{OWNER_MEMORY_BUDGET_BYTES} bytes committed)"
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


def _ensure_capsule_volume(cid: str, kind: str) -> None:
    name = network_policy.volume_name(cid, kind)
    try:
        volume = _docker.volumes.get(name)
    except docker.errors.NotFound:
        try:
            volume = _docker.volumes.create(
                name=name,
                driver="local",
                labels=network_policy.volume_labels(cid, kind),
            )
        except docker.errors.DockerException as exc:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "could not create the Capsule volume") from exc
    except docker.errors.DockerException as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "could not inspect the Capsule volume") from exc
    try:
        volume.reload()
    except docker.errors.DockerException as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "could not verify the Capsule volume") from exc
    if not network_policy.volume_identity_valid(volume.attrs, cid, kind):
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            f"Capsule isolation is blocked: invalid or foreign {kind} volume",
        )


def _require_capsule_volumes(cid: str) -> None:
    """Re-inspect both exact Docker-local volumes at every runnable isolation seam."""
    for kind in (network_policy.CONFIG_VOLUME_KIND, network_policy.WORKSPACE_VOLUME_KIND):
        try:
            volume = _docker.volumes.get(network_policy.volume_name(cid, kind))
            volume.reload()
        except docker.errors.DockerException as exc:
            raise ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                f"Capsule isolation is blocked: required {kind} volume is unavailable",
            ) from exc
        if not network_policy.volume_identity_valid(volume.attrs, cid, kind):
            raise ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                f"Capsule isolation is blocked: invalid or foreign {kind} volume",
            )


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
            "cannot verify Capsule network isolation",
        ) from exc
    else:
        return containers


def _require_network_policy(
    network,
    cid: str,
    kind: str,
    *,
    require_brain: bool,
    require_dependencies: bool,
) -> None:
    containers = _network_container_metadata(network)
    if not network_policy.network_members_valid(
        network.attrs,
        containers,
        cid,
        kind,
        require_brain=require_brain,
        require_dependencies=require_dependencies,
    ):
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            f"Capsule isolation is blocked: invalid or contaminated {kind} network",
        )


def _ensure_capsule_network_kind(cid: str, kind: str):
    net_name = network_policy.network_name(cid, kind)
    try:
        network = _docker.networks.get(net_name)
    except docker.errors.NotFound:
        try:
            network = _docker.networks.create(
                net_name,
                driver="bridge",
                internal=True,
                attachable=False,
                labels=network_policy.network_labels(cid, kind),
            )
        except docker.errors.DockerException as exc:
            raise ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                f"could not create the Capsule {kind} network",
            ) from exc
    except docker.errors.DockerException as exc:
        raise ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            f"could not inspect the Capsule {kind} network",
        ) from exc
    _require_network_policy(
        network,
        cid,
        kind,
        require_brain=False,
        require_dependencies=False,
    )
    return network


def _ensure_capsule_network(cid: str):
    return _ensure_capsule_network_kind(cid, network_policy.CORE_KIND)


def _ensure_brain_egress_network(cid: str):
    return _ensure_capsule_network_kind(cid, network_policy.BRAIN_EGRESS_KIND)


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
                f"failed to connect required service {container_name!r} to its Capsule network",
            ) from exc


def _wire_network_deps(network, dependencies: list[tuple[str, list[str]]]) -> None:
    for container_name, aliases in dependencies:
        _safe_connect(network, container_name, aliases=aliases, required=True)


def _teardown_capsule_network_kind(cid: str, kind: str) -> bool:
    """Remove one owned plane, or report residue without mutating any foreign identity."""
    try:
        network = _docker.networks.get(network_policy.network_name(cid, kind))
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
    if not network_policy.network_identity_valid(network.attrs, cid, kind):
        return False
    cleanup_complete = True
    for container_id in dict(network.attrs.get("Containers", {})):
        try:
            container = _docker.containers.get(container_id)
            container.reload()
        except docker.errors.DockerException:
            cleanup_complete = False
            continue
        if not network_policy.network_member_managed(container.attrs, cid, kind):
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


def _teardown_capsule_networks(cid: str) -> bool:
    results = [
        _teardown_capsule_network_kind(cid, kind)
        for kind in (network_policy.BRAIN_EGRESS_KIND, network_policy.CORE_KIND)
    ]
    return all(results)


def _describe(container) -> dict:
    brain = container.labels.get("capsule.brain", manifests.DEFAULT_BRAIN)
    return {
        "id": container.labels.get("capsule.id"),
        "name": container.labels.get("capsule.name"),
        "owner": container.labels.get("capsule.owner", ""),
        "brain": brain,
        "model": _brain_model(container, brain),
        "status": container.status,
        "container": container.name,
    }


@dataclass(frozen=True)
class _AuthorizationLease:
    cid: str
    container_id: str
    owner: str
    principal: tuple[str, str | None]
    cleanup_nonce: str = ""


def _cleanup_record(cid: str) -> cleanup_state.Record | None:
    try:
        return cleanup_state.load(cid)
    except cleanup_state.CleanupStateError as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Capsule cleanup state is unavailable") from exc


def _authorize_container(cid: str, principal: tuple[str, str | None], container) -> _AuthorizationLease:
    if not network_policy.brain_identity_valid(container.attrs, cid):
        raise ApiError(HTTPStatus.NOT_FOUND, f"capsule {cid!r} not found")
    owner = str(container.labels.get("capsule.owner", ""))
    kind, account_id = principal
    if kind != "operator" and owner != account_id:
        raise ApiError(HTTPStatus.NOT_FOUND, f"capsule {cid!r} not found")
    return _AuthorizationLease(
        cid=cid,
        container_id=container.id,
        owner=owner,
        principal=principal,
    )


def _authorize(cid: str, principal: tuple[str, str | None]) -> _AuthorizationLease:
    """Operator may touch any capsule; an account may only touch a capsule it owns.

    This first pass returns an identity lease. Every sensitive operation must revalidate it only after
    acquiring its lifecycle/chat lock: authorization that waited behind destroy/recreate is never
    transferable to the new container that happens to reuse the same CID.
    """
    container = _get_container(manifests.capsule_container_name(cid))
    if container is None:
        raise ApiError(HTTPStatus.NOT_FOUND, f"capsule {cid!r} not found")
    return _authorize_container(cid, principal, container)


def _authorize_destroy(cid: str, principal: tuple[str, str | None]) -> _AuthorizationLease:
    """Authorize against the Brain, or its durable non-runnable cleanup successor."""
    container = _get_container(manifests.capsule_container_name(cid))
    if container is not None:
        return _authorize_container(cid, principal, container)
    record = _cleanup_record(cid)
    if record is None or not cleanup_state.principal_authorized(record, principal):
        raise ApiError(HTTPStatus.NOT_FOUND, f"capsule {cid!r} not found")
    return _AuthorizationLease(
        cid=cid,
        container_id=record.brain_id,
        owner=record.owner,
        principal=principal,
        cleanup_nonce=record.nonce,
    )


def _require_cleanup_authorization(cid: str, lease: _AuthorizationLease) -> cleanup_state.Record:
    """Revalidate the exact durable ownership record after acquiring the lifecycle lock."""
    record = _cleanup_record(cid)
    if (
        not lease.cleanup_nonce
        or lease.cid != cid
        or record is None
        or record.nonce != lease.cleanup_nonce
        or record.owner != lease.owner
        or record.brain_id != lease.container_id
        or not cleanup_state.principal_authorized(record, lease.principal)
        or _get_container(manifests.capsule_container_name(cid)) is not None
    ):
        raise ApiError(HTTPStatus.NOT_FOUND, f"capsule {cid!r} not found")
    return record


def _require_current_authorization(
    cid: str,
    lease: _AuthorizationLease,
    *,
    require_isolation: bool = True,
    allow_pending_cleanup: bool = False,
):
    """Revalidate owner + immutable Docker identity; caller already holds the operation lock."""
    container = _get_container(manifests.capsule_container_name(cid))
    if (
        lease.cleanup_nonce
        or lease.cid != cid
        or container is None
        or not network_policy.brain_identity_valid(container.attrs, cid)
        or container.id != lease.container_id
        or str(container.labels.get("capsule.owner", "")) != lease.owner
    ):
        # Accounts must not learn that a different tenant recreated this name. Operators receive the
        # same retry-safe 404 contract instead of accidentally mutating an object they never selected.
        raise ApiError(HTTPStatus.NOT_FOUND, f"capsule {cid!r} not found")
    kind, account_id = lease.principal
    if kind != "operator" and lease.owner != account_id:
        raise ApiError(HTTPStatus.NOT_FOUND, f"capsule {cid!r} not found")
    if not allow_pending_cleanup and _cleanup_record(cid) is not None:
        raise ApiError(HTTPStatus.CONFLICT, f"capsule {cid!r} has an incomplete teardown; retry destroy")
    if require_isolation:
        _require_capsule_isolation(container)
    return container


# ── installed apps (the P4 deploy arm) ───────────────────────────────────────
def _capsule_app_containers(cid: str) -> list:
    """Every installed-app container of capsule `cid` (its OWN label set — never `capsule.driver`)."""
    return _docker.containers.list(all=True, filters={"label": ["capsule.app.driver", f"capsule.id={cid}"]})


def _app_egress_token(cid: str, app_id: str) -> str:
    """The app instance's stable egress token (its Proxy-Authorization to app-egress-proxy).

    Kept in the policy volume (drivers + proxy only) and reused across reinstalls, exactly like
    shimpz-driver's per-app tokens — the proxy maps token → this instance's own allowlist.
    """
    tdir = APP_EGRESS_POLICY_DIR / ".tokens"
    tdir.mkdir(parents=True, exist_ok=True)
    tf = tdir / f"{manifests.capsule_app_container_name(cid, app_id)}.token"
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


def _remove_egress_policy(cid: str, app_id: str) -> bool:
    """Remove an App's policy and token without losing the token needed for a retry."""
    tf = APP_EGRESS_POLICY_DIR / ".tokens" / f"{manifests.capsule_app_container_name(cid, app_id)}.token"
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
    cid: str,
    app_id: str,
    *,
    container=None,
    drop_db: bool = True,
) -> _CleanupResult:
    """Remove one exact managed App, retaining retry state whenever cleanup is incomplete."""
    if container is None:
        try:
            container = _get_container(manifests.capsule_app_container_name(cid, app_id))
        except docker.errors.DockerException:
            return _CleanupResult(False, not drop_db)

    if container is not None:
        try:
            container.reload()
        except docker.errors.DockerException:
            return _CleanupResult(False, not drop_db)
        if not network_policy.app_identity_valid(container.attrs, cid, app_id):
            # A deterministic-name collision or drifted ownership label is not ours to delete.
            return _CleanupResult(False, not drop_db)
        db_label = container.labels.get("capsule.app.db")
        if db_label not in (None, "0", "1"):
            return _CleanupResult(False, not drop_db)
        # Missing means a legacy App from before the label existed; conservatively assume it has a DB.
        drop_db = drop_db and db_label != "0"

    policy_removed = _remove_egress_policy(cid, app_id)
    container_removed = container is None
    if container is not None and policy_removed:
        container_removed = _remove_capsule_container(container)
    elif container is not None:
        # Preserve the labeled retry anchor, but do not leave tenant code running after a failed removal.
        with contextlib.suppress(ApiError):
            _fail_stop_capsule(container)

    artifacts_removed = policy_removed and container_removed
    if not drop_db:
        return _CleanupResult(artifacts_removed, True)
    if not artifacts_removed:
        # Keep the DB registration intact until the retryable container/policy phase has completed.
        return _CleanupResult(False, False)
    try:
        pgdriver_client.drop_app_db(cid, app_id)
    except pgdriver_client.PgDriverError, http.client.HTTPException, OSError, ValueError:
        return _CleanupResult(True, False)
    return _CleanupResult(True, True)


def _install_app(
    cid: str,
    app_id: str,
    spec: marketplace.AppSpec,
    owner: str,
    lease: _AuthorizationLease,
) -> dict:
    with _lock_for(cid):
        capsule = _require_current_authorization(cid, lease)
        if owner != lease.owner:
            raise ApiError(HTTPStatus.NOT_FOUND, f"capsule {cid!r} not found")
        capsule_name = capsule.labels.get("capsule.name", "")
        existing = _get_container(manifests.capsule_app_container_name(cid, app_id))
        if existing is not None:  # idempotent only for this exact, still-isolated installed App
            try:
                existing.reload()
            except docker.errors.DockerException as exc:
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    f"cannot verify installed app {app_id!r}",
                ) from exc
            expected_labels = {
                "capsule.app.driver": "1",
                "capsule.id": cid,
                "capsule.app": app_id,
                "capsule.owner": owner,
            }
            if any(str(existing.labels.get(key, "")) != value for key, value in expected_labels.items()):
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    f"existing container for app {app_id!r} has invalid ownership metadata; uninstall it first",
                )
            _require_capsule_isolation(existing)
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
            return {"capsule": cid, "app": app_id, "status": status, "installed": False}
        if len(_capsule_app_containers(cid)) >= MAX_APPS_PER_CAPSULE:
            raise ApiError(HTTPStatus.TOO_MANY_REQUESTS, f"app limit reached for {cid!r} ({MAX_APPS_PER_CAPSULE})")
        key = f"app:{cid}:{app_id}"
        with _reserve_capacity(key, owner, manifests.APP_MEM_LIMIT_BYTES, capsule_slot=False):
            # Return the same explicit admission error as Capsule create instead of relying on a
            # lower-level Docker create failure when the hostile-tenant runtime is unavailable.
            _require_capsule_runtime()
            # Transactional like _create: on ANY failure the app's own artifacts are rolled back (the
            # capsule itself is never touched) — no orphan DB, policy file, or half-started container.
            # Only the reservation remains global while these external calls run; unrelated Capsules
            # can reserve/provision concurrently.
            try:
                database_url = ""
                if spec.db:
                    database_url = pgdriver_client.create_app_db(cid, app_id)["database_url"]
                network = _ensure_capsule_network(cid)
                proxy_env: dict[str, str] = {}
                if spec.egress:
                    token = _app_egress_token(cid, app_id)
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
                kwargs = manifests.build_capsule_app_kwargs(
                    cid,
                    app_id,
                    spec,
                    database_url=database_url,
                    proxy_env=proxy_env,
                    owner=owner,
                    capsule_name=capsule_name,
                )
                _require_capsule_runtime()
                container = _docker.containers.create(**kwargs)
                # Re-attach with the app aliases (create can't set them): the capsule brain and sibling
                # apps reach it as http://<app-id>:<port>, and as http://<app-id>.capsule:<port> — the
                # `.capsule` form tail-matches the NO_PROXY suffix baked into every capsule container, so
                # proxied clients skip egress. The App remains attached only to the internal core plane.
                network.disconnect(container)
                network.connect(container, aliases=[app_id, f"{app_id}.capsule"])
                _start_capsule_with_isolation(container)
                healthy, reason = _wait_app_healthy(container, spec.port, spec.health_path)
                if not healthy:
                    raise ApiError(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        f"app {app_id!r} failed its health probe ({reason}; rolled back)",
                    )
                _require_capsule_isolation(container)
                ready, committed_status = _app_ready_now(container, spec.port, spec.health_path)
                if not ready:
                    raise ApiError(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        f"app {app_id!r} lost readiness before install commit ({committed_status}; rolled back)",
                    )
            except Exception as exc:
                cleanup = _teardown_app(cid, app_id, drop_db=spec.db)
                if not cleanup.complete:
                    raise ApiError(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "app install failed and rollback is incomplete; retry uninstall or contact the operator",
                    ) from exc
                if isinstance(exc, ApiError):
                    raise
                raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "app install failed and was rolled back") from exc
        return {
            "capsule": cid,
            "app": app_id,
            "status": committed_status,
            "installed": True,
            **({"database": manifests.capsule_app_db_project(cid, app_id)} if spec.db else {}),
        }


def _uninstall_app(cid: str, app_id: str, lease: _AuthorizationLease) -> dict:
    with _lock_for(cid):
        # Removal is a remediation operation and must remain available for a legacy blocked Capsule.
        _require_current_authorization(cid, lease, require_isolation=False)
        cleanup = _teardown_app(cid, app_id)
        if not cleanup.complete:
            raise ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "app teardown is incomplete; retry uninstall or contact the operator",
            )
        return {"capsule": cid, "app": app_id, "uninstalled": True, "db_dropped": cleanup.db_dropped}


def _list_apps(cid: str, lease: _AuthorizationLease) -> dict:
    with _lock_for(cid):
        # Read-only inventory lets the owner see and remove residual Apps without executing tenant code.
        _require_current_authorization(cid, lease, require_isolation=False)
        apps = [
            {
                "app": c.labels.get("capsule.app"),
                "status": c.status,
                "container": c.name,
            }
            for c in _capsule_app_containers(cid)
        ]
        return {"capsule": cid, "apps": apps}


# ── the Captain's chat (ADR-0004): named exec ops into the capsule's OWN brain ──────────────────
# No new network path — the store forwards, ownership is enforced here, and the brain is reached the
# same way an operator would reach it: an exec, as the runtime user, inside that capsule only.
CHAT_TIMEOUT_SECONDS = int(os.environ.get("SHIMPZ_CAPSULE_CHAT_TIMEOUT", "170"))
CHAT_OUTPUT_CAP = 60000
MAX_STREAM_LINE_BYTES = 256 * 1024
MAX_STREAM_RAW_BYTES = 2 * 1024 * 1024
INBOX_DIR = "/config/workspace/inbox"
MAX_INBOX_FILE_BYTES = 30 * 1024 * 1024  # the store caps uploads well below Cloudflare's 100 MB
# Base64 expands by 4/3; leave a small fixed envelope for the JSON keys and filename.
MAX_FILE_BODY_BYTES = 4 * ((MAX_INBOX_FILE_BYTES + 2) // 3) + 8192
_BRAIN_USER = "abc"  # every provider image's fixed runtime user (uid 1000)
_CHAT_CONTROL_USER = "root"  # root subreaper; provider child is dropped to abc before exec
_AUTH_MARKERS = (
    "please run /login",
    "not logged in",
    "invalid api key",
    "unauthorized",
    "credentials",
    "please log in",
    "oauth",
)
_EXEC_CAPTURE_CAP = CHAT_OUTPUT_CAP + 16 * 1024
_EXEC_OUTPUT_LIMIT_EXIT = 70
MAX_CHAT_ASKS = 16
MAX_CHAT_ASK_TEXT_CHARS = 2_000
MAX_CHAT_ASK_OPTIONS = 8
MAX_CHAT_ASK_OPTION_CHARS = 200
MAX_CHAT_ASK_JSON_BYTES = 48 * 1024
if MAX_CHAT_ASK_JSON_BYTES >= _EXEC_CAPTURE_CAP:
    raise ValueError("chat ask JSON budget must fit inside the bounded Docker exec capture")
_PROVIDER_ARTIFACTS = {
    "claude-code": (
        ".shimpz/brain-credential.sh",
        ".claude/.credentials.json",
        ".shimpz/login",
    ),
    "codex": (".codex/auth.json", ".shimpz/codex-thread-id"),
}
_PROVIDER_VOLUME_SCRIPT = r"""
import os
import re
import shutil
import sys
from pathlib import Path

ARTIFACTS = {
    "claude-code": (".shimpz/brain-credential.sh", ".claude/.credentials.json", ".shimpz/login"),
    "codex": (".codex/auth.json", ".shimpz/codex-thread-id"),
}
MAX_CONFIG_ENTRIES = 4096
QUARANTINE_NAME = re.compile(r"[.]shimpz-provider-swap-[a-f0-9]{32}")


def remove(path):
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


action, provider, token = sys.argv[1:]
if action not in {"hide", "restore", "discard", "purge", "purge-all"} or provider not in ARTIFACTS:
    raise SystemExit(64)
if action not in {"purge", "purge-all"} and re.fullmatch(r"[a-f0-9]{32}", token) is None:
    raise SystemExit(64)

config = Path("/config")
base = config / f".shimpz-provider-swap-{token}"
paths = tuple(config / relative for relative in ARTIFACTS[provider])
if action == "purge-all":
    for relative in sorted({relative for provider_paths in ARTIFACTS.values() for relative in provider_paths}):
        remove(config / relative)
    quarantines = []
    with os.scandir(config) as entries:
        for count, entry in enumerate(entries, 1):
            if count > MAX_CONFIG_ENTRIES:
                raise SystemExit("config entry limit exceeded during credential purge")
            if QUARANTINE_NAME.fullmatch(entry.name):
                quarantines.append(Path(entry.path))
    for path in quarantines:
        remove(path)
elif action == "purge":
    for path in paths:
        remove(path)
elif action == "hide":
    if base.exists():
        raise SystemExit("provider swap quarantine already exists")
    base.mkdir(mode=0o700)
    moved = []
    try:
        for index, path in enumerate(paths):
            if os.path.lexists(path):
                destination = base / str(index)
                os.replace(path, destination)
                moved.append((destination, path))
    except BaseException:
        for source, destination in reversed(moved):
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.replace(source, destination)
        shutil.rmtree(base, ignore_errors=True)
        raise
elif action == "restore":
    if not base.is_dir():
        raise SystemExit("provider swap quarantine is missing")
    for index, path in enumerate(paths):
        source = base / str(index)
        if source.exists() or source.is_symlink():
            remove(path)
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.replace(source, path)
    shutil.rmtree(base)
else:
    shutil.rmtree(base)
"""


@dataclass(frozen=True)
class _BrainAdapter:
    """All provider-specific behavior in one registry entry, never request-shaped branching."""

    status: Callable[[object], tuple[bool, bool]]
    configure: Callable[[object, str, str], None]
    deconfigure: Callable[[object], None]
    invocation: Callable[[str, bool, bool, str], tuple[list[str], bytes | None]]
    parse_stream_line: Callable[[bytes], tuple[str, str] | None]
    no_session_markers: tuple[str, ...]
    interactive_login: bool
    requires_completion_event: bool


_BRAIN_ADAPTERS: dict[str, _BrainAdapter]


@dataclass
class _ProviderStreamState:
    """Bounded provider-neutral state accumulated from one JSONL turn."""

    adapter: _BrainAdapter
    final: str = ""
    completed: bool = False
    protocol_error: str = ""
    stop_requested: bool = False

    def fail(self, detail: str) -> None:
        if not self.protocol_error:
            self.protocol_error = detail
            self.stop_requested = True

    def consume(self, line: bytes) -> dict | None:
        if len(line) > MAX_STREAM_LINE_BYTES:
            self.fail("brain stream line exceeded its size limit")
            return None
        if not line.strip():
            return None
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError, UnicodeDecodeError:
            self.fail("brain stream contained malformed JSONL")
            return None
        if not isinstance(decoded, dict):
            self.fail("brain stream contained a non-object JSONL event")
            return None
        event = self.adapter.parse_stream_line(line)
        if event is None:
            return None
        kind, value = event
        if kind == "text":
            self.final = value[:CHAT_OUTPUT_CAP]
            return {"t": "text", "text": self.final}
        if kind == "tool":
            return {"t": "tool", "label": value[:200]}
        if kind == "result":
            self.final = (value or self.final)[:CHAT_OUTPUT_CAP]
            self.completed = True
        elif kind == "complete":
            self.completed = True
        elif kind == "error":
            self.fail("provider reported a failed turn")
        return None


def _apply_stream_stop(state: _ProviderStreamState, container, token: str) -> None:
    if state.stop_requested:
        container.exec_run(["shimpz-chat-stop", token], user=_CHAT_CONTROL_USER)
        state.stop_requested = False


def _brain_exec(container, cmd: list[str], *, user: str = _BRAIN_USER) -> tuple[int, str]:
    exec_id = _docker.api.exec_create(
        container.id,
        cmd,
        stdin=False,
        stdout=True,
        stderr=True,
        user=user,
        workdir="/config/workspace",
        environment={"HOME": "/config"},
    )["Id"]
    stream = _docker.api.exec_start(exec_id, socket=True)
    rc, stdout, stderr = _bounded_exec_output(exec_id, stream)
    return rc, stdout + stderr


def _close_exec_stream(stream) -> None:
    """Close docker-py's owning HTTP response before its raw socket (Python 3.14 safe)."""
    response = getattr(stream, "_response", None)
    if response is not None:
        response.close()
    else:
        stream.close()


def _bounded_exec_output(exec_id: str, stream) -> tuple[int | None, str, str]:
    """Drain one Docker exec while retaining at most one fixed aggregate output budget."""
    stdout = bytearray()
    stderr = bytearray()
    truncated = False
    try:
        for stream_id, chunk in docker_socket.frames_iter(stream, tty=False):
            remaining = _EXEC_CAPTURE_CAP - len(stdout) - len(stderr)
            if remaining > 0:
                target = stdout if stream_id == docker_socket.STDOUT else stderr
                target.extend(chunk[:remaining])
            if len(chunk) > max(0, remaining):
                truncated = True
    finally:
        _close_exec_stream(stream)
    rc = _docker.api.exec_inspect(exec_id).get("ExitCode")
    if truncated:
        return _EXEC_OUTPUT_LIMIT_EXIT, "", "Capsule exec output exceeded its capture limit"
    return rc, stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace")


def _open_brain_exec_stdin(container, cmd: list[str], payload: bytes, *, user: str = _BRAIN_USER):
    """Start one fixed exec and write its sensitive input over the attach socket only."""
    exec_id = _docker.api.exec_create(
        container.id,
        cmd,
        stdin=True,
        stdout=True,
        stderr=True,
        user=user,
        workdir="/config/workspace",
        environment={"HOME": "/config"},
    )["Id"]
    stream = _docker.api.exec_start(exec_id, socket=True)
    ready = False
    try:
        raw_socket = getattr(stream, "_sock", None)
        if raw_socket is None:
            raise OSError("Docker exec attach socket does not support a stdin half-close")
        raw_socket.sendall(payload)
        raw_socket.shutdown(socket.SHUT_WR)
        ready = True
    finally:
        if not ready:
            _close_exec_stream(stream)
    return exec_id, stream


def _brain_exec_stdin(
    container, cmd: list[str], payload: bytes, *, user: str = _BRAIN_USER
) -> tuple[int | None, str, str]:
    """Run a provider op with stdin, returning bounded and demultiplexed stdout/stderr."""
    exec_id, stream = _open_brain_exec_stdin(container, cmd, payload, user=user)
    return _bounded_exec_output(exec_id, stream)


def _brain_id(container) -> str:
    return container.labels.get("capsule.brain", manifests.DEFAULT_BRAIN)


def _brain_model(container, brain: str | None = None) -> str:
    provider = brain or _brain_id(container)
    try:
        return manifests.model_for_brain(provider, container.labels.get("capsule.model"))
    except ValueError as exc:
        raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "Capsule has invalid Brain model metadata") from exc


def _adapter_for(container) -> tuple[str, _BrainAdapter]:
    brain = _brain_id(container)
    try:
        return brain, _BRAIN_ADAPTERS[brain]
    except KeyError as exc:
        raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, f"unsupported Capsule brain {brain!r}") from exc


def _brain_status_from_container(cid: str, container) -> dict:
    """Return local configuration + a driver-observed successful-turn verdict."""
    brain, adapter = _adapter_for(container)
    configured, _provider_claim = adapter.status(container)
    authenticated = configured and _brain_authenticated(cid, container)
    title = manifests.BRAINS.get(brain, {}).get("title", brain)
    return {
        "capsule": cid,
        "brain": brain,
        "model": _brain_model(container, brain),
        "title": title,
        "configured": configured,
        "authenticated": authenticated,
    }


def _brain_status(cid: str, lease: _AuthorizationLease) -> dict:
    """Serialize public status with lifecycle/config mutations for a coherent snapshot."""
    with _lock_for(cid):
        container = _require_current_authorization(cid, lease)
        container.reload()
        if container.status != "running":
            raise ApiError(HTTPStatus.CONFLICT, f"capsule {cid!r} is not running (status={container.status})")
        return _brain_status_from_container(cid, container)


def _require_brain_configured(cid: str, container) -> None:
    status = _brain_status_from_container(cid, container)
    if not status["configured"]:
        raise ApiError(
            HTTPStatus.CONFLICT,
            "configure the Brain before starting a chat",
        )


def _purge_credential_or_fail_stop(container, brain: str, *, failure: str) -> None:
    """Prove credential bytes purged, or prove their workload cannot keep running."""
    try:
        _provider_volume_action(container, "purge", brain)
    except ApiError as purge_exc:
        try:
            _fail_stop_capsule(container)
        except ApiError as stop_exc:
            raise ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"{failure}; credential cleanup and Capsule stop could not be proved",
            ) from stop_exc
        raise ApiError(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            f"{failure}; credential cleanup failed and the Capsule was stopped",
        ) from purge_exc


def _assert_credential_generation(container, account_id: str, brain: str, generation: int) -> None:
    """Linearize injection against account revocation; stale bytes are purged before failure."""
    check_error = None
    try:
        current = brain_credentials_client.generation_is_current(account_id, brain, generation)
    except brain_credentials_client.BrainCredentialError as exc:
        current = False
        check_error = exc
    if current:
        return
    cid = str(container.labels.get("capsule.id", ""))
    if cid:
        _clear_brain_authentication(cid)
    _purge_credential_or_fail_stop(
        container,
        brain,
        failure="Brain credential generation changed",
    )
    if check_error is not None:
        raise ApiError(
            HTTPStatus.BAD_GATEWAY,
            "Brain credential generation could not be verified",
        ) from check_error
    raise ApiError(
        HTTPStatus.CONFLICT,
        "Brain credential changed or was revoked during injection; retry",
    )


def _configure_brain(cid: str, account_id: str, lease: _AuthorizationLease) -> dict:
    """Inject this account's encrypted-at-rest provider credential into this Capsule only."""
    with _lock_for(cid):
        container = _require_current_authorization(cid, lease)
        if account_id != lease.owner:
            raise ApiError(HTTPStatus.NOT_FOUND, f"capsule {cid!r} not found")
        container.reload()
        if container.status != "running":
            raise ApiError(HTTPStatus.CONFLICT, f"capsule {cid!r} is not running (status={container.status})")
        brain, adapter = _adapter_for(container)
        _require_no_credential_mutation(cid)
        try:
            credential = brain_credentials_client.resolve(account_id, brain)
        except brain_credentials_client.BrainCredentialError as exc:
            raise ApiError(HTTPStatus.BAD_GATEWAY, str(exc)) from exc
        if credential is None:
            raise ApiError(
                HTTPStatus.CONFLICT,
                f"configure the {brain!r} account Brain credential first",
            )
        auth_type, secret, generation = credential
        chat_lock = _chat_lock_for(cid)
        if not chat_lock.acquire(blocking=False):
            raise ApiError(HTTPStatus.CONFLICT, "stop the active chat turn before changing its Brain credential")
        try:
            _require_no_durable_chat_turn(container)
            # A rotation is unverified even if injection later fails. Never carry a success observed
            # with credential N over to credential N+1.
            _clear_brain_authentication(cid)
            try:
                _wait_brain_ready(container, brain)
            except (brain_credentials_client.BrainCredentialError, docker.errors.DockerException, OSError) as exc:
                raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "Brain credential injection failed") from exc
            try:
                adapter.configure(container, auth_type, secret)
            except (brain_credentials_client.BrainCredentialError, docker.errors.DockerException, OSError) as exc:
                try:
                    _purge_credential_or_fail_stop(
                        container,
                        brain,
                        failure="Brain credential injection failed",
                    )
                except ApiError as cleanup_exc:
                    raise cleanup_exc from exc
                raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "Brain credential injection failed") from exc
            _assert_credential_generation(container, account_id, brain, generation)
        finally:
            chat_lock.release()
    trace_id = audit.log("brain_configure", cid, result="ok", brain=brain)
    return {
        "capsule": cid,
        "brain": brain,
        "configured": True,
        "trace_id": trace_id,
    }


def _deconfigure_brain(cid: str, lease: _AuthorizationLease) -> dict:
    """Stop all provider work, revoke its files, then restore the Capsule's prior run state."""
    with _lock_for(cid):
        # Once the configured sandbox is available, revocation may purge a credential from a legacy
        # Capsule through the sandboxed helper. The legacy Capsule stays stopped; only a correctly
        # isolated container may be restored to its prior running state.
        container = _require_current_authorization(cid, lease, require_isolation=False)
        try:
            _require_capsule_isolation(container)
        except ApiError:
            isolated = False
        else:
            isolated = True
        brain, _adapter = _adapter_for(container)
        container.reload()
        was_running = container.status == "running"
        chat_lock = _chat_lock_for(cid)
        purge_proved = False
        try:
            if was_running:
                _fail_stop_capsule(container, timeout=30)
            if not chat_lock.acquire(timeout=30):
                raise ApiError(HTTPStatus.CONFLICT, "the active chat turn did not stop in time")
            try:
                with _active_chat_guard:
                    _credential_mutations.discard(cid)
                    _credential_mutation_execs.pop(cid, None)
                    _verified_brains.pop(cid, None)
                _provider_volume_action(container, "purge", brain)
                purge_proved = True
            finally:
                chat_lock.release()
        except (docker.errors.DockerException, OSError) as exc:
            raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "Brain credential removal failed") from exc
        finally:
            # A failed/ambiguous purge leaves the prior credential potentially usable. Never make
            # that workload runnable again; only a proved revocation may restore its prior run state.
            if was_running and isolated and purge_proved:
                try:
                    _require_capsule_runtime()
                    container.reload()
                    if container.status != "running":
                        _start_capsule_with_isolation(container)
                    else:
                        _require_running_capsule_isolation(container)
                except ApiError:
                    raise
                except docker.errors.DockerException as exc:
                    raise ApiError(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "Brain credential was removed but the Capsule could not be restarted",
                    ) from exc
    trace_id = audit.log("brain_deconfigure", cid, result="ok", brain=brain)
    return {
        "capsule": cid,
        "brain": brain,
        "configured": False,
        "isolation_blocked": not isolated,
        "trace_id": trace_id,
    }


# The brain's QUESTIONS surface in the web chat: shimpz-ask (unchanged) drops `<rid>.req` into
# $SHIMPZ_HOME/ipc and blocks on `<rid>.resp` — in a Capsule there is NO Telegram gateway, so the
# web chat is the responder. Both ops run shimpzipc itself inside the capsule (the protocol's single
# source of truth), as the runtime user, via the fixed venv python — never a caller-shaped command.
_IPC_LIST_SCRIPT = (
    "import json, os, sys\n"
    "sys.path.insert(0, '/opt/shimpz-lib')\n"
    "import shimpzipc\n"
    "ipc = os.path.join(os.environ.get('SHIMPZ_HOME', '/config/.shimpz'), 'ipc')\n"
    "asks = []\n"
    "encoded = '[]'\n"
    "token = sys.argv[1]\n"
    "for rid, _req, payload in shimpzipc.pending_for_chat(ipc, token):\n"
    "    if payload.get('type') == 'ask':\n"
    f"        raw_options = payload.get('options') or []\n"
    f"        options = [str(value)[:{MAX_CHAT_ASK_OPTION_CHARS}] for value in raw_options"
    f"[:{MAX_CHAT_ASK_OPTIONS}]] if isinstance(raw_options, list) else []\n"
    "        raw_default = payload.get('default')\n"
    "        default = raw_default if type(raw_default) is int and 1 <= raw_default <= len(options) else None\n"
    f"        candidate = {{'rid': rid, 'text': str(payload.get('text', ''))[:{MAX_CHAT_ASK_TEXT_CHARS}],\n"
    "                     'options': options, 'default': default}\n"
    "        candidate_encoded = json.dumps([*asks, candidate], separators=(',', ':'))\n"
    f"        if len(candidate_encoded.encode('utf-8')) > {MAX_CHAT_ASK_JSON_BYTES}:\n"
    "            break\n"
    "        asks.append(candidate)\n"
    "        encoded = candidate_encoded\n"
    f"        if len(asks) >= {MAX_CHAT_ASKS}:\n"
    "            break\n"
    "print(encoded)\n"
)
_IPC_ANSWER_SCRIPT = (
    "import json, os, sys\n"
    "sys.path.insert(0, '/opt/shimpz-lib')\n"
    "import shimpzipc\n"
    "ipc = os.path.join(os.environ.get('SHIMPZ_HOME', '/config/.shimpz'), 'ipc')\n"
    "payload = json.load(sys.stdin)\n"
    "rid, answer = payload['rid'], payload['answer']\n"
    "wrote = shimpzipc.answer_for_chat(ipc, rid, {'answer': answer}, sys.argv[1])\n"
    "print(json.dumps({'answered': bool(wrote)}))\n"
)


def _chat_asks(cid: str, lease: _AuthorizationLease) -> dict:
    """The brain's pending shimpz-ask questions — what the web chat renders as option cards."""
    with _lock_for(cid):
        container = _require_current_authorization(cid, lease)
        container.reload()
        if container.status != "running":
            raise ApiError(HTTPStatus.CONFLICT, f"capsule {cid!r} is not running (status={container.status})")
        token = _durable_active_chat_token(container)
        if token is None:
            return {"capsule": cid, "asks": []}
        rc, out = _brain_exec(container, ["/opt/venv/bin/python", "-c", _IPC_LIST_SCRIPT, token])
        asks = []
        if rc == 0:
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                asks = json.loads(out or "[]")
        return {"capsule": cid, "asks": asks}


def _chat_answer(cid: str, body: dict, lease: _AuthorizationLease) -> dict:
    """Answer one pending ask — unblocks the shimpz-ask the brain is waiting on mid-turn."""
    rid = validate.validate_ask_rid(body.get("rid"))
    answer = validate.validate_chat_message(body.get("answer"))
    with _lock_for(cid):
        container = _require_current_authorization(cid, lease)
        container.reload()
        if container.status != "running":
            raise ApiError(HTTPStatus.CONFLICT, f"capsule {cid!r} is not running (status={container.status})")
        token = _durable_active_chat_token(container)
        if token is None:
            return {"capsule": cid, "answered": False}
        rc, out, _stderr = _brain_exec_stdin(
            container,
            ["/opt/venv/bin/python", "-c", _IPC_ANSWER_SCRIPT, token],
            json.dumps({"rid": rid, "answer": answer}, separators=(",", ":")).encode(),
        )
        answered = False
        if rc == 0:
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                answered = bool(json.loads(out or "{}").get("answered"))
        audit.log("chat_answer", cid, result="ok" if answered else "denied", rid=rid)
        return {"capsule": cid, "rid": rid, "answered": answered}


# The Claude-subscription OAuth flow, PER CAPSULE — the exact mirror of shimpz-driver's brain-login
# (drivers/apps/app.py): only ever the FIXED binary `shimpz-login` (baked into the brain image every
# capsule runs), owner-enforced upstream, audited, and the pasted code carried only over stdin.
def _interactive_login_capsule(cid: str, lease: _AuthorizationLease):
    container = _require_current_authorization(cid, lease)
    container.reload()
    if container.status != "running":
        raise ApiError(HTTPStatus.CONFLICT, f"capsule {cid!r} is not running (status={container.status})")
    brain, adapter = _adapter_for(container)
    if not adapter.interactive_login:
        raise ApiError(
            HTTPStatus.CONFLICT,
            f"{brain!r} uses the account Brain credential flow, not the Claude login bridge",
        )
    return container


def _credential_writer_finished(cid: str, container) -> bool:
    """Prove the exact detached OAuth writer exited before releasing the mutation gate."""
    with _active_chat_guard:
        identity = _credential_mutation_execs.get(cid)
    if identity is None or identity[0] != container.id:
        return False
    try:
        metadata = _docker.api.exec_inspect(identity[1])
    except docker.errors.DockerException:
        return False
    return (
        isinstance(metadata, dict) and metadata.get("ContainerID") == container.id and metadata.get("Running") is False
    )


def _capsule_login_start(cid: str, lease: _AuthorizationLease) -> dict:
    with _lock_for(cid):
        container = _interactive_login_capsule(cid, lease)
        chat_lock = _chat_lock_for(cid)
        if not chat_lock.acquire(blocking=False):
            raise ApiError(HTTPStatus.CONFLICT, "stop the active chat turn before changing its Brain credential")
        try:
            _require_no_durable_chat_turn(container)
            _clear_brain_authentication(cid)
            with _active_chat_guard:
                if cid in _credential_mutations:
                    raise ApiError(HTTPStatus.CONFLICT, "Brain credential change is already in progress")
                _credential_mutations.add(cid)
            # The bridge blocks holding the PKCE state until the Captain pastes the code. Keep its
            # exact Engine exec identity: a lost start response may still have launched the writer.
            exec_id = ""
            try:
                created = _docker.api.exec_create(
                    container.id,
                    ["shimpz-login", "run"],
                    user=_BRAIN_USER,
                    environment={"HOME": "/config"},
                )
                exec_id = created.get("Id") if isinstance(created, dict) else ""
                if not isinstance(exec_id, str) or not exec_id:
                    raise docker.errors.DockerException("Docker returned an invalid login exec identity")
                with _active_chat_guard:
                    _credential_mutation_execs[cid] = (container.id, exec_id)
                _docker.api.exec_start(exec_id, detach=True)
            except docker.errors.DockerException as exc:
                if not exec_id:
                    with _active_chat_guard:
                        _credential_mutations.discard(cid)
                        _credential_mutation_execs.pop(cid, None)
                    raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "Brain login writer could not be created") from exc
                with _active_chat_guard:
                    _blocked_chat_capsules.add(cid)
                try:
                    _fail_stop_capsule(container)
                except ApiError as stop_exc:
                    raise ApiError(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "Brain login writer start was ambiguous and the Capsule stop could not be proved",
                    ) from stop_exc
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Brain login writer start was ambiguous; Capsule stopped",
                ) from exc
        finally:
            chat_lock.release()
    trace_id = audit.log("brain_login", cid, result="ok", step="start")
    return {"capsule": cid, "started": True, "trace_id": trace_id}


def _capsule_login_url(cid: str, lease: _AuthorizationLease) -> dict:
    with _lock_for(cid):
        container = _interactive_login_capsule(cid, lease)
        rc, out = _brain_exec(
            container,
            ["sh", "-c", 'cat "${SHIMPZ_HOME:-/config/.shimpz}/login/url" 2>/dev/null'],
        )
    url = out.strip() if rc == 0 else ""
    audit.log("brain_login", cid, result="ok", step="url", has_url=bool(url))
    return {"capsule": cid, "url": url} if url else {"capsule": cid, "pending": True}


def _capsule_login_code(cid: str, body: dict, lease: _AuthorizationLease) -> dict:
    code = validate.validate_login_code(body.get("code"))
    with _lock_for(cid):
        container = _interactive_login_capsule(cid, lease)
        chat_lock = _chat_lock_for(cid)
        if not chat_lock.acquire(blocking=False):
            raise ApiError(HTTPStatus.CONFLICT, "stop the active chat turn before changing its Brain credential")
        try:
            _require_no_durable_chat_turn(container)
            _clear_brain_authentication(cid)
            rc, _stdout, _stderr = _brain_exec_stdin(
                container,
                ["shimpz-login", "submit"],
                code.encode("ascii"),
            )
        finally:
            chat_lock.release()
    ok = rc == 0
    audit.log("brain_login", cid, result="ok" if ok else "error", step="code")
    return {"capsule": cid, "ok": ok}


def _capsule_login_status(cid: str, lease: _AuthorizationLease) -> dict:
    with _lock_for(cid):
        container = _interactive_login_capsule(cid, lease)
        chat_lock = _chat_lock_for(cid)
        if not chat_lock.acquire(blocking=False):
            raise ApiError(HTTPStatus.CONFLICT, "stop the active chat turn before checking Brain login")
        try:
            _require_no_durable_chat_turn(container)
            rc, out = _brain_exec(container, ["shimpz-login", "status", "--json"])
            result: dict = {"loggedIn": False}
            if rc == 0:
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    parsed = json.loads(out or "{}")
                    result = {"loggedIn": bool(parsed.get("loggedIn")), "email": parsed.get("email")}
            # Read the bridge's own verdict. A successful interactive OAuth exchange supersedes any old
            # API-key shell file; this is a credential rotation and invalidates prior turn verification.
            verdict: dict = {}
            rc, raw = _brain_exec(
                container,
                ["sh", "-c", 'cat "${SHIMPZ_HOME:-/config/.shimpz}/login/result" 2>/dev/null'],
            )
            if rc == 0 and raw.strip():
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        verdict = parsed
            if result["loggedIn"] and verdict.get("ok"):
                _clear_brain_authentication(cid)
                stale_rc, _stale_output = _brain_exec(
                    container,
                    ["rm", "-f", "/config/.shimpz/brain-credential.sh"],
                )
                if stale_rc != 0:
                    with _active_chat_guard:
                        _blocked_chat_capsules.add(cid)
                    try:
                        _fail_stop_capsule(container)
                    except ApiError as stop_exc:
                        raise ApiError(
                            HTTPStatus.INTERNAL_SERVER_ERROR,
                            "OAuth succeeded but stale credential removal and Capsule stop could not be proved",
                        ) from stop_exc
                    raise ApiError(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "OAuth succeeded but stale credential removal failed; Capsule stopped",
                    )
            elif not result["loggedIn"] and verdict.get("message"):
                result["last_error"] = str(verdict["message"])[:300]
            if verdict and _credential_writer_finished(cid, container):
                with _active_chat_guard:
                    _credential_mutations.discard(cid)
                    _credential_mutation_execs.pop(cid, None)
        finally:
            chat_lock.release()
    audit.log("brain_login", cid, result="ok", step="status", logged_in=result["loggedIn"])
    return {"capsule": cid, **result}


def _claude_status(container) -> tuple[bool, bool]:
    rc, _ = _brain_exec(
        container,
        ["sh", "-c", "[ -s /config/.shimpz/brain-credential.sh ] || [ -s /config/.claude/.credentials.json ]"],
    )
    configured = rc == 0
    # This is intentionally ONLY local presence. `claude auth status` reports an API-key source as
    # logged in without proving that the provider accepts it. The driver owns the truthful second bit
    # and sets it only after a real turn completes successfully.
    return configured, False


def _codex_status(container) -> tuple[bool, bool]:
    rc, _output = _brain_exec(container, ["test", "-s", "/config/.codex/auth.json"])
    return rc == 0, False


def _configure_claude(container, auth_type: str, secret: str) -> None:
    target, _content = brain_credentials_client.credential_file("claude-code", auth_type, secret)
    archive = brain_credentials_client.credential_archive("claude-code", auth_type, secret)
    if not container.put_archive("/config", archive):
        raise brain_credentials_client.BrainCredentialError("Brain credential injection failed")
    stale = {".shimpz/brain-credential.sh", ".claude/.credentials.json"} - {target}
    rc, _output = _brain_exec(container, ["rm", "-f", *(f"/config/{path}" for path in sorted(stale))])
    if rc != 0:
        raise brain_credentials_client.BrainCredentialError("stale Brain credential cleanup failed")


def _deconfigure_claude(container) -> None:
    rc, _output = _brain_exec(
        container,
        ["rm", "-f", "/config/.shimpz/brain-credential.sh", "/config/.claude/.credentials.json"],
    )
    if rc != 0:
        raise OSError("Claude credential removal failed")


def _configure_codex(container, auth_type: str, secret: str) -> None:
    operation = {"api_key": "api-key", "oauth": "oauth-cache"}.get(auth_type)
    if operation is None:
        raise brain_credentials_client.BrainCredentialError("Brain credential metadata is unsupported")
    payload = secret.encode("utf-8") + (b"\n" if auth_type == "api_key" else b"")
    rc, _stdout, _stderr = _brain_exec_stdin(
        container,
        ["timeout", "30", "shimpz-codex-auth", operation],
        payload,
    )
    configured, _authenticated = _codex_status(container)
    if rc != 0 or not configured:
        raise brain_credentials_client.BrainCredentialError("Brain credential injection failed")


def _deconfigure_codex(container) -> None:
    _brain_exec(container, ["shimpz-codex-auth", "logout"])
    rc, _output = _brain_exec(container, ["rm", "-f", "/config/.shimpz/codex-thread-id"])
    if rc != 0:
        raise OSError("Codex session removal failed")


def _wait_brain_ready(container, brain: str) -> None:
    readycheck = manifests.BRAINS[brain]["readycheck"]
    for _attempt in range(HEALTH_RETRIES):
        container.reload()
        if container.status != "running":
            break
        rc, _ = _brain_exec(container, ["sh", "-c", readycheck])
        if rc == 0:
            return
        time.sleep(HEALTH_DELAY_SECONDS)
    raise brain_credentials_client.BrainCredentialError(f"{brain!r} Brain did not become ready")


def _claude_cmd(*, resume: bool, stream: bool, model: str = "") -> list[str]:
    """The headless brain invocation, mirroring shimpzchat.py's own flags EXACTLY.

    `--dangerously-skip-permissions` is MANDATORY here: with no interactive approver in `-p` mode, a
    tool call (Bash, shimpz-ask, …) would otherwise BLOCK forever on a permission prompt no one can
    answer — the same reason the Telegram brain runs with it (as the unprivileged `abc` user, which
    is what makes the flag acceptable; the capsule is already sandboxed: no docker.sock, no host, its
    own net + scoped DB). `stream` uses stream-json (the same read loop the Telegram brain relays) so
    the web chat can update the reply LIVE and show tool status; else plain text (the fallback path).
    """
    cmd = ["timeout", "--kill-after=5", str(CHAT_TIMEOUT_SECONDS), "claude", "-p"]
    if resume:
        cmd.append("--continue")
    cmd.append("--dangerously-skip-permissions")
    if model:
        cmd += ["--model", model]
    if stream:
        cmd += ["--output-format", "stream-json", "--verbose", "--include-partial-messages"]
    else:
        cmd += ["--output-format", "text"]
    # The provider secret lives only in the private config volume. Source it inside the Capsule
    # immediately before exec so it never appears in Docker environment metadata or argv.
    return [
        "sh",
        "-c",
        "set -a; [ ! -r /config/.shimpz/brain-credential.sh ] || "
        '. /config/.shimpz/brain-credential.sh; set +a; exec "$@"',
        "shimpz-brain",
        *cmd,
    ]


def _claude_invocation(message: str, resume: bool, stream: bool, model: str = "") -> tuple[list[str], bytes | None]:
    return _claude_cmd(resume=resume, stream=stream, model=model), message.encode("utf-8")


def _codex_invocation(message: str, resume: bool, stream: bool, model: str = "") -> tuple[list[str], bytes | None]:
    """Fixed runner argv plus prompt bytes carried exclusively over Docker exec stdin."""
    command = [
        "timeout",
        "--kill-after=5",
        str(CHAT_TIMEOUT_SECONDS),
        "shimpz-codex-run",
        "resume" if resume else "new",
        "json" if stream else "text",
    ]
    if model:
        command.extend(("--model", model))
    return command, message.encode("utf-8")


def _brain_invocation(container, message: str, *, resume: bool, stream: bool) -> tuple[list[str], bytes | None]:
    brain, adapter = _adapter_for(container)
    return adapter.invocation(message, resume, stream, _brain_model(container, brain))


def _chat_command(container, command: list[str], token: str) -> list[str]:
    return ["shimpz-chat-exec", token, container.id, *command]


def _run_brain_once(container, message: str, *, resume: bool, token: str) -> tuple[int | None, str]:
    command, payload = _brain_invocation(container, message, resume=resume, stream=False)
    command = _chat_command(container, command, token)
    if payload is None:
        return _brain_exec(container, command, user=_CHAT_CONTROL_USER)
    rc, stdout, stderr = _brain_exec_stdin(container, command, payload, user=_CHAT_CONTROL_USER)
    return rc, stdout if rc == 0 else f"{stdout}\n{stderr}"


def _chat(cid: str, message: str, lease: _AuthorizationLease) -> dict:
    """One provider-adapted Captain exchange; first configured turn verifies authentication."""
    # The slot comes first. A losing concurrent request must not run even the local credential probe,
    # much less provider status or a second provider CLI.
    with _exclusive_chat_turn(cid, lease) as (token, container):
        _brain, adapter = _adapter_for(container)
        _clear_brain_authentication(cid)
        _require_brain_configured(cid, container)
        rc, out = _run_brain_once(container, message, resume=True, token=token)
        if (
            rc not in {130, 137, 143}
            and not _token_cancelled(token)
            and any(marker in out.lower() for marker in adapter.no_session_markers)
        ):
            rc, out = _run_brain_once(container, message, resume=False, token=token)
        if not _commit_chat_terminal(cid, token):
            raise ApiError(HTTPStatus.CONFLICT, "brain turn stopped")
        if rc == 124:
            raise ApiError(HTTPStatus.GATEWAY_TIMEOUT, f"the brain did not answer within {CHAT_TIMEOUT_SECONDS}s")
        if rc != 0:
            lowered = out.lower()
            if rc == _CHAT_BUSY_EXIT and _CHAT_BUSY_MARKER in lowered:
                raise ApiError(HTTPStatus.CONFLICT, "capsule already has an active chat turn")
            if any(marker in lowered for marker in _AUTH_MARKERS):
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    "brain not authenticated — configure or refresh the account Brain credential",
                )
            raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, f"brain execution failed (rc={rc})")
        _mark_brain_authenticated(cid, container)
        return {"capsule": cid, "reply": out.strip()[:CHAT_OUTPUT_CAP]}


def _stream_events(container, message: str, *, resume: bool, token: str):
    """Run one provider JSON stream and yield provider-neutral chat events."""
    _brain, adapter = _adapter_for(container)
    command, payload = _brain_invocation(container, message, resume=resume, stream=True)
    command = _chat_command(container, command, token)
    attached = None
    if payload is None:
        exec_id = _docker.api.exec_create(
            container.id,
            command,
            stdin=False,
            stdout=True,
            stderr=True,
            user=_CHAT_CONTROL_USER,
            workdir="/config/workspace",
            environment={"HOME": "/config"},
        )["Id"]
        frames = ((docker_socket.STDOUT, chunk) for chunk in _docker.api.exec_start(exec_id, stream=True))
    else:
        exec_id, attached = _open_brain_exec_stdin(container, command, payload, user=_CHAT_CONTROL_USER)
        frames = docker_socket.frames_iter(attached, tty=False)
    buf = b""
    state = _ProviderStreamState(adapter)
    raw_bytes = 0
    tail = ""  # last bit of raw output, so an auth failure (no text, non-zero exit) is still classifiable
    try:
        for stream_id, chunk in frames:
            raw_bytes += len(chunk)
            tail = (tail + chunk.decode("utf-8", "replace"))[-2000:]
            if raw_bytes > MAX_STREAM_RAW_BYTES:
                state.fail("brain stream exceeded its total output limit")
                _apply_stream_stop(state, container, token)
                continue
            if stream_id != docker_socket.STDOUT:
                continue
            buf += chunk
            if len(buf) > MAX_STREAM_LINE_BYTES and b"\n" not in buf:
                state.fail("brain stream line exceeded its size limit")
                buf = b""
                _apply_stream_stop(state, container, token)
                continue
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                outgoing = state.consume(line)
                if outgoing is not None:
                    yield outgoing
            _apply_stream_stop(state, container, token)
        if buf and not state.protocol_error:
            outgoing = state.consume(buf)
            if outgoing is not None:
                yield outgoing
            _apply_stream_stop(state, container, token)
    finally:
        if attached is not None:
            _close_exec_stream(attached)
    rc = _docker.api.exec_inspect(exec_id).get("ExitCode")
    yield {
        "t": "_end",
        "rc": rc,
        "final": state.final,
        "tail": tail,
        "completed": state.completed,
        "requires_completion": adapter.requires_completion_event,
        "protocol_error": state.protocol_error,
        "cancelled": _token_cancelled(token),
    }


def _parse_claude_stream_line(line: bytes):
    """One stream-json line → ('text'|'tool'|'result', value) or None. Never raises."""
    text = line.decode("utf-8", "replace").strip()
    if not text:
        return None
    try:
        evt = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(evt, dict):
        return None
    etype = evt.get("type")
    if etype == "assistant":
        message = evt.get("message")
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        if not isinstance(content, list):
            return None
        for blk in content:
            if not isinstance(blk, dict):
                continue
            block_text = blk.get("text")
            if blk.get("type") == "text" and isinstance(block_text, str) and block_text.strip():
                return ("text", block_text.strip())
            if blk.get("type") == "tool_use":
                return ("tool", str(blk.get("name") or "tool"))
    elif etype == "result":
        result = evt.get("result")
        if isinstance(result, str):
            return ("result", result.strip())
    return None


def _codex_tool_label(item: dict) -> str | None:
    item_type = item.get("type")
    if item_type == "command_execution":
        return str(item.get("command") or "Running command")
    if item_type == "mcp_tool_call":
        return str(item.get("tool") or item.get("name") or "Calling tool")
    if item_type == "web_search":
        return str(item.get("query") or "Searching the web")
    if item_type == "file_change":
        return "Editing files"
    if item_type == "plan_update":
        return "Updating plan"
    return None


def _parse_codex_stream_line(line: bytes):
    """One official ``codex exec --json`` JSONL event → a provider-neutral event."""
    text = line.decode("utf-8", "replace").strip()
    if not text:
        return None
    try:
        event = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict):
        return None
    event_type = event.get("type")
    if event_type == "thread.started":
        thread_id = event.get("thread_id")
        return ("thread", thread_id) if isinstance(thread_id, str) else None
    if event_type == "turn.completed":
        return ("complete", "")
    if event_type in {"turn.failed", "error"}:
        return ("error", "")
    item = event.get("item")
    if not isinstance(item, dict):
        return None
    if event_type == "item.completed" and item.get("type") == "agent_message":
        message = item.get("text")
        if isinstance(message, str) and message.strip():
            return ("text", message.strip())
        return None
    if event_type != "item.started":
        return None
    label = _codex_tool_label(item)
    return ("tool", label[:200]) if label is not None else None


_BRAIN_ADAPTERS = {
    "claude-code": _BrainAdapter(
        status=_claude_status,
        configure=_configure_claude,
        deconfigure=_deconfigure_claude,
        invocation=_claude_invocation,
        parse_stream_line=_parse_claude_stream_line,
        no_session_markers=("no conversation",),
        interactive_login=True,
        requires_completion_event=True,
    ),
    "codex": _BrainAdapter(
        status=_codex_status,
        configure=_configure_codex,
        deconfigure=_deconfigure_codex,
        invocation=_codex_invocation,
        parse_stream_line=_parse_codex_stream_line,
        no_session_markers=("no saved session", "no session found", "no rollout found", "no thread found"),
        interactive_login=False,
        requires_completion_event=True,
    ),
}


def _stream_terminal_event(end: dict | None) -> dict:
    """Map the real Docker exec verdict to exactly one terminal WebSocket event."""
    if end is None:
        return {"type": "error", "status": 502, "detail": "brain stream ended without an exit status"}
    if end.get("protocol_error"):
        return {"type": "error", "status": 502, "detail": str(end["protocol_error"])}
    rc = end.get("rc")
    if rc in {130, 137, 143}:
        return {"type": "error", "status": 500, "detail": "brain process terminated unexpectedly"}
    if rc == 0:
        if end.get("requires_completion") and not end.get("completed"):
            return {"type": "error", "status": 502, "detail": "brain stream ended without a completion event"}
        if not str(end.get("final", "")).strip():
            return {"type": "error", "status": 502, "detail": "brain stream completed without an answer"}
        return {"type": "done", "reply": str(end.get("final", ""))[:CHAT_OUTPUT_CAP]}
    if rc == 124:
        return {
            "type": "error",
            "status": 504,
            "detail": f"the brain did not answer within {CHAT_TIMEOUT_SECONDS}s",
        }
    tail = str(end.get("tail", ""))
    if rc == _CHAT_BUSY_EXIT and _CHAT_BUSY_MARKER in tail.lower():
        return {"type": "error", "status": 409, "detail": "capsule already has an active chat turn"}
    if any(marker in tail.lower() for marker in _AUTH_MARKERS):
        return {"type": "error", "status": 409, "detail": "brain not authenticated"}
    return {"type": "error", "status": 500, "detail": f"brain error (rc={rc})"}


def _fail_stop_chat_container(container, cid: str) -> bool:
    """Prove extinction, then best-effort restore only through the complete isolation gate."""
    _fail_stop_capsule(container)
    try:
        _require_capsule_runtime()
        _start_capsule_with_isolation(container)
    except ApiError, docker.errors.DockerException:
        return False
    with _active_chat_guard:
        _blocked_chat_capsules.discard(cid)
    return True


def _terminate_chat_token(
    container,
    cid: str,
    token: str | None,
    *,
    durable_recovered: bool = False,
) -> tuple[bool, bool, bool]:
    """Stop one captured token; container fallback is allowed only while that token is active."""
    confirmed = False
    forced_restart = False
    accepted = False
    if token is not None:
        try:
            result = container.exec_run(["shimpz-chat-stop", token], user=_CHAT_CONTROL_USER)
            accepted = result.exit_code in {0, 3}
            confirmed = result.exit_code == 0
        except docker.errors.DockerException:
            accepted = False
        if not accepted:
            with _active_chat_guard:
                still_active = (
                    _active_chat_tokens.get(cid) == token and _active_chat_container_ids.get(cid) == container.id
                )
            if not still_active and durable_recovered:
                try:
                    still_active = _durable_active_chat_token(container) == token
                except ApiError:
                    # The authenticated Stop captured a valid durable token. If its termination and
                    # subsequent identity read are both ambiguous, fail-stop this exact container.
                    still_active = True
            try:
                current = _get_container(manifests.capsule_container_name(cid))
                container.reload()
                same_running_container = (
                    current is not None and current.id == container.id and container.status == "running"
                )
            except docker.errors.DockerException:
                same_running_container = False
            if still_active and same_running_container:
                try:
                    forced_restart = _fail_stop_chat_container(container, cid)
                    confirmed = True
                    accepted = True
                except ApiError:
                    confirmed = False
    return accepted, confirmed, forced_restart


def _abort_chat(container, cid: str, token: str) -> None:
    """Internal transport abort: terminate work without pretending the Captain pressed Stop."""
    accepted, _confirmed, _forced_restart = _terminate_chat_token(container, cid, token)
    audit.log("chat_abort", cid, result="ok" if accepted else "error")


def _stop_chat(cid: str, lease: _AuthorizationLease) -> dict:
    """Kill this Capsule's one allowed in-flight provider process (the Stop button)."""
    # Stop cannot take the chat lock (that would deadlock behind the turn it must terminate). The
    # lifecycle lock protects the authorization recheck + termination as one operation; the active
    # token's immutable container id additionally prevents a stale Stop from targeting a replacement.
    with _lock_for(cid):
        container = _require_current_authorization(cid, lease)
        container.reload()
        if container.status != "running":
            raise ApiError(HTTPStatus.CONFLICT, f"capsule {cid!r} is not running (status={container.status})")
        try:
            token, recovered = _request_chat_stop(cid, container)
        except ApiError as state_error:
            if state_error.status != HTTPStatus.SERVICE_UNAVAILABLE:
                raise
            # Corrupt/lost control state cannot identify a token safely. An authenticated owner Stop
            # therefore proves extinction at the exact authorized container boundary; a failed stop
            # remains unavailable instead of pretending success.
            try:
                forced_restart = _fail_stop_chat_container(container, cid)
            except ApiError as stop_error:
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Capsule chat state is ambiguous and the Capsule stop could not be proved",
                ) from stop_error
            audit.log("chat_stop", cid, result="ok", recovery="fail-stop")
            return {
                "capsule": cid,
                "requested": True,
                "accepted": True,
                "confirmed": True,
                "forced_restart": forced_restart,
            }
        try:
            accepted, confirmed, forced_restart = _terminate_chat_token(
                container,
                cid,
                token,
                durable_recovered=recovered,
            )
        finally:
            if recovered and token is not None:
                with _active_chat_guard:
                    # Recovery can race a just-claimed same-driver turn before it publishes its
                    # in-memory mapping. If that mapping appeared meanwhile, the turn itself must
                    # consume the cancellation and emit `stopped`; only an unmapped recovered token
                    # is safe to retire here.
                    if _active_chat_tokens.get(cid) != token and not _chat_lock_for(cid).locked():
                        _cancelled_chat_tokens.discard(token)
    audit.log("chat_stop", cid, result="ok" if token is not None and accepted else "denied")
    return {
        "capsule": cid,
        "requested": token is not None,
        "accepted": accepted,
        "confirmed": confirmed,
        "forced_restart": forced_restart,
    }


def _put_inbox_file(
    cid: str,
    filename: str,
    content_b64: str,
    lease: _AuthorizationLease,
) -> dict:
    """Land an uploaded file in the capsule's OWN workspace inbox (chat references it by path)."""
    safe_name = validate.validate_inbox_filename(filename)
    try:
        data = base64.b64decode(content_b64 or "", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid base64 content: {exc}") from exc
    if not data or len(data) > MAX_INBOX_FILE_BYTES:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"file must be 1..{MAX_INBOX_FILE_BYTES} bytes")
    with _lock_for(cid):
        container = _require_current_authorization(cid, lease)
        container.reload()
        if container.status != "running":
            raise ApiError(HTTPStatus.CONFLICT, f"capsule {cid!r} is not running (status={container.status})")
        _brain_exec(container, ["mkdir", "-p", INBOX_DIR])
        container.put_archive(INBOX_DIR, manifests.build_inbox_tar(safe_name, data))
        return {"capsule": cid, "path": f"{INBOX_DIR}/{safe_name}", "bytes": len(data)}


# ── operations ───────────────────────────────────────────────────────────────
def _remove_volume(cid: str, kind: str) -> bool:
    name = network_policy.volume_name(cid, kind)
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
    if not network_policy.volume_identity_valid(volume.attrs, cid, kind):
        return False
    try:
        volume.remove(force=True)
    except docker.errors.NotFound:
        return True
    except docker.errors.DockerException:
        return False
    return True


def _owned_teardown_brain(cid: str, owner: str, brain_id: str):
    try:
        brain = _get_container(manifests.capsule_container_name(cid))
    except docker.errors.DockerException:
        return False, None
    if brain is None:
        return True, None
    try:
        brain.reload()
    except docker.errors.DockerException:
        return False, None
    valid = (
        network_policy.brain_identity_valid(brain.attrs, cid)
        and brain.id == brain_id
        and str(brain.labels.get("capsule.owner", "")) == owner
    )
    return valid, brain


def _stop_teardown_brain(brain) -> bool:
    if brain is None:
        return True
    try:
        _fail_stop_capsule(brain, timeout=30)
    except ApiError:
        return False
    return True


def _teardown_apps(cid: str) -> bool:
    try:
        app_containers = _capsule_app_containers(cid)
    except docker.errors.DockerException:
        return False
    cleanup_complete = True
    for app_container in app_containers:
        app_id = app_container.labels.get("capsule.app", "")
        if not isinstance(app_id, str) or marketplace.APP_ID_RE.fullmatch(app_id) is None:
            cleanup_complete = False
            continue
        # The Capsule-level database drop removes every registered App database in one scoped call.
        result = _teardown_app(cid, app_id, container=app_container, drop_db=False)
        cleanup_complete = result.artifacts_removed and cleanup_complete
    return cleanup_complete


def _teardown_network_planes(cid: str) -> bool:
    return _teardown_capsule_networks(cid)


def _remove_teardown_brain(brain) -> bool:
    if brain is None:
        return True
    return _remove_capsule_container(brain, timeout=30)


def _purge_teardown_credentials(brain) -> bool:
    """Purge every supported provider artifact and swap quarantine before releasing the Brain."""
    if brain is None:
        return True
    try:
        _provider_volume_action(brain, "purge-all", manifests.DEFAULT_BRAIN)
    except ApiError:
        return False
    return True


def _teardown_volumes(cid: str) -> bool:
    results = [
        _remove_volume(cid, kind) for kind in (network_policy.CONFIG_VOLUME_KIND, network_policy.WORKSPACE_VOLUME_KIND)
    ]
    return all(results)


def _retire_teardown_r2(cid: str) -> bool:
    """Cut off every new Capsule R2 operation before deleting any tenant artifact."""
    try:
        r2driver_client.retire_capsule(cid)
    except r2driver_client.R2DriverError:
        return False
    return True


def _drop_teardown_database(cid: str, record: cleanup_state.Record) -> cleanup_state.Record | None:
    if record.db_dropped:
        return record
    try:
        pgdriver_client.drop_capsule(cid)
        return cleanup_state.mark_db_dropped(record)
    except (
        pgdriver_client.PgDriverError,
        cleanup_state.CleanupStateError,
        http.client.HTTPException,
        OSError,
        ValueError,
    ):
        return None


def _finalize_teardown(cid: str, record: cleanup_state.Record) -> bool:
    try:
        # R2 destroys its encrypted bundles and hashed principal first. Its local cleartext principal
        # is removed only after that authenticated 200; both finalizers remain safe to replay.
        r2driver_client.finalize_capsule_drop(cid)
        pgdriver_client.finalize_capsule_drop(cid)
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


def _teardown(cid: str, *, owner: str, brain_id: str) -> _CleanupResult:
    """Remove every Capsule artifact, preserving a durable owner-bound retry anchor throughout."""
    brain_valid, brain = _owned_teardown_brain(cid, owner, brain_id)
    if not brain_valid:
        return _CleanupResult(False, False)

    # Persist the immutable tenant/Brain identity before the first mutation. Once Docker releases the
    # Brain's volume references this record—not a runnable workload—authorizes only a retrying DELETE.
    try:
        record = cleanup_state.begin(cid, owner, brain_id)
    except cleanup_state.CleanupStateError:
        return _CleanupResult(False, False)
    if not _stop_teardown_brain(brain):
        return _CleanupResult(False, record.db_dropped)
    # Fail closed while the durable cleanup record and stopped Brain still exist. No credential,
    # volume, network or database artifact is removed until R2 has revoked this Capsule principal.
    if not _retire_teardown_r2(cid):
        return _CleanupResult(False, record.db_dropped)
    if not _purge_teardown_credentials(brain):
        return _CleanupResult(False, record.db_dropped)
    if not _teardown_apps(cid) or not _teardown_network_planes(cid):
        return _CleanupResult(False, record.db_dropped)
    if not _remove_teardown_brain(brain) or not _teardown_volumes(cid):
        return _CleanupResult(False, record.db_dropped)
    record = _drop_teardown_database(cid, record)
    if record is None:
        return _CleanupResult(False, False)
    # pg-driver keeps a retired, idempotent principal until this provisioner-authorized finalizer;
    # only then is the controller's cleartext principal removed. Both operations are retry-safe.
    if not _finalize_teardown(cid, record):
        return _CleanupResult(False, True)
    return _CleanupResult(True, True)


def _database_url_from(container) -> str:
    for value in container.attrs.get("Config", {}).get("Env", []):
        if value.startswith("DATABASE_URL="):
            database_url = value.split("=", 1)[1]
            if database_url:
                return database_url
    raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "existing Capsule has no scoped database credential")


def _provider_config_mount(container):
    container.reload()
    cid = str(container.labels.get("capsule.id", ""))
    expected = network_policy.volume_name(cid, network_policy.CONFIG_VOLUME_KIND)
    for mount in container.attrs.get("Mounts", []):
        if (
            mount.get("Destination") != "/config"
            or mount.get("Type") != "volume"
            or mount.get("Name") != expected
            or mount.get("RW") is not True
        ):
            continue
        _require_capsule_volumes(cid)
        return docker.types.Mount(target="/config", source=expected, type="volume", read_only=False)
    raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "Capsule has no exact writable private config volume")


def _provider_volume_action(container, action: str, provider: str, token: str | None = None) -> None:
    """Run one fixed root-only file transaction against a Capsule config volume, even while stopped."""
    if action not in {"hide", "restore", "discard", "purge", "purge-all"} or provider not in _PROVIDER_ARTIFACTS:
        raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "invalid provider credential transaction")
    if action not in {"purge", "purge-all"} and (
        token is None or len(token) != 32 or any(char not in "0123456789abcdef" for char in token)
    ):
        raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "invalid provider credential transaction")
    transaction_token = token or "-"
    provider_spec = manifests.BRAINS.get(provider)
    image_ref = provider_spec.get("image") if provider_spec is not None else None
    if not isinstance(image_ref, str) or not image_ref:
        raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "invalid provider credential transaction")
    image_id = _trusted_image_id(image_ref)
    # Resolve release-owned control-plane code independently of the possibly drifted workload image.
    # The writable mount still contains hostile tenant data, so keep this short-lived helper inside
    # the configured sandbox as well.
    _require_capsule_runtime()
    try:
        _docker.containers.run(
            image_id,
            command=[action, provider, transaction_token],
            entrypoint=["/opt/venv/bin/python3", "-c", _PROVIDER_VOLUME_SCRIPT],
            environment={"PYTHONDONTWRITEBYTECODE": "1"},
            mounts=[_provider_config_mount(container)],
            runtime=manifests.RUNTIME,
            network_mode="none",
            read_only=True,
            cap_drop=["ALL"],
            cap_add=["DAC_OVERRIDE", "FOWNER"],
            security_opt=["no-new-privileges:true"],
            user="0:0",
            remove=True,
            stdout=False,
            # docker-py must attach at least one stream when detach=False; disabling both makes it
            # raise after an otherwise-successful helper exits and leaves the transaction orphaned.
            stderr=True,
        )
    except docker.errors.DockerException as exc:
        raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "provider credential transaction failed") from exc


def _remove_replacement_before_credential_restore(replacement) -> bool:
    """Return true only after the replacement identity is proved absent from Engine."""
    return _remove_capsule_container(replacement)


def _restore_old_brain_topology(
    existing,
    *,
    original_name: str,
    renamed: bool,
    core_network,
    egress_network,
) -> bool:
    """Reconcile the old Brain's exact name/endpoints after any ambiguous rollback response."""
    cid = original_name
    try:
        existing.reload()
        if renamed:
            with contextlib.suppress(docker.errors.DockerException):
                existing.rename(original_name)
            # docker-py keeps the pre-rename name cached until an explicit reload by immutable ID.
            existing.reload()
        if existing.name != original_name:
            with contextlib.suppress(OSError):
                audit.log("brain_rollback_topology", cid, result="denied", reason="name")
            return False
    except docker.errors.DockerException:
        with contextlib.suppress(OSError):
            audit.log("brain_rollback_topology", cid, result="denied", reason="container-inspect")
        return False
    for network in (core_network, egress_network):
        connected = False
        for _attempt in range(2):
            try:
                _safe_connect(network, original_name, required=True)
            except ApiError:
                continue
            connected = True
            break
        if not connected:
            with contextlib.suppress(OSError):
                audit.log("brain_rollback_topology", cid, result="denied", reason="network-connect")
            return False
    # Network connect/rename responses may precede the Engine's converged inspect view. Re-read the
    # immutable container ID and require the complete policy; a bounded retry is reconciliation, not
    # an omission allowance. No credential is restored unless one attempt proves the exact topology.
    for attempt in range(3):
        try:
            _require_capsule_isolation(existing)
        except (ApiError, docker.errors.DockerException) as exc:
            if attempt < 2:
                time.sleep(0.1)
            else:
                reason = exc.message if isinstance(exc, ApiError) else "docker-reconciliation-error"
                with contextlib.suppress(OSError):
                    audit.log("brain_rollback_topology", cid, result="denied", reason=reason)
        else:
            return True
    return False


def _raise_brain_replacement_failure(
    cause: Exception,
    *,
    existing,
    replacement,
    original_name: str,
    old_brain: str,
    new_brain: str,
    quarantine: str,
    hide_intent: bool,
    renamed: bool,
    old_was_running: bool,
    core_network,
    egress_network,
) -> None:
    """Restore the old container and opaque credential files, or leave it safely stopped."""
    candidate_absent = replacement is None or _remove_replacement_before_credential_restore(replacement)
    old_stopped = False
    if candidate_absent:
        try:
            _fail_stop_capsule(existing, timeout=30)
        except ApiError:
            pass
        else:
            old_stopped = True
    topology_restored = candidate_absent and _restore_old_brain_topology(
        existing,
        original_name=original_name,
        renamed=renamed,
        core_network=core_network,
        egress_network=egress_network,
    )
    credential_restored = not hide_intent
    # Never restore a usable secret while a candidate may retain this mount, the old workload is not
    # proved stopped, or its owner-visible identity/topology could not be reconciled.
    if candidate_absent and old_stopped and topology_restored and hide_intent:
        try:
            _provider_volume_action(existing, "purge", new_brain)
            _provider_volume_action(existing, "restore", old_brain, quarantine)
        except ApiError:
            credential_restored = False
        else:
            credential_restored = True
    rollback_complete = candidate_absent and old_stopped and topology_restored and credential_restored
    if rollback_complete and old_was_running:
        try:
            _start_capsule_with_isolation(existing)
        except ApiError:
            rollback_complete = False
    if not rollback_complete:
        with contextlib.suppress(OSError):
            audit.log(
                "brain_rollback",
                original_name,
                result="denied",
                candidate_absent=candidate_absent,
                old_stopped=old_stopped,
                topology_restored=topology_restored,
                credential_restored=credential_restored,
            )
        raise ApiError(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Brain provider replacement failed; candidate removal or credential rollback could not be proved",
        ) from cause
    if isinstance(cause, ApiError):
        raise cause
    raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "Brain provider replacement failed; rolled back") from cause


def _replace_brain(existing, cid: str, name: str, owner: str, brain: str, model: str) -> dict:
    """Atomically replace the provider/model container, preserving DB/config/workspace/network."""
    old_brain = _brain_id(existing)
    old_model = _brain_model(existing, old_brain)
    try:
        credential = brain_credentials_client.resolve(owner, brain) if owner else None
    except brain_credentials_client.BrainCredentialError as exc:
        raise ApiError(HTTPStatus.BAD_GATEWAY, str(exc)) from exc
    _require_capsule_runtime()
    database_url = _database_url_from(existing)
    original_name = manifests.capsule_container_name(cid)
    retiring_name = f"{original_name}__brain_{secrets.token_hex(4)}"
    replacement = None
    existing.reload()
    core_network = _docker.networks.get(manifests.capsule_network_name(cid))
    egress_network = _docker.networks.get(manifests.capsule_brain_egress_network_name(cid))
    old_was_running = existing.status == "running"
    chat_lock = _chat_lock_for(cid)
    quarantine = secrets.token_hex(16)
    hide_intent = False
    committed = False
    chat_lock_acquired = False
    renamed = False
    try:
        if old_was_running:
            _fail_stop_capsule(existing, timeout=30)
        if not chat_lock.acquire(timeout=30):
            raise ApiError(HTTPStatus.CONFLICT, "the active chat turn did not stop in time")
        chat_lock_acquired = True
        with _active_chat_guard:
            _credential_mutations.discard(cid)
            _credential_mutation_execs.pop(cid, None)
            _verified_brains.pop(cid, None)
        # Detach the stopped old identity before renaming it. This keeps the strict network member
        # policy valid for rollback-safe pre-start verification of the replacement. Rollback restores
        # both attachments before it is allowed to restart the old Brain.
        core_network.disconnect(existing)
        egress_network.disconnect(existing)
        existing.rename(retiring_name)
        renamed = True
        # Intent precedes the helper call: a lost response may still have committed the quarantine.
        hide_intent = True
        _provider_volume_action(existing, "hide", old_brain, quarantine)
        _require_capsule_runtime()
        replacement = _docker.containers.create(
            **manifests.build_capsule_kwargs(
                cid,
                name,
                database_url=database_url,
                owner=owner,
                brain=brain,
                model=model,
            )
        )
        _safe_connect(egress_network, replacement.name, required=True)
        _start_capsule_with_isolation(replacement)
        _wait_brain_ready(replacement, brain)
        if credential is not None:
            auth_type, secret, generation = credential
            _BRAIN_ADAPTERS[brain].configure(replacement, auth_type, secret)
            _assert_credential_generation(replacement, owner, brain, generation)
        _require_capsule_isolation(replacement)
        if not _remove_capsule_container(existing, timeout=30):
            raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "old Brain removal could not be proved")
        committed = True
        _provider_volume_action(replacement, "discard", old_brain, quarantine)
        if not old_was_running:
            _fail_stop_capsule(replacement, timeout=30)
    except Exception as exc:
        if committed:
            if replacement is not None:
                try:
                    _fail_stop_capsule(replacement)
                except ApiError as stop_exc:
                    raise ApiError(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "Brain provider changed but cleanup failed and its Capsule stop could not be proved",
                    ) from stop_exc
            raise ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Brain provider changed but credential cleanup failed; Capsule stopped for safety",
            ) from exc
        _raise_brain_replacement_failure(
            exc,
            existing=existing,
            replacement=replacement,
            original_name=original_name,
            old_brain=old_brain,
            new_brain=brain,
            quarantine=quarantine,
            hide_intent=hide_intent,
            renamed=renamed,
            old_was_running=old_was_running,
            core_network=core_network,
            egress_network=egress_network,
        )
    finally:
        if chat_lock_acquired:
            chat_lock.release()
    audit.log(
        "brain_replace",
        cid,
        result="ok",
        old_brain=old_brain,
        old_model=old_model,
        brain=brain,
        model=model,
    )
    replacement.reload()
    return {
        "capsule": cid,
        "name": name,
        "brain": brain,
        "model": model,
        "status": replacement.status,
        "created": False,
        "brain_replaced": True,
        "brain_configured": credential is not None,
    }


def _create(cid: str, body: dict, owner: str = "") -> dict:
    name = str(body.get("name") or cid).strip() or cid
    brain = str(body.get("brain") or manifests.DEFAULT_BRAIN).strip()
    if brain not in manifests.BRAINS:
        raise ApiError(
            HTTPStatus.BAD_REQUEST, f"unknown brain {brain!r} — this Space accepts: {sorted(manifests.BRAINS)}"
        )
    try:
        model = manifests.model_for_brain(brain, body.get("model"))
    except ValueError as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    with _lock_for(cid):
        pending_cleanup = _cleanup_record(cid)
        if pending_cleanup is not None:
            if owner and pending_cleanup.owner != owner:
                raise ApiError(HTTPStatus.NOT_FOUND, f"capsule {cid!r} not found")
            raise ApiError(
                HTTPStatus.CONFLICT,
                f"capsule {cid!r} has an incomplete teardown; retry destroy before creating it",
            )
        existing = _get_container(manifests.capsule_container_name(cid))
        if existing is not None:
            # An account may only "re-create" (get) its OWN capsule; a name collision with a different
            # owner is invisible (404), never a hijack of someone else's capsule.
            existing_owner = existing.labels.get("capsule.owner", "")
            if owner and existing_owner != owner:
                raise ApiError(HTTPStatus.NOT_FOUND, f"capsule {cid!r} not found")
            # Upgrade fail-close: idempotent create must not bless a legacy runc container. Test
            # data can be destroyed/recreated; production migration must be an explicit release step.
            _require_capsule_runtime()
            _require_capsule_isolation(existing)
            existing_brain = _brain_id(existing)
            existing_model = _brain_model(existing, existing_brain)
            if existing_brain != brain or existing_model != model:
                return _replace_brain(existing, cid, name, existing_owner, brain, model)
            return {
                "capsule": cid,
                "name": name,
                "brain": existing_brain,
                "model": existing_model,
                "status": existing.status,
                "created": False,
            }
        # Reserve count + memory atomically, then let unrelated Capsules enter admission while the
        # runtime check, credential service, Postgres, Docker start and health work proceed.
        with _reserve_capacity(f"capsule:{cid}", owner, manifests.MEM_LIMIT_BYTES, capsule_slot=True):
            # Quotas are an admission decision of their own: an owner already at the limit must receive
            # 429 even while the hostile-tenant runtime is unavailable. A different owner reaches this
            # independent fail-closed host gate and still cannot provision without the required runtime.
            _require_capsule_runtime()
            try:
                credential = brain_credentials_client.resolve(owner, brain) if owner else None
            except brain_credentials_client.BrainCredentialError as exc:
                raise ApiError(HTTPStatus.BAD_GATEWAY, str(exc)) from exc
            # Transactional: on ANY failure, roll back everything partially created before surfacing —
            # never leak an orphan DB/role, network, or volume for an operator to hunt down later.
            container = None
            try:
                db = pgdriver_client.provision_capsule(cid)
                try:
                    # Principal registration precedes every runnable/public Brain artifact. A retry
                    # reuses the same local principal and the R2 lifecycle endpoint is idempotent.
                    r2driver_client.provision_capsule(cid)
                except r2driver_client.R2DriverError as exc:
                    raise ApiError(exc.status, exc.message) from exc
                _ensure_capsule_volume(cid, network_policy.CONFIG_VOLUME_KIND)
                _ensure_capsule_volume(cid, network_policy.WORKSPACE_VOLUME_KIND)
                network = _ensure_capsule_network(cid)
                egress_network = _ensure_brain_egress_network(cid)
                _wire_network_deps(network, manifests.core_deps())
                _wire_network_deps(egress_network, manifests.brain_egress_deps())
                _require_network_policy(
                    network,
                    cid,
                    network_policy.CORE_KIND,
                    require_brain=False,
                    require_dependencies=True,
                )
                _require_network_policy(
                    egress_network,
                    cid,
                    network_policy.BRAIN_EGRESS_KIND,
                    require_brain=False,
                    require_dependencies=True,
                )
                kwargs = manifests.build_capsule_kwargs(
                    cid,
                    name,
                    database_url=db["database_url"],
                    owner=owner,
                    brain=brain,
                    model=model,
                )
                _require_capsule_runtime()
                container = _docker.containers.create(**kwargs)
                _safe_connect(egress_network, container.name, required=True)
                _start_capsule_with_isolation(container)
                _wait_brain_ready(container, brain)
                if credential is not None:
                    auth_type, secret, generation = credential
                    _BRAIN_ADAPTERS[brain].configure(container, auth_type, secret)
                    _assert_credential_generation(container, owner, brain, generation)
            except Exception as exc:
                cleanup = _teardown(
                    cid,
                    owner=owner,
                    brain_id=container.id if container is not None else "",
                )
                if not cleanup.complete:
                    raise ApiError(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "Capsule create failed and rollback is incomplete; contact the operator",
                    ) from exc
                if isinstance(exc, ApiError):
                    raise
                raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "Capsule create failed and was rolled back") from exc
        return {
            "capsule": cid,
            "name": name,
            "brain": brain,
            "model": model,
            "status": "running",
            "created": True,
            "database": manifests.capsule_db_project(cid),
            "brain_configured": credential is not None,
        }


def _destroy(cid: str, lease: _AuthorizationLease) -> dict:
    with _lock_for(cid):
        # Destruction is the supported remediation for a legacy or drifted runtime.
        if lease.cleanup_nonce:
            _require_cleanup_authorization(cid, lease)
            container = None
        else:
            container = _require_current_authorization(
                cid,
                lease,
                require_isolation=False,
                allow_pending_cleanup=True,
            )
            # A running chat is terminated by stopping the Brain before its lock can drain. Commit
            # the retry authorization first so even a timeout or ambiguous Docker stop leaves the
            # owner with a durable path back into DELETE.
            try:
                cleanup_state.begin(cid, lease.owner, lease.container_id)
            except cleanup_state.CleanupStateError as exc:
                raise ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Capsule cleanup state is unavailable",
                ) from exc
        chat_lock = _chat_lock_for(cid)
        if container is not None:
            container.reload()
            if container.status == "running":
                _fail_stop_capsule(container, timeout=30)
        if not chat_lock.acquire(timeout=30):
            raise ApiError(HTTPStatus.CONFLICT, "the active chat turn did not stop in time")
        try:
            cleanup = _teardown(cid, owner=lease.owner, brain_id=lease.container_id)
            _clear_cid_runtime_state(cid)
            if not cleanup.complete:
                raise ApiError(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "Capsule teardown is incomplete; retry destroy or contact the operator",
                )
            return {"capsule": cid, "destroyed": True, "db_dropped": cleanup.db_dropped}
        finally:
            chat_lock.release()


def _list(owner: str | None = None) -> dict:
    """All capsules for the operator; only the account's own when `owner` is set."""
    caps = _docker.containers.list(all=True, filters={"label": "capsule.driver"})
    if owner is not None:
        caps = [c for c in caps if c.labels.get("capsule.owner", "") == owner]
    return {"capsules": [_describe(c) for c in caps]}


def _status(cid: str, lease: _AuthorizationLease) -> dict:
    with _lock_for(cid):
        # Status remains readable so the UI can offer Stop/Destroy remediation.
        return _describe(_require_current_authorization(cid, lease, require_isolation=False))


def _logs(cid: str, lines: int, lease: _AuthorizationLease) -> dict:
    with _lock_for(cid):
        container = _require_current_authorization(cid, lease, require_isolation=False)
        return {"capsule": cid, "logs": container.logs(tail=lines).decode("utf-8", "replace")}


def _lifecycle(cid: str, op: str, lease: _AuthorizationLease) -> dict:
    with _lock_for(cid):
        # Stop is always available as remediation. Start/restart require both an exact per-container
        # runtime and a currently registered daemon runtime; Docker may never fall back to runc.
        container = _require_current_authorization(cid, lease, require_isolation=op != "stop")
        if op in {"start", "restart"}:
            _require_capsule_runtime()
        container.reload()
        # Stop first so the provider tree cannot keep mutating the volume, then wait for the turn's
        # cleanup/finally to relinquish its slot. Start takes the same slot before making the container
        # runnable. The lifecycle lock serializes all of this with configure/deconfigure/destroy.
        if op in {"stop", "restart"} and container.status == "running":
            _fail_stop_capsule(container, timeout=30)
        chat_lock = _chat_lock_for(cid)
        if not chat_lock.acquire(timeout=30):
            raise ApiError(HTTPStatus.CONFLICT, "the active chat turn did not stop in time")
        try:
            container.reload()
            if op in {"start", "restart"} and container.status != "running":
                _start_capsule_with_isolation(container)
            with _active_chat_guard:
                if op in {"stop", "start", "restart"}:
                    _credential_mutations.discard(cid)
                    _credential_mutation_execs.pop(cid, None)
                if op in {"start", "restart"}:
                    _blocked_chat_capsules.discard(cid)
        finally:
            chat_lock.release()
    return {"capsule": cid, "op": op, "status": "ok"}


def _r2_driver_operation(
    cid: str,
    lease: _AuthorizationLease,
    operation: Callable[[], dict[str, object]],
) -> dict[str, object]:
    """Revalidate the exact Capsule, lazily provision its principal, then make one fixed R2 call."""
    with _lock_for(cid):
        _require_current_authorization(cid, lease)
        try:
            # Existing Capsules acquire R2 only here, after their owner and immutable container id
            # have both been rechecked inside the lifecycle lock. Provision is deliberately replayable.
            r2driver_client.ensure_provisioned(cid)
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
    server_version = "capsule-driver/1.0"

    def log_message(self, *_args) -> None:  # audit.log is the ONLY log source
        pass

    def _principal(self) -> tuple[str, str | None] | None:
        """('operator', None) for the admin bearer; ('account', <id>) for a valid account token; else None.

        The operator token (the admin panel) has full access. A store-forwarded account token is verified
        against the accounts service and scopes every op to that account's OWN capsules — the store holds
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

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream_chat(self, cid: str, message: str, lease: _AuthorizationLease) -> None:
        """Chunked NDJSON stream of a live brain turn — one JSON event per line, flushed as it happens.

        The store reads this line-by-line and relays each event over the Captain's WebSocket. On the
        first-message case (no session to --continue) the brain emits a 'no conversation' error with no
        text; we transparently restart fresh so the Captain never sees that internal retry.
        """
        with _exclusive_chat_turn(cid, lease) as (token, container):
            _brain, adapter = _adapter_for(container)
            _clear_brain_authentication(cid)
            _require_brain_configured(cid, container)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def emit(obj: dict) -> None:
                line = (json.dumps(obj, ensure_ascii=False) + "\n").encode()
                self.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n")
                self.wfile.flush()

            produced = False
            end = None
            terminal = {"type": "error", "status": 500, "detail": "brain stream failed"}
            stream_error = None
            try:
                for resume in (True, False):
                    end = None
                    events = _stream_events(container, message, resume=resume, token=token)
                    try:
                        for evt in events:
                            if evt.get("t") == "_end":
                                end = evt
                                break
                            produced = True
                            emit({"type": evt["t"], **{k: v for k, v in evt.items() if k != "t"}})
                    finally:
                        events.close()
                    no_session = end and any(
                        marker in str(end.get("tail", "")).lower() for marker in adapter.no_session_markers
                    )
                    stopped = end and (end.get("cancelled") or end.get("rc") in {130, 137, 143})
                    if produced or not no_session or stopped or (end and end.get("protocol_error")):
                        break  # got output, or a real (non-first-message) failure — don't retry
                terminal = _stream_terminal_event(end) if _commit_chat_terminal(cid, token) else {"type": "stopped"}
                if terminal["type"] == "done":
                    _mark_brain_authenticated(cid, container)
                emit(terminal)
            except (docker.errors.APIError, OSError) as exc:
                stream_error = type(exc).__name__
                _abort_chat(container, cid, token)
                terminal = (
                    {"type": "error", "status": 500, "detail": "brain stream failed"}
                    if _commit_chat_terminal(cid, token)
                    else {"type": "stopped"}
                )
                if not isinstance(exc, OSError):
                    with contextlib.suppress(OSError):
                        emit(terminal)
            finally:
                with contextlib.suppress(OSError):
                    self.wfile.write(b"0\r\n\r\n")  # terminating chunk
                    self.wfile.flush()
            audit.log(
                "chat",
                cid,
                result="ok" if terminal["type"] in {"done", "stopped"} else "error",
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

        if method == "GET" and path == "/v1/capsules":
            self._send_json(HTTPStatus.OK, _list(owner=account_id if kind == "account" else None))
            return

        if len(parts) >= 3 and parts[0] == "v1" and parts[1] == "capsules":
            cid = validate.validate_capsule_id(parts[2])
            sub = parts[3] if len(parts) > 3 else ""
            if method == "POST" and sub == "create":
                _enforce_rate("create", principal)
                body = self._read_body()
                # an account owns what it creates; an operator may create-on-behalf via an explicit owner
                owner = account_id or str(body.get("owner", "")).strip()
                result = _create(cid, body, owner)
                trace = audit.log("create", cid, result="ok", created=result.get("created"), owner=owner)
                self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
                return
            if method == "DELETE" and sub == "":
                # A completed Brain removal may leave a bounded durable cleanup record while volume
                # deletion is retried. Only Destroy may authorize against that non-runnable successor.
                lease = _authorize_destroy(cid, principal)
                result = _destroy(cid, lease)
                trace = audit.log("destroy", cid, result="ok", db_dropped=result["db_dropped"])
                self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
                return
            # every other op acts on an EXISTING capsule → gate on ownership first (404 if not yours)
            lease = _authorize(cid, principal)
            if sub == "drivers":
                self._route_driver(method, parts, cid, lease)
                return
            if sub == "apps":
                self._route_apps(method, parts, cid, principal, lease)
                return
            if sub == "brain":
                self._route_brain(method, parts, cid, principal, lease)
                return
            if sub == "chat":
                self._route_chat(method, parts, cid, principal, lease)
                return
            if method == "POST" and sub == "files":
                body = self._read_body(max_bytes=MAX_FILE_BODY_BYTES)
                result = _put_inbox_file(cid, body.get("filename"), body.get("content_b64"), lease)
                trace = audit.log("inbox_file", cid, result="ok", path=result["path"], bytes=result["bytes"])
                self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
                return
            if method == "GET" and sub == "status":
                self._send_json(HTTPStatus.OK, _status(cid, lease))
                return
            if method == "GET" and sub == "logs":
                self._send_json(HTTPStatus.OK, _logs(cid, int(query.get("lines", "200")), lease))
                return
            if method == "POST" and sub in ("stop", "start", "restart"):
                result = _lifecycle(cid, sub, lease)
                audit.log(sub, cid, result="ok")
                self._send_json(HTTPStatus.OK, result)
                return

        raise ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} {path}")

    def _route_driver(
        self,
        method: str,
        parts: list[str],
        cid: str,
        lease: _AuthorizationLease,
    ) -> None:
        """Closed Admin surface for the single proven Driver implementation: Cloudflare R2."""
        if len(parts) < 5 or parts[4] != "r2":
            raise ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} /{'/'.join(parts)}")
        if method == "GET" and len(parts) == 5:
            result = _r2_driver_operation(cid, lease, lambda: r2driver_client.driver_document(cid))
            self._send_json(HTTPStatus.OK, result)
            return
        if method == "POST" and len(parts) == 6 and parts[5] == "credentials":
            body = self._read_driver_body({"profile_id", "label", "values", "idempotency_key"})
            result = _r2_driver_operation(cid, lease, lambda: r2driver_client.create_credential(cid, body))
            audit.log("driver_credential_create", cid, result="ok", driver="r2", credential=result.get("id"))
            self._send_json(HTTPStatus.OK, result)
            return
        if len(parts) == 7 and parts[5] == "credentials":
            credential_id = parts[6]
            if method == "PUT":
                body = self._read_driver_body(
                    {"profile_id", "label", "values", "expected_generation"},
                )
                result = _r2_driver_operation(
                    cid,
                    lease,
                    lambda: r2driver_client.rotate_credential(cid, credential_id, body),
                )
                audit.log(
                    "driver_credential_rotate",
                    cid,
                    result="ok",
                    driver="r2",
                    credential=result.get("id"),
                )
                self._send_json(HTTPStatus.OK, result)
                return
            if method == "DELETE":
                body = self._read_driver_body({"expected_generation"})
                result = _r2_driver_operation(
                    cid,
                    lease,
                    lambda: r2driver_client.remove_credential(cid, credential_id, body),
                )
                audit.log(
                    "driver_credential_remove",
                    cid,
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
                cid,
                lease,
                lambda: r2driver_client.verify_credential(cid, credential_id),
            )
            audit.log(
                "driver_credential_verify",
                cid,
                result="ok",
                driver="r2",
                credential=result.get("id"),
            )
            self._send_json(HTTPStatus.OK, result)
            return
        raise ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} /{'/'.join(parts)}")

    def _route_brain(
        self,
        method: str,
        parts: list[str],
        cid: str,
        principal: tuple[str, str | None],
        lease: _AuthorizationLease,
    ) -> None:
        """Provider status/config plus the Claude-only legacy interactive OAuth bridge.

        Ownership was already enforced by _authorize. Login operations are rejected for every
        provider whose adapter does not explicitly expose the fixed `shimpz-login` bridge.
        """
        if method == "GET" and len(parts) == 4:
            self._send_json(HTTPStatus.OK, _brain_status(cid, lease))
            return
        if method == "POST" and len(parts) == 5 and parts[4] == "configure":
            owner = principal[1] or lease.owner
            if not owner:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    "this Capsule has no account owner for Brain configuration",
                )
            self._send_json(HTTPStatus.OK, _configure_brain(cid, owner, lease))
            return
        if method == "POST" and len(parts) == 5 and parts[4] == "deconfigure":
            self._send_json(HTTPStatus.OK, _deconfigure_brain(cid, lease))
            return
        if len(parts) == 6 and parts[4] == "login":
            step = parts[5]
            if method == "POST" and step == "start":
                self._send_json(HTTPStatus.OK, _capsule_login_start(cid, lease))
                return
            if method == "GET" and step == "url":
                self._send_json(HTTPStatus.OK, _capsule_login_url(cid, lease))
                return
            if method == "POST" and step == "code":
                self._send_json(HTTPStatus.OK, _capsule_login_code(cid, self._read_body(), lease))
                return
            if method == "GET" and step == "status":
                self._send_json(HTTPStatus.OK, _capsule_login_status(cid, lease))
                return
        raise ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} /{'/'.join(parts)}")

    def _route_chat(
        self,
        method: str,
        parts: list[str],
        cid: str,
        principal: tuple[str, str | None],
        lease: _AuthorizationLease,
    ) -> None:
        """/v1/capsules/{cid}/chat[/stream|/stop|/asks|/answer] — the Captain's brain conversation.

        Ownership was already enforced by _authorize. `chat` (bare) is the non-streaming fallback;
        `chat/stream` is the live NDJSON turn; the rest are the shimpz-ask surface + the Stop control.
        """
        sub2 = parts[4] if len(parts) > 4 else ""
        if method == "POST" and not sub2:
            _enforce_rate("chat", principal)
            message = validate.validate_chat_message(self._read_body().get("message"))
            result = _chat(cid, message, lease)
            audit.log("chat", cid, result="ok", chars_in=len(message), chars_out=len(result["reply"]))
            self._send_json(HTTPStatus.OK, result)
            return
        if method == "POST" and sub2 == "stream":
            _enforce_rate("stream", principal)
            self._stream_chat(cid, validate.validate_chat_message(self._read_body().get("message")), lease)
            return
        if method == "POST" and sub2 == "stop":
            _enforce_rate("stop", principal)
            self._send_json(HTTPStatus.OK, _stop_chat(cid, lease))
            return
        if method == "GET" and sub2 == "asks":
            _enforce_rate("asks", principal)
            self._send_json(HTTPStatus.OK, _chat_asks(cid, lease))
            return
        if method == "POST" and sub2 == "answer":
            _enforce_rate("answer", principal)
            self._send_json(HTTPStatus.OK, _chat_answer(cid, self._read_body(), lease))
            return
        raise ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} /{'/'.join(parts)}")

    def _route_apps(
        self,
        method: str,
        parts: list[str],
        cid: str,
        principal: tuple[str, str | None],
        lease: _AuthorizationLease,
    ) -> None:
        """/v1/capsules/{cid}/apps[/{app}] — the P4 deploy arm. Ownership was already enforced."""
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
            result = _install_app(cid, app_id, spec, owner, lease)
            trace = audit.log("install", cid, result="ok", app=app_id, installed=result["installed"])
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        if method == "GET" and len(parts) == 4:
            self._send_json(HTTPStatus.OK, _list_apps(cid, lease))
            return
        if method == "DELETE" and len(parts) == 5:
            # Shape-validated only — NOT resolved: an app later pulled from the registry must still
            # be uninstallable from every capsule that has it.
            app_id = marketplace.validate_app_id(parts[4])
            result = _uninstall_app(cid, app_id, lease)
            trace = audit.log("uninstall", cid, result="ok", app=app_id, db_dropped=result["db_dropped"])
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        raise ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} /{'/'.join(parts)}")


def main() -> None:
    _BoundedThreadingHTTPServer((ALL_INTERFACES, LISTEN_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
