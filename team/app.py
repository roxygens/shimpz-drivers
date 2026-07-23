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
import functools
import http.client
import ipaddress
import math
import os
import secrets
import sys
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import replace
from http import HTTPStatus
from pathlib import Path
from typing import NoReturn

import assistant_account_challenges
import assistant_account_flow
import assistant_genesis
import assistant_manifest
import assistant_secret_challenges
import assistant_secret_flow
import assistant_secret_store
import audit
import brain_runtime_client
import chat_orchestrator
import chat_turn_engine
import cleanup_state
import docker
import docker.errors
import inference_config
import manifests
import marketplace
import oauth_account_service
import oauth_account_store
import oauth_http_client
import oauth_pkce_challenges
import pgdriver_client
import power_execution
import power_journal
import team_storage
import token_store
import validate
from assistant_human import approval_challenges as assistant_approval_challenges
from assistant_human import approval_flow as assistant_approval_flow
from assistant_human import approval_grants as assistant_approval_grants
from assistant_human import input_challenges as assistant_input_challenges
from assistant_human import input_flow as assistant_input_flow
from container_policy import network as network_policy

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
MAX_ASSISTANT_SECRET_BODY_BYTES = 512 * 1024
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
APP_EGRESS_POLICY_GID = 10017
TEAM_STORAGE_ROOT = Path("/var/lib/team-driver/storage")
POWER_JOURNAL_PATH = Path(
    os.environ.get(
        "SHIMPZ_TEAM_POWER_JOURNAL_PATH",
        "/var/lib/team-driver/power-journal/journal.sqlite3",
    )
)
ASSISTANT_SECRET_STATE_PATH = Path(
    os.environ.get(
        "SHIMPZ_TEAM_ASSISTANT_SECRET_STATE_PATH",
        "/var/lib/team-driver/assistant-secrets/state/secrets.json",
    )
)
ASSISTANT_SECRET_KEY_PATH = Path(
    os.environ.get(
        "SHIMPZ_TEAM_ASSISTANT_SECRET_KEY_PATH",
        "/var/lib/team-driver/assistant-secrets/key/aes256.key",
    )
)
ASSISTANT_ACCOUNT_STATE_PATH = Path(
    os.environ.get(
        "SHIMPZ_TEAM_ASSISTANT_ACCOUNT_STATE_PATH",
        "/var/lib/team-driver/assistant-accounts/state/accounts.json",
    )
)
ASSISTANT_ACCOUNT_KEY_PATH = Path(
    os.environ.get(
        "SHIMPZ_TEAM_ASSISTANT_ACCOUNT_KEY_PATH",
        "/var/lib/team-driver/assistant-accounts/key/aes256.key",
    )
)
ASSISTANT_APPROVAL_GRANTS_PATH = Path(
    os.environ.get(
        "SHIMPZ_TEAM_ASSISTANT_APPROVAL_GRANTS_PATH",
        "/var/lib/team-driver/assistant-approvals/grants.sqlite3",
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
_assistant_genesis_cache = assistant_genesis.GenesisCache()
_assistant_allowed_hosts_cache = assistant_manifest.ManifestContractCache()
_assistant_machine_contract_cache = assistant_manifest.MachineContractCache()
_assistant_secrets = assistant_secret_store.AssistantSecretStore(
    ASSISTANT_SECRET_STATE_PATH,
    ASSISTANT_SECRET_KEY_PATH,
)
_assistant_secret_challenges = assistant_secret_challenges.SecretChallengeStore()
_assistant_accounts = oauth_account_store.OAuthAccountStore(
    ASSISTANT_ACCOUNT_STATE_PATH,
    ASSISTANT_ACCOUNT_KEY_PATH,
)
_assistant_account_challenges = assistant_account_challenges.AccountChallengeStore()
# Hosted interaction challenges are process-local by contract: a restart invalidates them and the
# client retries the turn; encrypted restart durability belongs only to the local Controller profile.
_assistant_approval_challenges = assistant_approval_challenges.ApprovalChallengeStore()
_assistant_approval_grants = assistant_approval_grants.ApprovalGrantStore(ASSISTANT_APPROVAL_GRANTS_PATH)
_assistant_input_challenges = assistant_input_challenges.InputChallengeStore()
_oauth_pkce_challenges = oauth_pkce_challenges.OAuthPKCEChallengeStore()
_oauth_http = oauth_http_client.OAuthHTTPClient()
_cloudflare_oauth_client_id = os.environ.get("SHIMPZ_CLOUDFLARE_OAUTH_CLIENT_ID")
_cloudflare_oauth_client_secret = os.environ.get("SHIMPZ_CLOUDFLARE_OAUTH_CLIENT_SECRET")
_oauth_accounts = oauth_account_service.OAuthAccountService(
    client_id=_cloudflare_oauth_client_id,
    client_secret=_cloudflare_oauth_client_secret,
    redirect_uri=oauth_http_client.HOSTED_REDIRECT_URI,
    challenge=_oauth_pkce_challenges,
    store=_assistant_accounts,
    http=_oauth_http,
)


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
    "secret": _FixedWindowRateLimiter(CHAT_RATE_LIMIT, CHAT_RATE_WINDOW_SECONDS),
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


def _serialize_against_team_chat(operation: Callable[..., dict]) -> Callable[..., dict]:
    """Reject lifecycle mutation before its first side effect while a Team turn owns the slot."""

    @functools.wraps(operation)
    def guarded(team_id: str, *args, **kwargs) -> dict:
        lock = _chat_lock_for(team_id)
        if not lock.acquire(blocking=False):
            raise ApiError(HTTPStatus.CONFLICT, "Team lifecycle cannot change during an active chat turn")
        try:
            return operation(team_id, *args, **kwargs)
        finally:
            lock.release()

    return guarded


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
from http_boundary import controller_binding

controller_binding.bind(sys.modules[__name__])

from assistant_human.hosted_assistants import (
    ASSISTANT_RPC_TIMEOUT_SECONDS as ASSISTANT_RPC_TIMEOUT_SECONDS,
)
from assistant_human.hosted_assistants import (
    CHAT_OUTPUT_CAP as CHAT_OUTPUT_CAP,
)
from assistant_human.hosted_assistants import (
    CHAT_PAUSED_STATUSES as CHAT_PAUSED_STATUSES,
)
from assistant_human.hosted_assistants import (
    MAX_ASSISTANT_RPC_OUTPUT_BYTES as MAX_ASSISTANT_RPC_OUTPUT_BYTES,
)
from assistant_human.hosted_assistants import (
    MAX_CHAT_ASSISTANTS as MAX_CHAT_ASSISTANTS,
)
from assistant_human.hosted_assistants import (
    MAX_CHAT_FILES as MAX_CHAT_FILES,
)
from assistant_human.hosted_assistants import (
    MAX_FILE_BODY_BYTES as MAX_FILE_BODY_BYTES,
)
from assistant_human.hosted_assistants import (
    MAX_INBOX_FILE_BYTES as MAX_INBOX_FILE_BYTES,
)
from assistant_human.hosted_assistants import (
    AssistantRpcRequest as AssistantRpcRequest,
)
from assistant_human.hosted_assistants import (
    PowerInvocationRequest as PowerInvocationRequest,
)
from assistant_human.hosted_assistants import (
    _active_team_assistants as _active_team_assistants,
)
from assistant_human.hosted_assistants import (
    _ActiveAssistant as _ActiveAssistant,
)
from assistant_human.hosted_assistants import (
    _admit_app_contract as _admit_app_contract,
)
from assistant_human.hosted_assistants import (
    _assistant_account_inventory as _assistant_account_inventory,
)
from assistant_human.hosted_assistants import (
    _assistant_help as _assistant_help,
)
from assistant_human.hosted_assistants import (
    _assistant_rpc as _assistant_rpc,
)
from assistant_human.hosted_assistants import (
    _assistant_rpc_exchange as _assistant_rpc_exchange,
)
from assistant_human.hosted_assistants import (
    _assistant_secret_inventory as _assistant_secret_inventory,
)
from assistant_human.hosted_assistants import (
    _chat_assistant_ids as _chat_assistant_ids,
)
from assistant_human.hosted_assistants import (
    _chat_file_metadata as _chat_file_metadata,
)
from assistant_human.hosted_assistants import (
    _close_exec_stream as _close_exec_stream,
)
from assistant_human.hosted_assistants import (
    _contains_secret as _contains_secret,
)
from assistant_human.hosted_assistants import (
    _fail_stop_power as _fail_stop_power,
)
from assistant_human.hosted_assistants import (
    _hosted_power_identity as _hosted_power_identity,
)
from assistant_human.hosted_assistants import (
    _hosted_secret_spec as _hosted_secret_spec,
)
from assistant_human.hosted_assistants import (
    _HostedAssistantSecretBinding as _HostedAssistantSecretBinding,
)
from assistant_human.hosted_assistants import (
    _HostedAssistantSecretSpec as _HostedAssistantSecretSpec,
)
from assistant_human.hosted_assistants import (
    _HostedPowerSecretSpec as _HostedPowerSecretSpec,
)
from assistant_human.hosted_assistants import (
    _installed_assistant as _installed_assistant,
)
from assistant_human.hosted_assistants import (
    _installed_assistant_secret_specs as _installed_assistant_secret_specs,
)
from assistant_human.hosted_assistants import (
    _invoke_assistant_power as _invoke_assistant_power,
)
from assistant_human.hosted_assistants import (
    _model_credential as _model_credential,
)
from assistant_human.hosted_assistants import (
    _pending_chat_secrets as _pending_chat_secrets,
)
from assistant_human.hosted_assistants import (
    _PendingHostedChat as _PendingHostedChat,
)
from assistant_human.hosted_assistants import (
    _power_account_generations as _power_account_generations,
)
from assistant_human.hosted_assistants import (
    _power_operation as _power_operation,
)
from assistant_human.hosted_assistants import (
    _power_secret_generations as _power_secret_generations,
)
from assistant_human.hosted_assistants import (
    _raise_assistant_secret_error as _raise_assistant_secret_error,
)
from assistant_human.hosted_assistants import (
    _raise_if_rpc_cancelled as _raise_if_rpc_cancelled,
)
from assistant_human.hosted_assistants import (
    _read_rpc_frames as _read_rpc_frames,
)
from assistant_human.hosted_assistants import (
    _refresh_oauth_account as _refresh_oauth_account,
)
from assistant_human.hosted_assistants import (
    _register_active_power as _register_active_power,
)
from assistant_human.hosted_assistants import (
    _register_optional_power as _register_optional_power,
)
from assistant_human.hosted_assistants import (
    _release_active_power as _release_active_power,
)
from assistant_human.hosted_assistants import (
    _release_optional_power as _release_optional_power,
)
from assistant_human.hosted_assistants import (
    _replace_assistant_secrets as _replace_assistant_secrets,
)
from assistant_human.hosted_assistants import (
    _require_assistant_allowed_hosts as _require_assistant_allowed_hosts,
)
from assistant_human.hosted_assistants import (
    _require_assistant_genesis as _require_assistant_genesis,
)
from assistant_human.hosted_assistants import (
    _require_hosted_power_rpc_envelope as _require_hosted_power_rpc_envelope,
)
from assistant_human.hosted_assistants import (
    _require_model_credential_current as _require_model_credential_current,
)
from assistant_human.hosted_assistants import (
    _resolve_power_accounts as _resolve_power_accounts,
)
from assistant_human.hosted_assistants import (
    _resolve_power_secrets as _resolve_power_secrets,
)
from assistant_human.hosted_assistants import (
    _revoke_assistant_approval_grants as _revoke_assistant_approval_grants,
)
from assistant_human.hosted_assistants import (
    _revoke_team_approval_grants as _revoke_team_approval_grants,
)
from assistant_human.hosted_assistants import (
    _secret_bindings as _secret_bindings,
)
from assistant_human.hosted_assistants import (
    _select_team_assistants as _select_team_assistants,
)
from assistant_human.hosted_assistants import (
    _validate_assistant_power_input as _validate_assistant_power_input,
)
from container_policy.hosted_apps import (
    _activate_admitted_egress as _activate_admitted_egress,
)
from container_policy.hosted_apps import (
    _app_egress_token as _app_egress_token,
)
from container_policy.hosted_apps import (
    _app_ready_now as _app_ready_now,
)
from container_policy.hosted_apps import (
    _egress_proxy_environment as _egress_proxy_environment,
)
from container_policy.hosted_apps import (
    _egress_store as _egress_store,
)
from container_policy.hosted_apps import (
    _install_app as _install_app,
)
from container_policy.hosted_apps import (
    _list_apps as _list_apps,
)
from container_policy.hosted_apps import (
    _probe_app_health as _probe_app_health,
)
from container_policy.hosted_apps import (
    _raise_egress_error as _raise_egress_error,
)
from container_policy.hosted_apps import (
    _remove_egress_policy as _remove_egress_policy,
)
from container_policy.hosted_apps import (
    _reserve_egress_environment as _reserve_egress_environment,
)
from container_policy.hosted_apps import (
    _retain_admitted_assistant_accounts as _retain_admitted_assistant_accounts,
)
from container_policy.hosted_apps import (
    _retain_admitted_assistant_private_state as _retain_admitted_assistant_private_state,
)
from container_policy.hosted_apps import (
    _retain_admitted_assistant_secrets as _retain_admitted_assistant_secrets,
)
from container_policy.hosted_apps import (
    _team_app_containers as _team_app_containers,
)
from container_policy.hosted_apps import (
    _teardown_app as _teardown_app,
)
from container_policy.hosted_apps import (
    _uninstall_app as _uninstall_app,
)
from container_policy.hosted_apps import (
    _validate_admitted_egress as _validate_admitted_egress,
)
from container_policy.hosted_apps import (
    _validate_assistant_proxy_environment as _validate_assistant_proxy_environment,
)
from container_policy.hosted_apps import (
    _validate_egress_policy as _validate_egress_policy,
)
from container_policy.hosted_apps import (
    _wait_app_healthy as _wait_app_healthy,
)
from container_policy.hosted_apps import (
    _write_egress_policy as _write_egress_policy,
)
from container_policy.hosted_resources import (
    _admitted_resource_containers as _admitted_resource_containers,
)
from container_policy.hosted_resources import (
    _already_connected as _already_connected,
)
from container_policy.hosted_resources import (
    _AuthorizationLease as _AuthorizationLease,
)
from container_policy.hosted_resources import (
    _authorize as _authorize,
)
from container_policy.hosted_resources import (
    _authorize_container as _authorize_container,
)
from container_policy.hosted_resources import (
    _authorize_destroy as _authorize_destroy,
)
from container_policy.hosted_resources import (
    _capacity_key as _capacity_key,
)
from container_policy.hosted_resources import (
    _CapacityReservation as _CapacityReservation,
)
from container_policy.hosted_resources import (
    _cleanup_record as _cleanup_record,
)
from container_policy.hosted_resources import (
    _CleanupResult as _CleanupResult,
)
from container_policy.hosted_resources import (
    _describe as _describe,
)
from container_policy.hosted_resources import (
    _ensure_team_network as _ensure_team_network,
)
from container_policy.hosted_resources import (
    _ensure_team_network_kind as _ensure_team_network_kind,
)
from container_policy.hosted_resources import (
    _fail_stop_team as _fail_stop_team,
)
from container_policy.hosted_resources import (
    _get_container as _get_container,
)
from container_policy.hosted_resources import (
    _memory_usage as _memory_usage,
)
from container_policy.hosted_resources import (
    _MemoryUsage as _MemoryUsage,
)
from container_policy.hosted_resources import (
    _network_container_metadata as _network_container_metadata,
)
from container_policy.hosted_resources import (
    _physical_teams as _physical_teams,
)
from container_policy.hosted_resources import (
    _prepare_marketplace_image as _prepare_marketplace_image,
)
from container_policy.hosted_resources import (
    _remove_team_container as _remove_team_container,
)
from container_policy.hosted_resources import (
    _require_cleanup_authorization as _require_cleanup_authorization,
)
from container_policy.hosted_resources import (
    _require_current_authorization as _require_current_authorization,
)
from container_policy.hosted_resources import (
    _require_network_policy as _require_network_policy,
)
from container_policy.hosted_resources import (
    _require_running_team_isolation as _require_running_team_isolation,
)
from container_policy.hosted_resources import (
    _require_team_isolation as _require_team_isolation,
)
from container_policy.hosted_resources import (
    _require_team_isolation_mode as _require_team_isolation_mode,
)
from container_policy.hosted_resources import (
    _require_team_runtime as _require_team_runtime,
)
from container_policy.hosted_resources import (
    _reserve_capacity as _reserve_capacity,
)
from container_policy.hosted_resources import (
    _safe_connect as _safe_connect,
)
from container_policy.hosted_resources import (
    _start_team_with_isolation as _start_team_with_isolation,
)
from container_policy.hosted_resources import (
    _team_not_running as _team_not_running,
)
from container_policy.hosted_resources import (
    _team_runtime as _team_runtime,
)
from container_policy.hosted_resources import (
    _teardown_team_network_kind as _teardown_team_network_kind,
)
from container_policy.hosted_resources import (
    _teardown_team_networks as _teardown_team_networks,
)
from container_policy.hosted_resources import (
    _trusted_image_id as _trusted_image_id,
)
from container_policy.hosted_resources import (
    _trusted_workload_image as _trusted_workload_image,
)
from container_policy.hosted_resources import (
    _wire_network_deps as _wire_network_deps,
)


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


def _hosted_chat_setup(
    team_id: str,
    file_ids: object,
    assistant_ids: tuple[str, ...],
    container,
    owner: str,
) -> tuple[
    str,
    tuple[_ActiveAssistant, ...],
    list[dict[str, object]],
    inference_config.InferenceConfig,
    str,
    int,
    tuple[object, ...],
]:
    team_name = _team_name_from_anchor(container)
    assistants = _select_team_assistants(_active_team_assistants(team_id), assistant_ids)
    files = _chat_file_metadata(team_id, file_ids)
    try:
        config = _inference_store.load(team_id)
    except inference_config.InferenceConfigError as exc:
        raise ApiError(HTTPStatus.CONFLICT, "configure this Team's model provider before chatting") from exc
    api_key, generation = _model_credential(owner, config.provider)
    _require_model_credential_current(owner, config.provider, generation)
    identity = (
        container.id,
        owner,
        team_name,
        tuple((active.assistant_id, active.container.id) for active in assistants),
        files,
        config,
        generation,
    )
    return team_name, assistants, files, config, api_key, generation, identity


def _raise_hosted_chat_problem(reason: str, exc: BaseException | None) -> NoReturn:
    if reason == "invalid-continuation" or reason == "invalid-suspension":
        raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, f"invalid chat {reason.removeprefix('invalid-')}")
    if reason == "context-changed":
        raise ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry")
    if isinstance(exc, power_journal.PowerJournalError):
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Team Power execution state is unavailable") from exc
    if isinstance(exc, chat_orchestrator.ChatStoppedError):
        raise ApiError(HTTPStatus.CONFLICT, "brain turn stopped") from exc
    if isinstance(exc, chat_orchestrator.ChatOrchestrationError):
        raise ApiError(HTTPStatus.BAD_GATEWAY, "Brain could not complete the Assistant turn") from exc
    if isinstance(exc, brain_runtime_client.BrainRuntimeError):
        raise ApiError(HTTPStatus.BAD_GATEWAY, "Brain runtime is unavailable") from exc
    raise AssertionError(f"unknown hosted chat failure: {reason}")


def _hosted_private_requirements(
    team_id: str,
    bindings: dict[str, _ActiveAssistant],
    requests: tuple[brain_runtime_client.PowerRequest, ...],
) -> tuple[
    tuple[assistant_account_challenges.AccountRequirement, ...],
    tuple[assistant_secret_challenges.SecretRequirement, ...],
]:
    try:
        accounts = assistant_account_flow.requirements_for_batch(
            team_id,
            _secret_bindings(bindings),
            requests,
            _assistant_accounts,
        )
    except (
        assistant_account_flow.AccountFlowError,
        oauth_account_store.OAuthAccountStoreError,
    ) as exc:
        raise ApiError(HTTPStatus.CONFLICT, "Assistant account contract is unavailable") from exc
    if accounts:
        return accounts, ()
    try:
        secrets_required = assistant_secret_flow.requirements_for_batch(
            team_id,
            _secret_bindings(bindings),
            requests,
            _assistant_secrets,
        )
    except assistant_secret_store.AssistantSecretError as exc:
        _raise_assistant_secret_error(exc)
    except assistant_secret_flow.SecretFlowError as exc:
        raise ApiError(HTTPStatus.CONFLICT, "Assistant secret contract is unavailable") from exc
    return (), secrets_required


def _hosted_approval_requirement(
    team_id: str,
    interactions: tuple[chat_orchestrator.HumanInteraction, ...],
    answers_by_interrupt: dict[str, tuple[object, ...]],
    bindings: dict[str, _ActiveAssistant],
) -> tuple[assistant_approval_challenges.ApprovalRequirement | None, bool]:
    if not interactions:
        return None, False
    if len(interactions) != 1:
        raise ApiError(HTTPStatus.BAD_GATEWAY, "Assistant human approval request is invalid")
    interaction = interactions[0]
    answers = answers_by_interrupt.get(interaction.request.interrupt_id, ())
    active = bindings.get(interaction.request.assistant_id)
    if active is None:
        raise ApiError(HTTPStatus.CONFLICT, "Brain requested an unavailable Assistant")
    try:
        requirement = assistant_approval_flow.requirement(
            interaction,
            active.assistant_id.replace("-", " ").title(),
            _hosted_power_identity(active)[1],
            len(answers),
        )
        granted = requirement.runs == "once" and _assistant_approval_grants.is_granted(
            team_id,
            requirement.assistant_id,
            requirement.power_id,
            requirement.assistant_image,
            requirement.ordinal,
        )
    except assistant_approval_flow.ApprovalFlowError as exc:
        raise ApiError(HTTPStatus.BAD_GATEWAY, "Assistant human approval request is invalid") from exc
    except assistant_approval_grants.ApprovalGrantError as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Assistant approval state is unavailable") from exc
    return requirement, granted


def _hosted_answer_log(
    answer_logs: tuple[tuple[str, tuple[object, ...]], ...],
) -> dict[str, tuple[object, ...]]:
    answers_by_interrupt = dict(answer_logs)
    if len(answers_by_interrupt) != len(answer_logs):
        raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "invalid chat answer log")
    return answers_by_interrupt


def _run_hosted_chat_segment(
    team_id: str,
    file_ids: object,
    assistant_ids: tuple[str, ...],
    token: str,
    container,
    owner: str,
    *,
    message: str | None = None,
    continuation: chat_orchestrator.ChatContinuation | None = None,
    expected_identity: tuple[object, ...] | None = None,
    answer_logs: tuple[tuple[str, tuple[object, ...]], ...] = (),
) -> chat_turn_engine.SegmentResult:
    answers_by_interrupt = _hosted_answer_log(answer_logs)
    bindings: dict[str, _ActiveAssistant] = {}
    initial_identity: tuple[object, ...] = ()
    config: inference_config.InferenceConfig | None = None
    generation = 0

    def require_current_credential() -> None:
        if config is None:
            raise AssertionError("hosted chat segment was not prepared")
        _require_model_credential_current(owner, config.provider, generation)

    def validate_power(assistant_id: str, power: str, power_input) -> object:
        return _validate_assistant_power_input(bindings, assistant_id, power, power_input)

    def execute_power(request: brain_runtime_client.PowerRequest) -> object:
        require_current_credential()
        active = bindings.get(request.assistant_id)
        if active is None:
            raise ApiError(HTTPStatus.CONFLICT, "Brain requested an unavailable Assistant")
        invocation = _invoke_assistant_power(
            PowerInvocationRequest(
                team_id=team_id,
                token=token,
                assistant_id=request.assistant_id,
                contract=active.contract,
                container=active.container,
                power=request.power,
                payload=request.input,
                answers=answers_by_interrupt.get(request.interrupt_id, ()),
            )
        )
        if "suspend" in invocation:
            return power_execution.RpcSuspension(invocation["suspend"])
        return invocation["result"]

    def prepare() -> chat_turn_engine.PreparedSegment:
        nonlocal bindings, config, generation, initial_identity
        team_name, assistants, files, config, api_key, generation, initial_identity = _hosted_chat_setup(
            team_id,
            file_ids,
            assistant_ids,
            container,
            owner,
        )
        genesis_by_id = {active.assistant_id: _require_assistant_genesis(active.container) for active in assistants}
        context = brain_runtime_client.RuntimeContext(
            thread_id=_brain_thread_id(team_id, container.id),
            team_name=team_name,
            assistants=tuple(
                brain_runtime_client.RuntimeAssistant(
                    id=active.assistant_id,
                    genesis=genesis_by_id[active.assistant_id],
                    powers=tuple(
                        brain_runtime_client.RuntimePower(
                            id=power_id,
                            summary=power.summary,
                            input_schema=power.input_schema,
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
        bindings = {active.assistant_id: active for active in assistants}
        batch = power_execution.PowerBatch(
            _power_execution_journal,
            container.id,
            context.thread_id,
            bindings,
            power_execution.PowerBatchStrategy(
                _hosted_power_identity,
                execute_power,
                lambda request: _require_hosted_power_rpc_envelope(
                    team_id,
                    bindings,
                    request,
                    answers_by_interrupt.get(request.interrupt_id, ()),
                ),
                lambda request: _power_secret_generations(team_id, bindings[request.assistant_id], request.power),
                lambda request: _power_account_generations(team_id, bindings[request.assistant_id], request.power),
            ),
        )
        return chat_turn_engine.PreparedSegment(team_name, initial_identity, context, files, batch)

    def pause_for_private_inputs(
        requests: tuple[object, ...],
        requirements: chat_turn_engine.SegmentRequirements,
    ) -> bool:
        requirements.accounts, requirements.secrets = _hosted_private_requirements(
            team_id,
            bindings,
            requests,
        )
        return bool(requirements.accounts or requirements.secrets)

    def validate_context() -> None:
        current_anchor = _current_team_anchor(team_id, container.id, owner)
        *_unused, current_identity = _hosted_chat_setup(
            team_id,
            file_ids,
            assistant_ids,
            current_anchor,
            owner,
        )
        if current_identity != initial_identity:
            raise ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry")

    current_message = message
    current_continuation = continuation
    current_identity = expected_identity
    while True:
        team_name, identity, outcome, requirements = chat_turn_engine.run_segment(
            chat_turn_engine.SegmentStrategy(
                runtime=_brain_runtime,
                prepare=prepare,
                validate_power=validate_power,
                pause_for_private_inputs=pause_for_private_inputs,
                cancelled=lambda: _token_cancelled(token),
                validate_context=validate_context,
                raise_problem=_raise_hosted_chat_problem,
                finalize=require_current_credential,
            ),
            message=current_message,
            continuation=current_continuation,
            expected_identity=current_identity,
        )
        approval_requirements: tuple[assistant_approval_challenges.ApprovalRequirement, ...] = ()
        requirement, granted = _hosted_approval_requirement(
            team_id,
            requirements.approvals,
            answers_by_interrupt,
            bindings,
        )
        if requirement is not None and granted:
            answers = answers_by_interrupt.get(requirement.interrupt_id, ())
            answers_by_interrupt[requirement.interrupt_id] = (*answers, True)
            if not isinstance(outcome, chat_orchestrator.ChatSuspension):
                raise AssertionError("approval requirement did not suspend")
            current_message = None
            current_continuation = outcome.continuation
            current_identity = identity
            continue
        if requirement is not None:
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


def _pause_hosted_chat(
    team_id: str,
    token: str,
    outcome: chat_orchestrator.ChatSuspension,
    requirements: tuple[assistant_secret_challenges.SecretRequirement, ...],
    pending: _PendingHostedChat,
) -> dict[str, object]:
    try:
        challenge = _assistant_secret_challenges.create(team_id, requirements, pending)
    except assistant_secret_challenges.SecretChallengeError as exc:
        raise ApiError(HTTPStatus.CONFLICT, "Assistant secret request is already pending") from exc
    _commit_hosted_suspension(team_id, token, outcome, pending, _assistant_secret_challenges)
    return assistant_secret_flow.challenge_payload(challenge)


def _commit_hosted_suspension(
    team_id: str,
    token: str,
    outcome: chat_orchestrator.ChatSuspension,
    pending: _PendingHostedChat,
    challenge_store: object,
) -> None:
    chat_turn_engine.commit_suspension(
        outcome.continuation,
        pending.continuation,
        lambda: _commit_chat_terminal(team_id, token),
        lambda: challenge_store.cancel_team(team_id),
        lambda: ApiError(HTTPStatus.CONFLICT, "brain turn stopped"),
    )


def _hosted_account_challenge_payload(
    challenge: assistant_account_challenges.PendingAccountChallenge,
) -> dict[str, object]:
    bindings: dict[str, _HostedAssistantSecretBinding] = {}
    try:
        for requirement in challenge.requirements:
            assistant_id, contract, container = _installed_assistant(
                challenge.team_id,
                requirement.assistant_id,
            )
            active = _ActiveAssistant(assistant_id, contract, container)
            bindings[assistant_id] = _HostedAssistantSecretBinding(_hosted_secret_spec(active))
        return assistant_account_flow.challenge_payload(challenge, bindings)
    except (marketplace.MarketplaceError, assistant_account_flow.AccountFlowError) as exc:
        raise ApiError(HTTPStatus.CONFLICT, "Assistant account contract changed; retry the message") from exc


def _pause_hosted_connection(
    team_id: str,
    token: str,
    outcome: chat_orchestrator.ChatSuspension,
    requirements: tuple[assistant_account_challenges.AccountRequirement, ...],
    pending: _PendingHostedChat,
) -> dict[str, object]:
    try:
        challenge = _assistant_account_challenges.create(team_id, requirements, pending)
    except assistant_account_challenges.AccountChallengeError as exc:
        raise ApiError(HTTPStatus.CONFLICT, "Assistant account request is already pending") from exc
    _commit_hosted_suspension(team_id, token, outcome, pending, _assistant_account_challenges)
    return _hosted_account_challenge_payload(challenge)


def _pause_hosted_input(
    team_id: str,
    token: str,
    outcome: chat_orchestrator.ChatSuspension,
    requirements: tuple[chat_orchestrator.HumanInteraction, ...],
    pending: _PendingHostedChat,
) -> dict[str, object]:
    if len(requirements) != 1:
        raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "invalid human input suspension")
    interaction = requirements[0]
    answers = dict(pending.answer_logs).get(interaction.request.interrupt_id, ())
    try:
        assistant_id, contract, container = _installed_assistant(
            team_id,
            interaction.request.assistant_id,
        )
        requirement = assistant_input_flow.requirement(
            interaction,
            _hosted_power_identity(_ActiveAssistant(assistant_id, contract, container))[1],
            len(answers),
        )
        challenge = _assistant_input_challenges.create(team_id, requirement, pending)
    except (
        marketplace.MarketplaceError,
        assistant_input_challenges.InputChallengeError,
        assistant_input_flow.InputFlowError,
    ) as exc:
        raise ApiError(HTTPStatus.BAD_GATEWAY, "Assistant human input request is invalid") from exc
    _commit_hosted_suspension(team_id, token, outcome, pending, _assistant_input_challenges)
    return assistant_input_flow.challenge_payload(challenge)


def _pause_hosted_approval(
    team_id: str,
    token: str,
    outcome: chat_orchestrator.ChatSuspension,
    requirements: tuple[assistant_approval_challenges.ApprovalRequirement, ...],
    pending: _PendingHostedChat,
) -> dict[str, object]:
    try:
        challenge = _assistant_approval_challenges.create(team_id, requirements, pending)
    except assistant_approval_challenges.ApprovalChallengeError as exc:
        raise ApiError(HTTPStatus.CONFLICT, "Assistant approval is already pending") from exc
    _commit_hosted_suspension(team_id, token, outcome, pending, _assistant_approval_challenges)
    return assistant_approval_flow.challenge_payload(challenge)


def _hosted_segment_response(
    team_id: str,
    token: str,
    segment: chat_turn_engine.SegmentResult,
    assistant_ids: tuple[str, ...],
    file_ids: tuple[str, ...],
    owner: str,
) -> dict[str, object]:
    def pending(suspension: chat_orchestrator.ChatSuspension) -> _PendingHostedChat:
        return _PendingHostedChat(
            continuation=suspension.continuation,
            assistant_ids=assistant_ids,
            file_ids=file_ids,
            owner=owner,
            identity=segment.identity,
            answer_logs=segment.answer_logs,
        )

    def complete(terminal: chat_orchestrator.ChatOutcome) -> dict[str, object]:
        if not _commit_chat_terminal(team_id, token):
            raise ApiError(HTTPStatus.CONFLICT, "brain turn stopped")
        return {
            "team_id": team_id,
            "team_name": segment.team_name,
            "reply": terminal.reply[:CHAT_OUTPUT_CAP],
        }

    try:
        return chat_turn_engine.dispatch(
            segment.outcome,
            segment.requirement_groups(),
            pending,
            (
                lambda suspension, requirements, state: _pause_hosted_connection(
                    team_id, token, suspension, requirements, state
                ),
                lambda suspension, requirements, state: _pause_hosted_chat(
                    team_id, token, suspension, requirements, state
                ),
                lambda suspension, requirements, state: _pause_hosted_input(
                    team_id, token, suspension, requirements, state
                ),
                lambda suspension, requirements, state: _pause_hosted_approval(
                    team_id, token, suspension, requirements, state
                ),
            ),
            complete,
        )
    except ValueError as exc:
        raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc)) from exc


def _chat_in_turn(
    team_id: str,
    message: str,
    file_ids: object,
    assistant_ids: tuple[str, ...],
    token: str,
    container,
    owner: str,
) -> dict[str, object]:
    segment = _run_hosted_chat_segment(
        team_id,
        file_ids,
        assistant_ids,
        token,
        container,
        owner,
        message=message,
    )
    return _hosted_segment_response(
        team_id,
        token,
        segment,
        assistant_ids,
        tuple(file_ids) if isinstance(file_ids, list) else (),
        owner,
    )


def _chat(
    team_id: str,
    message: str,
    file_ids: object,
    assistant_ids: tuple[str, ...],
    lease: _AuthorizationLease,
) -> dict:
    """Run one bounded Team turn across the explicit Controller-brokered Assistant scope."""
    pending = _pending_hosted_chat(team_id)
    if pending is not None:
        return pending
    # The slot comes first. A losing concurrent request must not run even the local credential probe,
    # much less provider status or a second provider CLI.
    with _exclusive_chat_turn(team_id, lease) as (token, container):
        pending = _pending_hosted_chat(team_id)
        if pending is not None:
            return pending
        return _chat_in_turn(team_id, message, file_ids, assistant_ids, token, container, lease.owner)


def _pending_hosted_chat(team_id: str) -> dict[str, object] | None:
    account = _assistant_account_challenges.current(team_id)
    secret = _assistant_secret_challenges.current(team_id)
    input_challenge = _assistant_input_challenges.current(team_id)
    approval = _assistant_approval_challenges.current(team_id)
    if sum(item is not None for item in (account, secret, input_challenge, approval)) > 1:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Team chat continuation state is unavailable")
    if account is not None:
        return _hosted_account_challenge_payload(account)
    if secret is not None:
        return assistant_secret_flow.challenge_payload(secret)
    if input_challenge is not None:
        return assistant_input_flow.challenge_payload(input_challenge)
    if approval is not None:
        return assistant_approval_flow.challenge_payload(approval)
    return None


def _current_account_declaration(team_id: str, assistant_id: str, account_id: str) -> object:
    try:
        installed_id, contract, _container = _installed_assistant(team_id, assistant_id)
        declaration = contract.accounts.get(account_id)
        if installed_id != assistant_id or declaration is None:
            raise ApiError(HTTPStatus.CONFLICT, "Assistant account declaration changed")
    except ApiError, marketplace.MarketplaceError:
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
    lease: _AuthorizationLease,
) -> dict[str, object]:
    _require_current_authorization(team_id, lease, require_isolation=False)
    try:
        challenge = _assistant_account_challenges.get(team_id, challenge_id)
    except assistant_account_challenges.AccountChallengeNotFoundError as exc:
        raise ApiError(HTTPStatus.CONFLICT, "Assistant account request expired; retry the message") from exc
    pending = challenge.payload
    if not isinstance(pending, _PendingHostedChat) or pending.owner != lease.owner:
        raise ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry")
    try:
        authorization_url = _oauth_accounts.authorization_url(challenge, session_binding)
    except oauth_account_service.OAuthAccountUnavailableError as exc:
        raise ApiError(HTTPStatus.CONFLICT, "Assistant accounts are already configured") from exc
    except oauth_account_service.OAuthAccountServiceError as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Assistant account could not be started") from exc
    return {"authorization_url": authorization_url}


def _complete_oauth_account(
    body: object,
    principal: tuple[str, str | None],
) -> dict[str, object]:
    if not isinstance(body, dict) or set(body) != {"state", "code", "session_binding"}:
        raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "OAuth callback is invalid")
    try:
        completion = _oauth_accounts.complete(
            body["state"],
            body["code"],
            body["session_binding"],
            _current_account_declaration,
        )
    except oauth_account_service.OAuthAccountServiceError as exc:
        raise ApiError(HTTPStatus.BAD_GATEWAY, "Assistant account could not be completed") from exc
    try:
        _authorize(completion.team_id, principal)
    except Exception:
        with contextlib.suppress(oauth_account_service.OAuthAccountServiceError):
            _oauth_accounts.disconnect(
                completion.team_id,
                completion.assistant_id,
                completion.account_id,
            )
        raise
    pending = _assistant_account_challenges.current(completion.team_id)
    return {
        "connected": True,
        "team_id": completion.team_id,
        "assistant_id": completion.assistant_id,
        "account_id": completion.account_id,
        "provider": completion.provider,
        "scopes": list(completion.scopes),
        "challenge_id": pending.id if pending is not None else None,
    }


@_serialize_against_team_chat
def _disconnect_oauth_account(
    team_id: str,
    assistant_id: str,
    account_id: str,
    lease: _AuthorizationLease,
) -> dict[str, object]:
    with _lock_for(team_id):
        _require_current_authorization(team_id, lease, require_isolation=False)
        _current_account_declaration(team_id, assistant_id, account_id)
        _assistant_account_challenges.cancel_team(team_id)
        try:
            disconnected = _oauth_accounts.disconnect(team_id, assistant_id, account_id)
        except oauth_account_service.OAuthAccountServiceError as exc:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Assistant account could not be disconnected") from exc
    return {"disconnected": disconnected}


def _resume_chat_accounts(
    team_id: str,
    challenge_id: object,
    lease: _AuthorizationLease,
) -> dict[str, object]:
    with _exclusive_chat_turn(team_id, lease) as (token, container):

        def inspect(pending: object) -> chat_turn_engine.AccountResumeContext:
            if not isinstance(pending, _PendingHostedChat):
                raise AssertionError("invalid hosted account continuation")
            _, assistants, _files, _config, _key, _generation, current_identity = _hosted_chat_setup(
                team_id,
                list(pending.file_ids),
                pending.assistant_ids,
                container,
                lease.owner,
            )
            bindings = {active.assistant_id: active for active in assistants}
            return chat_turn_engine.AccountResumeContext(
                current_identity,
                _secret_bindings(bindings),
                pending.continuation.turn.powers,
            )

        admission = chat_turn_engine.admit_account_resume(
            chat_turn_engine.AccountResumeStrategy(
                store=_assistant_account_challenges,
                team_id=team_id,
                challenge_id=challenge_id,
                pending_valid=lambda pending: isinstance(pending, _PendingHostedChat) and pending.owner == lease.owner,
                pending_identity=lambda pending: pending.identity,
                inspect=inspect,
                account_store=_assistant_accounts,
                challenge_response=_hosted_account_challenge_payload,
                expired_error=lambda: ApiError(
                    HTTPStatus.CONFLICT,
                    "Assistant account request expired; retry the message",
                ),
                context_error=lambda: ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry"),
                contract_error=lambda: ApiError(HTTPStatus.CONFLICT, "Assistant account contract is unavailable"),
            )
        )
        if admission.response is not None:
            return admission.response
        pending = admission.pending
        if not isinstance(pending, _PendingHostedChat):
            raise AssertionError("shared account resume returned invalid state")

        segment = _run_hosted_chat_segment(
            team_id,
            list(pending.file_ids),
            pending.assistant_ids,
            token,
            container,
            lease.owner,
            continuation=pending.continuation,
            expected_identity=pending.identity,
            answer_logs=pending.answer_logs,
        )
        return _hosted_segment_response(
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
    lease: _AuthorizationLease,
) -> dict[str, object]:
    try:
        challenge_id = body.get("challenge_id") if isinstance(body, dict) else None
        challenge = _assistant_secret_challenges.get(team_id, challenge_id)
        values = assistant_secret_flow.submission_values(challenge, body)
    except assistant_secret_challenges.SecretChallengeNotFoundError as exc:
        raise ApiError(HTTPStatus.CONFLICT, "Assistant secret request expired; retry the message") from exc
    except (assistant_secret_challenges.SecretChallengeError, assistant_secret_flow.SecretFlowError) as exc:
        raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "Assistant secret submission is invalid") from exc
    pending = challenge.payload
    if not isinstance(pending, _PendingHostedChat) or pending.owner != lease.owner:
        raise ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry")

    with _exclusive_chat_turn(team_id, lease) as (token, container):
        *_unused, current_identity = _hosted_chat_setup(
            team_id,
            list(pending.file_ids),
            pending.assistant_ids,
            container,
            lease.owner,
        )
        if current_identity != pending.identity:
            _assistant_secret_challenges.cancel_team(team_id)
            raise ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry")

        def commit_secret_transaction(current) -> None:
            if current is not challenge:
                raise assistant_secret_challenges.SecretChallengeNotFoundError("secret challenge is unavailable")
            _assistant_secrets.put_for_assistants(team_id, values)

        try:
            claimed = _assistant_secret_challenges.claim_after(
                team_id,
                challenge.id,
                commit_secret_transaction,
            )
            if claimed is not challenge:
                raise assistant_secret_challenges.SecretChallengeNotFoundError("secret challenge is unavailable")
        except assistant_secret_challenges.SecretChallengeNotFoundError as exc:
            raise ApiError(HTTPStatus.CONFLICT, "Assistant secret request expired; retry the message") from exc
        except assistant_secret_store.AssistantSecretError as exc:
            _raise_assistant_secret_error(exc)

        segment = _run_hosted_chat_segment(
            team_id,
            list(pending.file_ids),
            pending.assistant_ids,
            token,
            container,
            lease.owner,
            continuation=pending.continuation,
            expected_identity=pending.identity,
            answer_logs=pending.answer_logs,
        )
        return _hosted_segment_response(
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
    lease: _AuthorizationLease,
) -> dict[str, object]:
    challenge_id = body.get("challenge_id") if isinstance(body, dict) else None
    try:
        challenge = _assistant_input_challenges.get(team_id, challenge_id)
        answer = assistant_input_flow.submitted_answer(challenge, body)
    except assistant_input_challenges.InputChallengeNotFoundError as exc:
        raise ApiError(HTTPStatus.CONFLICT, "Assistant input request expired; retry the message") from exc
    except (assistant_input_challenges.InputChallengeError, assistant_input_flow.InputFlowError) as exc:
        raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "Assistant input submission is invalid") from exc
    pending = challenge.payload
    if not isinstance(pending, _PendingHostedChat) or pending.owner != lease.owner:
        raise ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry")

    with _exclusive_chat_turn(team_id, lease) as (token, container):
        *_unused, current_identity = _hosted_chat_setup(
            team_id,
            list(pending.file_ids),
            pending.assistant_ids,
            container,
            lease.owner,
        )
        if current_identity != pending.identity:
            _assistant_input_challenges.cancel_team(team_id)
            raise ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry")
        answer_logs = dict(pending.answer_logs)
        existing = answer_logs.get(challenge.requirement.interrupt_id, ())
        if len(existing) != challenge.requirement.ordinal:
            _assistant_input_challenges.cancel_team(team_id)
            raise ApiError(HTTPStatus.CONFLICT, "Assistant input replay changed; retry the message")
        try:
            claimed = _assistant_input_challenges.claim(team_id, challenge.id)
        except assistant_input_challenges.InputChallengeNotFoundError as exc:
            raise ApiError(HTTPStatus.CONFLICT, "Assistant input request expired; retry the message") from exc
        if claimed is not challenge:
            raise ApiError(HTTPStatus.CONFLICT, "Assistant input request expired; retry the message")
        answer_logs[challenge.requirement.interrupt_id] = (*existing, answer)
        resumed = replace(pending, answer_logs=tuple(sorted(answer_logs.items())))

        segment = _run_hosted_chat_segment(
            team_id,
            list(resumed.file_ids),
            resumed.assistant_ids,
            token,
            container,
            lease.owner,
            continuation=resumed.continuation,
            expected_identity=resumed.identity,
            answer_logs=resumed.answer_logs,
        )
        return _hosted_segment_response(
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
    lease: _AuthorizationLease,
) -> dict[str, object]:
    challenge_id = body.get("challenge_id") if isinstance(body, dict) else None
    try:
        challenge = _assistant_approval_challenges.get(team_id, challenge_id)
        answer = assistant_approval_flow.submitted_answer(challenge, body)
    except assistant_approval_challenges.ApprovalChallengeNotFoundError as exc:
        raise ApiError(HTTPStatus.CONFLICT, "Assistant approval expired; retry the message") from exc
    except (assistant_approval_challenges.ApprovalChallengeError, assistant_approval_flow.ApprovalFlowError) as exc:
        raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "Assistant approval submission is invalid") from exc
    pending = challenge.payload
    if not isinstance(pending, _PendingHostedChat) or pending.owner != lease.owner:
        raise ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry")

    with _exclusive_chat_turn(team_id, lease) as (token, container):
        *_unused, current_identity = _hosted_chat_setup(
            team_id,
            list(pending.file_ids),
            pending.assistant_ids,
            container,
            lease.owner,
        )
        if current_identity != pending.identity:
            _assistant_approval_challenges.cancel_team(team_id)
            raise ApiError(HTTPStatus.CONFLICT, "Team capabilities changed; retry")
        requirement = challenge.requirements[0]
        answer_logs = dict(pending.answer_logs)
        existing = answer_logs.get(requirement.interrupt_id, ())
        if len(existing) != requirement.ordinal:
            _assistant_approval_challenges.cancel_team(team_id)
            raise ApiError(HTTPStatus.CONFLICT, "Assistant approval replay changed; retry the message")
        try:
            claimed = _assistant_approval_challenges.claim(team_id, challenge.id)
            if claimed is not challenge:
                raise assistant_approval_challenges.ApprovalChallengeNotFoundError("approval challenge is unavailable")
            if requirement.runs == "once":
                _assistant_approval_grants.grant_many(
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
            raise ApiError(HTTPStatus.CONFLICT, "Assistant approval expired; retry the message") from exc
        except assistant_approval_grants.ApprovalGrantError as exc:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Assistant approval state is unavailable") from exc
        answer_logs[requirement.interrupt_id] = (*existing, answer)
        resumed = replace(pending, answer_logs=tuple(sorted(answer_logs.items())))

        segment = _run_hosted_chat_segment(
            team_id,
            list(resumed.file_ids),
            resumed.assistant_ids,
            token,
            container,
            lease.owner,
            continuation=resumed.continuation,
            expected_identity=resumed.identity,
            answer_logs=resumed.answer_logs,
        )
        return _hosted_segment_response(
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
    secret_cancelled = _assistant_secret_challenges.cancel_team(team_id)
    account_cancelled = _assistant_account_challenges.cancel_team(team_id)
    input_cancelled = _assistant_input_challenges.cancel_team(team_id)
    approval_cancelled = _assistant_approval_challenges.cancel_team(team_id)
    challenge_cancelled = secret_cancelled or account_cancelled or input_cancelled or approval_cancelled
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


def _teardown_assistant_secrets(team_id: str) -> bool:
    _assistant_secret_challenges.cancel_team(team_id)
    _assistant_input_challenges.cancel_team(team_id)
    _assistant_approval_challenges.cancel_team(team_id)
    try:
        _assistant_secrets.delete_team(team_id)
    except assistant_secret_store.AssistantSecretError:
        return False
    return True


def _teardown_assistant_accounts(team_id: str) -> bool:
    _assistant_account_challenges.cancel_team(team_id)
    try:
        _assistant_accounts.delete_team(team_id)
    except oauth_account_store.OAuthAccountStoreError:
        return False
    return _revoke_team_approval_grants(team_id)


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
        pgdriver_client.finalize_team_drop(team_id)
        cleanup_state.finish(record)
    except (
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
    if (
        not _teardown_apps(team_id)
        or not _teardown_storage(team_id)
        or not _teardown_inference(team_id)
        or not _teardown_assistant_secrets(team_id)
        or not _teardown_assistant_accounts(team_id)
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


@_serialize_against_team_chat
def _lifecycle(team_id: str, op: str, lease: _AuthorizationLease) -> dict:
    with _lock_for(team_id):
        # Stop is always available as remediation. Start/restart require both an exact per-container
        # runtime and a currently registered daemon runtime; Docker may never fall back to runc.
        container = _require_current_authorization(team_id, lease, require_isolation=op != "stop")
        if op in {"start", "restart"}:
            _require_team_runtime()
        container.reload()
        # The outer Team chat slot proves no turn can observe a partially changed runtime.
        if op in {"stop", "restart"} and container.status == "running":
            _fail_stop_team(container, timeout=30)
        container.reload()
        if op in {"start", "restart"} and container.status != "running":
            _start_team_with_isolation(container)
    return {"team_id": team_id, "op": op, "status": "ok"}


from http_boundary import hosted_controller as hosted_http

Handler = hosted_http.Handler
_BoundedThreadingHTTPServer = hosted_http._BoundedThreadingHTTPServer
main = hosted_http.main
hosted_http.bind_controller(sys.modules[__name__])

if __name__ == "__main__":
    main()
