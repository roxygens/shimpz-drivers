"""team-driver — a socket-holding sidecar dedicated to Team lifecycle.

Besides shimpz-driver, this is the ONLY container holding /var/run/docker.sock — and it exposes ONLY
named operations (create/list/status/logs/stop/start/restart/destroy), never a generic Docker
passthrough. A Team is one isolated `shimpz-brain`: its OWN internal network, its OWN config+workspace
volumes, and a SCOPED Postgres database (provisioned via pg-driver — this driver never holds the
superuser). Every mutating call is bearer-gated → validated → mutated → audited (trace_id returned).
A compromised caller can only ever request what validate.py permits.
"""

from __future__ import annotations

import contextlib
import functools
import ipaddress
import math
import os
import secrets
import sys
import threading
import time
import weakref
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path

import assistant_account_challenges
import assistant_account_flow as assistant_account_flow
import assistant_genesis
import assistant_manifest
import assistant_secret_challenges
import assistant_secret_flow as assistant_secret_flow
import assistant_secret_store
import audit as audit
import brain_runtime_client
import chat_orchestrator as chat_orchestrator
import chat_turn_engine as chat_turn_engine
import cleanup_state as cleanup_state
import docker
import docker.errors
import inference_config
import manifests
import marketplace as marketplace
import oauth_account_service
import oauth_account_store
import oauth_http_client
import oauth_pkce_challenges
import pgdriver_client as pgdriver_client
import power_execution as power_execution
import power_journal
import team_storage
import token_store
import validate
from assistant_human import approval_challenges as assistant_approval_challenges
from assistant_human import approval_grants as assistant_approval_grants
from assistant_human import input_challenges as assistant_input_challenges
from container_policy import network as _network_policy

network_policy = _network_policy

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
# The capacity lock protects reservation snapshots and mutations. Docker inventory and slow
# provisioning run after this lock is released; a generation counter detects snapshot churn.
_capacity_lock = threading.Lock()
_capacity_generation = 0
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
    _secret_bindings as _secret_bindings,
)
from assistant_human.hosted_assistants import (
    _select_team_assistants as _select_team_assistants,
)
from assistant_human.hosted_assistants import (
    _teardown_team_approval_grants as _teardown_team_approval_grants,
)
from assistant_human.hosted_assistants import (
    _validate_assistant_power_input as _validate_assistant_power_input,
)
from assistant_human.hosted_chat_api import (
    _chat as _chat,
)
from assistant_human.hosted_chat_api import (
    _complete_oauth_account as _complete_oauth_account,
)
from assistant_human.hosted_chat_api import (
    _current_account_declaration as _current_account_declaration,
)
from assistant_human.hosted_chat_api import (
    _disconnect_oauth_account as _disconnect_oauth_account,
)
from assistant_human.hosted_chat_api import (
    _pending_hosted_chat as _pending_hosted_chat,
)
from assistant_human.hosted_chat_api import (
    _resume_chat_accounts as _resume_chat_accounts,
)
from assistant_human.hosted_chat_api import (
    _start_oauth_account as _start_oauth_account,
)
from assistant_human.hosted_chat_api import (
    _stop_active_power as _stop_active_power,
)
from assistant_human.hosted_chat_api import (
    _stop_chat as _stop_chat,
)
from assistant_human.hosted_chat_api import (
    _submit_chat_approval as _submit_chat_approval,
)
from assistant_human.hosted_chat_api import (
    _submit_chat_input as _submit_chat_input,
)
from assistant_human.hosted_chat_api import (
    _submit_chat_secrets as _submit_chat_secrets,
)
from assistant_human.hosted_chat_segment import (
    HostedChatSegmentRequest as HostedChatSegmentRequest,
)
from assistant_human.hosted_chat_segment import (
    _chat_in_turn as _chat_in_turn,
)
from assistant_human.hosted_chat_segment import (
    _commit_hosted_suspension as _commit_hosted_suspension,
)
from assistant_human.hosted_chat_segment import (
    _current_team_anchor as _current_team_anchor,
)
from assistant_human.hosted_chat_segment import (
    _hosted_account_challenge_payload as _hosted_account_challenge_payload,
)
from assistant_human.hosted_chat_segment import (
    _hosted_answer_log as _hosted_answer_log,
)
from assistant_human.hosted_chat_segment import (
    _hosted_approval_requirement as _hosted_approval_requirement,
)
from assistant_human.hosted_chat_segment import (
    _hosted_chat_current_identity as _hosted_chat_current_identity,
)
from assistant_human.hosted_chat_segment import (
    _hosted_chat_setup as _hosted_chat_setup,
)
from assistant_human.hosted_chat_segment import (
    _hosted_private_requirements as _hosted_private_requirements,
)
from assistant_human.hosted_chat_segment import (
    _hosted_segment_response as _hosted_segment_response,
)
from assistant_human.hosted_chat_segment import (
    _pause_hosted_approval as _pause_hosted_approval,
)
from assistant_human.hosted_chat_segment import (
    _pause_hosted_chat as _pause_hosted_chat,
)
from assistant_human.hosted_chat_segment import (
    _pause_hosted_connection as _pause_hosted_connection,
)
from assistant_human.hosted_chat_segment import (
    _pause_hosted_input as _pause_hosted_input,
)
from assistant_human.hosted_chat_segment import (
    _raise_hosted_chat_problem as _raise_hosted_chat_problem,
)
from assistant_human.hosted_chat_segment import (
    _run_hosted_chat_segment as _run_hosted_chat_segment,
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
from container_policy.hosted_lifecycle import (
    _configure_inference as _configure_inference,
)
from container_policy.hosted_lifecycle import (
    _create as _create,
)
from container_policy.hosted_lifecycle import (
    _delete_team_file as _delete_team_file,
)
from container_policy.hosted_lifecycle import (
    _destroy as _destroy,
)
from container_policy.hosted_lifecycle import (
    _drop_teardown_database as _drop_teardown_database,
)
from container_policy.hosted_lifecycle import (
    _finalize_teardown as _finalize_teardown,
)
from container_policy.hosted_lifecycle import (
    _inference_status as _inference_status,
)
from container_policy.hosted_lifecycle import (
    _lifecycle as _lifecycle,
)
from container_policy.hosted_lifecycle import (
    _list as _list,
)
from container_policy.hosted_lifecycle import (
    _list_team_files as _list_team_files,
)
from container_policy.hosted_lifecycle import (
    _logs as _logs,
)
from container_policy.hosted_lifecycle import (
    _owned_teardown_brain as _owned_teardown_brain,
)
from container_policy.hosted_lifecycle import (
    _put_inbox_file as _put_inbox_file,
)
from container_policy.hosted_lifecycle import (
    _remove_teardown_brain as _remove_teardown_brain,
)
from container_policy.hosted_lifecycle import (
    _remove_volume as _remove_volume,
)
from container_policy.hosted_lifecycle import (
    _status as _status,
)
from container_policy.hosted_lifecycle import (
    _stop_teardown_brain as _stop_teardown_brain,
)
from container_policy.hosted_lifecycle import (
    _teardown as _teardown,
)
from container_policy.hosted_lifecycle import (
    _teardown_apps as _teardown_apps,
)
from container_policy.hosted_lifecycle import (
    _teardown_assistant_accounts as _teardown_assistant_accounts,
)
from container_policy.hosted_lifecycle import (
    _teardown_assistant_secrets as _teardown_assistant_secrets,
)
from container_policy.hosted_lifecycle import (
    _teardown_inference as _teardown_inference,
)
from container_policy.hosted_lifecycle import (
    _teardown_network_planes as _teardown_network_planes,
)
from container_policy.hosted_lifecycle import (
    _teardown_storage as _teardown_storage,
)
from container_policy.hosted_lifecycle import (
    _teardown_volumes as _teardown_volumes,
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
from http_boundary import hosted_controller as hosted_http

Handler = hosted_http.Handler
_BoundedThreadingHTTPServer = hosted_http._BoundedThreadingHTTPServer
main = hosted_http.main
hosted_http.bind_controller(sys.modules[__name__])

if __name__ == "__main__":
    main()
