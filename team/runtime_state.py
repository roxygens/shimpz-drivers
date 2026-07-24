"""Process-wide state and environment configuration for the hosted Team controller."""

from __future__ import annotations

import functools
import ipaddress
import math
import os
import threading
import time
import weakref
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path

import assistant_account_challenges
import assistant_genesis
import assistant_manifest
import assistant_secret_challenges
import assistant_secret_store
import brain_runtime_client
import docker
import inference_config
import manifests
import oauth_account_service
import oauth_account_store
import oauth_http_client
import oauth_pkce_challenges
import power_journal
import team_storage
import token_store
from assistant_human import approval_challenges as assistant_approval_challenges
from assistant_human import approval_grants as assistant_approval_grants
from assistant_human import input_challenges as assistant_input_challenges

ALL_INTERFACES = str(ipaddress.IPv4Address(0))


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


LISTEN_PORT = int(os.environ.get("SHIMPZ_TEAMDRIVER_PORT", "7077"))
# The host has 125 GiB and each Team has a 2 GiB hard ceiling. The default leaves roughly half the
# host for the platform, installed apps, and Docker overhead; operators may lower these quotas.
MAX_TEAMS = _positive_int_env("SHIMPZ_MAX_TEAMS", 32)
MAX_TEAMS_PER_OWNER = _positive_int_env("SHIMPZ_MAX_TEAMS_PER_OWNER", 1)
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
# One token-gated proxy serves every app, with each token confined to its own allowlist in this volume.
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
# Weak maps retain one per-Team lock while a holder or waiter references it, without leaking entries
# after terminal operations or allowing an old locked object and a new unlocked object to coexist.
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
# Docker inventory and slow provisioning run outside this lock. The generation detects snapshot churn.
_capacity_lock = threading.Lock()
_capacity_reservations: dict[str, object] = {}
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
# Hosted interaction challenges are process-local: a restart invalidates them and the client retries.
# Encrypted restart durability belongs only to the local Controller profile.
_assistant_approval_challenges = assistant_approval_challenges.ApprovalChallengeStore()
_assistant_approval_grants = assistant_approval_grants.ApprovalGrantStore(ASSISTANT_APPROVAL_GRANTS_PATH)
_assistant_input_challenges = assistant_input_challenges.InputChallengeStore()
_oauth_pkce_challenges = oauth_pkce_challenges.OAuthPKCEChallengeStore()
_oauth_http = oauth_http_client.OAuthHTTPClient()
_oauth_accounts = oauth_account_service.OAuthAccountService(
    client_id=os.environ.get("SHIMPZ_CLOUDFLARE_OAUTH_CLIENT_ID"),
    client_secret=os.environ.get("SHIMPZ_CLOUDFLARE_OAUTH_CLIENT_SECRET"),
    redirect_uri=oauth_http_client.HOSTED_REDIRECT_URI,
    challenge=_oauth_pkce_challenges,
    store=_assistant_accounts,
    http=_oauth_http,
)
_inference_store = inference_config.InferenceConfigStore()


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class _UnsupportedAssistantRpcPathError(RuntimeError):
    """The fixed Assistant RPC adapter rejected a path it does not implement."""


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


def _lock_for(team_id: str) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(team_id)
        if lock is None:
            lock = threading.Lock()
            _locks[team_id] = lock
        return lock


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
