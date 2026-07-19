"""Turn a validated team id into docker-py kwargs for an isolated Team anchor.

The ONE place that decides what a Team container actually gets. Every security-relevant field
(security_opt, network, mounts, limits, host/browser access OFF) is a hardcoded constant here; the caller
never carries any of them, so there is nothing to override. A Team is a `shimpz-brain` with:
its OWN internal network and resource envelope, but no provider runtime, credential, Docker socket,
filesystem, browser, or application authority. Inference runs in the separate LangGraph service.
"""

from __future__ import annotations

import hashlib
import io
import ipaddress
import os
import re
import tarfile
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath

import docker
import docker.types
import network_policy
from marketplace import AppSpec

# Multi-instance (R137): SHIMPZ_SUFFIX names this Space's resources; empty (the default) is prod.
SUFFIX = os.environ.get("SHIMPZ_SUFFIX", "")
IMAGE = os.environ.get(
    "SHIMPZ_TEAM_IMAGE",
    "registry.k8s.io/pause:3.10.1@sha256:278fb9dbcca9518083ad1e11276933a2e96f23de604a3a08cc3c80002767d24c",
)
# Hostile-tenant Teams are unconditionally locked to gVisor. This is deliberately not an
# environment setting: Docker rejects create when runsc is unavailable, and the driver refuses
# lifecycle mutations until the daemon registry preserves its exact handler path, built-in security
# defaults, and every existing workload proves this exact runtime.
RUNTIME = "runsc"
RUNTIME_PATH = network_policy.TEAM_RUNTIME_PATH
CONTAINER_ALL_INTERFACES = str(ipaddress.IPv4Address(0))
CONTAINER_TMP = str(PurePosixPath("/") / "tmp")

# The lifecycle identity is intentionally not a model provider. Keeping this small trusted registry
# lets existing isolation code resolve the exact immutable image while provider/model live elsewhere.
BRAINS: dict[str, dict[str, str]] = {
    "runtime": {
        "image": IMAGE,
        "title": "Team runtime",
        "default_model": "",
    },
}
DEFAULT_BRAIN = "runtime"


def build_inbox_tar(filename: str, data: bytes) -> bytes:
    """A single-file tar for put_archive into the team's workspace inbox.

    Owned by the runtime user (uid/gid 1000 = abc) so the brain can read AND clean it up.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name=filename)
        info.size = len(data)
        info.mode = 0o644
        info.uid = info.gid = 1000
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


# Shared-service identities are suffix-aware. Postgres and installed Apps live on the Team core
# network; inference egress belongs to the separate shared LangGraph runtime.
POSTGRES_CONTAINER = network_policy.POSTGRES_CONTAINER

TEAM_PREFIX = network_policy.TEAM_PREFIX
NET_PREFIX = network_policy.CORE_NETWORK_PREFIX

# Per-team envelope. The hard cap is charged in full against team-driver's global/owner
# admission budget before Docker provisioning begins; the lower cgroup reservation is only runtime
# reclaim protection, never the capacity-accounting unit. cgroup v2: mem_reservation ≈ memory.low,
# mem_limit ≈ memory.max.
MEM_LIMIT = os.environ.get("SHIMPZ_TEAM_MEM_LIMIT", "64m")
MEM_RESERVATION = os.environ.get("SHIMPZ_TEAM_MEM_RESERVATION", "16m")
NANO_CPUS = int(os.environ.get("SHIMPZ_TEAM_NANO_CPUS", str(100_000_000)))
# runsc needs roughly 100 host tasks to establish even the inert pause sandbox; 32/64/96 fail
# before the Team process starts. 128 is the measured minimum and remains a tight hard ceiling.
PIDS_LIMIT = int(os.environ.get("SHIMPZ_TEAM_PIDS_LIMIT", "128"))


def hard_memory_bytes(value: str | int | float, *, setting: str) -> int:
    """Parse one Docker hard-memory setting once and reject an absent/unbounded value."""
    if isinstance(value, bool):
        raise ValueError(f"{setting} must be a valid positive Docker memory size")
    match = re.fullmatch(
        r"(?P<number>[0-9]+(?:\.[0-9]+)?)(?P<unit>[kmgtp]?)(?:i?b)?",
        str(value).strip(),
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"{setting} must be a valid positive Docker memory size")
    try:
        parsed = Decimal(match.group("number")) * Decimal(1024 ** "bkmgtp".index(match.group("unit").lower() or "b"))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{setting} must be a valid positive Docker memory size") from exc
    if parsed <= 0 or parsed != parsed.to_integral_value():
        raise ValueError(f"{setting} must be a valid positive Docker memory size")
    return int(parsed)


MEM_LIMIT_BYTES = hard_memory_bytes(MEM_LIMIT, setting="SHIMPZ_TEAM_MEM_LIMIT")

MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


def model_for_brain(brain: str, value: object = None) -> str:
    """Return one Team's validated provider model, or that provider's explicit default."""
    if brain not in BRAINS:
        raise ValueError(f"unsupported brain: {brain!r}")
    if value is None or value == "":
        return BRAINS[brain]["default_model"]
    if not isinstance(value, str):
        raise ValueError("model must be a string")
    model = value.strip()
    if not model:
        return BRAINS[brain]["default_model"]
    if MODEL_RE.fullmatch(model) is None:
        raise ValueError("model must be 1-128 safe identifier characters")
    return model


# Per-team APP envelope (mirrors drivers/apps: 1g because real uvicorn backends idle near 500 MiB —
# R125; 0.5 vCPU; pids capped). One installed app = one container INSIDE the team's own network.
APP_MEM_LIMIT = os.environ.get("SHIMPZ_TEAM_APP_MEM_LIMIT", "1g")
APP_NANO_CPUS = int(os.environ.get("SHIMPZ_TEAM_APP_NANO_CPUS", str(500_000_000)))
APP_PIDS_LIMIT = int(os.environ.get("SHIMPZ_TEAM_APP_PIDS_LIMIT", "256"))
APP_MEM_LIMIT_BYTES = hard_memory_bytes(APP_MEM_LIMIT, setting="SHIMPZ_TEAM_APP_MEM_LIMIT")
# The MANY-tenant egress proxy (per-app token-gated) is connected into a Team's core network only
# when an installed App declares egress.
APP_EGRESS_CONTAINER = network_policy.APP_EGRESS_CONTAINER

# Vector reads Docker's json-file logs and derives the team from the line's own label (no Docker API).
# Keep the required json-file driver, but never inherit its unbounded default: a hostile workload can
# otherwise fill the host filesystem without exceeding its cgroup memory/PID admission envelope.
TEAM_LOG_MAX_SIZE = "5m"
TEAM_LOG_MAX_FILE = "2"
TEAM_LOG_CONFIG = docker.types.LogConfig(
    type=docker.types.LogConfig.types.JSON,
    config={
        "labels": "team.id",
        "max-size": TEAM_LOG_MAX_SIZE,
        "max-file": TEAM_LOG_MAX_FILE,
    },
)


def team_container_name(team_id: str) -> str:
    return network_policy.team_container_name(team_id)


def team_network_name(team_id: str) -> str:
    return network_policy.network_name(team_id, network_policy.CORE_KIND)


def team_network_labels(team_id: str, kind: str) -> dict[str, str]:
    return network_policy.network_labels(team_id, kind)


def team_config_volume(team_id: str) -> str:
    return network_policy.volume_name(team_id, network_policy.CONFIG_VOLUME_KIND)


def team_workspace_volume(team_id: str) -> str:
    return network_policy.volume_name(team_id, network_policy.WORKSPACE_VOLUME_KIND)


def team_db_project(team_id: str) -> str:
    return f"team_{team_id}"


def team_app_sane(app_id: str) -> str:
    """The catalog id ('notification-center') as a Docker/Postgres-safe token ('notification_center')."""
    return app_id.replace("-", "_")


def team_app_container_name(team_id: str, app_id: str) -> str:
    return network_policy.team_app_container_name(team_id, app_id)


def team_app_db_project(team_id: str, app_id: str) -> str:
    """The per-(team, app) DB project: 'team_<sha10(team_id)>_<app>'.

    Deterministic (uninstall/teardown re-derive it with no lookup) and always within pg-driver's
    58-char project cap: a readable 'team_<team_id>_<app>' would overflow at the 40-char team-id
    maximum, so the team contributes a fixed 10-hex digest instead.
    """
    digest = hashlib.sha256(team_id.encode()).hexdigest()[:10]
    return f"team_{digest}_{team_app_sane(app_id)}"


def core_deps() -> list[tuple[str, list[str]]]:
    """Shared services allowed on a Team's app/data plane."""
    return [(POSTGRES_CONTAINER, ["postgres"])]


def build_team_kwargs(
    team_id: str,
    team_name: str,
    *,
    database_url: str,
    owner: str = "",
    brain: str = DEFAULT_BRAIN,
    model: object = None,
) -> dict:
    """Kwargs for docker-py's low-level `containers.create` — never `run`.

    `run` would risk an accidental host-port publish or default-network attach; the whole isolation
    model depends on create + one explicit network. `brain` picks the agent runtime image from the
    trusted BRAINS registry (validated by the caller) and is recorded as the team.brain label.
    """
    selected_model = model_for_brain(brain, model)
    env = {
        "SHIMPZ_TEAM_ID": team_id,
        "SHIMPZ_TEAM_NAME": team_name,
    }
    return {
        "image": BRAINS[brain]["image"],
        "name": team_container_name(team_id),
        "hostname": team_id,
        "runtime": RUNTIME,
        "environment": env,
        "security_opt": ["no-new-privileges:true", "apparmor=docker-default"],
        "privileged": False,
        "read_only": True,
        "ipc_mode": "private",
        "cgroupns": "private",
        "cap_drop": ["ALL"],
        "cap_add": [],
        # The anchor and installed Apps share only the Team's internal core network.
        "network": team_network_name(team_id),
        "mounts": [],
        "tmpfs": {CONTAINER_TMP: "size=16m,mode=1777"},
        "mem_limit": MEM_LIMIT,
        # Equal memory and memory+swap ceilings disable swap for this hostile workload. Leaving
        # MemorySwap unset lets Docker grant an additional swap allowance on swap-enabled hosts.
        "memswap_limit": MEM_LIMIT,
        "mem_reservation": MEM_RESERVATION,
        "nano_cpus": NANO_CPUS,
        "pids_limit": PIDS_LIMIT,
        "ulimits": [docker.types.Ulimit(name="nofile", soft=256, hard=256)],
        # Hostile workloads may only become runnable through the driver's static+live proof. Docker
        # daemon startup or a natural process crash must never auto-start them behind that gate.
        "restart_policy": {"Name": "no"},
        "labels": {
            "team.driver": "1",
            "team.id": team_id,
            "team.name": team_name,
            "team.owner": owner,
            "team.brain": brain,
            "team.model": selected_model,
        },
        "log_config": TEAM_LOG_CONFIG,
        "detach": True,
    }


def build_team_app_kwargs(
    team_id: str,
    app_id: str,
    spec: AppSpec,
    *,
    database_url: str = "",
    proxy_env: dict[str, str] | None = None,
    owner: str = "",
    team_name: str = "",
) -> dict:
    """Kwargs for an installed APP container inside team `team_id`'s own core/data network.

    Tighter than the team brain (the packaging contract allows it): non-root fixed uid, cap_drop ALL,
    read-only rootfs with a /tmp tmpfs, no mounts at all — the app's ONLY state is its scoped DB, so an
    app container is disposable by construction. `proxy_env` is the app-egress lock (HTTPS_PROXY with the
    app's own token) — injected here by app.py only when the registry spec declares egress, never
    caller-suppliable. NOTE: the label is `team.app.driver`, NOT `team.driver` — app containers must
    never count against the team quota or appear in the team list.
    """
    env = {
        # The contract: the app answers HTTP on $PORT on its own interface (see sdk packaging docs).
        "PORT": str(spec.port),
        "HOST": CONTAINER_ALL_INTERFACES,
        "SHIMPZ_TEAM_ID": team_id,
        # The team's DISPLAY name — the owner-given identity ("the hero's name"), so every app can
        # speak AS its team ("Zyon asks your approval") instead of leaking an internal id.
        "SHIMPZ_TEAM_NAME": team_name or team_id,
        "SHIMPZ_APP": app_id,
        **({"DATABASE_URL": database_url} if database_url else {}),
        **(proxy_env or {}),
    }
    return {
        "image": spec.image,
        "name": team_app_container_name(team_id, app_id),
        "runtime": RUNTIME,
        "environment": env,
        "user": "10001:10001",
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true", "apparmor=docker-default"],
        "privileged": False,
        "ipc_mode": "private",
        "cgroupns": "private",
        # ONE network at create: the team's OWN internal bridge (app.py re-attaches with the app-id
        # alias so the team brain reaches it as http://<app-id>:<port>). Never a shared app net —
        # apps are per-Team (ADR-0002); a shared instance would mix tenant data.
        "network": team_network_name(team_id),
        "read_only": True,
        "tmpfs": {CONTAINER_TMP: "size=256m"},
        "mem_limit": APP_MEM_LIMIT,
        "memswap_limit": APP_MEM_LIMIT,
        "nano_cpus": APP_NANO_CPUS,
        "pids_limit": APP_PIDS_LIMIT,
        "ulimits": [docker.types.Ulimit(name="nofile", soft=4096, hard=4096)],
        "restart_policy": {"Name": "no"},
        "labels": {
            "team.app.driver": "1",
            "team.id": team_id,
            "team.app": app_id,
            "team.app.db": "1" if spec.db else "0",
            "team.owner": owner,
        },
        "log_config": TEAM_LOG_CONFIG,
        "detach": True,
    }
