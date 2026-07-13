"""Turn a validated capsule id into docker-py container kwargs for an isolated brain.

The ONE place that decides what a Capsule container actually gets. Every security-relevant field
(security_opt, network, mounts, limits, Telegram/browser OFF) is a hardcoded constant here; the caller
never carries any of them, so there is nothing to override. A Capsule is a `shimpz-brain` with:
its OWN internal network, its OWN config+workspace volumes, a SCOPED Postgres DSN, no docker.sock,
no secrets keyring, no browser, no Telegram — reached out to the internet ONLY via egress-proxy.
"""

from __future__ import annotations

import os

import docker
import docker.types

# Multi-instance (R137): SHIMPZ_SUFFIX names this Space's resources; empty (the default) is prod.
SUFFIX = os.environ.get("SHIMPZ_SUFFIX", "")
IMAGE = os.environ.get("SHIMPZ_CAPSULE_IMAGE", "shimpz-brain:shimpz-local")

# Shared-plane container names (suffix-aware) that get CONNECTED into each capsule's own internal net —
# passed from compose exactly like shimpz-driver receives SHIMPZ_POSTGRES_CONTAINER etc. Deliberately
# MINIMAL: only egress-proxy (guarded to refuse internal destinations) + postgres (authz-isolated by the
# per-capsule proj_ role). victorialogs is NOT connected — the shared, unauthenticated log store would
# let any capsule read/forge every other capsule's logs; capsule logs still ship out via Vector's
# json-file tail, so nothing is lost.
EGRESS_CONTAINER = os.environ.get("SHIMPZ_EGRESS_PROXY_CONTAINER", f"egress-proxy{SUFFIX}")
POSTGRES_CONTAINER = os.environ.get("SHIMPZ_POSTGRES_CONTAINER", f"shimpz-postgres{SUFFIX}")

CAPSULE_PREFIX = f"capsule{SUFFIX}_"
NET_PREFIX = f"net_capsule{SUFFIX}_"

# Per-capsule envelope. A limit is not a reservation (the host has 125 GiB); it caps a runaway while
# keeping the marginal footprint small enough to pack hundreds of Capsules per Space. cgroup v2:
# mem_reservation ≈ memory.low (protect this much), mem_limit ≈ memory.max (hard cap).
MEM_LIMIT = os.environ.get("SHIMPZ_CAPSULE_MEM_LIMIT", "2g")
MEM_RESERVATION = os.environ.get("SHIMPZ_CAPSULE_MEM_RESERVATION", "384m")
NANO_CPUS = int(os.environ.get("SHIMPZ_CAPSULE_NANO_CPUS", str(4_000_000_000)))  # 4 vCPU ceiling; idle ≈ 0
PIDS_LIMIT = int(os.environ.get("SHIMPZ_CAPSULE_PIDS_LIMIT", "2048"))

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SHIMPZ_MODEL = os.environ.get("SHIMPZ_MODEL", "claude-sonnet-5")

# Vector reads Docker's json-file logs and derives the capsule from the line's own label (no Docker API).
CAP_LOG_CONFIG = docker.types.LogConfig(type=docker.types.LogConfig.types.JSON, config={"labels": "capsule.id"})


def capsule_container_name(cid: str) -> str:
    return f"{CAPSULE_PREFIX}{cid}"


def capsule_network_name(cid: str) -> str:
    return f"{NET_PREFIX}{cid}"


def capsule_config_volume(cid: str) -> str:
    return f"{CAPSULE_PREFIX}{cid}_config"


def capsule_workspace_volume(cid: str) -> str:
    return f"{CAPSULE_PREFIX}{cid}_workspace"


def capsule_db_project(cid: str) -> str:
    return f"capsule_{cid}"


def shared_deps() -> list[tuple[str, list[str]]]:
    """(container_name, [aliases]) to connect into each capsule's OWN internal net.

    Aliases matter: these containers carry a SHIMPZ_SUFFIX in their real name, but the capsule brain
    addresses them by the bare compose service name (its HTTPS_PROXY/DATABASE_URL do), so the alias is
    what its DNS actually resolves. Minimal set: its sole route out (egress-proxy, which refuses any
    internal destination) and its own database (postgres, authz-isolated by its proj_ role) — nothing
    else is reachable, so there is no cross-capsule or capsule→brain L3 path.
    """
    return [
        (EGRESS_CONTAINER, ["egress-proxy"]),
        (POSTGRES_CONTAINER, ["postgres"]),
    ]


def build_capsule_kwargs(cid: str, name: str, *, database_url: str, owner: str = "") -> dict:
    """Kwargs for docker-py's low-level `containers.create` — never `run`.

    `run` would risk an accidental host-port publish or default-network attach; the whole isolation
    model depends on create + one explicit network.
    """
    env = {
        "PUID": "1000",
        "PGID": "1000",
        "TZ": "America/Sao_Paulo",
        "TITLE": f"Capsule {name}",
        "SHIMPZ_HOME": "/config/.shimpz",
        "SHIMPZ_CAPSULE_ID": cid,
        "SHIMPZ_CAPSULE_NAME": name,
        # DEFAULT-DENY EGRESS (same posture as the main brain): off any default route, the ONLY way out
        # is the allowlist CONNECT proxy. NO_PROXY lists the in-cluster hosts it reaches directly.
        "HTTPS_PROXY": "http://egress-proxy:8888",
        "HTTP_PROXY": "http://egress-proxy:8888",
        "https_proxy": "http://egress-proxy:8888",
        "http_proxy": "http://egress-proxy:8888",
        "NO_PROXY": "localhost,127.0.0.1,::1,egress-proxy,postgres",
        "no_proxy": "localhost,127.0.0.1,::1,egress-proxy,postgres",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        # The capsule's OWN scoped database — a least-privilege proj_ role, never the superuser.
        "DATABASE_URL": database_url,
        "SHIMPZ_MODEL": SHIMPZ_MODEL,
        # Infinity-memory + run knobs (mirror the main brain).
        "SHIMPZ_MEMORY_DIR": "/config/.shimpz/memory",
        "SHIMPZ_MEM_TTL_DAYS": "90",
        "SHIMPZ_RECENT_TURNS": "6",
        "SHIMPZ_PONYTAIL": "1",
        "SHIMPZ_CTX_MAX_BYTES": "1500000",
        "SHIMPZ_THINKING_TOKENS": "10000",
        "SHIMPZ_MAX_TURNS": "80",
        "SHIMPZ_AUTO_CONTINUE": "3",
        # Thread caps sized to this capsule's CPU envelope (the kernel still shows all 96 host cores).
        "LP_NUM_THREADS": "4",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
        # Telegram OFF — a Capsule is not the owner's phone-facing brain (empty = gateway off).
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_ALLOWED_USERS": "",
    }
    if ANTHROPIC_API_KEY:
        env["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
    return {
        "image": IMAGE,
        "name": capsule_container_name(cid),
        "hostname": cid,
        "environment": env,
        # Hardened, identical to the main brain minus the browser's elevated caps: no new privileges,
        # not privileged, no docker.sock, no secrets keyring. (cap_drop:ALL would break the LSIO s6 init
        # that must chown/setuid to drop to the runtime user — so we match the brain, not the L2 apps.)
        "security_opt": ["no-new-privileges:true"],
        "privileged": False,
        # ONE network at create: the capsule's OWN internal bridge. The shared plane (egress-proxy /
        # postgres / victorialogs) is CONNECTED INTO it afterward — the capsule shares a net with NO
        # other capsule and NOT with the main brain, so it can never resolve or reach them.
        "network": capsule_network_name(cid),
        "mounts": [
            docker.types.Mount(target="/config", source=capsule_config_volume(cid), type="volume"),
            docker.types.Mount(target="/config/workspace", source=capsule_workspace_volume(cid), type="volume"),
        ],
        "tmpfs": {"/tmp": "size=2g,mode=1777"},  # noqa: S108 — the CAPSULE's own scratch mount, not this process's
        "mem_limit": MEM_LIMIT,
        "mem_reservation": MEM_RESERVATION,
        "nano_cpus": NANO_CPUS,
        "pids_limit": PIDS_LIMIT,
        "ulimits": [docker.types.Ulimit(name="nofile", soft=65536, hard=65536)],
        "restart_policy": {"Name": "unless-stopped"},
        "healthcheck": docker.types.Healthcheck(
            test=["CMD-SHELL", "claude --version >/dev/null"],
            interval=30 * 10**9,
            timeout=10 * 10**9,
            retries=3,
            start_period=60 * 10**9,
        ),
        "labels": {"capsule.driver": "1", "capsule.id": cid, "capsule.name": name, "capsule.owner": owner},
        "log_config": CAP_LOG_CONFIG,
        "detach": True,
    }
