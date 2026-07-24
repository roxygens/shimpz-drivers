"""Turn a validated DeployRequest into docker-py container kwargs.

The ONE place that decides what a container actually gets. Every security-relevant field
(user, caps, network, mounts) is a hardcoded constant here; validate.py's DeployRequest
never carries any of them, so there is nothing for a caller to override.
"""

from __future__ import annotations

import os

import docker
import docker.types
from validate import DeployRequest

RUNTIME_UID = 10001
WORKSPACE_VOLUME = os.environ.get("SHIMPZ_WORKSPACE_VOLUME", "shimpz_shimpz-workspace")
# Each app's network exists ONLY for that app — no shared bridge, so an app can never resolve or
# reach another app's container. app.py connects shimpz-caddy
# (always) and PostgreSQL (only when the app's own env declares a database) to it.
APP_NETWORK_PREFIX = os.environ.get("SHIMPZ_APP_NETWORK_PREFIX", "net_app_")
# Multi-instance (R137): SHIMPZ_SUFFIX names this instance's resources; empty (the default) keeps
# every generated name byte-identical to the single-instance era — prod is untouched by design.
APP_CONTAINER_PREFIX = f"app{os.environ.get('SHIMPZ_SUFFIX', '')}_"
CONTAINER_ALL_INTERFACES = "0.0.0.0"
CONTAINER_TMP = "/tmp"

# 1g, not 512m: real uvicorn backends idle near 500 MiB — three sat pinned at 91–99% of the old
# 512m cap for days, one allocation from an OOM-kill (Round 125). A limit is not a reservation
# (the host has 125 GiB); fleet-health flags any container that crosses 90% of its cap.
MEM_LIMIT = "1g"
NANO_CPUS = 500_000_000  # 0.5 vCPU
PIDS_LIMIT = 256

# The Docker daemon forwards only app stdout/stderr to Vector's loopback-bound Fluent input. Async
# delivery keeps app startup fail-open for observability when Vector is restarting; the bounded
# buffer prevents a hostile logger from consuming unbounded daemon memory. Vector needs neither the
# Docker socket nor the host's container-log tree.
APP_LOG_CONFIG = docker.types.LogConfig(
    type="fluentd",
    config={
        "fluentd-address": "127.0.0.1:24224",
        "fluentd-async": "true",
        "fluentd-buffer-limit": "4096",
        "labels": "shimpz.app",
        "tag": "shimpz.app",
    },
)


def resolve_host_projects_root(client: docker.DockerClient) -> str:
    """The `shimpz-workspace` named volume's real host path.

    Discovered via the Docker API (no hardcoded storage-driver path) — app containers are
    siblings created via the socket, so their bind mounts need a literal host path; this is
    the one place that's derived.
    """
    mountpoint = client.volumes.get(WORKSPACE_VOLUME).attrs["Mountpoint"]
    return f"{mountpoint}/projects"


def container_name(app_name: str) -> str:
    return f"{APP_CONTAINER_PREFIX}{app_name}"


def volume_name(app_name: str) -> str:
    return f"{APP_CONTAINER_PREFIX}{app_name}_data"


def app_network_name(app_name: str) -> str:
    return f"{APP_NETWORK_PREFIX}{app_name}"


def build_mounts(req: DeployRequest, host_projects_root: str) -> list[docker.types.Mount]:
    host_src = f"{host_projects_root}/{req.run_subpath}"
    mounts = [docker.types.Mount(target="/app", source=host_src, type="bind", read_only=True)]
    if req.persist:
        mounts.append(docker.types.Mount(target="/data", source=volume_name(req.name), type="volume", read_only=False))
    return mounts


def build_container_kwargs(req: DeployRequest, host_projects_root: str, extra_env: dict | None = None) -> dict:
    """Build kwargs for docker-py's low-level `containers.create`.

    Never `run` — that would risk an accidental host-port publish or default network. `extra_env` overlays
    the container environment (used ONLY by the L2 egress lock to inject HTTPS_PROXY/NO_PROXY when active;
    None = today's behavior exactly).
    """
    return {
        "image": req.image,
        "name": container_name(req.name),
        "command": req.entrypoint,
        # The WHOLE project is mounted at /app (see build_mounts) so a role-specific process can
        # still resolve a relative `../.env` at the project root (confirmed against the real
        # laudoctor-backend's own config.py) — working_dir is what actually picks the role's
        # subdirectory to run FROM, equivalent to supervisord's old per-role `directory=`.
        "working_dir": req.working_dir,
        # UV_PROJECT_ENVIRONMENT points `uv run`/`uv sync` at a writable tmpfs OUTSIDE /app:
        # /app is a read-only BIND mount, so a tmpfs can't be nested inside it (Docker can't
        # create a mountpoint through an already-mounted read-only filesystem at container
        # start — confirmed by an actual failed container start, not a theoretical concern).
        # HOME=/tmp: the runtime image's user has NO home directory at all, so every uv state
        # dir that defaults under $HOME (.cache/uv, .local/share/uv/python, .local/share/uv/tools,
        # …) crashed outright ("Read-only file system") one at a time until redirected here, to
        # the already-writable /tmp tmpfs — confirmed against real failed container starts, not
        # a theoretical concern; simpler than enumerating every individual UV_* override. Both
        # unused/harmless for the node or static-file runtime.
        # HOST=0.0.0.0 is the APP CONTAINER's own bind address (it must answer on its network
        # interface for shimpz-caddy/health-checks to reach it) — not a bind in THIS process.
        "environment": {
            **req.env,
            "PORT": str(req.port),
            "HOST": CONTAINER_ALL_INTERFACES,
            "UV_PROJECT_ENVIRONMENT": "/venv",
            "HOME": CONTAINER_TMP,
            **(extra_env or {}),  # Mandatory L2 lock: the driver always supplies tokened proxy/NO_PROXY values.
        },
        "user": f"{RUNTIME_UID}:{RUNTIME_UID}",
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true"],
        "privileged": False,
        "network": app_network_name(req.name),
        "mounts": build_mounts(req, host_projects_root),
        "read_only": True,
        # `exec` is required on BOTH: Docker's tmpfs default is noexec, but uv legitimately
        # executes a managed CPython build + venv binaries here as normal operation (confirmed —
        # without it, every `uv run` failed with "Permission denied" on its own downloaded
        # interpreter, not a chmod issue). This doesn't touch the container's other protections
        # (read-only /app, non-root, cap_drop:ALL, no-new-privileges, network isolation) — an
        # attacker who can already write+run arbitrary code in /tmp has RCE in the app process
        # regardless of this flag; it isn't a new escalation path, just what uv needs to function.
        "tmpfs": {CONTAINER_TMP: "size=256m,exec", "/venv": "size=512m,uid=10001,gid=10001,exec"},
        "mem_limit": MEM_LIMIT,
        "nano_cpus": NANO_CPUS,
        "pids_limit": PIDS_LIMIT,
        "restart_policy": {"Name": "unless-stopped"},
        "labels": {"shimpz.driver": "1", "shimpz.app": req.name, "shimpz.port": str(req.port)},
        "log_config": APP_LOG_CONFIG,
        "detach": True,
    }
