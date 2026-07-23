#!/opt/venv/bin/python
"""shimpz-driver — the host-side control plane for per-app containers.

The Brain and Assistants never touch the Docker API. The trusted control plane calls this restricted,
allowlisted, audited HTTP API instead. Every request is validated (see validate.py) BEFORE any Docker
call, so a compromised caller can only request the closed operations implemented here.

Endpoints (all require `Authorization: Bearer <token>` — see token_store.py):
  POST   /v1/apps/<name>/deploy         {image_kind, entrypoint, port, env, persist}
  POST   /v1/apps/<name>/stop|start|restart
  GET    /v1/apps/<name>/status
  GET    /v1/apps/<name>/logs?lines=N
  GET    /v1/apps/<name>/health
  DELETE /v1/apps/<name>[?purge_volume=1]
  POST   /v1/routes/apply               {fqdn, target, web_port, api_port, ws_port}
  DELETE /v1/routes/<fqdn>
  POST   /v1/stack/recreate             {service, env}   (C2: recreate a whitelisted stateless sidecar)
"""

from __future__ import annotations

import contextlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import sys
import threading
import time
from collections import defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import audit
import caddy_routes
import docker
import docker.errors
import egress_lock
import manifests
import token_store
import validate

WORKSPACE_PROJECTS_ROOT = Path(os.environ.get("SHIMPZ_WORKSPACE_PROJECTS_ROOT", "/workspace-root/projects"))
LISTEN_PORT = int(os.environ.get("SHIMPZ_DRIVER_PORT", "7070"))
MAX_BODY_BYTES = 96 * 1024

# Per-app network isolation: shimpz-caddy is connected to EVERY app's own network (it's the only
# thing that needs to reach every app); PostgreSQL is connected only when the app declares a database.
CADDY_CONTAINER = os.environ.get("SHIMPZ_CADDY_CONTAINER", "shimpz-caddy")
POSTGRES_CONTAINER = os.environ.get("SHIMPZ_POSTGRES_CONTAINER", "shimpz-postgres")

# Shimpz L2 is a mandatory invariant, not a feature toggle. Missing means the secure default; any
# explicit value other than the exact string "1" aborts startup before the Docker client is used.
# Each app's OWN network is internal and the proxy is attached to that private network; apps never
# join a shared bridge. The proxy's public-destination pinning prevents it becoming a cross-net pivot.
egress_lock.require_enabled()
APP_EGRESS_PROXY = os.environ.get(
    "SHIMPZ_APP_EGRESS_PROXY_CONTAINER", f"app-egress-proxy{os.environ.get('SHIMPZ_SUFFIX', '')}"
)
APP_EGRESS_POLICY_DIR = Path(os.environ.get("SHIMPZ_APP_EGRESS_POLICY_DIR", "/app-egress-policy"))
# shimpz-caddy joins each app network at DEPLOY time; a recreated caddy (daemon restart, compose
# recreate, crash) comes back with NONE of them and silently 502s every app domain (a real prod
# outage, 2026-07-10). The driver reconciles that invariant on startup AND on this loop, so a
# bare caddy self-heals within one interval — no manual `docker network connect` ever again.
CADDY_RECONCILE_SECONDS = int(os.environ.get("SHIMPZ_CADDY_RECONCILE_SECONDS", "30"))

# Blue-green redeploy: the candidate is created and health-checked under a DIFFERENT name before
# ever touching the currently-serving container — the rename-swap below IS the route cutover
# (Caddy resolves app_<name> via Docker's embedded DNS, so Caddy's config never changes).
CANDIDATE_SUFFIX = "__candidate"
RETIRING_SUFFIX = "__retiring"
# 40×1.5s ≈ 60s window: an app container cold-starts with a full `uv sync` into its tmpfs venv
# (every candidate downloads its deps — nothing is cached across containers on purpose), and a
# heavier dependency sets can take 20-40s before
# uvicorn can answer. 10×1.5s ≈ 15s lost that race and rolled back a HEALTHY build. Both waiters
# exit EARLY on success or a definitive crash (exited/restarting), so the wider window only costs
# time on a genuinely slow start — never on a healthy or a crashed candidate.
HEALTH_RETRIES = int(os.environ.get("SHIMPZ_HEALTH_RETRIES", "40"))
HEALTH_DELAY_SECONDS = float(os.environ.get("SHIMPZ_HEALTH_DELAY_SECONDS", "1.5"))

_APP_ROUTE = re.compile(r"^/v1/apps/([^/]+)/(deploy|stop|start|restart|status|logs|health)$")
_APP_DELETE_ROUTE = re.compile(r"^/v1/apps/([^/]+)$")
_ROUTE_DELETE_ROUTE = re.compile(r"^/v1/routes/([^/]+)$")

_docker = docker.from_env()
_token = token_store.ensure_token()
_host_projects_root = manifests.resolve_host_projects_root(_docker)
_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
_locks_guard = threading.Lock()


def _lock_for(name: str) -> threading.Lock:
    with _locks_guard:
        return _locks[name]


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _list_apps() -> dict:
    """Every app container's name/port.

    The ONLY way `shimpz-brain` (no Docker access of its own) can resolve which app owns a given port,
    e.g. for shimpz-publish to build Caddy route targets.
    """
    containers = _docker.containers.list(all=True, filters={"label": "shimpz.driver=1"})
    return {
        "apps": [
            {"name": c.labels.get("shimpz.app"), "port": c.labels.get("shimpz.port"), "status": c.status}
            for c in containers
        ]
    }


def _get_by_container_name(container_name: str):
    try:
        return _docker.containers.get(container_name)
    except docker.errors.NotFound:
        return None


def _get_or_none(name: str):
    return _get_by_container_name(manifests.container_name(name))


def _probe_health(container, port: str) -> tuple[bool, str]:
    """Mirror the pre-container `_smoke` probe exactly.

    Try /api/health, /health, / in order, stop at the first path whose code isn't 404. All-404
    counts as alive (a bare API with no matching route still answers); only a connection failure
    (000) or a 5xx is unhealthy.
    """
    code = "000"
    for path in ("/api/health", "/health", "/"):
        rc, out = container.exec_run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "3", f"http://127.0.0.1:{port}{path}"]
        )
        code = out.decode(errors="replace").strip() if rc == 0 else "000"
        if code and code != "404":
            break
    healthy = bool(code) and code != "000" and not code.startswith("5")
    return healthy, code


def _wait_healthy(container, port: str) -> tuple[bool, str]:
    code = "000"
    for attempt in range(HEALTH_RETRIES):
        container.reload()
        if container.status != "running":
            return False, f"container not running (status={container.status})"
        healthy, code = _probe_health(container, port)
        if healthy:
            return True, code
        if attempt < HEALTH_RETRIES - 1:
            time.sleep(HEALTH_DELAY_SECONDS)
    return False, code


def _wait_running(container) -> tuple[bool, str]:
    """For --worker apps, "healthy" means "stays running past startup", never an HTTP probe.

    No HTTP surface by contract — the same contract shimpz-app's own pre-container
    `_worker_smoke` used.
    """
    status = "unknown"
    for attempt in range(HEALTH_RETRIES):
        container.reload()
        status = container.status
        if status == "running":
            return True, status
        if status in ("exited", "dead"):
            return False, status
        if attempt < HEALTH_RETRIES - 1:
            time.sleep(HEALTH_DELAY_SECONDS)
    return False, status


def _wait_recreated_healthy(container, hc_test) -> tuple[bool, str]:
    """Health-gate a recreated CORE sidecar by exec'ing its OWN Docker HEALTHCHECK (rc==0).

    NOT _wait_healthy: that curls an HTTP path, but the sidecar images ship no curl — their
    healthcheck is `python3 /app/healthcheck.py` (proving the 403 auth gate is live). No healthcheck
    at all → fall back to "stays running".
    """
    if not hc_test:
        return _wait_running(container)
    if hc_test[0] == "CMD-SHELL":
        argv = ["sh", "-c", hc_test[1]]
    elif hc_test[0] == "CMD":
        argv = hc_test[1:]
    else:
        argv = hc_test
    detail = "unknown"
    for attempt in range(HEALTH_RETRIES):
        container.reload()
        if container.status != "running":
            return False, f"not running (status={container.status})"
        rc, _ = container.exec_run(argv)
        if rc == 0:
            return True, "healthy"
        detail = f"healthcheck rc={rc}"
        if attempt < HEALTH_RETRIES - 1:
            time.sleep(HEALTH_DELAY_SECONDS)
    return False, detail


def _ensure_app_network(name: str):
    """Get-or-create this app's OWN network — never the old shared app_net.

    An app can never resolve or reach another app's container at all: there is no shared
    bridge left to enumerate or scan. The network is ALWAYS internal (no NAT); the app's only
    internet egress is the token-authenticated app-egress-proxy.
    """
    net_name = manifests.app_network_name(name)
    try:
        network = _docker.networks.get(net_name)
    except docker.errors.NotFound:
        return _docker.networks.create(net_name, driver="bridge", internal=True)
    network.reload()
    if not network.attrs.get("Internal", False):
        if network.attrs.get("Containers"):
            raise ApiError(
                HTTPStatus.CONFLICT,
                f"legacy network {net_name!r} has public NAT; remove/redeploy its disposable app before launch",
            )
        network.remove()
        return _docker.networks.create(net_name, driver="bridge", internal=True)
    return network


def _egress_token(name: str) -> str:
    """The app's stable per-app egress token (the Proxy-Authorization it presents to app-egress-proxy).

    Generated once and reused across redeploys, kept in the policy volume (driver + proxy only) so
    the app's HTTPS_PROXY stays valid and the proxy can map the token → the app's allowlist.
    """
    tdir = APP_EGRESS_POLICY_DIR / ".tokens"
    tdir.mkdir(parents=True, exist_ok=True)
    tf = tdir / f"{name}.token"
    with contextlib.suppress(OSError):
        tok = tf.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    tok = secrets.token_hex(16)
    tf.write_text(tok, encoding="utf-8")
    return tok


def _write_egress_policy(token: str, egress: list[str]) -> None:
    """Publish the app's allowlist the proxy reads: <token>.json = its effective_egress (sorted hosts)."""
    APP_EGRESS_POLICY_DIR.mkdir(parents=True, exist_ok=True)
    (APP_EGRESS_POLICY_DIR / f"{token}.json").write_text(json.dumps(sorted(egress)), encoding="utf-8")


def _no_proxy_for(req: validate.DeployRequest) -> str:
    """Hosts the app reaches DIRECTLY, never via the egress proxy.

    PostgreSQL and shimpz-caddy stay in-cluster; external HTTPS goes through the proxy.
    """
    hosts = ["localhost", "127.0.0.1", "postgres", POSTGRES_CONTAINER, CADDY_CONTAINER]
    return ",".join(hosts)


def _already_connected(exc: docker.errors.APIError) -> bool:
    """True only for the ONE expected idempotent case: this container is already on this network.

    Confirmed against the real Docker API (not guessed): a repeat `network.connect()` on an
    already-connected container raises APIError with response status 403 and explanation
    "endpoint with name <container> already exists in network <network>" — a completely different
    shape from every other real failure (network not found, permission, daemon error, etc.), so
    this check is what makes `required=True` safe to also make idempotent.
    """
    resp = exc.response
    return (
        resp is not None
        and resp.status_code == HTTPStatus.FORBIDDEN
        and "already exists in network" in (exc.explanation or "")
    )


def _safe_connect(network, container_name: str, *, aliases: list[str] | None = None, required: bool) -> None:
    """Connect `container_name` to `network`, idempotent for the one real no-op case.

    A wiring failure must never be swallowed: `required=True` (every current call site) raises
    `ApiError`, ABORTING the deploy — the same failure-before-any-candidate-created point as every
    other pre-flight check in `_deploy`, so nothing needs rolling back yet.

    `aliases` matters: compose auto-aliases services to their names on `edge`, but this network is
    created via the raw Docker API, so that alias does NOT carry over — without it apps could only
    resolve `shimpz-postgres`, while every project's DATABASE_URL is written against "postgres".
    """
    try:
        container = _docker.containers.get(container_name)
    except docker.errors.NotFound as exc:
        if required:
            raise ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR, f"required dependency container {container_name!r} not found"
            ) from exc
        return
    try:
        network.connect(container, aliases=aliases)
    except docker.errors.APIError as exc:
        if _already_connected(exc):
            return
        if required:
            raise ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"failed to connect required dependency {container_name!r} to the app network: {exc}",
            ) from exc


def _wire_network_deps(network, req: validate.DeployRequest) -> None:
    """Connect ONLY what this specific app actually declared it needs.

    shimpz-caddy is the one exception — every app must be reachable via Caddy, always, so it's
    ALWAYS required. PostgreSQL is connected (and, once declared, equally required) only
    when the app's own env says it uses them — an app that never declared DATABASE_URL can't even
    resolve `shimpz-postgres`, let alone reach it.
    """
    _safe_connect(network, CADDY_CONTAINER, required=True)
    _safe_connect(network, APP_EGRESS_PROXY, aliases=["app-egress-proxy"], required=True)
    if "DATABASE_URL" in req.env:
        _safe_connect(network, POSTGRES_CONTAINER, aliases=["postgres"], required=True)


def _teardown_app_network(name: str) -> None:
    net_name = manifests.app_network_name(name)
    try:
        network = _docker.networks.get(net_name)
    except docker.errors.NotFound:
        return
    network.reload()
    for container_id in network.attrs.get("Containers", {}):
        with contextlib.suppress(docker.errors.APIError):
            network.disconnect(container_id, force=True)
    with contextlib.suppress(docker.errors.APIError):
        network.remove()


def _deploy(name: str, body: dict) -> dict:
    """Blue-green deploy: create+health-check a CANDIDATE before touching the serving container.

    On any failure — create, start, or health check — the candidate is torn down and the
    previous container (if any) is left running, untouched, still serving. Only after the
    candidate proves healthy does the rename-swap (the actual route cutover — Caddy resolves
    app_<name> by Docker's own embedded DNS) happen, and the previous container is removed
    only once that swap has succeeded.
    """
    req = validate.validate_deploy_request(name, body, WORKSPACE_PROJECTS_ROOT)
    final_name = manifests.container_name(name)
    candidate_name = f"{final_name}{CANDIDATE_SUFFIX}"
    retiring_name = f"{final_name}{RETIRING_SUFFIX}"

    with _lock_for(name):
        # Defensive cleanup: a leftover candidate/retiring container from a previous crashed
        # attempt must never block this one — a transactional redeploy must be safely retryable.
        for stale_name in (candidate_name, retiring_name):
            stale = _get_by_container_name(stale_name)
            if stale is not None:
                stale.remove(force=True)

        if req.persist:
            vol_name = manifests.volume_name(name)
            try:
                _docker.volumes.get(vol_name)
            except docker.errors.NotFound:
                _docker.volumes.create(name=vol_name)

        network = _ensure_app_network(name)
        _wire_network_deps(network, req)

        # Publish the app's allowlist and route egress through the per-app-token proxy. The proxy,
        # not the app, is dynamically attached to this private network; no shared app bridge exists.
        token = _egress_token(name)
        _write_egress_policy(token, req.egress)
        no_proxy = _no_proxy_for(req)
        proxy = f"http://{token}@app-egress-proxy:8889"
        proxy_env = {"HTTPS_PROXY": proxy, "https_proxy": proxy, "NO_PROXY": no_proxy, "no_proxy": no_proxy}

        kwargs = manifests.build_container_kwargs(req, _host_projects_root, extra_env=proxy_env)
        kwargs["name"] = candidate_name  # never the final name yet — that's the whole point
        try:
            candidate = _docker.containers.create(**kwargs)
            candidate.start()
        except docker.errors.APIError as exc:
            # create/start failed outright — nothing to roll back (the previous container, if
            # any, was never touched), but clean up any half-created candidate immediately
            # rather than leaving it dangling for the NEXT deploy's defensive cleanup to find.
            with contextlib.suppress(docker.errors.APIError, NameError):
                candidate.remove(force=True)
            audit.log("deploy", name, result="error", reason=f"candidate create/start failed: {exc}")
            raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, f"candidate create/start failed: {exc}") from exc

        healthy, code = _wait_running(candidate) if req.worker else _wait_healthy(candidate, str(req.port))
        if not healthy:
            log_tail = candidate.logs(tail=40).decode(errors="replace")
            candidate.remove(force=True)
            audit.log("deploy", name, result="denied", reason=f"candidate failed health check (code={code})")
            raise ApiError(
                HTTPStatus.BAD_GATEWAY,
                f"candidate for {name!r} failed its health check (code={code}) — rolled back, "
                f"the previous container (if any) was never touched. Log tail:\n{log_tail}",
            )

        old = _get_or_none(name)
        try:
            if old is not None:
                old.rename(retiring_name)
            candidate.rename(final_name)
        except docker.errors.APIError as exc:
            # Roll back the rename attempt itself: restore the old container's name so it keeps
            # serving, and discard the (already-healthy, but never cut over) candidate.
            if old is not None:
                with contextlib.suppress(docker.errors.APIError):
                    old.rename(final_name)
            with contextlib.suppress(docker.errors.APIError):
                candidate.remove(force=True)
            audit.log("deploy", name, result="error", reason=f"cutover rename failed: {exc}")
            raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, f"cutover failed, rolled back: {exc}") from exc

        if old is not None:
            old.remove(force=True)

    trace_id = audit.log(
        "deploy",
        name,
        result="ok",
        image=req.image,
        allowed_env_keys=sorted(req.env),
        mounts=["/app:ro"] + (["/data"] if req.persist else []),
    )
    return {"status": "deployed", "container_id": candidate.id, "trace_id": trace_id}


def _recreate(body: dict) -> dict:
    """Blue-green recreate of ONE whitelisted stateless sidecar with a new env overlay (Phase C2).

    Mirrors _deploy's create-candidate → health-gate → rename-swap, but rebuilds the create kwargs
    from the RUNNING container's own `.attrs` (manifests.build_recreate_kwargs) instead of a manifest
    — the caller supplies ONLY the env, already constrained by validate.RECREATABLE. Keep-old-on-
    failure: any failure tears down the candidate and leaves the old sidecar serving, untouched.

    DNS cutover is the rename itself: a sidecar's service alias EQUALS its container name (e.g.
    `example-driver`), so renaming the healthy candidate to that name moves Docker's embedded DNS to
    it — the same reason _deploy needs no Caddy reconfig.
    """
    req = validate.validate_recreate_request(body)
    name = req.container_name
    candidate_name = f"{name}{CANDIDATE_SUFFIX}"
    retiring_name = f"{name}{RETIRING_SUFFIX}"

    with _lock_for(name):
        for stale_name in (candidate_name, retiring_name):
            stale = _get_by_container_name(stale_name)
            if stale is not None:
                stale.remove(force=True)

        old = _get_by_container_name(name)
        if old is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"service {req.service!r} ({name}) is not running")
        old.reload()
        attrs = old.attrs
        nets = list((attrs.get("NetworkSettings", {}).get("Networks") or {}).keys())
        hc_test = (attrs.get("Config", {}).get("Healthcheck") or {}).get("Test")

        kwargs = manifests.build_recreate_kwargs(attrs, req.env)
        kwargs["name"] = candidate_name
        if nets:
            kwargs["network"] = nets[0]  # create attaches to ONE network; the rest are connected next
        try:
            candidate = _docker.containers.create(**kwargs)
            for extra in nets[1:]:
                _docker.networks.get(extra).connect(candidate)
            candidate.start()
        except docker.errors.APIError as exc:
            with contextlib.suppress(docker.errors.APIError, NameError):
                candidate.remove(force=True)
            audit.log("stack_recreate", req.service, result="error", reason=f"candidate create/start failed: {exc}")
            raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, f"candidate create/start failed: {exc}") from exc

        healthy, detail = _wait_recreated_healthy(candidate, hc_test)
        if not healthy:
            log_tail = candidate.logs(tail=40).decode(errors="replace")
            candidate.remove(force=True)
            audit.log("stack_recreate", req.service, result="denied", reason=f"candidate unhealthy: {detail}")
            raise ApiError(
                HTTPStatus.BAD_GATEWAY,
                f"recreated {req.service!r} failed its health check ({detail}) — rolled back, the "
                f"previous sidecar was never touched. Log tail:\n{log_tail}",
            )

        try:
            old.rename(retiring_name)
            candidate.rename(name)
        except docker.errors.APIError as exc:
            with contextlib.suppress(docker.errors.APIError):
                old.rename(name)
            with contextlib.suppress(docker.errors.APIError):
                candidate.remove(force=True)
            audit.log("stack_recreate", req.service, result="error", reason=f"cutover rename failed: {exc}")
            raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, f"cutover failed, rolled back: {exc}") from exc

        old.remove(force=True)

    trace_id = audit.log("stack_recreate", req.service, result="ok", env_keys=sorted(req.env), health=detail)
    return {"status": "recreated", "service": req.service, "health": detail, "trace_id": trace_id}


def _lifecycle(name: str, op: str) -> dict:
    validate.validate_name(name)
    container = _get_or_none(name)
    if container is None:
        raise ApiError(HTTPStatus.NOT_FOUND, f"no app container for {name!r}")
    with _lock_for(name):
        getattr(container, op)()
    trace_id = audit.log(op, name, result="ok")
    return {"status": op, "trace_id": trace_id}


def _status(name: str) -> dict:
    validate.validate_name(name)
    container = _get_or_none(name)
    if container is None:
        raise ApiError(HTTPStatus.NOT_FOUND, f"no app container for {name!r}")
    container.reload()
    return {
        "state": container.attrs.get("State", {}),
        "restart_count": container.attrs.get("RestartCount", 0),
        "id": container.id,
    }


def _logs(name: str, lines: int) -> dict:
    validate.validate_name(name)
    container = _get_or_none(name)
    if container is None:
        raise ApiError(HTTPStatus.NOT_FOUND, f"no app container for {name!r}")
    return {"logs": container.logs(tail=lines).decode(errors="replace")}


def _health(name: str) -> dict:
    validate.validate_name(name)
    container = _get_or_none(name)
    if container is None:
        raise ApiError(HTTPStatus.NOT_FOUND, f"no app container for {name!r}")
    container.reload()
    if container.status != "running":
        return {"healthy": False, "state": container.status}
    healthy, code = _probe_health(container, container.labels.get("shimpz.port"))
    return {"healthy": healthy, "code": code}


def _remove(name: str, purge_volume: bool) -> dict:
    validate.validate_name(name)
    container = _get_or_none(name)
    with _lock_for(name):
        if container is not None:
            container.remove(force=True)
        if purge_volume:
            with contextlib.suppress(docker.errors.NotFound):
                _docker.volumes.get(manifests.volume_name(name)).remove()
        # This app's network is 1:1 with the app (nothing else ever shares it) — always torn
        # down on removal, independent of purge_volume (a different, data-persistence concern).
        _teardown_app_network(name)
    trace_id = audit.log("rm", name, result="ok", purge_volume=purge_volume)
    return {"status": "removed", "trace_id": trace_id}


def _route_apply(body: dict) -> dict:
    req = validate.validate_route_request(body)
    caddy_routes.apply_route(_docker, req)
    trace_id = audit.log(
        "route_apply",
        req.fqdn,
        result="ok",
        web_target=req.web_target,
        api_target=req.api_target,
        ws_target=req.ws_target,
    )
    return {"status": "applied", "trace_id": trace_id}


def _route_delete(fqdn: str) -> dict:
    fqdn = validate.validate_fqdn(fqdn)
    caddy_routes.remove_route(_docker, fqdn)
    trace_id = audit.log("route_del", fqdn, result="ok")
    return {"status": "removed", "trace_id": trace_id}


def _route_list() -> dict:
    return {"routes": caddy_routes.list_routes()}


class Handler(BaseHTTPRequestHandler):
    server_version = "shimpz-driver/1.0"

    def _authed(self) -> bool:
        auth = self.headers.get("Authorization", "")
        return hmac.compare_digest(auth, f"Bearer {_token}")

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        raw_length = self.headers.get("Content-Length", "0") or "0"
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "invalid Content-Length") from exc
        if length < 0 or length > MAX_BODY_BYTES:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body is too large")
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid JSON body: {exc}") from exc

    def _dispatch(self, method: str) -> None:
        if not self._authed():
            # 127.0.0.1 = this container's own Docker HEALTHCHECK proving the 403 gate is live
            # (an unauthenticated probe every 30s BY DESIGN) — keep the audit line but at info,
            # so warn/error carries only real denials, never a heartbeat.
            if self.client_address[0] == "127.0.0.1":
                audit.log("auth", self.path, result="denied", level="info", source="loopback-probe")
            else:
                audit.log("auth", self.path, result="denied")
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "invalid or missing bearer token"})
            return
        try:
            self._route(method)
        except ApiError as exc:
            audit.log(method.lower(), self.path, result="denied", reason=exc.message)
            self._send_json(exc.status, {"error": exc.message})
        except validate.ValidationError as exc:
            audit.log(method.lower(), self.path, result="denied", reason=str(exc))
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            audit.log(method.lower(), self.path, result="error", reason=type(exc).__name__)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal error"})

    def _route(self, method: str) -> None:
        path = self.path.split("?", 1)[0]
        query = self.path.split("?", 1)[1] if "?" in self.path else ""

        if method == "POST" and (m := _APP_ROUTE.match(path)) and m.group(2) == "deploy":
            self._send_json(HTTPStatus.OK, _deploy(m.group(1), self._body()))
            return
        if method == "POST" and path == "/v1/stack/recreate":
            self._send_json(HTTPStatus.OK, _recreate(self._body()))
            return
        if method == "POST" and (m := _APP_ROUTE.match(path)) and m.group(2) in ("stop", "start", "restart"):
            self._send_json(HTTPStatus.OK, _lifecycle(m.group(1), m.group(2)))
            return
        if method == "GET" and (m := _APP_ROUTE.match(path)) and m.group(2) == "status":
            self._send_json(HTTPStatus.OK, _status(m.group(1)))
            return
        if method == "GET" and (m := _APP_ROUTE.match(path)) and m.group(2) == "logs":
            lines = int(dict(p.split("=") for p in query.split("&") if "=" in p).get("lines", "80"))
            self._send_json(HTTPStatus.OK, _logs(m.group(1), lines))
            return
        if method == "GET" and (m := _APP_ROUTE.match(path)) and m.group(2) == "health":
            self._send_json(HTTPStatus.OK, _health(m.group(1)))
            return
        if method == "DELETE" and (m := _APP_DELETE_ROUTE.match(path)):
            purge = "purge_volume=1" in query
            self._send_json(HTTPStatus.OK, _remove(m.group(1), purge))
            return
        if method == "GET" and path == "/v1/apps":
            self._send_json(HTTPStatus.OK, _list_apps())
            return
        if method == "GET" and path == "/v1/routes":
            self._send_json(HTTPStatus.OK, _route_list())
            return
        if method == "POST" and path == "/v1/routes/apply":
            self._send_json(HTTPStatus.OK, _route_apply(self._body()))
            return
        if method == "DELETE" and (m := _ROUTE_DELETE_ROUTE.match(path)) and path.startswith("/v1/routes/"):
            self._send_json(HTTPStatus.OK, _route_delete(m.group(1)))
            return
        raise ApiError(HTTPStatus.NOT_FOUND, f"no route for {method} {path}")

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def log_message(self, fmt: str, *args: object) -> None:
        # Suppress BaseHTTPRequestHandler's default stderr access log — audit.log() is the
        # single source of truth for what happened, in the schema logq expects.
        pass


def reconcile_caddy_networks(caddy_name: str = CADDY_CONTAINER, prefix: str | None = None) -> int:
    """Ensure `caddy_name` is attached to EVERY app network (name starts with `prefix`).

    Self-heals the outage class where a recreated shimpz-caddy loses all the per-app networks it joins
    at deploy time and stops routing every app domain. Cheap and idempotent (already-connected is a
    no-op); scoped to THIS instance by the prefix, so it can never touch another instance's caddy.
    Returns the number of (re)connections made this pass. Never raises for a per-network hiccup.
    """
    if prefix is None:
        prefix = manifests.APP_NETWORK_PREFIX
    try:
        caddy = _docker.containers.get(caddy_name)
    except docker.errors.NotFound:
        return 0  # caddy itself is gone (mid-recreate) — the next pass will catch it
    reconnected = 0
    for net in _docker.networks.list(filters={"name": prefix}):
        if not net.name.startswith(prefix):
            continue  # docker's name filter is a loose substring — enforce the real prefix boundary
        try:
            net.connect(caddy)
        except docker.errors.APIError as exc:
            if not _already_connected(exc):
                print(f"caddy-reconcile: failed to connect {caddy_name} -> {net.name}: {exc}", file=sys.stderr)
            continue
        reconnected += 1
        print(f"caddy-reconcile: reconnected {caddy_name} -> {net.name}", file=sys.stderr)
    return reconnected


def _caddy_reconcile_loop() -> None:
    """Periodic self-heal so a caddy recreated WHILE the driver runs is repaired too."""
    while True:
        time.sleep(CADDY_RECONCILE_SECONDS)
        try:
            reconcile_caddy_networks()
        except Exception as exc:
            print(f"caddy-reconcile: pass errored (continuing): {exc}", file=sys.stderr)


def main() -> None:
    # Startup self-heal FIRST: a daemon restart / driver recreate that left caddy bare is
    # repaired before we accept any request. Then keep watch for a caddy recreated at runtime.
    n = reconcile_caddy_networks()
    print(f"shimpz-driver: startup caddy-reconcile connected {n} app network(s)", file=sys.stderr)
    threading.Thread(target=_caddy_reconcile_loop, daemon=True).start()
    server = ThreadingHTTPServer((str(ipaddress.IPv4Address(0)), LISTEN_PORT), Handler)
    print(f"shimpz-driver listening on :{LISTEN_PORT}", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
