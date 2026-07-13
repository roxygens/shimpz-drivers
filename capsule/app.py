"""capsule-driver — a socket-holding sidecar dedicated to Capsule lifecycle.

Besides shimpz-driver, this is the ONLY container holding /var/run/docker.sock — and it exposes ONLY
named operations (create/list/status/logs/stop/start/restart/destroy), never a generic Docker
passthrough. A Capsule is one isolated `shimpz-brain`: its OWN internal network, its OWN config+workspace
volumes, and a SCOPED Postgres database (provisioned via pg-driver — this driver never holds the
superuser). Every mutating call is bearer-gated → validated → mutated → audited (trace_id returned).
A compromised caller can only ever request what validate.py permits.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from collections import defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import accounts_client
import audit
import docker
import docker.errors
import manifests
import pgdriver_client
import token_store
import validate

LISTEN_PORT = int(os.environ.get("SHIMPZ_CAPSULEDRIVER_PORT", "7077"))
# Hard ceiling on live capsules per Space — a runaway/hostile caller can't exhaust host RAM/disk/IPs.
MAX_CAPSULES = int(os.environ.get("SHIMPZ_MAX_CAPSULES", "200"))

_docker = docker.from_env()
_token = token_store.ensure_token()

# Per-capsule lock: create/destroy of the SAME capsule must serialize; different capsules run parallel.
_locks_guard = threading.Lock()
_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)


def _lock_for(cid: str) -> threading.Lock:
    with _locks_guard:
        return _locks[cid]


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# ── docker helpers ───────────────────────────────────────────────────────────
def _get_container(name: str):
    try:
        return _docker.containers.get(name)
    except docker.errors.NotFound:
        return None


def _ensure_volume(name: str) -> None:
    try:
        _docker.volumes.get(name)
    except docker.errors.NotFound:
        _docker.volumes.create(name=name)


def _ensure_capsule_network(cid: str):
    net_name = manifests.capsule_network_name(cid)
    try:
        return _docker.networks.get(net_name)
    except docker.errors.NotFound:
        # internal=True: the capsule has NO NAT of its own — its only route out is egress-proxy.
        return _docker.networks.create(net_name, driver="bridge", internal=True)


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
    try:
        network.connect(container, aliases=aliases)
    except docker.errors.APIError as exc:
        if _already_connected(exc):
            return
        if required:
            raise ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"failed to connect {container_name!r} to the capsule network: {exc}",
            ) from exc


def _wire_capsule_deps(network) -> None:
    for container_name, aliases in manifests.shared_deps():
        _safe_connect(network, container_name, aliases=aliases, required=True)


def _teardown_capsule_network(cid: str) -> None:
    try:
        network = _docker.networks.get(manifests.capsule_network_name(cid))
    except docker.errors.NotFound:
        return
    network.reload()
    for container_id in network.attrs.get("Containers", {}):
        with contextlib.suppress(docker.errors.APIError):
            network.disconnect(container_id, force=True)
    with contextlib.suppress(docker.errors.APIError):
        network.remove()


def _describe(container) -> dict:
    return {
        "id": container.labels.get("capsule.id"),
        "name": container.labels.get("capsule.name"),
        "owner": container.labels.get("capsule.owner", ""),
        "status": container.status,
        "container": container.name,
    }


def _owner_of(cid: str) -> str | None:
    """The account_id that owns capsule `cid`, or None if the capsule does not exist."""
    container = _get_container(manifests.capsule_container_name(cid))
    return container.labels.get("capsule.owner", "") if container is not None else None


def _authorize(cid: str, principal: tuple[str, str | None]) -> None:
    """Operator may touch any capsule; an account may only touch a capsule it owns.

    Raises 404 (not 403) for an account acting on someone else's / a missing capsule — an account must
    not even be able to tell whether another account's capsule exists.
    """
    kind, account_id = principal
    if kind == "operator":
        if _owner_of(cid) is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"capsule {cid!r} not found")
        return
    if _owner_of(cid) != account_id:
        raise ApiError(HTTPStatus.NOT_FOUND, f"capsule {cid!r} not found")


# ── operations ───────────────────────────────────────────────────────────────
def _teardown(cid: str) -> bool:
    """Idempotently remove a capsule's container + network + scoped DB + BOTH volumes.

    Returns whether the DB drop succeeded. Used by _destroy AND by _create's rollback — a destroyed
    capsule leaves NO remanence, so a later capsule whose name collides to the same cid can never
    inherit prior data.
    """
    container = _get_container(manifests.capsule_container_name(cid))
    if container is not None:
        with contextlib.suppress(docker.errors.APIError):
            container.remove(force=True)
    _teardown_capsule_network(cid)
    dropped = True
    try:
        pgdriver_client.drop_db(manifests.capsule_db_project(cid))
    except Exception:  # noqa: BLE001 — surfaced by the caller's audit line; teardown proceeds regardless
        dropped = False
    for vol in (manifests.capsule_config_volume(cid), manifests.capsule_workspace_volume(cid)):
        with contextlib.suppress(docker.errors.APIError, docker.errors.NotFound):
            _docker.volumes.get(vol).remove(force=True)
    return dropped


def _create(cid: str, body: dict, owner: str = "") -> dict:
    name = str(body.get("name") or cid).strip() or cid
    with _lock_for(cid):
        existing = _get_container(manifests.capsule_container_name(cid))
        if existing is not None:
            # An account may only "re-create" (get) its OWN capsule; a name collision with a different
            # owner is invisible (404), never a hijack of someone else's capsule.
            if owner and existing.labels.get("capsule.owner", "") != owner:
                raise ApiError(HTTPStatus.NOT_FOUND, f"capsule {cid!r} not found")
            return {"capsule": cid, "name": name, "status": existing.status, "created": False}
        # Hard quota — an authenticated caller must not be able to exhaust host RAM/disk or the Docker
        # network address pool by creating capsules without bound.
        current = len(_docker.containers.list(all=True, filters={"label": "capsule.driver"}))
        if current >= MAX_CAPSULES:
            raise ApiError(HTTPStatus.TOO_MANY_REQUESTS, f"capsule limit reached ({current}/{MAX_CAPSULES})")
        # Transactional: on ANY failure, roll back everything partially created before surfacing — never
        # leak an orphan DB/role, network, or volume for an operator to hunt down later.
        try:
            db = pgdriver_client.create_db(manifests.capsule_db_project(cid))
            _ensure_volume(manifests.capsule_config_volume(cid))
            _ensure_volume(manifests.capsule_workspace_volume(cid))
            network = _ensure_capsule_network(cid)
            _wire_capsule_deps(network)
            kwargs = manifests.build_capsule_kwargs(cid, name, database_url=db["database_url"], owner=owner)
            container = _docker.containers.create(**kwargs)
            container.start()
        except Exception as exc:
            _teardown(cid)
            if isinstance(exc, ApiError):
                raise
            raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, f"capsule create failed (rolled back): {exc}") from exc
        return {
            "capsule": cid,
            "name": name,
            "status": "running",
            "created": True,
            "database": manifests.capsule_db_project(cid),
        }


def _destroy(cid: str) -> dict:
    with _lock_for(cid):
        dropped = _teardown(cid)
        return {"capsule": cid, "destroyed": True, "db_dropped": dropped}


def _list(owner: str | None = None) -> dict:
    """All capsules for the operator; only the account's own when `owner` is set."""
    caps = _docker.containers.list(all=True, filters={"label": "capsule.driver"})
    if owner is not None:
        caps = [c for c in caps if c.labels.get("capsule.owner", "") == owner]
    return {"capsules": [_describe(c) for c in caps]}


def _status(cid: str) -> dict:
    container = _get_container(manifests.capsule_container_name(cid))
    if container is None:
        raise ApiError(HTTPStatus.NOT_FOUND, f"capsule {cid!r} not found")
    return _describe(container)


def _logs(cid: str, lines: int) -> dict:
    container = _get_container(manifests.capsule_container_name(cid))
    if container is None:
        raise ApiError(HTTPStatus.NOT_FOUND, f"capsule {cid!r} not found")
    return {"capsule": cid, "logs": container.logs(tail=lines).decode("utf-8", "replace")}


def _lifecycle(cid: str, op: str) -> dict:
    container = _get_container(manifests.capsule_container_name(cid))
    if container is None:
        raise ApiError(HTTPStatus.NOT_FOUND, f"capsule {cid!r} not found")
    getattr(container, op)()
    return {"capsule": cid, "op": op, "status": "ok"}


# ── HTTP ─────────────────────────────────────────────────────────────────────
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
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid JSON body: {exc}") from exc

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

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
        except Exception as exc:  # noqa: BLE001 — fail loud, never leak a stack to the caller
            audit.log(method.lower(), self.path, result="error", reason=str(exc))
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

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
                body = self._read_body()
                # an account owns what it creates; an operator may create-on-behalf via an explicit owner
                owner = account_id or str(body.get("owner", "")).strip()
                result = _create(cid, body, owner)
                trace = audit.log("create", cid, result="ok", created=result.get("created"), owner=owner)
                self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
                return
            # every other op acts on an EXISTING capsule → gate on ownership first (404 if not yours)
            _authorize(cid, principal)
            if method == "DELETE" and sub == "":
                result = _destroy(cid)
                trace = audit.log("destroy", cid, result="ok", db_dropped=result["db_dropped"])
                self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
                return
            if method == "GET" and sub == "status":
                self._send_json(HTTPStatus.OK, _status(cid))
                return
            if method == "GET" and sub == "logs":
                self._send_json(HTTPStatus.OK, _logs(cid, int(query.get("lines", "200"))))
                return
            if method == "POST" and sub in ("stop", "start", "restart"):
                result = _lifecycle(cid, sub)
                audit.log(sub, cid, result="ok")
                self._send_json(HTTPStatus.OK, result)
                return

        raise ApiError(HTTPStatus.NOT_FOUND, f"no such operation: {method} {path}")


def main() -> None:
    ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), Handler).serve_forever()  # noqa: S104


if __name__ == "__main__":
    main()
