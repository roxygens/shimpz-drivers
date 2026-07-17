"""Allowlist validation for deploy/route requests — runs BEFORE any Docker or filesystem mutation.

Nothing here talks to Docker; it only decides yes/no and returns a validated, structured
request the caller (app.py) turns into Docker calls via manifests.py.

This is the actual security boundary: even a fully compromised caller can only request what this module allows.
"""

from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass
from pathlib import Path

APP_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
FQDN_RE = re.compile(r"^[A-Za-z0-9.-]+$")
PORT_MIN, PORT_MAX = 3100, 3999

# Every app runs one of these two operator-selected release images — never a client-supplied image
# string. The defaults preserve local development; an immutable release override injects each exact
# manifest digest into shimpz-driver so app containers cannot drift behind the control plane.
ALLOWED_IMAGES = {
    "python": os.environ.get("SHIMPZ_APP_RUNTIME_IMAGE", "shimpz-app-runtime:local"),
    "node": os.environ.get("SHIMPZ_APP_RUNTIME_NODE_IMAGE", "shimpz-app-runtime-node:local"),
}
ALLOWED_ENTRYPOINT_BINS = frozenset({"uv", "uvicorn", "python", "python3", "pnpm", "node"})
ARG_RE = re.compile(r"^[A-Za-z0-9_./:=@,+-]{1,200}$")

# App egress (Shimpz L2): a declared external host must be a lowercase hostname. A payment PROCESSOR is
# refused server-side too — defense-in-depth so a compromised brain can't inject api.stripe.com into an
# app's allowlist. `pay.shimpz.com` is NOT refused: it is exactly what a paid app's effective_egress
# legitimately carries (the sanctioned rail). Mirrors shimpzmanifest.PAYMENT_HOSTS minus PAY_HOST.
EGRESS_HOST_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$")
REFUSED_EGRESS_HOSTS = frozenset(
    {
        "api.stripe.com",
        "js.stripe.com",
        "checkout.stripe.com",
        "api.paypal.com",
        "api-m.paypal.com",
        "api.braintreegateway.com",
        "api.adyen.com",
        "checkout.adyen.com",
        "api.razorpay.com",
        "api.mercadopago.com",
        "api.pagar.me",
        "api.mollie.com",
        "api.paddle.com",
        "api.lemonsqueezy.com",
        "connect.squareup.com",
        "metadata.google.internal",
    }
)
REFUSED_EGRESS_SUFFIXES = (".home", ".internal", ".lan", ".local", ".localhost")

# Positive allowlist: an app env var must be one of these keys. Every global secret name is
# excluded BY CONSTRUCTION, not by a deny-list that could go stale (section 3).
ALLOWED_ENV_KEYS = frozenset(
    {
        "PORT",
        "HOST",
        "DATABASE_URL",
        "SHIMPZ_BUS_BROKERS",
        "SHIMPZ_BUS_SASL_USERNAME",
        "SHIMPZ_BUS_SASL_PASSWORD",
        "SHIMPZ_BUS_SASL_MECHANISM",
        "SECRET_KEY",
    }
)
# Named explicitly only so a rejection error can say exactly what was refused and why.
FORBIDDEN_ENV_KEYS = frozenset(
    {
        "SHIMPZ_CF_TOKEN",
        "SHIMPZ_CF_ACCOUNT",
        "CF_TUNNEL_TOKEN",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "SHIMPZ_OPENAI_MEDIA_API_KEY",
        "VOICE_TOOLS_OPENAI_KEY",
        "GITHUB_TOKEN",
        "SHIMPZ_PG_DSN",
        "SHIMPZ_BUS_ADMIN_USERNAME",
        "SHIMPZ_BUS_ADMIN_PASSWORD",
    }
)

DSN_RE = re.compile(r"^postgres(?:ql)?(?:\+\w+)?://proj_([a-z0-9_]+):[^@]+@postgres:5432/proj_([a-z0-9_]+)$")
# shimpz-bus provision <project> always names the SCRAM user proj_<project> (the same canonical
# proj_<name> identity as the scoped Postgres role) — the same cross-project-credential-mismatch gate
# as DATABASE_URL below. A project cannot declare another project's bus identity even if driver-side
# validation were ever the only line of defense.
BUS_USERNAME_RE = re.compile(r"^proj_([a-z0-9_]+)$")


class ValidationError(Exception):
    """A deploy/route request failed the allowlist — nothing was touched."""


def sanitize_proj(name: str) -> str:
    """Port of shimpzdetect.sh's _sanitize_proj — all app/control-plane validators must match exactly."""
    lowered = re.sub(r"[^a-z0-9_]+", "_", name.lower())
    return lowered.strip("_")


def validate_name(name: str) -> str:
    if not APP_NAME_RE.match(name):
        raise ValidationError(f"app name must match {APP_NAME_RE.pattern!r}: {name!r}")
    return name


def validate_port(port: object) -> int:
    if not isinstance(port, int) or isinstance(port, bool):
        raise ValidationError(f"port must be an integer: {port!r}")
    if not (PORT_MIN <= port <= PORT_MAX):
        raise ValidationError(f"port {port} outside the app range {PORT_MIN}-{PORT_MAX}")
    return port


def validate_image_kind(image_kind: object) -> str:
    if image_kind not in ALLOWED_IMAGES:
        raise ValidationError(f"image_kind must be one of {sorted(ALLOWED_IMAGES)}: {image_kind!r}")
    return ALLOWED_IMAGES[image_kind]


def validate_entrypoint(entrypoint: object) -> list[str]:
    if not isinstance(entrypoint, list) or not entrypoint or not all(isinstance(a, str) for a in entrypoint):
        raise ValidationError("entrypoint must be a non-empty list of strings")
    if entrypoint[0] not in ALLOWED_ENTRYPOINT_BINS:
        raise ValidationError(f"entrypoint[0] must be one of {sorted(ALLOWED_ENTRYPOINT_BINS)}: {entrypoint[0]!r}")
    # http.server writes its access log to STDERR, and the log pipeline rightly maps non-JSON
    # stderr to level=error — so every routine 200/404 flooded `logq errors`, double-logged
    # (Round 125). shimpz_static (baked into the python runtime) keeps the same argv surface with
    # structured JSON access logs at the right level. Refuse the noisy server outright.
    if "http.server" in entrypoint:
        raise ValidationError(
            "http.server logs every request to stderr at level=error: serve static builds with "
            "`python3 -m shimpz_static <port> --directory <dir>` (same args, structured access logs)"
        )
    for arg in entrypoint:
        if not ARG_RE.match(arg):
            raise ValidationError(f"entrypoint arg has disallowed characters: {arg!r}")
        # The app container mounts the project at /app and has NO /config (that's the BRAIN's home).
        # A /config/... arg is the brain's own view of the project — it does not exist in the container,
        # so http.server serves (and uvicorn imports) a path that isn't there: a 404 on every request /
        # a dead import, with the deploy otherwise reporting success. This was the recurring
        # salesnator-meta bug (a `--directory /config/workspace/.../frontend/build` that silently 404'd
        # every recreate). Reject it LOUDLY instead of shipping a dead container.
        if arg == "/config" or "/config/" in arg:
            raise ValidationError(
                f"entrypoint references the brain-only path {arg!r}: the app container mounts the project "
                "at /app and has no /config. Use a RELATIVE path (e.g. `--directory frontend/build`) or an "
                "/app/... path — never /config/workspace/..."
            )
    return list(entrypoint)


_ROLE_SUFFIXES = ("-backend", "-ws")


def project_name_for(app_name: str) -> str:
    """Strip the "-backend"/"-ws" role suffix, if any, to get the underlying PROJECT name.

    "laudoctor-backend" and "laudoctor" share one project (and one database, one .env) — every
    place that derives a project-scoped identifier (the expected proj_<name> database here, the
    mounted directory in resolve_run_dir below) MUST agree on this, or a real deploy of any
    suffixed app fails validation against the wrong name (confirmed against the real live
    `laudoctor-backend`, whose actual database is `proj_laudoctor`, not `proj_laudoctor_backend`).
    """
    for suffix in _ROLE_SUFFIXES:
        if app_name.endswith(suffix):
            return app_name.removesuffix(suffix)
    return app_name


def validate_env(env: object, app_name: str) -> dict[str, str]:
    if not isinstance(env, dict):
        raise ValidationError("env must be an object")
    out: dict[str, str] = {}
    expected = sanitize_proj(project_name_for(app_name))
    for key, value in env.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValidationError("env keys/values must be strings")
        if key in FORBIDDEN_ENV_KEYS:
            raise ValidationError(f"env key {key!r} is a global agent secret — never allowed in an app deploy")
        if key not in ALLOWED_ENV_KEYS:
            raise ValidationError(f"env key {key!r} is not allowlisted ({sorted(ALLOWED_ENV_KEYS)})")
        if key == "DATABASE_URL":
            m = DSN_RE.match(value)
            if not m or m.group(1) != expected or m.group(2) != expected:
                raise ValidationError(
                    f"DATABASE_URL must point at this app's own isolated database (proj_{expected} on "
                    "the local postgres service) — got a mismatched or foreign DSN"
                )
        if key == "SHIMPZ_BUS_SASL_USERNAME":
            m = BUS_USERNAME_RE.match(value)
            if not m or m.group(1) != expected:
                raise ValidationError(
                    f"SHIMPZ_BUS_SASL_USERNAME must be this app's own bus identity (proj_{expected}, from "
                    "`shimpz-bus provision`) — got a mismatched or foreign username"
                )
        out[key] = value
    return out


@dataclass(frozen=True, slots=True)
class RunLocation:
    project_dir: Path  # the WHOLE project — mounted as /app (never just the role subdir)
    working_subdir: str | None  # None (project root) or "backend" — becomes the container's cwd


def resolve_run_dir(workspace_projects_root: Path, app_name: str) -> RunLocation:
    """Resolve the app's project directory + working subdir from its name via the naming convention.

    A bare name ("laudoctor") is the project's WEB tier, working dir = the project root; a
    "<project>-backend" or "<project>-ws" name is a role-specific process of that SAME project,
    working dir = <project>/backend (the legacy workspace layout also placed the WebSocket gateway
    at backend/app/ws.py). This is NEVER guessed from which subdirectories happen to exist — a
    bare-named web app's project root routinely ALSO has its own backend/ subdirectory, so "does
    backend/ exist" is not a safe signal for which tier a bare name means (confirmed against the
    real live `laudoctor` + `laudoctor-backend` apps, which share one project dir this way).

    The MOUNT is always the WHOLE project directory, never just the role subdir: a real backend's
    own config routinely resolves its .env via a RELATIVE `../` path (confirmed against the real
    live laudoctor-backend: `SettingsConfigDict(env_file="../.env")`, one level above its own
    `backend/`) — mounting only `backend/` would put that `../.env` outside the mount entirely.
    Isolation is unaffected: still scoped to this ONE project's own directory, never another's.

    Only ever resolves within workspace_projects_root/<project_name> — never anything else,
    regardless of symlinks or '..' in app_name (already blocked by validate_name's charset, this is
    defense in depth for a future caller that might relax it).
    """
    root = workspace_projects_root.resolve()
    project_name = project_name_for(app_name)
    role = "backend" if project_name != app_name else None
    project_dir = (root / project_name).resolve()
    if project_dir.parent != root:
        raise ValidationError(f"resolved project path escaped the workspace root: {project_dir}")
    if not project_dir.is_dir():
        raise ValidationError(f"no project at {project_dir} (scaffold it first: shimpz-new {project_name})")
    if role is None:
        return RunLocation(project_dir, None)
    working_dir = (project_dir / role).resolve()
    if working_dir.parent != project_dir:
        raise ValidationError(f"resolved working dir escaped the project: {working_dir}")
    if not working_dir.is_dir():
        raise ValidationError(f"no {role}/ dir at {working_dir} for app {app_name!r}")
    return RunLocation(project_dir, role)


def validate_fqdn(fqdn: str) -> str:
    if not FQDN_RE.match(fqdn) or not fqdn:
        raise ValidationError(f"fqdn must be [A-Za-z0-9.-]: {fqdn!r}")
    return fqdn


def validate_optional_port(port: object, field: str) -> int | None:
    if port is None:
        return None
    return validate_port(port) if field != "target" else port


def validate_calls(calls: object, app_name: str) -> list[str]:
    """The app's DECLARED synchronous dependencies (`shimpzbus.call` targets), by app name.

    Declaration is what makes cross-service reach auditable and wirable: the driver
    connects each provider to THIS app's network (per-app isolation stays the default — no
    flat any-app-can-reach-any-app fabric), and fleet-health re-checks the attachment every
    turn. An undeclared call simply fails DNS, by design.
    """
    if calls is None:
        return []
    if not isinstance(calls, list) or not all(isinstance(c, str) for c in calls):
        raise ValidationError("calls must be a list of app names")
    for c in calls:
        if not APP_NAME_RE.match(c):
            raise ValidationError(f"calls entry must be an app name [A-Za-z0-9_-]{{1,40}}: {c!r}")
        if c == app_name:
            raise ValidationError(f"calls cannot include the app itself: {c!r}")
    return list(dict.fromkeys(calls))  # de-dup, order-preserving


def validate_egress(egress: object) -> list[str]:
    """The app's declared external egress hosts (its effective_egress) — the deny-by-default L2 allowlist.

    Deny-by-default is enforced by the per-app proxy; here we only re-validate shape and refuse a payment
    PROCESSOR host (the ShimpzPay lock, server-side defense-in-depth). None/absent → [] (no internet).
    """
    if egress is None:
        return []
    if not isinstance(egress, list) or not all(isinstance(h, str) for h in egress):
        raise ValidationError("egress must be a list of hostnames")
    for h in egress:
        if not EGRESS_HOST_RE.match(h):
            raise ValidationError(f"egress host must be a lowercase hostname: {h!r}")
        is_ip_literal = True
        try:
            ipaddress.ip_address(h)
        except ValueError:
            is_ip_literal = False
        if is_ip_literal:
            raise ValidationError(f"egress host must not be an IP literal: {h!r}")
        if h.endswith(REFUSED_EGRESS_SUFFIXES):
            raise ValidationError(f"egress host must not use a private DNS suffix: {h!r}")
        if h in REFUSED_EGRESS_HOSTS:
            raise ValidationError(f"egress host {h!r} is a payment processor — payment is locked to ShimpzPay")
    return list(dict.fromkeys(egress))  # de-dup, order-preserving


@dataclass(frozen=True, slots=True)
class DeployRequest:
    name: str
    image: str
    entrypoint: list[str]
    port: int
    env: dict[str, str]
    persist: bool
    run_subpath: str  # relative to the workspace projects root, e.g. "myapp" — ALWAYS the whole
    # project (never just a role subdir — see RunLocation's docstring for why)
    working_dir: str  # in-container absolute path: "/app" or "/app/backend"
    worker: bool  # a pure bus consumer with NO HTTP surface by contract (matches shimpz-app's
    # --worker flag) — the deploy health check must confirm "stays running", never probe HTTP
    calls: list[str]  # declared shimpzbus.call targets (validate_calls) — drives network wiring
    egress: list[str]  # declared external hosts (effective_egress) — the deny-by-default L2 allowlist


def validate_deploy_request(name: str, body: dict, workspace_projects_root: Path) -> DeployRequest:
    app_name = validate_name(name)
    loc = resolve_run_dir(workspace_projects_root, app_name)
    run_subpath = str(loc.project_dir.relative_to(workspace_projects_root.resolve()))
    working_dir = f"/app/{loc.working_subdir}" if loc.working_subdir else "/app"
    return DeployRequest(
        name=app_name,
        image=validate_image_kind(body.get("image_kind")),
        entrypoint=validate_entrypoint(body.get("entrypoint")),
        port=validate_port(body.get("port")),
        env=validate_env(body.get("env", {}), app_name),
        persist=bool(body.get("persist", False)),
        run_subpath=run_subpath,
        working_dir=working_dir,
        worker=bool(body.get("worker", False)),
        calls=validate_calls(body.get("calls"), app_name),
        egress=validate_egress(body.get("egress")),
    )


def _validate_target(target: object, field: str) -> str:
    """A route target must be a real app container — never `shimpz-brain` itself.

    The `shimpz-brain` migration-bridge carve-out (`add-legacy`) was removed once the live migration
    confirmed no route depends on it — a route to `shimpz-brain` would be a public-hostname path straight
    into the one container holding the whole credential keyring.
    """
    prefix = f"app{os.environ.get('SHIMPZ_SUFFIX', '')}_"  # this instance's app prefix (R137)
    if not isinstance(target, str) or not (
        target.startswith(prefix) and APP_NAME_RE.match(target.removeprefix(prefix))
    ):
        raise ValidationError(f"{field} must be '{prefix}<name>': {target!r}")
    return target


# ── stack recreate (Phase C2): the marketplace's live-apply of a saved secret ────────────────────
# The admin panel (which holds the .env secrets, never the socket) passes the new env for ONE
# stateless capability sidecar; the driver (which holds the socket, never the .env) recreates
# it. This positive allowlist is the security boundary: ONLY these three services, and for each ONLY
# its own env keys — `shimpz-brain`, `postgres`, `redpanda`, `cloudflared`, and every stray key are refused BY
# CONSTRUCTION (a bad request can never recreate the brain or a stateful datastore, nor inject PATH).
RECREATABLE: dict[str, frozenset[str]] = {
    "r2-driver": frozenset(
        {
            "R2_BUCKET",
            "RCLONE_CONFIG_R2_ACCESS_KEY_ID",
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY",
            "RCLONE_CONFIG_R2_ENDPOINT",
        }
    ),
    "openai-driver": frozenset({"OPENAI_API_KEY", "VOICE_TOOLS_OPENAI_KEY"}),
    "cf-driver": frozenset({"SHIMPZ_CF_TOKEN", "SHIMPZ_CF_ACCOUNT"}),
}


@dataclass(frozen=True, slots=True)
class RecreateRequest:
    service: str  # the logical/compose service name (also the DNS alias == container base name)
    container_name: str  # resolved with SHIMPZ_SUFFIX (R137) — the actual running container to recreate
    env: dict[str, str]  # the allowlisted env overlay (empty value = disable that key → inert boot)


def validate_recreate_request(body: dict) -> RecreateRequest:
    service = body.get("service")
    if service not in RECREATABLE:
        raise ValidationError(f"service {service!r} is not recreatable (allowed: {sorted(RECREATABLE)})")
    env = body.get("env", {})
    if not isinstance(env, dict):
        raise ValidationError("env must be an object")
    allowed = RECREATABLE[service]
    out: dict[str, str] = {}
    for key, value in env.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValidationError("env keys/values must be strings")
        if key not in allowed:
            raise ValidationError(f"env key {key!r} not allowed for {service} (allowed: {sorted(allowed)})")
        out[key] = value
    container_name = f"{service}{os.environ.get('SHIMPZ_SUFFIX', '')}"  # this instance's actual container (R137)
    return RecreateRequest(service=service, container_name=container_name, env=out)


@dataclass(frozen=True, slots=True)
class RouteRequest:
    fqdn: str
    # web/api/ws routinely point at THREE DIFFERENT app containers (a fullstack project's static
    # front, API backend, and ws gateway are three separate `shimpz-app deploy`s) — never assume a
    # single shared target.
    web_target: str
    web_port: int
    api_target: str | None
    api_port: int | None
    ws_target: str | None
    ws_port: int | None


def validate_route_request(body: dict) -> RouteRequest:
    fqdn = validate_fqdn(body.get("fqdn", ""))
    api_port = validate_optional_port(body.get("api_port"), "api_port")
    ws_port = validate_optional_port(body.get("ws_port"), "ws_port")
    api_target = _validate_target(body.get("api_target"), "api_target") if api_port is not None else None
    ws_target = _validate_target(body.get("ws_target"), "ws_target") if ws_port is not None else None
    return RouteRequest(
        fqdn=fqdn,
        web_target=_validate_target(body.get("web_target"), "web_target"),
        web_port=validate_port(body.get("web_port")),
        api_target=api_target,
        api_port=api_port,
        ws_target=ws_target,
        ws_port=ws_port,
    )
