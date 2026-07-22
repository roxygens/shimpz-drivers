"""Templated Caddy route apply/remove.

The driver's own template, mirroring rootfs/usr/local/bin/shimpz-route's Caddyfile block
exactly, just with `$target:$port` (a container DNS name on app_net) instead of
`127.0.0.1:$port`. Caddy's admin API is never network-exposed: this module writes the site
file to a volume shared with the `shimpz-caddy` container, then reloads it via `docker exec`
over the socket the driver already holds — so `shimpz-route`/`shimpz-publish` (running
inside `shimpz-brain`) never touch Caddy directly.

shimpz-caddy is connected to EVERY app's own network (so it can reverse_proxy to each app's
upstream), which means an app container can ALSO reach shimpz-caddy:8080 directly and send an
arbitrary Host header — routing purely by Host, Caddy would happily proxy that request to a
DIFFERENT app's upstream, a lateral-movement path per-app network isolation never closed
by itself. Every rendered site therefore opens with a `remote_ip`
gate: only requests whose source IP falls inside the `edge` network (where the ONLY legitimate
caller — cloudflared — lives) are served; anything else gets 403 before any reverse_proxy happens.
The edge subnet is never hardcoded: `_edge_subnet()` asks the Docker API for the network's CURRENT
actual subnet at apply time, so nothing can silently drift.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from validate import RouteRequest

SITES_DIR = Path(os.environ.get("SHIMPZ_DRIVER_CADDY_SITES_DIR", "/caddy-sites"))
CADDY_CONTAINER = os.environ.get("SHIMPZ_CADDY_CONTAINER", "shimpz-caddy")
EDGE_NETWORK = os.environ.get("SHIMPZ_EDGE_NETWORK", "shimpz_edge")
CADDY_NETWORK = os.environ.get("SHIMPZ_CADDY_NETWORK", "shimpz_caddy_net")  # item 8: the brain's shimpz-caddy path
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")

# Public responses share one conservative browser policy. SvelteKit's prerendered output includes
# inline hydration scripts and the storefront uses inline style attributes, so those two sources are
# explicitly allowed; eval remains forbidden. Keep shimpz-caddy/Caddyfile in lockstep with this tuple.
SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    ("Strict-Transport-Security", "max-age=31536000; includeSubDomains"),
    (
        "Content-Security-Policy",
        "default-src 'self'; base-uri 'self'; object-src 'none'; frame-ancestors 'none'; "
        "form-action 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; font-src 'self' data:; connect-src 'self' https: wss:; "
        "worker-src 'self' blob:; manifest-src 'self'; upgrade-insecure-requests",
    ),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "strict-origin-when-cross-origin"),
    (
        "Permissions-Policy",
        "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    ),
)


def _security_header_lines(indent: str = "\t\t") -> list[str]:
    lines = [f"{indent}header {{"]
    lines.extend(f'{indent}\t{name} "{value}"' for name, value in SECURITY_HEADERS)
    lines.append(f"{indent}}}")
    return lines


def _safe_filename(fqdn: str) -> str:
    return _UNSAFE.sub("_", fqdn)


def _edge_subnet(client) -> str:  # client: docker.DockerClient
    """The `edge` network's CURRENT real subnet, read live from the Docker API.

    Never hardcoded/pinned in compose (that would force a disruptive network recreate — confirmed
    by testing on a throwaway compose project: adding an `ipam.config` block, even one matching the
    network's own current auto-assigned subnet byte-for-byte, still makes `docker compose up`
    remove and recreate the network, disconnecting every attached container at once). Reading it
    live means this is correct on every call, including after a hypothetical future recreate that
    reassigns a different subnet.
    """
    net = client.networks.get(EDGE_NETWORK)
    return net.attrs["IPAM"]["Config"][0]["Subnet"]


def _allowed_source_cidrs(client) -> str:  # client: docker.DockerClient
    """The source subnets shimpz-caddy serves — everything else is 403'd (item 1's lateral-bypass gate).

    `edge` is where the public caller (cloudflared) lives; `caddy_net` is the brain's ONLY path to the
    shimpz-publish health gate now that item 8 took the brain off `edge`. Both are 2-member internal nets
    (no app is on either), so a forged-Host request from an app's own network still can't match either.
    Returns a space-joined list for Caddy's `remote_ip`. caddy_net is optional (a pre-item-8 stack).
    """
    cidrs = [_edge_subnet(client)]
    caddy = client.networks.list(names=[CADDY_NETWORK])
    if caddy:
        cidrs.append(caddy[0].attrs["IPAM"]["Config"][0]["Subnet"])
    return " ".join(cidrs)


def render(req: RouteRequest, allowed_cidrs: str) -> str:
    """Render one site file.

    Wrapped in an explicit `route {}` block on purpose: Caddyfile's automatic directive-sorting
    reorders `respond`/`handle`/etc. by a FIXED built-in precedence list, NOT by source order —
    confirmed empirically (a `respond @not_edge 403` written BEFORE an unconditional `handle {}`
    still let every request through, because the adapter silently moved the unconditional handle
    first). `route {}` is Caddy's own escape hatch: directives inside it execute in the EXACT order
    written, which is what makes the 403 gate actually gate anything.
    """
    lines = [
        f"http://{req.fqdn}:8080 {{",
        "\troute {",
        *_security_header_lines(),
        "",
        f"\t\t@not_edge not remote_ip {allowed_cidrs}",
        "\t\trespond @not_edge 403",
        "",
    ]
    if req.ws_port and req.ws_target:
        lines.append("\t\t@ws path /ws")
        lines.append(f"\t\treverse_proxy @ws {req.ws_target}:{req.ws_port}")
        lines.append("")
    if req.api_port and req.api_target:
        lines.append("\t\t@api path /api/*")
        lines.append("\t\turi @api strip_prefix /api")
        lines.append(f"\t\treverse_proxy @api {req.api_target}:{req.api_port}")
        lines.append("")
    lines.append(f"\t\treverse_proxy {req.web_target}:{req.web_port}")
    lines.append("\t}\n}")
    return "\n".join(lines) + "\n"


def _reload(client) -> None:  # client: docker.DockerClient (untyped: no static-typing gate in this repo)
    caddy = client.containers.get(CADDY_CONTAINER)
    rc, out = caddy.exec_run(
        ["caddy", "reload", "--config", "/etc/caddy/Caddyfile", "--adapter", "caddyfile", "--address", "localhost:2019"]
    )
    if rc != 0:
        raise RuntimeError(f"caddy reload failed (rc={rc}): {out.decode(errors='replace')}")


def apply_route(client, req: RouteRequest) -> None:  # client: docker.DockerClient
    SITES_DIR.mkdir(parents=True, exist_ok=True)
    path = SITES_DIR / f"{_safe_filename(req.fqdn)}.caddy"
    path.write_text(render(req, _allowed_source_cidrs(client)))
    _reload(client)


def remove_route(client, fqdn: str) -> None:  # client: docker.DockerClient
    path = SITES_DIR / f"{_safe_filename(fqdn)}.caddy"
    path.unlink(missing_ok=True)
    _reload(client)


def list_routes() -> list[str]:
    if not SITES_DIR.is_dir():
        return []
    return sorted(p.stem for p in SITES_DIR.glob("*.caddy"))
