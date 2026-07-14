"""The ONLY place SHIMPZ_CF_TOKEN is ever read or sent.

A thin wrapper over the real Cloudflare API v4, called ONLY by app.py's already-allowlisted
(validate.py) endpoint handlers. Never exposes a generic "method+path" call the way the old `cf`
helper did (SECURITY_ENGINEERING_PLAN.md item 3): every function here is one SPECIFIC Cloudflare
operation with a fixed shape.

stdlib-only (urllib) on purpose — this sidecar's whole job is holding one credential and making a
short list of specific calls; no reason to add a dependency surface for it.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

BASE = "https://api.cloudflare.com/client/v4"
TOKEN = os.environ.get("SHIMPZ_CF_TOKEN", "")
ACCOUNT = os.environ.get("SHIMPZ_CF_ACCOUNT", "")


class CFError(Exception):
    """Cloudflare returned success=false, or the HTTP call itself failed."""


def _call(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            parsed = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            parsed = json.loads(exc.read())
        # PEP 758 (Python 3.14, this repo's pinned version): a comma-separated except list needs no
        # parens — equivalent to `except (json.JSONDecodeError, ValueError):`. `ruff format` (this
        # repo's canonical formatter) itself rewrites a parenthesized tuple into exactly this shape.
        except json.JSONDecodeError, ValueError:
            raise CFError(f"{method} {path} -> HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise CFError(f"{method} {path} -> unreachable: {exc.reason}") from exc
    if not parsed.get("success"):
        raise CFError(f"{method} {path} -> {parsed.get('errors')}")
    return parsed


def list_zones() -> list[dict]:
    return _call("GET", "/zones?per_page=50")["result"]


def active_tunnel_id() -> str | None:
    """Return this Space's tunnel.

    SHIMPZ_CF_TUNNEL_ID pins it explicitly — REQUIRED on a host that hosts more
    than one tunnel (the account's `cfd_tunnel` list order is arbitrary, so 'the first' can be a SIBLING
    Space's tunnel, which is how a publish once landed on the wrong tunnel). Unset → the first (single
    -tunnel host, unchanged).
    """
    pinned = os.environ.get("SHIMPZ_CF_TUNNEL_ID", "").strip()
    if pinned:
        return pinned
    result = _call("GET", f"/accounts/{ACCOUNT}/cfd_tunnel?is_deleted=false")["result"]
    return result[0]["id"] if result else None


def get_tunnel_ingress(tunnel_id: str) -> dict:
    return _call("GET", f"/accounts/{ACCOUNT}/cfd_tunnel/{tunnel_id}/configurations")["result"]["config"]


def put_tunnel_ingress(tunnel_id: str, config: dict) -> None:
    _call("PUT", f"/accounts/{ACCOUNT}/cfd_tunnel/{tunnel_id}/configurations", {"config": config})


def find_dns_record(zone_id: str, fqdn: str, record_type: str) -> dict | None:
    result = _call("GET", f"/zones/{zone_id}/dns_records?name={fqdn}&type={record_type}")["result"]
    return result[0] if result else None


def create_dns_record(zone_id: str, fqdn: str, record_type: str, content: str) -> dict:
    body = {"type": record_type, "name": fqdn, "content": content, "proxied": True, "ttl": 1}
    return _call("POST", f"/zones/{zone_id}/dns_records", body)["result"]


def update_dns_record(zone_id: str, record_id: str, fqdn: str, record_type: str, content: str) -> None:
    # Cloudflare's DNS record PUT is a full replace, not a partial patch — confirmed against the
    # real API by the original `cf PUT .../dns_records/<id>` calls this replaces, which always
    # sent the complete {type,name,content,proxied,ttl} body, never just the changed field.
    body = {"type": record_type, "name": fqdn, "content": content, "proxied": True, "ttl": 1}
    _call("PUT", f"/zones/{zone_id}/dns_records/{record_id}", body)


def delete_dns_record(zone_id: str, record_id: str) -> None:
    _call("DELETE", f"/zones/{zone_id}/dns_records/{record_id}")


def list_access_apps_for_domain(fqdn: str) -> list[dict]:
    result = _call("GET", f"/accounts/{ACCOUNT}/access/apps?per_page=100")["result"]
    return [a for a in result if a.get("domain") == fqdn]


def create_access_app(body: dict) -> dict:
    return _call("POST", f"/accounts/{ACCOUNT}/access/apps", body)["result"]


def delete_access_app(app_id: str) -> None:
    _call("DELETE", f"/accounts/{ACCOUNT}/access/apps/{app_id}")
