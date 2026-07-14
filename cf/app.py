#!/opt/venv/bin/python
"""cf-driver — the ONLY container that holds SHIMPZ_CF_TOKEN/SHIMPZ_CF_ACCOUNT.

SECURITY_ENGINEERING_PLAN.md item 3: `shimpz-brain` never sees the Cloudflare token; it calls this
restricted, allowlisted, audited HTTP API instead. Every endpoint is one SPECIFIC operation
with a fixed request shape (validate.py) — never a generic "method+path" passthrough the way
the old `cf` helper worked, so a compromised `shimpz-brain` can only ever ask for one of these named
operations, never rewrite DNS/Tunnel/Access arbitrarily.

Endpoints (all require `Authorization: Bearer <token>` — see token_store.py):
  GET    /v1/zones/resolve?fqdn=<fqdn>            -> {zone_name, zone_id}
  GET    /v1/tunnel                               -> {tunnel_id}
  POST   /v1/tunnel/ingress-rule                  {hostname, service} -> {previous_service}
  DELETE /v1/tunnel/ingress-rule/<hostname>        -> {removed, previous_service}
  POST   /v1/dns/upsert                           {fqdn, type, content} -> {record_id, created, previous_content, zone}
  DELETE /v1/dns/record?fqdn=<fqdn>&type=<type>    -> {deleted}
  GET    /v1/access/apps?fqdn=<fqdn>              -> [app, ...]
  POST   /v1/access/private                       {fqdn, owner_email} -> {created, app_id}
  POST   /v1/access/public                        {fqdn} -> {removed: [app, ...]}
  POST   /v1/access/restore                       {app} -> {app_id}
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import audit
import cf_client
import token_store
import validate

LISTEN_PORT = int(os.environ.get("SHIMPZ_CFDRIVER_PORT", "7071"))
_INGRESS_RULE_ROUTE = re.compile(r"^/v1/tunnel/ingress-rule/([^/]+)$")

_token = token_store.ensure_token()


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _resolve_zone(fqdn: str) -> tuple[str, str]:
    match = validate.longest_matching_zone(fqdn, cf_client.list_zones())
    if match is None:
        raise ApiError(HTTPStatus.NOT_FOUND, f"no Cloudflare zone matches {fqdn!r}")
    return match


def _zones_resolve(fqdn: str) -> dict:
    zone_name, zone_id = _resolve_zone(validate.validate_fqdn(fqdn))
    return {"zone_name": zone_name, "zone_id": zone_id}


def _tunnel() -> dict:
    tunnel_id = cf_client.active_tunnel_id()
    if tunnel_id is None:
        raise ApiError(HTTPStatus.NOT_FOUND, "no active cfd tunnel")
    return {"tunnel_id": tunnel_id}


def _ingress_upsert(body: dict) -> dict:
    hostname = validate.validate_fqdn(body.get("hostname"))
    service = validate.validate_hostname_service(body.get("service"))
    tunnel_id = cf_client.active_tunnel_id()
    if tunnel_id is None:
        raise ApiError(HTTPStatus.NOT_FOUND, "no active cfd tunnel")
    config = cf_client.get_tunnel_ingress(tunnel_id)
    ingress = config.get("ingress", [])
    previous = next((r.get("service") for r in ingress if r.get("hostname") == hostname), None)
    named = [r for r in ingress if r.get("hostname") not in (None, hostname)]
    catch_all = [r for r in ingress if r.get("hostname") is None]
    new_ingress = named + [{"hostname": hostname, "service": service}] + (catch_all or [{"service": "http_status:404"}])
    cf_client.put_tunnel_ingress(tunnel_id, {**config, "ingress": new_ingress})
    return {"previous_service": previous, "rule_count": len(new_ingress)}


def _ingress_delete(hostname: str) -> dict:
    hostname = validate.validate_fqdn(hostname)
    tunnel_id = cf_client.active_tunnel_id()
    if tunnel_id is None:
        raise ApiError(HTTPStatus.NOT_FOUND, "no active cfd tunnel")
    config = cf_client.get_tunnel_ingress(tunnel_id)
    ingress = config.get("ingress", [])
    previous = next((r.get("service") for r in ingress if r.get("hostname") == hostname), None)
    if previous is None:
        return {"removed": False, "previous_service": None}
    new_ingress = [r for r in ingress if r.get("hostname") != hostname]
    cf_client.put_tunnel_ingress(tunnel_id, {**config, "ingress": new_ingress})
    return {"removed": True, "previous_service": previous}


def _dns_upsert(body: dict) -> dict:
    fqdn = validate.validate_fqdn(body.get("fqdn"))
    record_type = validate.validate_dns_type(body.get("type"))
    content = body.get("content")
    if not isinstance(content, str) or not content:
        raise ApiError(HTTPStatus.BAD_REQUEST, "content must be a non-empty string")
    zone_name, zone_id = _resolve_zone(fqdn)
    existing = cf_client.find_dns_record(zone_id, fqdn, record_type)
    if existing:
        previous_content = existing["content"]
        cf_client.update_dns_record(zone_id, existing["id"], fqdn, record_type, content)
        return {"record_id": existing["id"], "created": False, "previous_content": previous_content, "zone": zone_name}
    created = cf_client.create_dns_record(zone_id, fqdn, record_type, content)
    return {"record_id": created["id"], "created": True, "previous_content": None, "zone": zone_name}


def _dns_delete(fqdn: str, record_type: str) -> dict:
    fqdn = validate.validate_fqdn(fqdn)
    record_type = validate.validate_dns_type(record_type)
    match = validate.longest_matching_zone(fqdn, cf_client.list_zones())
    if match is None:
        return {"deleted": 0}
    _, zone_id = match
    existing = cf_client.find_dns_record(zone_id, fqdn, record_type)
    if existing is None:
        return {"deleted": 0}
    cf_client.delete_dns_record(zone_id, existing["id"])
    return {"deleted": 1}


def _access_list(fqdn: str) -> list:
    return cf_client.list_access_apps_for_domain(validate.validate_fqdn(fqdn))


def _access_private(body: dict) -> dict:
    fqdn = validate.validate_fqdn(body.get("fqdn"))
    owner_email = validate.validate_email(body.get("owner_email"))
    existing = cf_client.list_access_apps_for_domain(fqdn)
    if existing:
        return {"created": False, "app_id": existing[0]["id"]}
    app_body = {
        "name": f"Shimpz {fqdn}",
        "domain": fqdn,
        "type": "self_hosted",
        "session_duration": "24h",
        "policies": [{"name": "allow-owner", "decision": "allow", "include": [{"email": {"email": owner_email}}]}],
    }
    created = cf_client.create_access_app(app_body)
    return {"created": True, "app_id": created["id"]}


def _access_public(body: dict) -> dict:
    fqdn = validate.validate_fqdn(body.get("fqdn"))
    existing = cf_client.list_access_apps_for_domain(fqdn)
    removed = []
    for app in existing:
        cf_client.delete_access_app(app["id"])
        removed.append(app)
    return {"removed": removed}


def _access_restore(body: dict) -> dict:
    app_body = validate.validate_access_app_body(body.get("app"))
    created = cf_client.create_access_app(app_body)
    return {"app_id": created["id"]}


class Handler(BaseHTTPRequestHandler):
    server_version = "cf-driver/1.0"

    def _authed(self) -> bool:
        return self.headers.get("Authorization", "") == f"Bearer {_token}"

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
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
        except cf_client.CFError as exc:
            audit.log(method.lower(), self.path, result="error", reason=str(exc))
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
        except Exception as exc:
            audit.log(method.lower(), self.path, result="error", reason=str(exc))
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _route(self, method: str) -> None:
        split = urlsplit(self.path)
        path, query = split.path, parse_qs(split.query)
        trace = None

        if method == "GET" and path == "/v1/zones/resolve":
            result = _zones_resolve(query.get("fqdn", [""])[0])
            trace = audit.log("zones.resolve", query.get("fqdn", [""])[0], result="ok")
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        if method == "GET" and path == "/v1/tunnel":
            result = _tunnel()
            self._send_json(HTTPStatus.OK, result)
            return
        if method == "POST" and path == "/v1/tunnel/ingress-rule":
            body = self._body()
            result = _ingress_upsert(body)
            trace = audit.log("tunnel.ingress-rule.upsert", body.get("hostname", "?"), result="ok", **result)
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        if method == "DELETE" and (m := _INGRESS_RULE_ROUTE.match(path)):
            result = _ingress_delete(m.group(1))
            trace = audit.log("tunnel.ingress-rule.delete", m.group(1), result="ok", **result)
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        if method == "POST" and path == "/v1/dns/upsert":
            body = self._body()
            result = _dns_upsert(body)
            audit_fields = {k: v for k, v in result.items() if k != "previous_content"}
            trace = audit.log("dns.upsert", body.get("fqdn", "?"), result="ok", **audit_fields)
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        if method == "DELETE" and path == "/v1/dns/record":
            fqdn = query.get("fqdn", [""])[0]
            result = _dns_delete(fqdn, query.get("type", [""])[0])
            trace = audit.log("dns.delete", fqdn, result="ok", **result)
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        if method == "GET" and path == "/v1/access/apps":
            result = _access_list(query.get("fqdn", [""])[0])
            self._send_json(HTTPStatus.OK, result)
            return
        if method == "POST" and path == "/v1/access/private":
            body = self._body()
            result = _access_private(body)
            trace = audit.log("access.private", body.get("fqdn", "?"), result="ok", **result)
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        if method == "POST" and path == "/v1/access/public":
            body = self._body()
            result = _access_public(body)
            trace = audit.log("access.public", body.get("fqdn", "?"), result="ok", removed_count=len(result["removed"]))
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        if method == "POST" and path == "/v1/access/restore":
            body = self._body()
            result = _access_restore(body)
            trace = audit.log("access.restore", (body.get("app") or {}).get("domain", "?"), result="ok", **result)
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
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


def main() -> None:
    server = ThreadingHTTPServer((str(ipaddress.IPv4Address(0)), LISTEN_PORT), Handler)
    print(f"cf-driver listening on :{LISTEN_PORT}", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
