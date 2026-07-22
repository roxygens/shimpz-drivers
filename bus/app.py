#!/opt/venv/bin/python
"""bus-driver — the ONLY container that holds SHIMPZ_BUS_ADMIN_* (the Redpanda cluster superuser).

SECURITY_ENGINEERING_PLAN.md item 2: `shimpz-brain` never sees the cluster superuser credential; it calls
this restricted, allowlisted, audited HTTP API instead — the same pattern pg-driver already
proved. Every endpoint is one SPECIFIC operation with a fixed request shape
(validate.py), mirroring shimpz-bus's own existing admin-authenticated subcommands exactly
(health/topics/create/produce/tail/provision) — this is a credential relocation, not a new
capability. shimpz-bus's OTHER subcommands (new-worker/services/discover) never touched this
credential and are unaffected — they already run as a PROJECT's own least-privilege SHIMPZ_BUS_SASL_*
identity via the shimpzbus library.

Endpoints (all require `Authorization: Bearer <token>` — see token_store.py):
  GET    /v1/bus/health                       -> {brokers, topics, bootstrap}
  GET    /v1/bus/topics                       -> {topics: [...]}
  POST   /v1/bus/topics/create   {topic, partitions} -> {created, topic}
  POST   /v1/bus/produce         {topic, payload, key} -> {published, topic}
  GET    /v1/bus/tail?topic=<t>&n=<N>         -> {messages: [...], count}
  POST   /v1/bus/provision       {project}    -> {username, password, mechanism, topic_prefix}
  POST   /v1/bus/grant           {consumer, topic} -> {granted, principal, topic, operations}
"""

from __future__ import annotations

import ipaddress
import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import audit
import bus_client
import token_store
import validate

LISTEN_PORT = int(os.environ.get("SHIMPZ_BUSDRIVER_PORT", "7073"))

_token = token_store.ensure_token()


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _bus_health() -> dict:
    return bus_client.health()


def _bus_topics() -> dict:
    return {"topics": bus_client.topics()}


def _bus_topics_create(body: dict) -> dict:
    topic = validate.validate_topic(body.get("topic"))
    partitions = validate.validate_partitions(body.get("partitions", 1))
    return bus_client.create_topic(topic, partitions)


def _bus_produce(body: dict) -> dict:
    topic = validate.validate_topic(body.get("topic"))
    payload = body.get("payload")
    if not isinstance(payload, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST, "payload must be a JSON object")
    key = body.get("key")
    if key is not None and not isinstance(key, str):
        raise ApiError(HTTPStatus.BAD_REQUEST, "key must be a string")
    return bus_client.produce(topic, payload, key)


def _bus_tail(topic: str, n: str) -> dict:
    topic = validate.validate_topic(topic)
    try:
        n_int = int(n) if n else 10
    except ValueError as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"n must be an integer: {n!r}") from exc
    n_int = validate.validate_tail_n(n_int)
    return bus_client.tail(topic, n_int)


def _bus_provision(body: dict) -> dict:
    project = validate.validate_project(body.get("project"))
    return bus_client.provision(project)


def _bus_grant(body: dict) -> dict:
    consumer, topic = validate.validate_grant(body.get("consumer"), body.get("topic"))
    return bus_client.grant_consume(consumer, topic)


class Handler(BaseHTTPRequestHandler):
    server_version = "bus-driver/1.0"

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
        except bus_client.BusError as exc:
            audit.log(method.lower(), self.path, result="error", reason=str(exc))
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
        except Exception as exc:
            audit.log(method.lower(), self.path, result="error", reason=str(exc))
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _route(self, method: str) -> None:
        split = urlsplit(self.path)
        path, query = split.path, parse_qs(split.query)

        if method == "GET" and path == "/v1/bus/health":
            result = _bus_health()
            self._send_json(HTTPStatus.OK, result)
            return
        if method == "GET" and path == "/v1/bus/topics":
            result = _bus_topics()
            self._send_json(HTTPStatus.OK, result)
            return
        if method == "POST" and path == "/v1/bus/topics/create":
            body = self._body()
            result = _bus_topics_create(body)
            trace = audit.log("topics.create", body.get("topic", "?"), result="ok", **result)
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        if method == "POST" and path == "/v1/bus/produce":
            body = self._body()
            result = _bus_produce(body)
            trace = audit.log("produce", body.get("topic", "?"), result="ok", key=body.get("key"))
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        if method == "GET" and path == "/v1/bus/tail":
            topic = query.get("topic", [""])[0]
            result = _bus_tail(topic, query.get("n", [""])[0])
            trace = audit.log("tail", topic, result="ok", count=result["count"])
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        if method == "POST" and path == "/v1/bus/provision":
            body = self._body()
            result = _bus_provision(body)
            # never log the derived SASL password itself
            trace = audit.log("provision", body.get("project", "?"), result="ok", username=result["username"])
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        if method == "POST" and path == "/v1/bus/grant":
            body = self._body()
            result = _bus_grant(body)
            trace = audit.log(
                "grant", result["topic"], result="ok", principal=result["principal"], consumer=body.get("consumer")
            )
            self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})
            return
        raise ApiError(HTTPStatus.NOT_FOUND, f"no route for {method} {path}")

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def log_message(self, fmt: str, *args: object) -> None:
        # Suppress BaseHTTPRequestHandler's default stderr access log — audit.log() is the
        # single source of truth for what happened, in the schema logq expects.
        pass


def main() -> None:
    server = ThreadingHTTPServer((str(ipaddress.IPv4Address(0)), LISTEN_PORT), Handler)
    print(f"bus-driver listening on :{LISTEN_PORT}", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
