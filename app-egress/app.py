#!/usr/local/bin/python3
"""app-egress-proxy — the ONLY internet route for DEPLOYED APPS (Shimpz L2, deny-by-default egress).

Every app container is on an `internal:true` network (no NAT to the internet). Its ONLY egress is
`HTTPS_PROXY=http://<app-egress-token>@app-egress-proxy:8889`, reached over an internal proxy network.
Unlike the brain's single-tenant egress-proxy (one global allowlist, network-gated), this proxy serves
MANY apps, so it is PER-APP TOKEN-GATED: each app carries its own token (issued by shimpz-driver at
deploy) and the proxy forwards a CONNECT only to the hosts THAT app declared in `[needs].egress` (plus
`pay.shimpz.com` iff the app is paid — its `effective_egress`). The driver writes each app's
allowlist to the policy dir as `<token>.json`.

Deny-by-default and fail-closed: an unknown token, an app that declared no egress, an unlisted host, a
non-:443 port, or this process being down all mean the app reaches NOTHING external. Same CONNECT-only,
opaque-TLS, hostname-allowlist design as the brain proxy (no CA injection, no plaintext seen, DNS-tunnel
exfil closed because the app has no default route and can't resolve external names itself).
"""

from __future__ import annotations

import base64
import contextlib
import ipaddress
import json
import os
import select
import socket
import socketserver
import sys
import threading
from pathlib import Path

import audit

LISTEN_PORT = int(os.environ.get("SHIMPZ_APP_EGRESS_PORT", "8889"))
POLICY_DIR = Path(os.environ.get("SHIMPZ_APP_EGRESS_POLICY_DIR", "/policy"))
ALLOWED_PORTS = {443}  # HTTPS only — every legitimate app destination is TLS
CONNECT_TIMEOUT = 15
IDLE_TIMEOUT = 300
BUFSIZE = 65536
MAX_CONCURRENCY = int(os.environ.get("SHIMPZ_APP_EGRESS_MAX_CONCURRENCY", "64"))
MAX_SOURCE_CONCURRENCY = int(os.environ.get("SHIMPZ_APP_EGRESS_MAX_SOURCE_CONCURRENCY", "8"))
LISTEN_BACKLOG = int(os.environ.get("SHIMPZ_APP_EGRESS_LISTEN_BACKLOG", "16"))
if (
    not 1 <= MAX_CONCURRENCY <= 64
    or not 1 <= MAX_SOURCE_CONCURRENCY <= 8
    or MAX_SOURCE_CONCURRENCY > MAX_CONCURRENCY
    or not 1 <= LISTEN_BACKLOG <= 16
):
    raise ValueError("app egress proxy concurrency/backlog must stay inside the shipping resource envelope")
_STATUS = {
    200: "Connection established",
    400: "Bad Request",
    403: "Forbidden",
    405: "Method Not Allowed",
    407: "Proxy Authentication Required",
    502: "Bad Gateway",
    503: "Service Unavailable",
}


def load_policy(policy_dir: Path) -> dict[str, frozenset[str]]:
    """Read the per-app allowlists the driver wrote: `<token>.json` = a JSON list of hostnames.

    Returns {token: frozenset(lowercased hosts)}. A missing dir → {} (deny everything — fail-closed). An
    unreadable/garbage file is SKIPPED (that app gets no egress) rather than crashing the proxy for others.
    """
    policy: dict[str, frozenset[str]] = {}
    if not policy_dir.is_dir():
        return policy
    for f in policy_dir.glob("*.json"):
        token = f.stem
        try:
            hosts = json.loads(f.read_text(encoding="utf-8"))
        except OSError, ValueError:
            continue  # a bad policy file denies that app's egress — never opens another's
        if isinstance(hosts, list) and all(isinstance(h, str) for h in hosts):
            policy[token] = frozenset(h.lower().rstrip(".") for h in hosts)
    return policy


def permitted(token: str, host: str, port: int, policy: dict[str, frozenset[str]]) -> bool:
    """Forward a CONNECT to host:port iff the app (identified by `token`) declared exactly this host.

    Deny-by-default: unknown/empty token, an app with no allowlist, an unlisted host, or a non-:443 port
    all return False. `effective_egress` entries are EXACT hostnames, so this is an exact match (no wildcard
    — an app lists every host it needs); this is the enforcement kernel of the ShimpzPay/egress lock.
    """
    if port not in ALLOWED_PORTS:
        return False
    allow = policy.get(token)
    if not allow:
        return False
    return host.lower().rstrip(".") in allow


def resolve_public(host: str, port: int) -> tuple[int, tuple] | None:
    """Resolve once and return a public sockaddr; any non-public answer denies the whole target.

    Connecting to the returned sockaddr rather than the hostname closes the resolve/connect DNS
    rebinding window. Refusing a mixed public/private answer prevents an attacker from influencing
    address ordering to pivot this multi-homed proxy into another app network or the host.
    """
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return None
    public: list[tuple[int, tuple]] = []
    for family, _socktype, _proto, _canonname, sockaddr in infos:
        try:
            address = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            return None
        if not address.is_global:
            return None
        public.append((family, sockaddr))
    return public[0] if public else None


def extract_token(headers: str) -> str | None:
    """Pull the per-app token out of a `Proxy-Authorization: Basic base64(token:)` header (or None).

    The app's HTTPS_PROXY is `http://<token>@app-egress-proxy:8889`, so clients send Basic proxy auth
    with the token as the username. We take the username half; a missing/garbled header → None (→ 407).
    """
    for line in headers.split("\r\n"):
        name, sep, value = line.partition(":")
        if sep and name.strip().lower() == "proxy-authorization":
            scheme, _, creds = value.strip().partition(" ")
            if scheme.lower() != "basic":
                return None
            try:
                decoded = base64.b64decode(creds, validate=True).decode("latin1")
            except ValueError, UnicodeDecodeError:
                return None
            return decoded.split(":", 1)[0] or None
    return None


class Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        cli = self.request
        cli.settimeout(CONNECT_TIMEOUT)
        probe = self.client_address[0] == "127.0.0.1"  # the Docker HEALTHCHECK (a deliberate denied CONNECT)
        headers = self._read_request(cli)
        if headers is None:
            return
        request_line = headers.split("\r\n", 1)[0]
        parts = request_line.split(" ")
        if len(parts) < 2 or parts[0] != "CONNECT":
            self._reply(cli, 405)
            audit.log("connect", request_line[:80], result="denied", level="info" if probe else "warn", code=405)
            return
        host, port = self._split_target(parts[1])
        if host is None:
            self._reply(cli, 400)
            audit.log("connect", parts[1][:80], result="denied", level="info" if probe else "warn", code=400)
            return
        token = extract_token(headers)
        if token is None:
            self._reply(cli, 407)
            audit.log("connect", f"{host}:{port}", result="denied", level="info" if probe else "warn", code=407)
            return
        if not permitted(token, host, port, load_policy(POLICY_DIR)):
            self._reply(cli, 403)
            audit.log("connect", f"{host}:{port}", result="denied", level="warn", code=403, app=token[:12])
            return
        resolved = resolve_public(host, port)
        if resolved is None:
            self._reply(cli, 403)
            audit.log(
                "connect",
                f"{host}:{port}",
                result="denied",
                level="warn",
                code=403,
                reason="internal or unresolvable destination",
                app=token[:12],
            )
            return
        family, sockaddr = resolved
        upstream: socket.socket | None = None
        try:
            upstream = socket.socket(family, socket.SOCK_STREAM)
            upstream.settimeout(CONNECT_TIMEOUT)
            upstream.connect(sockaddr)
        except OSError as exc:
            if upstream is not None:
                upstream.close()
            self._reply(cli, 502)
            audit.log("connect", f"{host}:{port}", result="error", reason=str(exc), app=token[:12])
            return
        audit.log("connect", f"{host}:{port}", result="ok", app=token[:12])
        self._reply(cli, 200)
        self._tunnel(cli, upstream)

    @staticmethod
    def _read_request(sock: socket.socket) -> str | None:
        buf = b""
        while b"\r\n\r\n" not in buf:
            try:
                chunk = sock.recv(4096)
            except OSError:
                return None
            if not chunk:
                return None
            buf += chunk
            if len(buf) > BUFSIZE:
                return None
        return buf.split(b"\r\n\r\n", 1)[0].decode("latin1", "replace")

    @staticmethod
    def _split_target(target: str) -> tuple[str | None, int]:
        host, sep, port_s = target.rpartition(":")
        if not sep:
            return target or None, 443
        try:
            return (host or None), int(port_s)
        except ValueError:
            return None, 0

    def _reply(self, cli: socket.socket, code: int) -> None:
        with contextlib.suppress(OSError):
            extra = 'Proxy-Authenticate: Basic realm="app-egress"\r\n' if code == 407 else ""
            cli.sendall(f"HTTP/1.1 {code} {_STATUS[code]}\r\n{extra}\r\n".encode())

    @staticmethod
    def _tunnel(a: socket.socket, b: socket.socket) -> None:
        for s in (a, b):
            s.settimeout(None)
        try:
            while True:
                readable, _, errored = select.select([a, b], [], [a, b], IDLE_TIMEOUT)
                if errored or not readable:
                    return
                for src in readable:
                    data = src.recv(BUFSIZE)
                    if not data:
                        return
                    (b if src is a else a).sendall(data)
        except OSError:
            return
        finally:
            for s in (a, b):
                with contextlib.suppress(OSError):
                    s.shutdown(socket.SHUT_RDWR)
                s.close()


class Server(socketserver.ThreadingTCPServer):
    """Threaded CONNECT server with admission enforced before worker creation."""

    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = LISTEN_BACKLOG

    def __init__(
        self,
        *args,
        max_concurrency: int = MAX_CONCURRENCY,
        max_source_concurrency: int = MAX_SOURCE_CONCURRENCY,
        **kwargs,
    ) -> None:
        if not 1 <= max_source_concurrency <= max_concurrency <= MAX_CONCURRENCY:
            raise ValueError("invalid App egress proxy concurrency")
        self._request_slots = threading.BoundedSemaphore(max_concurrency)
        self._max_source_concurrency = max_source_concurrency
        self._source_guard = threading.Lock()
        self._source_counts: dict[str, int] = {}
        super().__init__(*args, **kwargs)

    def get_request(self):
        request, client_address = super().get_request()
        request.settimeout(CONNECT_TIMEOUT)
        return request, client_address

    def _acquire_request_slot(self, client_address) -> bool:
        if not self._request_slots.acquire(blocking=False):
            return False
        source = str(client_address[0])
        with self._source_guard:
            current = self._source_counts.get(source, 0)
            if current >= self._max_source_concurrency:
                self._request_slots.release()
                return False
            self._source_counts[source] = current + 1
        return True

    def _release_request_slot(self, client_address) -> None:
        source = str(client_address[0])
        with self._source_guard:
            remaining = self._source_counts[source] - 1
            if remaining:
                self._source_counts[source] = remaining
            else:
                self._source_counts.pop(source)
        self._request_slots.release()

    def process_request(self, request, client_address) -> None:
        if not self._acquire_request_slot(client_address):
            with contextlib.suppress(OSError):
                request.settimeout(1)
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n\r\n")
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._release_request_slot(client_address)
            self.shutdown_request(request)
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._release_request_slot(client_address)


def main() -> None:
    server = Server((str(ipaddress.IPv4Address(0)), LISTEN_PORT), Handler)
    print(f"app-egress-proxy listening on :{LISTEN_PORT}; policy_dir={POLICY_DIR}", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
