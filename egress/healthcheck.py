#!/usr/local/bin/python3
"""Docker HEALTHCHECK probe: the proxy is up AND processing requests correctly.

Same philosophy as drivers/cf/healthcheck.py — prove a control is live, not just that the port is
open. A plain (non-CONNECT) request must be refused with 405: this proxy only tunnels CONNECT, never
forward-proxies plain HTTP (which would be an `http://` exfil path). That check is independent of the
host allowlist, so it holds in BOTH postures (broad `*` and a tight allowlist) — unlike a host-based
probe, which a `*` allowlist would let through.
"""

import socket
import sys

try:
    sock = socket.create_connection(("127.0.0.1", 8888), timeout=3)
    sock.sendall(b"GET / HTTP/1.1\r\nHost: healthcheck\r\n\r\n")
    resp = sock.recv(128)
    sock.close()
except OSError:
    sys.exit(1)
else:
    sys.exit(0 if b" 405 " in resp else 1)  # 405 = CONNECT-only enforcement is live
