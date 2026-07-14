#!/usr/local/bin/python3
"""Docker HEALTHCHECK probe: the app-egress proxy is up AND enforcing correctly.

Same philosophy as drivers/egress/healthcheck.py — prove a control is live, not just that the port
is open. A plain (non-CONNECT) request must be refused with 405: this proxy only tunnels CONNECT, never
forward-proxies plain HTTP (an `http://` exfil path). That check is independent of any app's allowlist,
so it holds regardless of policy — unlike a host probe, which would need a real token + allowlist.
"""

import socket
import sys

try:
    sock = socket.create_connection(("127.0.0.1", 8889), timeout=3)
    sock.sendall(b"GET / HTTP/1.1\r\nHost: healthcheck\r\n\r\n")
    resp = sock.recv(128)
    sock.close()
except OSError:
    sys.exit(1)
else:
    sys.exit(0 if b" 405 " in resp else 1)  # 405 = CONNECT-only enforcement is live
