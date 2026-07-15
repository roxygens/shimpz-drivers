#!/usr/local/bin/python3
"""Docker HEALTHCHECK probe: private state is trusted and the operational auth gate is live.

No `curl` in this image on purpose — an unauthenticated operational GET must be refused with 403
(see app.py's bearer-token check). Read-only manifest discovery remains intentionally unauthenticated.
"""

import http.client
import sys


def probe(path: str) -> int:
    connection = http.client.HTTPConnection("127.0.0.1", 7075, timeout=3)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        response.read()
        return response.status
    finally:
        connection.close()


def main() -> int:
    try:
        healthy = probe("/healthz") == 200
        protected = probe("/v1/r2/list") == 403
    except OSError, http.client.HTTPException:
        return 1
    return 0 if healthy and protected else 1


if __name__ == "__main__":
    sys.exit(main())
