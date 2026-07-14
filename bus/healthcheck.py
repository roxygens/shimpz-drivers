#!/usr/local/bin/python3
"""Docker HEALTHCHECK probe: the server is up and its auth gate is live.

An unauthenticated GET must be refused with 403 (see app.py's bearer-token check) — same proof
the other drivers' own dedicated healthcheck makes for their endpoints.
"""

import sys
import urllib.error
import urllib.request

try:
    urllib.request.urlopen("http://127.0.0.1:7073/v1/bus/topics", timeout=3)
except urllib.error.HTTPError as exc:
    sys.exit(0 if exc.code == 403 else 1)
except OSError, ValueError:
    sys.exit(1)
else:
    sys.exit(1)  # a 2xx with no auth would mean the bearer-token gate isn't enforced at all
