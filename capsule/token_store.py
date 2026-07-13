"""The local, single-scope bearer token the control plane uses to call capsule-driver.

Generated once on first boot, on a volume shared only between the caller (the admin panel) and this
sidecar; never stored in .env. Same 0440 + shared-group scheme as every other driver's token.
"""

from __future__ import annotations

import grp
import os
import secrets
from pathlib import Path

TOKEN_PATH = Path(os.environ.get("SHIMPZ_CAPSULEDRIVER_TOKEN_FILE", "/run/shimpz-capsuledriver/token"))
# Group the CALLER (the admin panel's uid 1000 — not this driver's own uid 10001) is a member of, so it
# can read the token without owning it (a 0400 owner-only token was unreadable by the caller).
TOKEN_GROUP = os.environ.get("SHIMPZ_CAPSULEDRIVER_TOKEN_GROUP", "shimpzcapsuledriver-token")


def _group_readable(path: Path) -> None:
    """Enforce 0440 + TOKEN_GROUP on `path`, every time — idempotent and self-healing."""
    gid = grp.getgrnam(TOKEN_GROUP).gr_gid
    os.chown(path, -1, gid)
    path.chmod(0o440)


def ensure_token() -> str:
    if TOKEN_PATH.exists():
        token = TOKEN_PATH.read_text().strip()
        if token:
            _group_readable(TOKEN_PATH)
            return token
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(32)
    TOKEN_PATH.write_text(token)
    _group_readable(TOKEN_PATH)
    return token
