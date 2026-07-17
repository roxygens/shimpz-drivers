"""The local, single-scope bearer token protecting this private driver.

Generated once on first boot and stored only in the driver token volume, never in ``.env``. No
Assistant Power currently receives it.
"""

from __future__ import annotations

import grp
import os
import secrets
from pathlib import Path

TOKEN_PATH = Path(os.environ.get("SHIMPZ_OPENAIDRIVER_TOKEN_FILE", "/run/shimpz-openaidriver/token"))
TOKEN_GROUP = os.environ.get("SHIMPZ_OPENAIDRIVER_TOKEN_GROUP", "shimpzopenaidriver-token")


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
