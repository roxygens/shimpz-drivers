"""Persistent bearer token for the local Admin-to-controller boundary."""

from __future__ import annotations

import grp
import os
import secrets
import stat
from pathlib import Path

TOKEN_PATH = Path("/run/shimpz-local/token")
LOCAL_ACCESS_GROUP = "shimpzteamdriver-token"
TOKEN_BYTES = 32


def _read_checked(path: Path, expected_gid: int) -> str:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.geteuid()
        or metadata.st_gid != expected_gid
        or stat.S_IMODE(metadata.st_mode) != 0o440
        or metadata.st_size != TOKEN_BYTES * 2
    ):
        raise RuntimeError("the local controller token has unsafe metadata")
    token = path.read_text(encoding="ascii")
    if len(token) != TOKEN_BYTES * 2:
        raise RuntimeError("the local controller token is invalid")
    try:
        bytes.fromhex(token)
    except ValueError as exc:
        raise RuntimeError("the local controller token is invalid") from exc
    return token


def ensure_token(path: Path = TOKEN_PATH) -> str:
    """Create the token once, then fail closed on metadata drift."""
    expected_gid = grp.getgrnam(LOCAL_ACCESS_GROUP).gr_gid
    if path.exists() or path.is_symlink():
        return _read_checked(path, expected_gid)

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        token = secrets.token_hex(TOKEN_BYTES)
        os.write(descriptor, token.encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        # The setgid token directory assigns the Admin-readable group at create time.
        # This remains valid when the controller itself drops every Linux capability.
        if temporary.stat().st_gid != expected_gid:
            raise RuntimeError("the local controller token directory has unsafe ownership")
        temporary.chmod(0o440)
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _read_checked(path, expected_gid)
