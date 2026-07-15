"""The local, single-scope bearer token `shimpz-brain` uses to call this driver.

Generated once on first boot, on a volume shared only between `shimpz-brain` and this sidecar; never stored in .env.
"""

from __future__ import annotations

import grp
import os
import re
import secrets
import stat
from pathlib import Path

TOKEN_PATH = Path(os.environ.get("SHIMPZ_R2DRIVER_TOKEN_FILE", "/run/shimpz-r2driver/token"))
TOKEN_GROUP = os.environ.get("SHIMPZ_R2DRIVER_TOKEN_GROUP", "shimpzr2driver-token")
_PRIVATE_TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")


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


def _ensure_secure_token(path: Path, *, mode: int, parent_mode: int, group: str | None) -> str:
    owner = os.geteuid()
    group_id = os.getegid() if group is None else grp.getgrnam(group).gr_gid
    path.parent.mkdir(parents=True, exist_ok=True, mode=parent_mode)
    parent_info = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or parent_info.st_uid != owner
        or stat.S_IMODE(parent_info.st_mode) != parent_mode
        or (group is not None and parent_info.st_gid != group_id)
    ):
        raise RuntimeError(f"unsafe capability token directory: {path.parent}")

    read_flags = os.O_RDONLY | os.O_CLOEXEC
    create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        read_flags |= os.O_NOFOLLOW
        create_flags |= os.O_NOFOLLOW

    def read_existing() -> str:
        try:
            fd = os.open(path, read_flags)
        except OSError as exc:
            raise RuntimeError(f"cannot securely open capability token: {path}") from exc
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != owner
                or info.st_gid != group_id
                or stat.S_IMODE(info.st_mode) != mode
                or info.st_nlink != 1
                or info.st_size != 64
            ):
                raise RuntimeError(f"unsafe capability token file: {path}")
            raw_token = os.read(fd, 65)
            after = os.fstat(fd)
            stable = (
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
                info.st_mode,
                info.st_uid,
                info.st_gid,
                info.st_nlink,
            ) == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_mode,
                after.st_uid,
                after.st_gid,
                after.st_nlink,
            )
            if not stable or len(raw_token) != 64:
                raise RuntimeError(f"capability token changed while it was read: {path}")
            try:
                token = raw_token.decode("ascii")
            except UnicodeDecodeError as exc:
                raise RuntimeError(f"invalid capability token contents: {path}") from exc
        finally:
            if fd >= 0:
                os.close(fd)
        if not _PRIVATE_TOKEN_RE.fullmatch(token):
            raise RuntimeError(f"invalid capability token contents: {path}")
        return token

    token = secrets.token_hex(32)
    try:
        fd = os.open(path, create_flags, mode)
    except FileExistsError:
        return read_existing()
    except OSError as exc:
        raise RuntimeError(f"cannot securely create capability token: {path}") from exc
    try:
        os.fchown(fd, -1, group_id)
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w") as stream:
            fd = -1
            stream.write(token)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return token


def ensure_private_token(path: Path) -> str:
    """Atomically create or securely read a service-private 0400 capability."""
    return _ensure_secure_token(path, mode=0o400, parent_mode=0o700, group=None)


def ensure_group_token(path: Path, group: str) -> str:
    """Create a 0440 capability readable only by its owner and one dedicated group."""
    return _ensure_secure_token(path, mode=0o440, parent_mode=0o750, group=group)
