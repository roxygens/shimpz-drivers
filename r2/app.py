#!/usr/local/bin/python3
"""r2-driver — the ONLY container that holds the Cloudflare R2 credentials (RCLONE_CONFIG_R2_*).

SECURITY_ENGINEERING_PLAN.md item 7: `shimpz-brain` (the brain) never sees the R2 secret; it calls this
restricted, allowlisted, audited HTTP API instead. Every endpoint is one SPECIFIC operation with a
fixed request shape (validate.py) — never a generic "run rclone" passthrough. Before this split a
prompt-injected brain could `rclone delete` the whole bucket or exfiltrate the access key; now it can
only ever ask for one of: upload one file (get a presigned link), list a prefix, download one small
object. A separate loopback-only operator capability handles immutable encrypted backup upload and
bounded-range recovery without widening any Brain-facing operation.

Mandatory controls (same contract as the other sidecars):
  - Auth fails closed on every operational endpoint. Only health and definition-only discovery are
    unauthenticated on the private driver network; neither can return credential values or inventory.
  - No CORS, ever: this API is for `shimpz-brain`'s own r2send/r2ls/r2get wrappers, never a page in Chrome.
  - No execution endpoint: r2_client.py shells rclone with a FIXED argv (never a shell string), so a
    bucket key can't inject a command. There is no "arbitrary rclone" endpoint by design.
  - Redacted audit: only keys/prefixes/sizes — never file bytes, never the presigned link itself
    (a live download credential), never the R2 secret.

Streaming both directions (no base64, no shared volume): the upload body IS the raw file bytes; the
download response IS the raw file bytes — neither is ever fully buffered in memory, so a multi-GB R2
object (the whole reason R2 exists over kclient) transfers with bounded memory.

Endpoints (all require `Authorization: Bearer <token>`):
  POST /v1/r2/upload   body=<raw bytes>  headers: X-R2-Filename, X-R2-Expire? -> {key, link, size}
  POST /v1/r2/backup/upload body=<raw backup bytes> headers: X-Backup-SHA256,
       X-Backup-Created-At -> {key, sha256, size} under the immutable backups/v1 prefix
  GET  /v1/r2/backup/download ?key=<exact backup key>&offset=<n>&length=<n> -> <raw range>
  GET  /v1/r2/list     ?prefix=<prefix>  -> {prefix, entries: [{size, modtime, path}, ...]}
  GET  /v1/r2/get      ?key=<key>        -> <raw bytes>  (X-R2-Size header)

Private-network discovery (unauthenticated, definitions only):
  GET  /healthz                  -> {status}
  GET  /v1/driver                -> Driver Spec v1 manifest
  GET  /v1/driver/credentials    -> closed credential form schema, never values or inventory
"""

from __future__ import annotations

import contextlib
import hashlib
import ipaddress
import os
import re
import stat
import sys
import tempfile
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO
from urllib.parse import parse_qs, urlsplit

import audit
import backup_gate
import driver_manifest
import r2_client
import token_store
import validate

DRIVER = driver_manifest.load()
CREDENTIALS = driver_manifest.load_credentials()
LISTEN_PORT = int(os.environ.get("SHIMPZ_R2DRIVER_PORT", str(DRIVER.port)))
LISTEN_HOST = str(ipaddress.IPv4Address(0))
_CHUNK = 1024 * 1024
_BACKUP_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_BACKUP_CREATED_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_BACKUP_SPOOL_PREFIX = ".shimpz-r2backup-"
BACKUP_SPOOL_DIR = Path(os.environ.get("SHIMPZ_R2DRIVER_BACKUP_SPOOL_DIR", "/var/lib/shimpz-r2backup"))
_backup_transfer_gate = backup_gate.BackupTransferGate()


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _prepare_backup_spool() -> None:
    """Validate the ciphertext-only spool and recover exact leftovers from a crashed process."""
    if not BACKUP_SPOOL_DIR.is_absolute():
        raise RuntimeError("SHIMPZ_R2DRIVER_BACKUP_SPOOL_DIR must be absolute")
    BACKUP_SPOOL_DIR.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = BACKUP_SPOOL_DIR.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise RuntimeError(f"unsafe backup spool directory: {BACKUP_SPOOL_DIR}")
    removed = False
    for stale in BACKUP_SPOOL_DIR.glob(f"{_BACKUP_SPOOL_PREFIX}*"):
        stale_info = stale.lstat()
        if (
            not stat.S_ISREG(stale_info.st_mode)
            or stale_info.st_uid != os.geteuid()
            or stat.S_IMODE(stale_info.st_mode) != 0o600
        ):
            raise RuntimeError(f"unsafe backup spool leftover: {stale}")
        stale.unlink()
        removed = True
    if removed:
        _fsync_directory(BACKUP_SPOOL_DIR)


@contextlib.contextmanager
def _backup_spool_file():
    fd, temporary_name = tempfile.mkstemp(prefix=_BACKUP_SPOOL_PREFIX, suffix=".sbk", dir=BACKUP_SPOOL_DIR)
    temporary = Path(temporary_name)
    _fsync_directory(BACKUP_SPOOL_DIR)
    try:
        with os.fdopen(fd, "w+b") as stream:
            fd = -1
            yield stream
    finally:
        if fd >= 0:
            os.close(fd)
        temporary.unlink(missing_ok=True)
        _fsync_directory(BACKUP_SPOOL_DIR)


@contextlib.contextmanager
def _exclusive_backup_transfer():
    try:
        with _backup_transfer_gate.claim():
            yield
    except backup_gate.BackupTransferBusyError as exc:
        raise ApiError(HTTPStatus.TOO_MANY_REQUESTS, "another private backup transfer is active") from exc


_token = token_store.ensure_token()
_backup_token = token_store.ensure_private_token(
    Path(os.environ.get("SHIMPZ_R2DRIVER_BACKUP_TOKEN_FILE", "/run/shimpz-r2backup/token"))
)


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _date_key(filename: str) -> str:
    return f"uploads/{time.strftime('%Y/%m/%d', time.gmtime())}/{filename}"


def _backup_key(created_at: str, sha256: str) -> str:
    if not _BACKUP_CREATED_RE.fullmatch(created_at):
        raise ApiError(HTTPStatus.BAD_REQUEST, "X-Backup-Created-At must be an exact UTC timestamp")
    try:
        created = time.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, "X-Backup-Created-At is not a real UTC timestamp") from exc
    if not _BACKUP_SHA_RE.fullmatch(sha256):
        raise ApiError(HTTPStatus.BAD_REQUEST, "X-Backup-SHA256 must be a lowercase SHA-256 digest")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", created)
    return f"backups/v1/{time.strftime('%Y/%m/%d', created)}/{stamp}-{sha256}.sbk"


class Handler(BaseHTTPRequestHandler):
    server_version = f"{DRIVER.id}-driver/{DRIVER.version}"

    def _authed(self) -> bool:
        path = urlsplit(self.path).path
        if path in {"/v1/r2/backup/upload", "/v1/r2/backup/download"}:
            # The backup capability is exercisable only by an explicitly approved host-side
            # `docker exec`, whose HTTP hop is loopback. A network peer cannot use this operation
            # even if the private bearer is accidentally disclosed later.
            if self.client_address[0] != "127.0.0.1":
                return False
            expected = _backup_token
        else:
            expected = _token
        return self.headers.get("Authorization", "") == f"Bearer {expected}"

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        import json

        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # NEVER an Access-Control-Allow-Origin header — this API is not browser-callable.
        self.end_headers()
        self.wfile.write(body)

    def _peer_disconnected(self) -> bool:
        """Observe FIN/RST without consuming any pipelined request byte."""
        return r2_client.peer_disconnected(self.connection)

    def _stream_body_to(
        self,
        destination: BinaryIO,
        *,
        backup: bool = False,
        deadline: float | None = None,
    ) -> tuple[int, str]:
        """Stream the raw request body to `dest` in bounded chunks, enforcing the upload cap."""
        remaining = int(self.headers.get("Content-Length", "0") or "0")
        if backup:
            validate.validate_backup_upload_size(remaining)
        else:
            validate.validate_upload_size(remaining)
        written = 0
        digest = hashlib.sha256()
        destination.seek(0)
        destination.truncate(0)
        while remaining > 0:
            if deadline is not None:
                seconds_left = deadline - time.monotonic()
                if seconds_left <= 0:
                    raise ApiError(
                        HTTPStatus.REQUEST_TIMEOUT,
                        "private backup request exceeded its total deadline",
                    )
                self.connection.settimeout(seconds_left)
            try:
                chunk = self.rfile.read(min(_CHUNK, remaining))
            except TimeoutError as exc:
                raise ApiError(
                    HTTPStatus.REQUEST_TIMEOUT,
                    "private backup request exceeded its total deadline",
                ) from exc
            if not chunk:
                break
            destination.write(chunk)
            digest.update(chunk)
            written += len(chunk)
            remaining -= len(chunk)
        destination.flush()
        os.fsync(destination.fileno())
        if written == 0:
            raise ApiError(HTTPStatus.BAD_REQUEST, "empty upload body")
        if remaining:
            raise ApiError(HTTPStatus.BAD_REQUEST, "upload body ended before Content-Length")
        return written, digest.hexdigest()

    def _dispatch(self, method: str) -> None:
        path = urlsplit(self.path).path
        if method == "GET" and path == DRIVER.health_path:
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if method == "GET" and path == DRIVER.metadata_path:
            self._send_json(HTTPStatus.OK, DRIVER.public())
            return
        if method == "GET" and path == DRIVER.credential_schema_path:
            self._send_json(HTTPStatus.OK, CREDENTIALS.public())
            return
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
        except r2_client.R2NotFoundError as exc:
            audit.log(method.lower(), self.path, result="denied", reason=str(exc))
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except r2_client.R2AlreadyExistsError as exc:
            audit.log(method.lower(), self.path, result="denied", reason=str(exc))
            self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except r2_client.R2Error as exc:
            audit.log(method.lower(), self.path, result="error", reason=str(exc))
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            audit.log(method.lower(), self.path, result="error", reason=str(exc))
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _route(self, method: str) -> None:
        split = urlsplit(self.path)
        path, query = split.path, parse_qs(split.query)

        if method == "POST" and path == "/v1/r2/upload":
            self._upload()
            return
        if method == "POST" and path == "/v1/r2/backup/upload":
            self._backup_upload()
            return
        if method == "GET" and path == "/v1/r2/backup/download":
            self._backup_download(
                validate.validate_backup_key((query.get("key") or [""])[0]),
                (query.get("offset") or [""])[0],
                (query.get("length") or [""])[0],
            )
            return
        if method == "GET" and path == "/v1/r2/list":
            prefix = validate.validate_prefix((query.get("prefix") or [""])[0])
            entries = [
                entry for entry in r2_client.list_prefix(prefix) if validate.generic_entry_visible(entry.get("path"))
            ]
            trace = audit.log("r2.list", prefix or "<root>", result="ok", count=len(entries))
            self._send_json(HTTPStatus.OK, {"prefix": prefix, "entries": entries, "trace_id": trace})
            return
        if method == "GET" and path == "/v1/r2/get":
            self._get(validate.validate_key((query.get("key") or [""])[0]))
            return
        raise ApiError(HTTPStatus.NOT_FOUND, f"no route for {method} {path}")

    def _upload(self) -> None:
        filename = validate.validate_filename(self.headers.get("X-R2-Filename"))
        expire = validate.validate_expire(self.headers.get("X-R2-Expire"))
        key = _date_key(filename)
        fd, tmp_str = tempfile.mkstemp(prefix="r2up-", dir="/tmp")
        tmp = Path(tmp_str)
        os.close(fd)
        try:
            with tmp.open("w+b") as destination:
                size, _sha256 = self._stream_body_to(destination)
            audit.log("r2.upload", key, result="attempt", level="info", size=size)
            r2_client.upload(str(tmp), key)
            url = r2_client.link(key, expire)
        finally:
            tmp.unlink(missing_ok=True)
        trace = audit.log("r2.upload", key, result="ok", size=size)
        self._send_json(HTTPStatus.OK, {"key": key, "link": url, "size": size, "trace_id": trace})

    def _backup_upload(self) -> None:
        expected_sha256 = self.headers.get("X-Backup-SHA256", "")
        created_at = self.headers.get("X-Backup-Created-At", "")
        budget = validate.validate_backup_deadline(self.headers.get("X-Backup-Deadline-Seconds"))
        deadline = time.monotonic() + budget
        key = _backup_key(created_at, expected_sha256)
        with _exclusive_backup_transfer(), _backup_spool_file() as source:
            size, actual_sha256 = self._stream_body_to(source, backup=True, deadline=deadline)
            if actual_sha256 != expected_sha256:
                raise ApiError(HTTPStatus.BAD_REQUEST, "backup body SHA-256 does not match X-Backup-SHA256")
            audit.log(
                "r2.backup.upload",
                key,
                result="attempt",
                level="info",
                size=size,
                sha256=actual_sha256,
            )
            uploaded_size = r2_client.backup_upload(
                source,
                key,
                expected_sha256,
                size,
                deadline=deadline,
                cancel_check=self._peer_disconnected,
            )
            if uploaded_size != size:
                raise ApiError(HTTPStatus.BAD_GATEWAY, "rclone reported an unexpected uploaded size")
        trace = audit.log("r2.backup.upload", key, result="ok", size=size, sha256=actual_sha256)
        self._send_json(
            HTTPStatus.OK,
            {"key": key, "sha256": actual_sha256, "size": size, "trace_id": trace},
        )

    def _backup_download(self, key: str, offset_value: str, length_value: str) -> None:
        with _exclusive_backup_transfer():
            object_size = r2_client.backup_size(key, cancel_check=self._peer_disconnected)
            offset, count = validate.validate_backup_range(offset_value, length_value, object_size)
            digest = validate.backup_key_sha256(key)
            # A range is bounded to 256 MiB and staged on the ciphertext-only backup volume. rclone must
            # finish with the exact byte count before headers are emitted, so an upstream failure can
            # still be returned as JSON rather than becoming an apparently successful truncated range.
            with tempfile.TemporaryFile(prefix=".shimpz-r2backup-range-", dir=BACKUP_SPOOL_DIR) as temporary:
                r2_client.backup_download_range(
                    key,
                    offset,
                    count,
                    temporary,
                    cancel_check=self._peer_disconnected,
                )
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(count))
                self.send_header("Content-Range", f"bytes {offset}-{offset + count - 1}/{object_size}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Backup-Size", str(object_size))
                self.send_header("X-Backup-SHA256", digest)
                self.end_headers()
                try:
                    while chunk := temporary.read(_CHUNK):
                        self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError) as exc:
                    audit.log(
                        "r2.backup.download",
                        key,
                        result="error",
                        reason=type(exc).__name__,
                        offset=offset,
                        length=count,
                    )
                    return
        audit.log(
            "r2.backup.download",
            key,
            result="ok",
            size=object_size,
            sha256=digest,
            offset=offset,
            length=count,
        )

    def _get(self, key: str) -> None:
        expected_size = r2_client.object_size(key)
        validate.validate_download_size(expected_size)
        fd, tmp_str = tempfile.mkstemp(prefix="r2dl-", dir="/tmp")
        tmp = Path(tmp_str)
        os.close(fd)
        try:
            size = r2_client.download(key, str(tmp), validate.DOWNLOAD_MAX_BYTES)
            validate.validate_download_size(size)
            if size != expected_size:
                raise r2_client.R2Error("generic object size changed during bounded download")
            audit.log("r2.get", key, result="ok", size=size)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.send_header("X-R2-Size", str(size))
            self.end_headers()
            with tmp.open("rb") as fh:
                while chunk := fh.read(_CHUNK):
                    self.wfile.write(chunk)
        finally:
            tmp.unlink(missing_ok=True)

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def log_message(self, fmt: str, *args: object) -> None:
        # Suppress BaseHTTPRequestHandler's default stderr access log — audit.log() is the single
        # source of truth for what happened, in the schema logq expects.
        pass


def main() -> None:
    _prepare_backup_spool()
    # IPv4Address(0) is INADDR_ANY. The container must serve its private Docker network as well as
    # loopback health/operator calls; Compose publishes no host port.
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    print(f"r2-driver listening on :{LISTEN_PORT}", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
