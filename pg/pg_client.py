"""The ONLY place SHIMPZ_PG_DSN is ever read or sent.

Shells out to fixed psql/createdb/dropdb CLI invocations inside the sole Postgres-superuser holder.
SQL is delivered on psql stdin so a derived tenant password never appears in process argv. Every
identifier interpolated into SQL comes from validate.py's sanitize_proj first (`[a-z0-9_]` only).
"""

from __future__ import annotations

import hmac
import os
import subprocess
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from urllib.parse import urlsplit

_dsn = urlsplit(os.environ.get("SHIMPZ_PG_DSN", ""))
PGHOST = _dsn.hostname or "postgres"
PGPORT = _dsn.port or 5432
PGUSER = _dsn.username or ""
PGPASSWORD = _dsn.password or ""

_PG_ARGS = ["-h", PGHOST, "-p", str(PGPORT), "-U", PGUSER]
_ENV = {**os.environ, "PGPASSWORD": PGPASSWORD}

LEGACY_GLOBAL_READER = "shimpz_ro"
_mutation_guard = threading.RLock()


class PgError(Exception):
    """A psql/createdb/dropdb invocation failed."""


@dataclass(frozen=True)
class ProvisionResult:
    """Postgres resources created by one provisioning attempt."""

    database_url: str
    database_created: bool
    role_created: bool

    def public(self) -> dict[str, object]:
        return {"database_url": self.database_url, "created": self.database_created}


@contextmanager
def mutation_lock() -> Iterator[None]:
    """Serialize Postgres and principal-registry mutations in the single driver process."""
    with _mutation_guard:
        yield


def _run(cmd: list[str], *, stdin: str | None = None) -> str:
    result = subprocess.run(cmd, env=_ENV, input=stdin, capture_output=True, text=True, timeout=20, check=False)
    if result.returncode != 0:
        # stderr can echo the failing SQL (including CREATE/ALTER ROLE PASSWORD). The numeric verdict
        # is sufficient for the private typed failure; command text and stderr never cross this seam.
        raise PgError(f"Postgres command failed (rc={result.returncode})")
    return result.stdout


def dbname(project: str) -> str:
    return f"proj_{project}"


def role_password(project: str) -> str:
    """Deterministic per-project password, no state file to keep in sync.

    HMAC-SHA256 keyed by the superuser secret, message = dbname — `database_url` recomputes exactly
    what `create_db_and_role` set without keeping a cleartext password registry.
    """
    return hmac.new(PGPASSWORD.encode(), dbname(project).encode(), sha256).hexdigest()[:32]


def database_url(project: str) -> str:
    db = dbname(project)
    return f"postgresql://{db}:{role_password(project)}@{PGHOST}:{PGPORT}/{db}"


def _psql(db: str, sql: str, variables: Mapping[str, str] | None = None) -> str:
    variable_args = [item for name, value in (variables or {}).items() for item in ("--set", f"{name}={value}")]
    return _run(
        ["psql", *_PG_ARGS, "-d", db, "-tA", "-v", "ON_ERROR_STOP=1", *variable_args, "-f", "-"],
        stdin=f"{sql}\n",
    )


def _role_exists(role: str) -> bool:
    return (
        _psql(
            "postgres",
            "SELECT 1 FROM pg_roles WHERE rolname = :'role_name'",
            {"role_name": role},
        ).strip()
        == "1"
    )


def _db_exists(db: str) -> bool:
    return (
        _psql(
            "postgres",
            "SELECT 1 FROM pg_database WHERE datname = :'database_name'",
            {"database_name": db},
        ).strip()
        == "1"
    )


def _cleanup_created_resources(project: str, *, database_created: bool, role_created: bool) -> None:
    db = dbname(project)
    failures: list[str] = []
    if database_created:
        try:
            _run(["dropdb", *_PG_ARGS, "--if-exists", db])
        except PgError as exc:
            failures.append(str(exc))
    if role_created:
        try:
            _psql("postgres", f'DROP ROLE IF EXISTS "{db}"')
        except PgError as exc:
            failures.append(str(exc))
    if failures:
        raise PgError("; ".join(failures))


def create_db_and_role(project: str) -> ProvisionResult:
    with mutation_lock():
        db = dbname(project)
        role = db
        pw = role_password(project)
        role_existed = _role_exists(role)
        database_existed = _db_exists(db)
        if database_existed and not role_existed:
            raise PgError(f'database "{db}" exists without its expected owner role')

        role_created = False
        database_created = False
        try:
            # 1) least-privilege LOGIN role (idempotent: create it, or re-sync the derived password).
            if role_existed:
                _psql("postgres", f"ALTER ROLE \"{role}\" LOGIN PASSWORD '{pw}'")
            else:
                _psql("postgres", f"CREATE ROLE \"{role}\" LOGIN PASSWORD '{pw}'")
                role_created = True
            # 2) database OWNED by that role — the project is never the superuser.
            if not database_existed:
                _run(["createdb", *_PG_ARGS, "-O", role, db])
                database_created = True
            # 3) lock it down: ONLY this role may connect; it owns public so it can create tables.
            _psql("postgres", f'REVOKE CONNECT ON DATABASE "{db}" FROM PUBLIC')
            _psql("postgres", f'GRANT ALL ON DATABASE "{db}" TO "{role}"')
            _psql(db, f'ALTER SCHEMA public OWNER TO "{role}"')
        except PgError as provision_error:
            try:
                _cleanup_created_resources(project, database_created=database_created, role_created=role_created)
            except PgError as cleanup_error:
                raise PgError(
                    f"Postgres provisioning failed ({provision_error}); compensation also failed ({cleanup_error})"
                ) from cleanup_error
            raise
        return ProvisionResult(database_url(project), database_created, role_created)


def rollback_provision(project: str, result: ProvisionResult) -> None:
    """Remove only resources created by `result`, preserving every preexisting object."""
    with mutation_lock():
        _cleanup_created_resources(
            project,
            database_created=result.database_created,
            role_created=result.role_created,
        )


def list_project_dbs() -> list[str]:
    out = _psql("postgres", "SELECT datname FROM pg_database WHERE left(datname, 5) = 'proj_' ORDER BY 1")
    return [line for line in out.splitlines() if line]


def revoke_legacy_global_reader() -> None:
    """Remove the historical cross-tenant reader and its database CONNECT grants.

    This migration is intentionally run before the driver listens. A dependency that prevents the
    role from being dropped is a launch-blocking error, not something to hide while serving traffic.
    """
    if not _role_exists(LEGACY_GLOBAL_READER):
        return
    for database in list_project_dbs():
        _psql(
            "postgres",
            'REVOKE CONNECT ON DATABASE :"database_name" FROM :"role_name"',
            {"database_name": database, "role_name": LEGACY_GLOBAL_READER},
        )
    role_variable = {"role_name": LEGACY_GLOBAL_READER}
    _psql("postgres", 'REVOKE pg_read_all_data FROM :"role_name"', role_variable)
    _psql("postgres", 'DROP ROLE :"role_name"', role_variable)


def drop_db_and_role(project: str) -> dict:
    with mutation_lock():
        db = dbname(project)
        role = db
        _run(["dropdb", *_PG_ARGS, "--if-exists", db])
        _psql("postgres", f'DROP ROLE IF EXISTS "{role}"')
        return {"dropped": db}


def project_resources_exist(project: str) -> bool:
    """Whether either exact Postgres artifact remains for an unregistered idempotent drop."""
    with mutation_lock():
        database = dbname(project)
        return _db_exists(database) or _role_exists(database)
