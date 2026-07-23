#!/opt/venv/bin/python
"""Tenant-scoped pg-driver: one hashed principal and exact DB set per Team."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import audit
import driver_manifest
import pg_client
import principal_store
import stdlib_http
import token_store
import validate

DRIVER = driver_manifest.load()
LISTEN_HOST = os.environ.get("SHIMPZ_PGDRIVER_HOST", "")
LISTEN_PORT = DRIVER.port
MAX_BODY_BYTES = int(os.environ.get("SHIMPZ_PGDRIVER_MAX_BODY_BYTES", str(64 * 1024)))
_provisioner_token = token_store.ensure_token()


ApiError = stdlib_http.HttpError

_ROUTES = (
    stdlib_http.Route("GET", re.compile(r"^/healthz$"), "health"),
    stdlib_http.Route("GET", re.compile(r"^/v1/driver$"), "metadata"),
    stdlib_http.Route("POST", re.compile(r"^/v1/teams/provision$"), "team.provision"),
    stdlib_http.Route("POST", re.compile(r"^/v1/teams/finalize$"), "team.finalize"),
    stdlib_http.Route("POST", re.compile(r"^/v1/teams/apps/create$"), "team.app.create"),
    stdlib_http.Route("POST", re.compile(r"^/v1/teams/apps/drop$"), "team.app.drop"),
    stdlib_http.Route("POST", re.compile(r"^/v1/teams/drop$"), "team.drop"),
)


def _provision_team(body: dict) -> dict:
    team_id = validate.validate_team_id(body.get("team_id"))
    principal_token = validate.validate_principal_token(body.get("principal_token"))
    project = validate.team_project(team_id)
    database = pg_client.dbname(project)
    with pg_client.mutation_lock():
        result = pg_client.create_db_and_role(
            project,
            allow_existing=principal_store.owns_database(team_id, database),
        )
        try:
            principal_store.register(team_id, principal_token, pg_client.dbname(project))
        except (principal_store.PrincipalError, principal_store.PrincipalStoreError) as registry_error:
            try:
                pg_client.rollback_provision(project, result)
            except pg_client.PgError as rollback_error:
                message = (
                    f"principal registry commit failed ({registry_error}); "
                    f"Postgres compensation failed ({rollback_error})"
                )
                raise pg_client.PgError(message) from rollback_error
            raise
        return result.public()


def _create_app(body: dict, token: str) -> dict:
    team_id = validate.validate_team_id(body.get("team_id"))
    app_id = validate.validate_app_id(body.get("app_id"))
    with pg_client.mutation_lock():
        scoped = principal_store.databases(token, team_id)
        project = validate.team_app_project(principal_store.database_namespace(token, team_id), app_id)
        database = pg_client.dbname(project)
        if database not in scoped and pg_client.project_resources_exist(project):
            raise principal_store.PrincipalError("unregistered App database artifacts already exist")
        result = pg_client.create_db_and_role(project, allow_existing=database in scoped)
        try:
            principal_store.add_database(token, team_id, database)
        except (principal_store.PrincipalError, principal_store.PrincipalStoreError) as registry_error:
            try:
                pg_client.rollback_provision(project, result)
            except pg_client.PgError as rollback_error:
                message = (
                    f"principal registry commit failed ({registry_error}); "
                    f"Postgres compensation failed ({rollback_error})"
                )
                raise pg_client.PgError(message) from rollback_error
            raise
        return result.public()


def _drop_app(body: dict, token: str) -> dict:
    team_id = validate.validate_team_id(body.get("team_id"))
    app_id = validate.validate_app_id(body.get("app_id"))
    with pg_client.mutation_lock():
        scoped = principal_store.databases(token, team_id)
        project = validate.team_app_project(principal_store.database_namespace(token, team_id), app_id)
        database = pg_client.dbname(project)
        if database not in scoped:
            # A response-lost retry and a declared no-DB App both reach this branch. Succeed only
            # after Postgres itself proves neither exact artifact exists; registry drift fails closed.
            if pg_client.project_resources_exist(project):
                raise principal_store.PrincipalError("unregistered App database artifacts still exist")
            return {"dropped": database, "already_absent": True}
        result = pg_client.drop_db_and_role(project)
        principal_store.remove_database(token, team_id, database)
        return result


def _drop_team(body: dict, token: str) -> dict:
    team_id = validate.validate_team_id(body.get("team_id"))
    with pg_client.mutation_lock():
        scoped = principal_store.databases(token, team_id, allow_retired=True)
        dropped: list[str] = []
        for database in sorted(scoped, key=lambda value: value.startswith("proj_team_")):
            project = database.removeprefix("proj_")
            pg_client.drop_db_and_role(project)
            principal_store.remove_database(token, team_id, database)
            dropped.append(database)
        principal_store.retire(token, team_id)
        return {"dropped": dropped}


def _finalize_team(body: dict) -> dict:
    team_id = validate.validate_team_id(body.get("team_id"))
    principal_store.finalize(team_id)
    return {"finalized": True}


def _http_failure(exc: Exception) -> stdlib_http.HttpFailure | None:
    if isinstance(exc, ApiError):
        failure = stdlib_http.HttpFailure(exc.status, exc.message, exc.message, "denied")
    elif isinstance(exc, validate.ValidationError):
        message = str(exc)
        failure = stdlib_http.HttpFailure(HTTPStatus.BAD_REQUEST, message, message, "denied")
    elif isinstance(exc, principal_store.PrincipalError):
        failure = stdlib_http.HttpFailure(HTTPStatus.FORBIDDEN, "principal scope denied", str(exc), "denied")
    elif isinstance(exc, principal_store.PrincipalStoreError):
        failure = stdlib_http.HttpFailure(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "principal registry unavailable",
            str(exc),
            "error",
        )
    elif isinstance(exc, pg_client.PgError):
        failure = stdlib_http.HttpFailure(
            HTTPStatus.BAD_GATEWAY,
            "database operation failed",
            "database operation failed",
            "error",
        )
    elif isinstance(exc, (OSError, RuntimeError, ValueError, subprocess.SubprocessError)):
        failure = stdlib_http.HttpFailure(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "internal driver error",
            type(exc).__name__,
            "error",
        )
    else:
        failure = None
    return failure


def _run_operation(operation: str, body: dict, token: str) -> dict:
    if operation == "team.provision":
        return _provision_team(body)
    if operation == "team.finalize":
        return _finalize_team(body)
    if operation == "team.app.create":
        return _create_app(body, token)
    if operation == "team.app.drop":
        return _drop_app(body, token)
    return _drop_team(body, token)


class Handler(BaseHTTPRequestHandler):
    server_version = f"{DRIVER.id}-driver/{DRIVER.version}"

    def _bearer(self) -> str:
        return stdlib_http.bearer_token(self.headers)

    def _is_provisioner(self) -> bool:
        return stdlib_http.bearer_authorized(self.headers, _provisioner_token)

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        stdlib_http.send_json(self, status, payload)

    def _body(self) -> dict:
        return stdlib_http.read_json_body(self.headers, self.rfile, max_bytes=MAX_BODY_BYTES)

    def _dispatch(self, method: str) -> None:
        stdlib_http.dispatch(
            lambda: self._route(method),
            classify=_http_failure,
            emit=lambda failure: self._emit_failure(method, failure),
            unexpected_message="internal driver error",
        )

    def _emit_failure(self, method: str, failure: stdlib_http.HttpFailure) -> None:
        audit.log(method.lower(), self.path, result=failure.result, reason=failure.audit_reason)
        self._send_json(failure.status, {"error": failure.public_message})

    def _route(self, method: str) -> None:
        route = stdlib_http.resolve_route(_ROUTES, method, self.path)
        if route.operation == "health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if route.operation == "metadata":
            self._send_json(HTTPStatus.OK, DRIVER.public())
            return
        body = self._body()
        token = self._bearer()
        if not token:
            raise ApiError(HTTPStatus.FORBIDDEN, "bearer required")
        if route.operation in {"team.provision", "team.finalize"}:
            if not self._is_provisioner():
                raise ApiError(HTTPStatus.FORBIDDEN, "provisioner bearer required")
            token = ""
        result = _run_operation(route.operation, body, token)
        trace = audit.log(route.operation, body.get("team_id", "?"), result="ok")
        self._send_json(HTTPStatus.OK, {**result, "trace_id": trace})

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def log_message(self, fmt: str, *args: object) -> None:
        pass


def main() -> None:
    pg_client.revoke_legacy_global_reader()
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    print(f"pg-driver listening on :{LISTEN_PORT}; tenant principals only", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
