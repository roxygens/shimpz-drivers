#!/opt/venv/bin/python
"""Tenant-scoped pg-driver: one hashed principal and exact DB set per Team."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import audit
import driver_manifest
import pg_client
import principal_store
import token_store
import validate

DRIVER = driver_manifest.load()
LISTEN_HOST = os.environ.get("SHIMPZ_PGDRIVER_HOST", "")
LISTEN_PORT = DRIVER.port
MAX_BODY_BYTES = int(os.environ.get("SHIMPZ_PGDRIVER_MAX_BODY_BYTES", str(64 * 1024)))
_provisioner_token = token_store.ensure_token()


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


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


class Handler(BaseHTTPRequestHandler):
    server_version = f"{DRIVER.id}-driver/{DRIVER.version}"

    def _bearer(self) -> str:
        scheme, separator, value = self.headers.get("Authorization", "").partition(" ")
        return value if separator and scheme == "Bearer" else ""

    def _is_provisioner(self) -> bool:
        return validate.tokens_equal(self._bearer(), _provisioner_token)

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        raw_length = self.headers.get("Content-Length", "0") or "0"
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "invalid Content-Length") from exc
        if length < 0 or length > MAX_BODY_BYTES:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"request body exceeds {MAX_BODY_BYTES} bytes")
        if length == 0:
            return {}
        try:
            body = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
        return body

    def _dispatch(self, method: str) -> None:
        try:
            self._route(method)
        except ApiError as exc:
            audit.log(method.lower(), self.path, result="denied", reason=exc.message)
            self._send_json(exc.status, {"error": exc.message})
        except validate.ValidationError as exc:
            audit.log(method.lower(), self.path, result="denied", reason=str(exc))
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except principal_store.PrincipalError as exc:
            audit.log(method.lower(), self.path, result="denied", reason=str(exc))
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "principal scope denied"})
        except principal_store.PrincipalStoreError as exc:
            audit.log(method.lower(), self.path, result="error", reason=str(exc))
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "principal registry unavailable"})
        except pg_client.PgError:
            # PgError is deliberately secret-free, but keep both the audit and public boundary
            # generic so a future SQL diagnostic cannot become a credential disclosure regression.
            audit.log(method.lower(), self.path, result="error", reason="database operation failed")
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "database operation failed"})
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
            audit.log(method.lower(), self.path, result="error", reason=type(exc).__name__)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal driver error"})

    def _route(self, method: str) -> None:
        path = urlsplit(self.path).path
        if method == "GET" and path == DRIVER.health_path:
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if method == "GET" and path == DRIVER.metadata_path:
            self._send_json(HTTPStatus.OK, DRIVER.public())
            return
        if not self._bearer():
            raise ApiError(HTTPStatus.FORBIDDEN, "bearer required")
        if method != "POST":
            raise ApiError(HTTPStatus.NOT_FOUND, f"no route for {method} {path}")
        body = self._body()
        if path in {"/v1/teams/provision", "/v1/teams/finalize"}:
            if not self._is_provisioner():
                raise ApiError(HTTPStatus.FORBIDDEN, "provisioner bearer required")
            if path == "/v1/teams/provision":
                result = _provision_team(body)
                operation = "team.provision"
            else:
                result = _finalize_team(body)
                operation = "team.finalize"
            trace = audit.log(operation, body.get("team_id", "?"), result="ok")
        else:
            token = self._bearer()
            if not token:
                raise ApiError(HTTPStatus.FORBIDDEN, "tenant bearer required")
            if path == "/v1/teams/apps/create":
                result = _create_app(body, token)
                trace = audit.log("team.app.create", body.get("team_id", "?"), result="ok")
            elif path == "/v1/teams/apps/drop":
                result = _drop_app(body, token)
                trace = audit.log("team.app.drop", body.get("team_id", "?"), result="ok")
            elif path == "/v1/teams/drop":
                result = _drop_team(body, token)
                trace = audit.log("team.drop", body.get("team_id", "?"), result="ok")
            else:
                raise ApiError(HTTPStatus.NOT_FOUND, f"no route for {method} {path}")
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
