from __future__ import annotations

import grp
import hmac
import http.client
import json
import os
import sys
import tempfile
import threading
import unittest
from hashlib import sha256
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

PG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PG))

MODULE_STATE = tempfile.TemporaryDirectory(prefix="pg-driver-module-test-")
os.environ.setdefault("SHIMPZ_PG_DSN", "postgresql://shimpz-brain:test-superuser-secret@postgres:5432/postgres")
os.environ["SHIMPZ_PGDRIVER_TOKEN_FILE"] = str(Path(MODULE_STATE.name) / "token")
os.environ["SHIMPZ_PGDRIVER_TOKEN_GROUP"] = grp.getgrgid(os.getgid()).gr_name
os.environ["SHIMPZ_PGDRIVER_PRINCIPALS_FILE"] = str(Path(MODULE_STATE.name) / "principals.json")
os.environ["SHIMPZ_PGDRIVER_AUDIT_LOG"] = str(Path(MODULE_STATE.name) / "audit.jsonl")

import app
import driver_manifest
import pg_client
import principal_store
import validate


class PgDriverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="pg-driver-test-")
        principal_store.STATE_PATH = Path(self.temporary.name) / "principals.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_project_validation_matches_postgres_identifier_limits(self) -> None:
        cases = {
            "Laudoctor": "laudoctor",
            "my project!!": "my_project",
            "  leading-trailing  ": "leading_trailing",
            "UP--PER": "up_per",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(validate.validate_project(raw), expected)

        for invalid in ("", None, 123, "!!!", "a" * 59):
            with self.subTest(invalid=invalid), self.assertRaises(validate.ValidationError):
                validate.validate_project(invalid)

    def test_team_app_and_principal_identifiers_are_server_derived(self) -> None:
        self.assertEqual(validate.team_project("captain_01"), "team_captain_01")
        first = validate.team_app_project("a" * 12, "notification-center")
        second = validate.team_app_project("b" * 12, "notification-center")
        self.assertTrue(first.startswith("team_") and first.endswith("_notification_center"))
        self.assertNotEqual(first, second)
        self.assertLessEqual(len(pg_client.dbname(validate.team_app_project("f" * 12, "a" * 40))), 63)

        token = "a" * 64
        self.assertEqual(validate.validate_principal_token(token), token)
        self.assertTrue(validate.tokens_equal(token, token))
        self.assertFalse(validate.tokens_equal(token, "b" * 64))
        for invalid in ("", "a" * 63, "A" * 64, "z" * 64, None):
            with self.subTest(invalid=invalid), self.assertRaises(validate.ValidationError):
                validate.validate_principal_token(invalid)

    def test_database_credentials_are_deterministic_and_keyed(self) -> None:
        password = pg_client.role_password("website")
        expected = hmac.new(
            pg_client.PGPASSWORD.encode(),
            pg_client.dbname("website").encode(),
            sha256,
        ).hexdigest()[:32]

        self.assertEqual(password, expected)
        self.assertEqual(len(password), 32)
        self.assertNotEqual(password, pg_client.role_password("other"))
        self.assertEqual(
            pg_client.database_url("website"),
            f"postgresql://proj_website:{password}@postgres:5432/proj_website",
        )

    def test_database_failures_do_not_reflect_commands_sql_or_stderr(self) -> None:
        completed = mock.Mock(returncode=23, stdout="", stderr="database-secret")
        with (
            mock.patch.object(pg_client.subprocess, "run", return_value=completed) as run,
            self.assertRaisesRegex(pg_client.PgError, r"^Postgres command failed \(rc=23\)$") as raised,
        ):
            pg_client._run(["psql", "command-secret"], stdin="sql-secret")

        detail = str(raised.exception)
        for secret in ("command-secret", "sql-secret", "database-secret"):
            self.assertNotIn(secret, detail)
        self.assertEqual(run.call_args.kwargs["input"], "sql-secret")

    def test_psql_sends_sql_on_stdin_with_fail_fast_literal_variables(self) -> None:
        with mock.patch.object(pg_client, "_run", return_value="ok") as run:
            result = pg_client._psql(
                "postgres",
                "SELECT 1 WHERE rolname = :'role_name'",
                {"role_name": "proj_website"},
            )

        self.assertEqual(result, "ok")
        command = run.call_args.args[0]
        self.assertNotIn("SELECT 1 WHERE rolname = :'role_name'", command)
        self.assertIn("ON_ERROR_STOP=1", command)
        self.assertEqual(command[-2:], ["-f", "-"])
        self.assertEqual(run.call_args.kwargs["stdin"], "SELECT 1 WHERE rolname = :'role_name'\n")
        self.assertIn("role_name=proj_website", command)

    def test_legacy_reader_revocation_uses_literal_variables_for_catalog_values(self) -> None:
        with (
            mock.patch.object(pg_client, "_role_exists", return_value=True),
            mock.patch.object(pg_client, "list_project_dbs", return_value=["proj_a", "proj_b"]),
            mock.patch.object(pg_client, "_psql", return_value="") as psql,
        ):
            pg_client.revoke_legacy_global_reader()

        calls = psql.call_args_list
        self.assertEqual(len(calls), 4)
        for database, call in zip(("proj_a", "proj_b"), calls[:2], strict=True):
            self.assertIn('REVOKE CONNECT ON DATABASE :"database_name"', call.args[1])
            self.assertEqual(
                call.args[2],
                {"database_name": database, "role_name": pg_client.LEGACY_GLOBAL_READER},
            )
        self.assertIn('REVOKE pg_read_all_data FROM :"role_name"', calls[2].args[1])
        self.assertIn('DROP ROLE :"role_name"', calls[3].args[1])

    def test_principal_registry_hashes_tokens_and_enforces_exact_scope(self) -> None:
        token_a, token_b = "a" * 64, "b" * 64
        main_database = "proj_team_alpha"
        app_database = "proj_team_hash_application"
        principal_store.register("alpha", token_a, main_database)
        namespace_a = principal_store.database_namespace(token_a, "alpha")

        stored = principal_store.STATE_PATH.read_text(encoding="utf-8")
        self.assertNotIn(token_a, stored)
        self.assertEqual(principal_store.STATE_PATH.stat().st_mode & 0o777, 0o600)
        self.assertEqual(principal_store.databases(token_a, "alpha"), {main_database})

        principal_store.add_database(token_a, "alpha", app_database)
        self.assertEqual(principal_store.databases(token_a, "alpha"), {main_database, app_database})
        with self.assertRaises(principal_store.PrincipalError):
            principal_store.databases(token_b, "alpha")
        with self.assertRaises(principal_store.PrincipalError):
            principal_store.databases(token_a, "other")

        principal_store.register("alpha", token_b, main_database)
        with self.assertRaises(principal_store.PrincipalError):
            principal_store.databases(token_a, "alpha")
        self.assertEqual(principal_store.databases(token_b, "alpha"), {main_database})
        self.assertEqual(principal_store.database_namespace(token_b, "alpha"), namespace_a)

        principal_store.register("beta", token_a, "proj_team_beta")
        self.assertNotEqual(principal_store.database_namespace(token_a, "beta"), namespace_a)

    def test_team_drop_is_idempotent_until_finalization(self) -> None:
        token = "c" * 64
        main_database = "proj_team_alpha"
        app_database = "proj_team_hash_application"
        principal_store.register("alpha", token, main_database)
        principal_store.add_database(token, "alpha", app_database)

        with mock.patch.object(
            pg_client,
            "drop_db_and_role",
            side_effect=lambda project: {"dropped": f"proj_{project}"},
        ) as drop:
            first = app._drop_team({"team_id": "alpha"}, token)
            retry = app._drop_team({"team_id": "alpha"}, token)

        self.assertEqual(set(first["dropped"]), {main_database, app_database})
        self.assertEqual(retry["dropped"], [])
        self.assertEqual(drop.call_count, 2)
        self.assertEqual(principal_store.databases(token, "alpha", allow_retired=True), set())
        with self.assertRaises(principal_store.PrincipalError):
            principal_store.databases(token, "alpha")
        self.assertEqual(app._finalize_team({"team_id": "alpha"}), {"finalized": True})
        self.assertEqual(app._finalize_team({"team_id": "alpha"}), {"finalized": True})

    def test_app_drop_requires_physical_absence_before_idempotent_success(self) -> None:
        token = "d" * 64
        principal_store.register("alpha", token, "proj_team_alpha")
        body = {"team_id": "alpha", "app_id": "notification-center"}

        with mock.patch.object(pg_client, "project_resources_exist", return_value=False):
            self.assertTrue(app._drop_app(body, token)["already_absent"])
        with (
            mock.patch.object(pg_client, "project_resources_exist", return_value=True),
            self.assertRaises(principal_store.PrincipalError),
        ):
            app._drop_app(body, token)

    def test_app_create_refuses_to_adopt_an_unregistered_database(self) -> None:
        token = "e" * 64
        team_id = "attacker"
        app_id = "notification-center"
        principal_store.register(team_id, token, "proj_team_attacker")
        project = validate.team_app_project(principal_store.database_namespace(token, team_id), app_id)

        with (
            mock.patch.object(pg_client, "project_resources_exist", return_value=True) as exists,
            mock.patch.object(pg_client, "create_db_and_role") as create,
            self.assertRaisesRegex(principal_store.PrincipalError, "unregistered App database"),
        ):
            app._create_app({"team_id": team_id, "app_id": app_id}, token)

        exists.assert_called_once_with(project)
        create.assert_not_called()

    def test_app_create_allows_a_registered_same_team_retry(self) -> None:
        token = "f" * 64
        team_id = "alpha"
        app_id = "notification-center"
        principal_store.register(team_id, token, "proj_team_alpha")
        project = validate.team_app_project(principal_store.database_namespace(token, team_id), app_id)
        database = pg_client.dbname(project)
        principal_store.add_database(token, team_id, database)
        provisioned = pg_client.ProvisionResult("postgresql://redacted", False, False)

        with (
            mock.patch.object(pg_client, "project_resources_exist") as exists,
            mock.patch.object(pg_client, "create_db_and_role", return_value=provisioned) as create,
        ):
            result = app._create_app({"team_id": team_id, "app_id": app_id}, token)

        exists.assert_not_called()
        create.assert_called_once_with(project, allow_existing=True)
        self.assertEqual(result, {"database_url": "postgresql://redacted", "created": False})

    def test_client_refuses_foreign_or_incomplete_existing_resources(self) -> None:
        with (
            mock.patch.object(pg_client, "_role_exists", return_value=True),
            mock.patch.object(pg_client, "_db_exists", return_value=True),
            mock.patch.object(pg_client, "_psql") as psql,
            self.assertRaisesRegex(pg_client.PgError, "without registry ownership"),
        ):
            pg_client.create_db_and_role("team_foreign_app")
        psql.assert_not_called()

        with (
            mock.patch.object(pg_client, "_role_exists", return_value=True),
            mock.patch.object(pg_client, "_db_exists", return_value=False),
            self.assertRaisesRegex(pg_client.PgError, "are incomplete"),
        ):
            pg_client.create_db_and_role("team_incomplete_app")

    def test_manifest_is_closed_and_public_metadata_contains_no_credentials(self) -> None:
        manifest = driver_manifest.load()
        self.assertEqual(manifest.id, "postgresql")
        self.assertEqual(manifest.scope, "space")
        self.assertEqual(manifest.credential_policy, "managed")
        self.assertEqual(manifest.data_plane, "direct")
        self.assertEqual(
            set(manifest.operations),
            {"team.provision", "team.finalize", "team.drop", "team.app.create", "team.app.drop"},
        )
        self.assertTrue({"credentials", "secrets"}.isdisjoint(manifest.public()))

        canonical = driver_manifest.MANIFEST_PATH.read_text(encoding="utf-8")
        invalid = (
            canonical.replace("[capabilities]", "unsupported = true\n\n[capabilities]"),
            canonical.replace('scope = "space"', 'scope = "team"'),
            canonical.replace('  "team.drop",', '  "team.drop",\n  "team.drop",'),
            "schema_version = [\n",
        )
        for index, source in enumerate(invalid):
            path = Path(self.temporary.name) / f"invalid-{index}.toml"
            path.write_text(source, encoding="utf-8")
            with self.subTest(index=index), self.assertRaises(driver_manifest.ManifestError):
                driver_manifest.load(path)

    def test_http_discovery_is_public_while_mutation_requires_a_bearer(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            for path, expected in (("/healthz", {"status": "ok"}), ("/v1/driver", app.DRIVER.public())):
                with self.subTest(path=path):
                    status, payload = self.http(server, "GET", path)
                    self.assertEqual(status, 200)
                    self.assertEqual(payload, expected)

            status, payload = self.http(server, "POST", "/v1/teams/provision", body={})
            self.assertEqual(status, 403)
            self.assertEqual(payload, {"error": "bearer required"})

            with mock.patch.object(app, "_provision_team", side_effect=pg_client.PgError("database-secret")):
                status, payload = self.http(
                    server,
                    "POST",
                    "/v1/teams/provision",
                    body={"team_id": "alpha", "principal_token": "a" * 64},
                    bearer=app._provisioner_token,
                )
            self.assertEqual(status, 502)
            self.assertEqual(payload, {"error": "database operation failed"})
            self.assertNotIn("database-secret", json.dumps(payload))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    @staticmethod
    def http(
        server: ThreadingHTTPServer,
        method: str,
        path: str,
        *,
        body: dict[str, object] | None = None,
        bearer: str | None = None,
    ) -> tuple[int, dict[str, object]]:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=3)
        try:
            encoded = None if body is None else json.dumps(body)
            headers = {} if body is None else {"Content-Type": "application/json"}
            if bearer is not None:
                headers["Authorization"] = f"Bearer {bearer}"
            connection.request(method, path, encoded, headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
