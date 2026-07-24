from __future__ import annotations

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path

APPS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APPS))

import egress_lock
import validate


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="app-driver-test-")
        self.projects = Path(self.temporary.name) / "projects"
        self.projects.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def project(self, name: str, *files: str) -> Path:
        project = self.projects / name
        project.mkdir(parents=True)
        for relative in files:
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test", encoding="utf-8")
        return project

    @staticmethod
    def deploy_body(**overrides: object) -> dict[str, object]:
        body: dict[str, object] = {
            "image_kind": "python",
            "entrypoint": ["python3", "main.py"],
            "port": 3100,
            "env": {},
        }
        body.update(overrides)
        return body

    def test_project_roles_mount_one_project_with_role_specific_working_directories(self) -> None:
        project = self.project("website", "backend/main.py")

        web = validate.resolve_run_dir(self.projects, "website")
        backend = validate.resolve_run_dir(self.projects, "website-backend")
        websocket = validate.resolve_run_dir(self.projects, "website-ws")

        self.assertEqual(web, validate.RunLocation(project, None))
        self.assertEqual(backend, validate.RunLocation(project, "backend"))
        self.assertEqual(websocket, validate.RunLocation(project, "backend"))

    def test_project_resolution_fails_closed_for_missing_or_escaping_paths(self) -> None:
        self.project("website")
        for name in ("missing", "website-backend", "../website"):
            with self.subTest(name=name), self.assertRaises(validate.ValidationError):
                validate.resolve_run_dir(self.projects, name)

    def test_deploy_request_derives_mount_and_worker_metadata(self) -> None:
        self.project("website", "backend/main.py")

        request = validate.validate_deploy_request(
            "website-backend",
            self.deploy_body(worker=True, persist=True, egress=["api.example.com"]),
            self.projects,
        )

        self.assertEqual(request.run_subpath, "website")
        self.assertEqual(request.working_dir, "/app/backend")
        self.assertTrue(request.worker)
        self.assertTrue(request.persist)
        self.assertEqual(request.egress, ["api.example.com"])

    def test_untrusted_deploy_fields_never_enter_the_request_contract(self) -> None:
        self.project("website", "main.py")
        dangerous = {
            "privileged": True,
            "mounts": [{"source": "/", "target": "/host"}],
            "network_mode": "host",
            "pid_mode": "host",
            "cap_add": ["SYS_ADMIN"],
            "devices": ["/dev/kmsg"],
            "security_opt": ["seccomp=unconfined"],
            "user": "root",
        }

        request = validate.validate_deploy_request(
            "website",
            self.deploy_body(**dangerous),
            self.projects,
        )

        request_fields = {field.name for field in dataclasses.fields(request)}
        self.assertTrue(request_fields.isdisjoint(dangerous))
        for field in dangerous:
            self.assertFalse(hasattr(request, field))

    def test_names_and_ports_accept_only_the_public_contract(self) -> None:
        self.assertEqual(validate.validate_name("website-backend"), "website-backend")
        self.assertEqual(validate.validate_port(3100), 3100)
        self.assertEqual(validate.validate_port(3999), 3999)
        for value in ("../etc", "has space", "", "a" * 41):
            with self.subTest(name=value), self.assertRaises(validate.ValidationError):
                validate.validate_name(value)
        for value in (80, 4000, True, "3100"):
            with self.subTest(port=value), self.assertRaises(validate.ValidationError):
                validate.validate_port(value)

    def test_environment_is_allowlisted_and_database_is_project_scoped(self) -> None:
        dsn = "postgresql://proj_website:secret@postgres:5432/proj_website"
        self.assertEqual(validate.validate_env({"DATABASE_URL": dsn}, "website-backend"), {"DATABASE_URL": dsn})

        invalid = (
            {"OPENAI_API_KEY": "secret"},
            {"UNREVIEWED": "value"},
            {"DATABASE_URL": "postgresql://proj_other:x@postgres:5432/proj_other"},
        )
        for environment in invalid:
            with self.subTest(environment=environment), self.assertRaises(validate.ValidationError):
                validate.validate_env(environment, "website-backend")

    def test_entrypoint_is_an_argument_allowlist_not_a_shell(self) -> None:
        valid = ["python3", "-m", "shimpz_static", "3100", "--directory", "frontend/build"]
        self.assertEqual(validate.validate_entrypoint(valid), valid)

        invalid = (
            ["bash", "-c", "id"],
            ["python3", "-m", "http.server", "3100"],
            ["python3", "-m", "shimpz_static", "3100", "--directory", "/config/workspace/app"],
            ["python3", "main.py;id"],
        )
        for entrypoint in invalid:
            with self.subTest(entrypoint=entrypoint), self.assertRaises(validate.ValidationError):
                validate.validate_entrypoint(entrypoint)

    def test_routes_require_app_targets_for_each_requested_port(self) -> None:
        request = validate.validate_route_request(
            {
                "fqdn": "app.example.com",
                "web_target": "app_web",
                "web_port": 3100,
                "api_target": "app_api",
                "api_port": 3101,
            }
        )
        self.assertEqual(request.api_target, "app_api")

        for body in (
            {"fqdn": "app.example.com", "web_target": "shimpz-brain", "web_port": 3100},
            {"fqdn": "app.example.com", "web_target": "app_web", "web_port": 3100, "api_port": 3101},
        ):
            with self.subTest(body=body), self.assertRaises(validate.ValidationError):
                validate.validate_route_request(body)

    def test_recreate_surface_does_not_exist(self) -> None:
        source = (APPS / "app.py").read_text(encoding="utf-8")

        self.assertNotIn("/v1/stack/recreate", source)
        self.assertFalse(hasattr(validate, "validate_recreate_request"))

    def test_egress_is_deduplicated_and_defaults_to_no_internet(self) -> None:
        self.assertEqual(validate.validate_egress(None), [])
        self.assertEqual(validate.validate_egress([]), [])
        self.assertEqual(
            validate.validate_egress(["api.example.com", "api.example.com", "pay.shimpz.com"]),
            ["api.example.com", "pay.shimpz.com"],
        )

    def test_egress_rejects_malformed_private_and_payment_processor_targets(self) -> None:
        invalid = (
            "api.example.com",
            [123],
            ["Bad_Host"],
            ["127.0.0.1"],
            ["metadata.google.internal"],
            ["api.stripe.com"],
            ["api.example.com", "connect.squareup.com"],
        )
        for egress in invalid:
            with self.subTest(egress=egress), self.assertRaises(validate.ValidationError):
                validate.validate_egress(egress)

    def test_egress_lock_has_one_fail_closed_configuration(self) -> None:
        self.assertTrue(egress_lock.require_enabled({}))
        self.assertTrue(egress_lock.require_enabled({egress_lock.ENV_NAME: "1"}))
        for value in ("0", "false", "true", "", "01", "1 "):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, egress_lock.ENV_NAME):
                egress_lock.require_enabled({egress_lock.ENV_NAME: value})


if __name__ == "__main__":
    unittest.main()
