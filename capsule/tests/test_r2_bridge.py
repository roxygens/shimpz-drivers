from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
import unittest
from http import HTTPStatus
from pathlib import Path
from unittest import mock

CAPSULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAPSULE))


class _DockerError(Exception):
    pass


class _NotFoundError(_DockerError):
    pass


class _APIError(_DockerError):
    pass


class _Passthru:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


class _LogConfig(_Passthru):
    types = types.SimpleNamespace(JSON="json-file")


class _EmptyCollection:
    @staticmethod
    def get(_identity):
        raise _NotFoundError

    @staticmethod
    def list(**_kwargs):
        return []


_engine = types.SimpleNamespace(
    containers=_EmptyCollection(),
    networks=_EmptyCollection(),
    volumes=_EmptyCollection(),
    images=_EmptyCollection(),
)
_docker_types = types.ModuleType("docker.types")
_docker_types.Mount = _Passthru
_docker_types.Ulimit = _Passthru
_docker_types.Healthcheck = _Passthru
_docker_types.LogConfig = _LogConfig
_docker_errors = types.ModuleType("docker.errors")
_docker_errors.DockerException = _DockerError
_docker_errors.NotFound = _NotFoundError
_docker_errors.APIError = _APIError
_docker_socket = types.ModuleType("docker.utils.socket")
_docker_utils = types.ModuleType("docker.utils")
_docker_utils.socket = _docker_socket
_docker = types.ModuleType("docker")
_docker.from_env = lambda: _engine
_docker.types = _docker_types
_docker.errors = _docker_errors
_docker.utils = _docker_utils
sys.modules.update(
    {
        "docker": _docker,
        "docker.types": _docker_types,
        "docker.errors": _docker_errors,
        "docker.utils": _docker_utils,
        "docker.utils.socket": _docker_socket,
    }
)


def _stub(name: str, **members):
    module = types.ModuleType(name)
    for key, value in members.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _BrainCredentialError(Exception):
    pass


class _PgDriverError(Exception):
    pass


_stub("accounts_client", verify=lambda _token: None)
_stub("audit", log=lambda *_args, **_kwargs: "trace")
_stub(
    "brain_credentials_client",
    BrainCredentialError=_BrainCredentialError,
    resolve=lambda *_args: None,
    generation_is_current=lambda *_args: True,
)
_stub(
    "pgdriver_client",
    PgDriverError=_PgDriverError,
    provision_capsule=lambda _cid: {"database_url": "postgres://scoped"},
    create_app_db=lambda *_args: {},
    drop_app_db=lambda *_args: {},
    drop_capsule=lambda *_args: {},
    finalize_capsule_drop=lambda *_args: {},
)
_stub("token_store", ensure_token=lambda: "operator-token")

spec = importlib.util.spec_from_file_location("capsule_app_r2_bridge_test", CAPSULE / "app.py")
app = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = app
spec.loader.exec_module(app)


@contextlib.contextmanager
def _patched(**replacements):
    originals = {name: getattr(app, name) for name in replacements}
    try:
        for name, replacement in replacements.items():
            setattr(app, name, replacement)
        yield
    finally:
        for name, original in originals.items():
            setattr(app, name, original)


class _RouteHarness:
    def __init__(self, body: dict | None = None) -> None:
        self.body = body
        self.read_count = 0
        self.sent: list[tuple[HTTPStatus, dict]] = []

    def _read_driver_body(self, keys: set[str]) -> dict:
        self.read_count += 1
        if self.body is None or set(self.body) != keys:
            raise AssertionError("unexpected body contract")
        return self.body

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        self.sent.append((status, payload))


class R2BridgeTests(unittest.TestCase):
    def test_driver_operation_rechecks_owner_inside_lock_before_lazy_provision(self) -> None:
        events: list[str] = []
        lease = object()

        @contextlib.contextmanager
        def locked(_cid):
            events.append("lock")
            yield

        def recheck(cid, supplied):
            self.assertEqual((cid, supplied), ("capsule_1", lease))
            events.append("recheck")
            return object()

        with (
            _patched(_lock_for=locked, _require_current_authorization=recheck),
            mock.patch.object(
                app.r2driver_client,
                "ensure_provisioned",
                side_effect=lambda _cid: events.append("provision"),
            ),
        ):
            result = app._r2_driver_operation(
                "capsule_1",
                lease,
                lambda: events.append("operation") or {"ok": True},
            )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(events, ["lock", "recheck", "provision", "operation"])

    def test_stale_or_cross_tenant_lease_cannot_reach_r2_after_body_parse(self) -> None:
        harness = _RouteHarness(
            {
                "profile_id": "s3-access-key",
                "label": "tenant-a",
                "values": {"secret_access_key": "never-forwarded"},
                "expected_generation": 1,
            }
        )

        @contextlib.contextmanager
        def locked(_cid):
            yield

        def reject(_cid, _lease):
            raise app.ApiError(HTTPStatus.NOT_FOUND, "capsule not found")

        with (
            _patched(_lock_for=locked, _require_current_authorization=reject),
            mock.patch.object(app.r2driver_client, "ensure_provisioned") as provision,
            mock.patch.object(app.r2driver_client, "rotate_credential") as rotate,
            self.assertRaises(app.ApiError) as caught,
        ):
            app.Handler._route_driver(
                harness,
                "PUT",
                ["v1", "capsules", "capsule_b", "drivers", "r2", "credentials", "primary"],
                "capsule_b",
                object(),
            )
        self.assertEqual(caught.exception.status, HTTPStatus.NOT_FOUND)
        self.assertEqual(harness.read_count, 1)
        provision.assert_not_called()
        rotate.assert_not_called()

    def test_route_allows_only_r2_and_forwards_closed_put(self) -> None:
        harness = _RouteHarness(
            {
                "profile_id": "s3-access-key",
                "label": "primary",
                "values": {"secret_access_key": "never-returned"},
                "expected_generation": 2,
            }
        )
        lease = object()
        calls: list[tuple] = []
        metadata = {
            "id": "primary",
            "profile_id": "s3-access-key",
            "label": "primary",
            "generation": 3,
            "status": "active",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:01Z",
        }

        def operation(cid, supplied_lease, callback):
            calls.append((cid, supplied_lease))
            return callback()

        with (
            _patched(_r2_driver_operation=operation),
            mock.patch.object(
                app.r2driver_client,
                "rotate_credential",
                side_effect=lambda *args: calls.append(args) or metadata,
            ),
        ):
            app.Handler._route_driver(
                harness,
                "PUT",
                ["v1", "capsules", "capsule_1", "drivers", "r2", "credentials", "primary"],
                "capsule_1",
                lease,
            )
        self.assertEqual(harness.sent, [(HTTPStatus.OK, metadata)])
        self.assertEqual(calls[0], ("capsule_1", lease))
        self.assertEqual(calls[1][0:2], ("capsule_1", "primary"))
        self.assertNotIn("values", harness.sent[0][1])

        with self.assertRaises(app.ApiError) as caught:
            app.Handler._route_driver(
                harness,
                "GET",
                ["v1", "capsules", "capsule_1", "drivers", "postgresql"],
                "capsule_1",
                lease,
            )
        self.assertEqual(caught.exception.status, HTTPStatus.NOT_FOUND)

    def test_create_provisions_r2_after_postgres_and_uses_common_rollback(self) -> None:
        events: list[str] = []

        @contextlib.contextmanager
        def reserve(*_args, **_kwargs):
            yield

        def pg_provision(_cid):
            events.append("postgres-provision")
            return {"database_url": "postgres://scoped"}

        def r2_provision(_cid):
            events.append("r2-provision")
            raise app.r2driver_client.R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver is unavailable", category="test")

        def rollback(*_args, **_kwargs):
            events.append("rollback")
            return app._CleanupResult(True, True)

        with (
            _patched(
                _cleanup_record=lambda _cid: None,
                _get_container=lambda _name: None,
                _reserve_capacity=reserve,
                _require_capsule_runtime=lambda: None,
                _teardown=rollback,
            ),
            mock.patch.object(app.brain_credentials_client, "resolve", return_value=None),
            mock.patch.object(app.pgdriver_client, "provision_capsule", side_effect=pg_provision),
            mock.patch.object(app.r2driver_client, "provision_capsule", side_effect=r2_provision),
            self.assertRaises(app.ApiError) as caught,
        ):
            app._create("capsule_1", {}, "owner-1")
        self.assertEqual(caught.exception.status, HTTPStatus.BAD_GATEWAY)
        self.assertEqual(events, ["postgres-provision", "r2-provision", "rollback"])

    def test_retire_failure_stops_teardown_before_artifact_removal(self) -> None:
        events: list[str] = []
        record = types.SimpleNamespace(db_dropped=False)
        brain = object()

        def stop(_brain):
            events.append("brain-stop")
            return True

        def retire(_cid):
            events.append("r2-retire")
            raise app.r2driver_client.R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver is unavailable", category="test")

        with (
            _patched(
                _owned_teardown_brain=lambda *_args: (True, brain),
                _stop_teardown_brain=stop,
                _purge_teardown_credentials=lambda _brain: events.append("artifact-remove") or True,
            ),
            mock.patch.object(app.cleanup_state, "begin", return_value=record),
            mock.patch.object(app.cleanup_state, "finish") as finish,
            mock.patch.object(app.r2driver_client, "retire_capsule", side_effect=retire),
        ):
            result = app._teardown("capsule_1", owner="owner-1", brain_id="brain-id")
        self.assertFalse(result.complete)
        self.assertEqual(events, ["brain-stop", "r2-retire"])
        finish.assert_not_called()

    def test_finalization_orders_r2_before_postgres_and_cleanup_record(self) -> None:
        events: list[str] = []
        record = object()
        with (
            mock.patch.object(
                app.r2driver_client,
                "finalize_capsule_drop",
                side_effect=lambda _cid: events.append("r2-finalize"),
            ),
            mock.patch.object(
                app.pgdriver_client,
                "finalize_capsule_drop",
                side_effect=lambda _cid: events.append("postgres-finalize"),
            ),
            mock.patch.object(app.cleanup_state, "finish", side_effect=lambda _record: events.append("cleanup-finish")),
        ):
            self.assertTrue(app._finalize_teardown("capsule_1", record))
        self.assertEqual(events, ["r2-finalize", "postgres-finalize", "cleanup-finish"])


if __name__ == "__main__":
    unittest.main()
