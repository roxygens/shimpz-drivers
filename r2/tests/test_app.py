from __future__ import annotations

import grp
import http.client
import importlib
import json
import os
import stat
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import credential_store
import principal_store
import r2_client
import token_store

MANAGED_TOKEN = "a" * 64
BACKUP_TOKEN = "b" * 64
PROVISIONER_TOKEN = "c" * 64
CAPSULE_A_TOKEN = "1" * 64
CAPSULE_B_TOKEN = "2" * 64

with (
    mock.patch.object(token_store, "ensure_token", return_value=MANAGED_TOKEN),
    mock.patch.object(token_store, "ensure_private_token", return_value=BACKUP_TOKEN),
    mock.patch.object(token_store, "ensure_group_token", return_value=PROVISIONER_TOKEN),
):
    app = importlib.import_module("app")


def values(*, bucket: str = "capsule-bucket", secret: str = "d" * 64) -> dict[str, str]:
    return {
        "account_id": "e" * 32,
        "access_key_id": "ACCESS_KEY_ID_1234567890",
        "secret_access_key": secret,
        "bucket": bucket,
    }


def create_body(idempotency_key: str, *, bucket: str = "capsule-bucket", secret: str = "d" * 64):
    return {
        "profile_id": "s3-access-key",
        "label": "Capsule R2",
        "values": values(bucket=bucket, secret=secret),
        "idempotency_key": idempotency_key,
    }


class AppLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        app.credential_store.STORE = credential_store.CredentialStore(
            root / "credentials" / "state.json",
            root / "keyring" / "aes256.key",
        )
        app.principal_store.STORE = principal_store.PrincipalStore(root / "principals" / "state.json")
        self.audit_events: list[tuple[object, ...]] = []
        self.audit_patch = mock.patch.object(app.audit, "log", side_effect=self.audit)
        self.audit_patch.start()
        self.probe_error: r2_client.R2Error | None = None
        self.probe_calls = 0
        self.probe_entered: threading.Event | None = None
        self.probe_release: threading.Event | None = None
        self.probe_patch = mock.patch.object(app.r2_client, "probe", side_effect=self.probe)
        self.probe_patch.start()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.01), daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.probe_patch.stop()
        self.audit_patch.stop()
        self.temporary.cleanup()

    def audit(self, *args: object, **kwargs: object) -> str:
        self.audit_events.append((*args, kwargs))
        return "f" * 32

    def probe(self, *, credentials: r2_client.R2Credentials | None = None) -> bool:
        self.assertIsNotNone(credentials)
        self.probe_calls += 1
        if self.probe_entered is not None and self.probe_release is not None:
            self.probe_entered.set()
            self.assertTrue(self.probe_release.wait(1))
        if self.probe_error is not None:
            raise self.probe_error
        return True

    def request(
        self,
        method: str,
        path: str,
        body: object | bytes | None = None,
        *,
        token: str | None = None,
    ) -> tuple[int, dict[str, object]]:
        headers = {"Accept": "application/json"}
        encoded: bytes | None
        if isinstance(body, bytes):
            encoded = body
            headers["Content-Type"] = "application/json"
        elif body is not None:
            encoded = json.dumps(body, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        else:
            encoded = None
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        connection.request(method, path, body=encoded, headers=headers)
        response = connection.getresponse()
        payload = response.read()
        connection.close()
        return response.status, json.loads(payload)

    def provision(self, capsule_id: str, principal_token: str) -> None:
        status, payload = self.request(
            "POST",
            "/v1/capsules/provision",
            {"capsule_id": capsule_id, "principal_token": principal_token},
            token=PROVISIONER_TOKEN,
        )
        self.assertEqual((status, payload), (200, {"status": "active"}))

    def test_many_credentials_are_isolated_atomic_and_never_publicly_secret(self) -> None:
        status, _ = self.request(
            "POST",
            "/v1/capsules/provision",
            {"capsule_id": "capsule_a", "principal_token": CAPSULE_A_TOKEN},
            token=MANAGED_TOKEN,
        )
        self.assertEqual(status, 403, "the managed data-plane token must not provision Capsules")
        self.provision("capsule_a", CAPSULE_A_TOKEN)
        self.provision("capsule_b", CAPSULE_B_TOKEN)
        status, _ = self.request("GET", "/v1/capsules/capsule_a/credentials")
        self.assertEqual(status, 403)

        first_body = create_body("11111111-1111-4111-8111-111111111111")
        status, first = self.request("POST", "/v1/capsules/capsule_a/credentials", first_body, token=CAPSULE_A_TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(set(first), {"id", "profile_id", "label", "generation", "status", "created_at", "updated_at"})
        self.assertLessEqual(len(str(first["id"])), 64)
        self.assertEqual(first["generation"], 1)
        first_probe_count = self.probe_calls

        self.probe_error = r2_client.R2Error("safe", category="network")
        with mock.patch.object(app.credential_store.STORE, "capsule_record_count", return_value=256):
            retry_status, retried = self.request(
                "POST", "/v1/capsules/capsule_a/credentials", first_body, token=CAPSULE_A_TOKEN
            )
        self.assertEqual((retry_status, retried), (200, first))
        self.assertEqual(self.probe_calls, first_probe_count, "an exact retry must not depend on provider availability")
        conflict_body = {**first_body, "label": "Changed request"}
        conflict_status, _ = self.request(
            "POST", "/v1/capsules/capsule_a/credentials", conflict_body, token=CAPSULE_A_TOKEN
        )
        self.assertEqual(conflict_status, 409)
        self.assertEqual(self.probe_calls, first_probe_count)
        self.probe_error = None

        status, second = self.request(
            "POST",
            "/v1/capsules/capsule_a/credentials",
            create_body("22222222-2222-4222-8222-222222222222", bucket="capsule-a-two"),
            token=CAPSULE_A_TOKEN,
        )
        self.assertEqual(status, 200)
        status, third = self.request(
            "POST",
            "/v1/capsules/capsule_b/credentials",
            create_body("11111111-1111-4111-8111-111111111111", bucket="capsule-b-one"),
            token=CAPSULE_B_TOKEN,
        )
        self.assertEqual(status, 200)
        self.assertEqual(len({first["id"], second["id"], third["id"]}), 3)

        status, listed_a = self.request("GET", "/v1/capsules/capsule_a/credentials", token=CAPSULE_A_TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual({item["id"] for item in listed_a["credentials"]}, {first["id"], second["id"]})
        status, _ = self.request("GET", "/v1/capsules/capsule_b/credentials", token=CAPSULE_A_TOKEN)
        self.assertEqual(status, 404, "a valid bearer for another Capsule must not disclose scope existence")

        rotate = {
            "profile_id": "s3-access-key",
            "label": "Rotated R2",
            "values": values(secret="f" * 64),
            "expected_generation": 1,
        }
        self.probe_error = r2_client.R2Error("safe", category="authentication")
        status, _ = self.request(
            "PUT", f"/v1/capsules/capsule_a/credentials/{first['id']}", rotate, token=CAPSULE_A_TOKEN
        )
        self.assertEqual(status, 422)
        self.assertEqual(app.credential_store.STORE.resolve("capsule_a", first["id"]).metadata.generation, 1)
        stale = {**rotate, "expected_generation": 99}
        calls_before_stale = self.probe_calls
        status, _ = self.request(
            "PUT", f"/v1/capsules/capsule_a/credentials/{first['id']}", stale, token=CAPSULE_A_TOKEN
        )
        self.assertEqual(status, 409)
        self.assertEqual(self.probe_calls, calls_before_stale, "stale CAS must fail before provider I/O")
        self.probe_error = None
        status, rotated = self.request(
            "PUT", f"/v1/capsules/capsule_a/credentials/{first['id']}", rotate, token=CAPSULE_A_TOKEN
        )
        self.assertEqual((status, rotated["generation"]), (200, 2))
        status, verified = self.request(
            "POST", f"/v1/capsules/capsule_a/credentials/{first['id']}/verify", {}, token=CAPSULE_A_TOKEN
        )
        self.assertEqual(
            (status, verified),
            (
                200,
                {"id": first["id"], "generation": 2, "verdict": "valid", "trace_id": "f" * 32},
            ),
        )

        status, removed = self.request(
            "DELETE",
            f"/v1/capsules/capsule_a/credentials/{first['id']}",
            {"expected_generation": 2},
            token=CAPSULE_A_TOKEN,
        )
        self.assertEqual((status, removed["status"], removed["generation"]), (200, "revoked", 3))
        retry_status, retry_removed = self.request(
            "DELETE",
            f"/v1/capsules/capsule_a/credentials/{first['id']}",
            {"expected_generation": 2},
            token=CAPSULE_A_TOKEN,
        )
        self.assertEqual((retry_status, retry_removed), (200, removed))

        state_bytes = app.credential_store.STORE.state_path.read_bytes()
        public_bytes = json.dumps([first, second, third, listed_a, rotated, verified, removed]).encode()
        audit_bytes = repr(self.audit_events).encode()
        for secret in (*values().values(), "f" * 64, first_body["idempotency_key"], CAPSULE_A_TOKEN):
            marker = str(secret).encode()
            self.assertNotIn(marker, state_bytes)
            self.assertNotIn(marker, public_bytes)
            self.assertNotIn(marker, audit_bytes)

    def test_provider_failure_does_not_persist_and_teardown_is_retry_safe(self) -> None:
        self.provision("capsule_b", CAPSULE_B_TOKEN)
        self.probe_error = r2_client.R2Error("raw provider detail", category="authentication")
        status, payload = self.request(
            "POST",
            "/v1/capsules/capsule_b/credentials",
            create_body("44444444-4444-4444-8444-444444444444"),
            token=CAPSULE_B_TOKEN,
        )
        self.assertEqual((status, payload), (422, {"error": "R2 rejected the credential bundle"}))
        self.assertEqual(app.credential_store.STORE.list_metadata("capsule_b"), ())
        self.probe_error = None
        status, created = self.request(
            "POST",
            "/v1/capsules/capsule_b/credentials",
            create_body("55555555-5555-4555-8555-555555555555"),
            token=CAPSULE_B_TOKEN,
        )
        self.assertEqual(status, 200)

        status, removed = self.request(
            "DELETE",
            f"/v1/capsules/capsule_b/credentials/{created['id']}",
            {"expected_generation": 1},
            token=CAPSULE_B_TOKEN,
        )
        self.assertEqual((status, removed["status"]), (200, "revoked"))
        credentials_before = app.credential_store.STORE.state_path.read_bytes()
        principals_before = app.principal_store.STORE.state_path.read_bytes()
        status, _ = self.request("POST", "/v1/capsules/finalize", {"capsule_id": "capsule_b"}, token=PROVISIONER_TOKEN)
        self.assertEqual(status, 409)
        self.assertEqual(app.credential_store.STORE.state_path.read_bytes(), credentials_before)
        self.assertEqual(app.principal_store.STORE.state_path.read_bytes(), principals_before)

        for _ in range(2):
            status, payload = self.request(
                "POST", "/v1/capsules/retire", {"capsule_id": "capsule_b"}, token=PROVISIONER_TOKEN
            )
            self.assertEqual((status, payload), (200, {"status": "retired"}))
        status, _ = self.request("GET", "/v1/capsules/capsule_b/credentials", token=CAPSULE_B_TOKEN)
        self.assertEqual(status, 404)
        state = json.loads(app.credential_store.STORE.state_path.read_text())
        self.assertIsNone(state["capsules"]["capsule_b"][created["id"]]["envelope"])
        for _ in range(2):
            status, payload = self.request(
                "POST", "/v1/capsules/finalize", {"capsule_id": "capsule_b"}, token=PROVISIONER_TOKEN
            )
            self.assertEqual((status, payload), (200, {"status": "finalized"}))
        self.assertNotIn("capsule_b", json.loads(app.credential_store.STORE.state_path.read_text())["capsules"])
        replay_status, _ = self.request(
            "POST",
            "/v1/capsules/provision",
            {"capsule_id": "capsule_b", "principal_token": CAPSULE_B_TOKEN},
            token=PROVISIONER_TOKEN,
        )
        self.assertEqual(replay_status, 409)
        self.provision("capsule_b", "3" * 64)

    def test_body_limit_is_413(self) -> None:
        self.provision("capsule_a", CAPSULE_A_TOKEN)
        status, _ = self.request(
            "POST", "/v1/capsules/capsule_a/credentials", b"{" + b" " * (64 * 1024), token=CAPSULE_A_TOKEN
        )
        self.assertEqual(status, 413)

    def test_health_and_startup_fail_closed_after_key_loss(self) -> None:
        self.provision("capsule_a", CAPSULE_A_TOKEN)
        status, _ = self.request(
            "POST",
            "/v1/capsules/capsule_a/credentials",
            create_body("77777777-7777-4777-8777-777777777777"),
            token=CAPSULE_A_TOKEN,
        )
        self.assertEqual(status, 200)
        app.credential_store.STORE.key_path.unlink()
        status, payload = self.request("GET", "/healthz")
        self.assertEqual((status, payload), (503, {"error": "credential storage is unavailable"}))
        with (
            mock.patch.object(app, "_prepare_backup_spool"),
            mock.patch.object(
                app.credential_store.STORE,
                "check_health",
                side_effect=credential_store.CredentialStoreError("private detail"),
            ),
            mock.patch.object(app, "ThreadingHTTPServer") as server,
            self.assertRaises(SystemExit),
        ):
            app.main()
        server.assert_not_called()

    def test_retire_waits_for_in_flight_probe_then_revokes_the_result(self) -> None:
        self.provision("capsule_a", CAPSULE_A_TOKEN)
        self.probe_entered = threading.Event()
        self.probe_release = threading.Event()
        results: dict[str, tuple[int, dict[str, object]]] = {}

        create = threading.Thread(
            target=lambda: results.setdefault(
                "create",
                self.request(
                    "POST",
                    "/v1/capsules/capsule_a/credentials",
                    create_body("66666666-6666-4666-8666-666666666666"),
                    token=CAPSULE_A_TOKEN,
                ),
            )
        )
        create.start()
        self.assertTrue(self.probe_entered.wait(1))
        retire = threading.Thread(
            target=lambda: results.setdefault(
                "retire",
                self.request(
                    "POST",
                    "/v1/capsules/retire",
                    {"capsule_id": "capsule_a"},
                    token=PROVISIONER_TOKEN,
                ),
            )
        )
        retire.start()
        retire.join(0.03)
        self.assertTrue(retire.is_alive(), "retire must not pass an in-flight scoped provider probe")
        self.probe_release.set()
        create.join(1)
        retire.join(1)
        self.assertEqual(results["create"][0], 200)
        self.assertEqual(results["retire"], (200, {"status": "retired"}))
        created_id = results["create"][1]["id"]
        state = json.loads(app.credential_store.STORE.state_path.read_text())
        self.assertIsNone(state["capsules"]["capsule_a"][created_id]["envelope"])
        status, _ = self.request("GET", "/v1/capsules/capsule_a/credentials", token=CAPSULE_A_TOKEN)
        self.assertEqual(status, 404)

    def test_provisioner_token_file_is_group_readable_only(self) -> None:
        root = Path(self.temporary.name) / "shared-token"
        root.mkdir(mode=0o750)
        group = grp.getgrgid(os.getegid()).gr_name
        token_path = root / "token"
        generated = token_store.ensure_group_token(token_path, group)
        info = token_path.stat()
        self.assertEqual(len(generated), 64)
        self.assertEqual(stat.S_IMODE(info.st_mode), 0o440)
        self.assertEqual(info.st_gid, os.getegid())
        self.assertEqual(token_store.ensure_group_token(token_path, group), generated)


if __name__ == "__main__":
    unittest.main()
