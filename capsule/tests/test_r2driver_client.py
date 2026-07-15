from __future__ import annotations

import os
import stat
import tempfile
import threading
import unittest
from http import HTTPStatus
from pathlib import Path
from typing import ClassVar
from unittest import mock

import r2driver_client as client


class _Response:
    def __init__(
        self,
        status: int = 200,
        body: bytes = b"{}",
        *,
        content_type: str = "application/json",
        content_length: str | None = None,
    ) -> None:
        self.status = status
        self._body = body
        self._headers = {
            "Content-Type": content_type,
            "Content-Length": content_length if content_length is not None else str(len(body)),
        }

    def getheader(self, name: str) -> str | None:
        return self._headers.get(name)

    def read(self, amount: int) -> bytes:
        return self._body[:amount]


class _Connection:
    response = _Response()
    requests: ClassVar[list[tuple]] = []

    def __init__(self, host: str, port: int, timeout: int) -> None:
        self.endpoint = (host, port, timeout)

    def request(self, *args, **kwargs) -> None:
        type(self).requests.append((self.endpoint, args, kwargs))

    def getresponse(self) -> _Response:
        return type(self).response

    def close(self) -> None:
        pass


class R2DriverClientTests(unittest.TestCase):
    def setUp(self) -> None:
        _Connection.requests = []
        _Connection.response = _Response()

    def test_endpoint_is_one_fixed_http_origin(self) -> None:
        self.assertEqual(client._parse_endpoint("http://r2-driver:7075"), client._Endpoint("r2-driver", 7075))
        for invalid in (
            "https://r2-driver:7075",
            "http://user@r2-driver:7075",
            "http://r2-driver:7075/path",
            "http://r2-driver:7075?next=evil",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                client._parse_endpoint(invalid)

    def test_request_is_bounded_json_and_never_logs_or_reflects_secret_errors(self) -> None:
        sentinel = "SENTINEL-R2-SECRET"
        _Connection.response = _Response(403, (f'{{"error":"{sentinel}"}}').encode())
        with (
            mock.patch.object(client.http.client, "HTTPConnection", _Connection),
            self.assertRaises(client.R2DriverError) as caught,
        ):
            client._call("POST", "/fixed", {"value": sentinel}, bearer="a" * 64)
        self.assertEqual(caught.exception.status, HTTPStatus.BAD_GATEWAY)
        self.assertNotIn(sentinel, str(caught.exception))
        endpoint, args, kwargs = _Connection.requests[0]
        self.assertEqual(endpoint, ("r2-driver", 7075, 30))
        self.assertEqual(args[:2], ("POST", "/fixed"))
        self.assertLessEqual(len(kwargs["body"]), client.MAX_JSON_REQUEST_BYTES)

        with self.assertRaises(client.R2DriverError) as oversized:
            client._call("POST", "/fixed", {"value": "x" * client.MAX_JSON_REQUEST_BYTES})
        self.assertEqual(oversized.exception.status, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)

    def test_only_safe_upstream_statuses_cross_boundary(self) -> None:
        for upstream, expected in (
            (400, 400),
            (404, 404),
            (409, 409),
            (413, 413),
            (422, 422),
            (429, 429),
            (401, 502),
            (403, 502),
            (503, 502),
            (302, 502),
        ):
            _Connection.response = _Response(upstream)
            with self.subTest(upstream=upstream), mock.patch.object(client.http.client, "HTTPConnection", _Connection):
                with self.assertRaises(client.R2DriverError) as caught:
                    client._call("GET", "/fixed")
                self.assertEqual(caught.exception.status, expected)

    def test_response_requires_bounded_json_object(self) -> None:
        responses = (
            _Response(200, b"[]"),
            _Response(200, b"{}", content_type="text/plain"),
            _Response(200, b"{}", content_length=str(client.MAX_JSON_RESPONSE_BYTES + 1)),
            _Response(200, b"x" * (client.MAX_JSON_RESPONSE_BYTES + 1)),
        )
        for response in responses:
            _Connection.response = response
            with self.subTest(response=response), mock.patch.object(client.http.client, "HTTPConnection", _Connection):
                with self.assertRaises(client.R2DriverError) as caught:
                    client._call("GET", "/fixed")
                self.assertEqual(caught.exception.status, HTTPStatus.BAD_GATEWAY)

    def test_principal_is_atomic_owner_only_and_stable_under_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(client, "PRINCIPAL_DIR", Path(directory)):
            Path(directory).chmod(0o700)
            tokens: list[str] = []

            def create() -> None:
                tokens.append(client._principal("capsule_1", create=True))

            threads = [threading.Thread(target=create) for _ in range(12)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(len(tokens), 12)
            self.assertEqual(len(set(tokens)), 1)
            token_path = Path(directory) / "capsule_1.token"
            metadata = token_path.stat()
            self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o600)
            self.assertEqual(metadata.st_nlink, 1)
            self.assertEqual(metadata.st_size, 64)
            self.assertEqual(list(Path(directory).glob(".*.tmp")), [])

    def test_principal_rejects_symlink_mode_and_hardlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(client, "PRINCIPAL_DIR", Path(directory)):
            root = Path(directory)
            root.chmod(0o700)
            target = root / "target"
            target.write_text("a" * 64, encoding="ascii")
            target.chmod(0o600)
            (root / "capsule_1.token").symlink_to(target)
            with self.assertRaises(client.R2DriverError):
                client._principal("capsule_1", create=False)

            (root / "capsule_1.token").unlink()
            target.chmod(0o640)
            os.link(target, root / "capsule_1.token")
            with self.assertRaises(client.R2DriverError):
                client._principal("capsule_1", create=False)

    def test_finalize_removes_principal_only_after_upstream_200(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(client, "PRINCIPAL_DIR", Path(directory)):
            root = Path(directory)
            root.chmod(0o700)
            client._principal("capsule_1", create=True)
            path = root / "capsule_1.token"
            with (
                mock.patch.object(client, "_read_provisioner", return_value="b" * 64),
                mock.patch.object(
                    client,
                    "_call",
                    side_effect=client.R2DriverError(HTTPStatus.BAD_GATEWAY, "unavailable", category="transport"),
                ),
                self.assertRaises(client.R2DriverError),
            ):
                client.finalize_capsule_drop("capsule_1")
            self.assertTrue(path.exists())

            with (
                mock.patch.object(client, "_read_provisioner", return_value="b" * 64),
                mock.patch.object(client, "_call", return_value={"status": "finalized"}),
            ):
                client.finalize_capsule_drop("capsule_1")
            self.assertFalse(path.exists())

    def test_metadata_projection_rejects_secret_fields(self) -> None:
        metadata = {
            "id": "primary",
            "profile_id": "s3-access-key",
            "label": "Primary",
            "generation": 1,
            "status": "active",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "values": {"secret_access_key": "SENTINEL"},
        }
        with self.assertRaises(client.R2DriverError):
            client._project_credential(metadata)


if __name__ == "__main__":
    unittest.main()
