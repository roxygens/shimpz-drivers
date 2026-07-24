"""Characterize the shared hosted/local HTTP boundary decisions."""

from __future__ import annotations

import sys
import unittest
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import local_app
from hosted_app_fixture import app, runtime_state
from http_boundary import strict as strict_http


class SharedStrictHttpTest(unittest.TestCase):
    @staticmethod
    def _handler(handler_type: type, body: bytes, headers: tuple[tuple[str, str], ...]):
        handler = object.__new__(handler_type)
        handler.headers = Message()
        for name, value in headers:
            handler.headers.add_header(name, value)
        handler.rfile = BytesIO(body)
        return handler

    def test_hosted_and_local_wrappers_make_the_same_body_decision(self) -> None:
        cases = (
            (b'{"a":1,"a":2}', (("Content-Type", "application/json"),), HTTPStatus.BAD_REQUEST),
            (b'{"a":NaN}', (("Content-Type", "application/json"),), HTTPStatus.BAD_REQUEST),
            (b"[]", (("Content-Type", "application/json"),), HTTPStatus.UNPROCESSABLE_ENTITY),
            (b"{}", (("Transfer-Encoding", "chunked"),), HTTPStatus.BAD_REQUEST),
        )
        for body, extra_headers, expected in cases:
            headers = (("Content-Length", str(len(body))), *extra_headers)
            hosted = self._handler(app.Handler, body, headers)
            local = self._handler(local_app.Handler, body, headers)
            with self.subTest(body=body):
                with self.assertRaises(runtime_state.ApiError) as hosted_error:
                    hosted._read_body()
                with self.assertRaises(local_app.ApiProblem) as local_error:
                    local._body()
                self.assertEqual((hosted_error.exception.status, local_error.exception.status), (expected, expected))

    def test_hosted_and_local_wrappers_reject_the_same_encoded_route(self) -> None:
        hosted = self._handler(app.Handler, b"", ())
        hosted.path = "/v1/teams/%74eam_1"
        local = self._handler(local_app.Handler, b"", ())
        local.path = hosted.path

        with self.assertRaises(runtime_state.ApiError) as hosted_error:
            hosted._route("GET", ("operator", None))
        with self.assertRaises(local_app.ApiProblem) as local_error:
            local._path_parts()

        self.assertEqual(
            (hosted_error.exception.status, local_error.exception.status),
            (HTTPStatus.BAD_REQUEST, HTTPStatus.BAD_REQUEST),
        )

    def test_hosted_and_local_wrappers_read_the_same_raw_file_contract(self) -> None:
        body = b"Team private data"
        headers = (
            ("Content-Length", str(len(body))),
            ("Content-Type", "text/plain"),
            ("X-Shimpz-Filename", "brief%20%E2%9C%93.txt"),
        )
        hosted = self._handler(app.Handler, body, headers)
        local = self._handler(local_app.Handler, body, headers)

        expected = ("brief ✓.txt", body, "text/plain")
        self.assertEqual(hosted._read_file_body(), expected)
        self.assertEqual(local._file_body(), expected)

    def test_file_size_is_rejected_from_content_length_before_body_read(self) -> None:
        class Unreadable:
            @staticmethod
            def read(_length: int) -> bytes:
                raise AssertionError("oversized body must not be read")

        headers = Message()
        headers.add_header("Content-Length", "11")
        headers.add_header("Content-Type", "text/plain")
        headers.add_header("X-Shimpz-Filename", "brief.txt")

        with self.assertRaises(strict_http.HttpContractError) as error:
            strict_http.read_file_upload(headers, Unreadable(), max_bytes=10)

        self.assertEqual(error.exception.status, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)


if __name__ == "__main__":
    unittest.main()
