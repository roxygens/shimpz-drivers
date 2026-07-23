from __future__ import annotations

import io
import re
import sys
import unittest
from email.message import Message
from http import HTTPStatus
from pathlib import Path
from unittest import mock

APPS = Path(__file__).resolve().parents[1]
PG_COPY = APPS.parent / "pg" / "stdlib_http.py"
sys.path.insert(0, str(APPS))

import stdlib_http


class StdlibHttpTests(unittest.TestCase):
    def test_vendored_pg_copy_is_byte_identical(self) -> None:
        self.assertEqual(Path(stdlib_http.__file__).read_bytes(), PG_COPY.read_bytes())

    def test_bearer_requires_one_header_and_uses_constant_time_comparison(self) -> None:
        headers = Message()
        headers["Authorization"] = "Bearer expected"
        with mock.patch.object(
            stdlib_http.hmac,
            "compare_digest",
            wraps=stdlib_http.hmac.compare_digest,
        ) as compare:
            self.assertTrue(stdlib_http.bearer_authorized(headers, "expected"))
        compare.assert_called_once_with("expected", "expected")

        headers["Authorization"] = "Bearer expected"
        self.assertFalse(stdlib_http.bearer_authorized(headers, "expected"))

    def test_json_reader_bounds_and_validates_objects(self) -> None:
        cases = (
            ("invalid", b"{}", HTTPStatus.BAD_REQUEST),
            ("-1", b"", HTTPStatus.REQUEST_ENTITY_TOO_LARGE),
            ("3", b"[]", HTTPStatus.BAD_REQUEST),
        )
        for length, body, expected in cases:
            with self.subTest(length=length, body=body), self.assertRaises(stdlib_http.HttpError) as caught:
                stdlib_http.read_json_body({"Content-Length": length}, io.BytesIO(body), max_bytes=16)
            self.assertEqual(caught.exception.status, expected)

        self.assertEqual(
            stdlib_http.read_json_body({"Content-Length": "7"}, io.BytesIO(b'{"a":1}'), max_bytes=16),
            {"a": 1},
        )

    def test_json_response_sets_exact_framing(self) -> None:
        handler = mock.Mock()
        handler.wfile = io.BytesIO()

        stdlib_http.send_json(handler, HTTPStatus.OK, {"ok": True})

        handler.send_response.assert_called_once_with(HTTPStatus.OK)
        self.assertEqual(
            handler.send_header.call_args_list,
            [
                mock.call("Content-Type", "application/json"),
                mock.call("Content-Length", "12"),
            ],
        )
        handler.end_headers.assert_called_once_with()
        self.assertEqual(handler.wfile.getvalue(), b'{"ok": true}')

    def test_declarative_route_resolves_named_params_and_query(self) -> None:
        routes = (stdlib_http.Route("GET", re.compile(r"^/v1/items/(?P<item>[^/]+)$"), "show"),)

        matched = stdlib_http.resolve_route(routes, "GET", "/v1/items/one?view=small")

        self.assertEqual(matched.operation, "show")
        self.assertEqual(matched.params, {"item": "one"})
        self.assertEqual(matched.query, {"view": ["small"]})
        with self.assertRaises(stdlib_http.HttpError) as caught:
            stdlib_http.resolve_route(routes, "POST", "/v1/items/one")
        self.assertEqual(caught.exception.status, HTTPStatus.NOT_FOUND)

    def test_dispatch_classifies_expected_and_redacts_unexpected_errors(self) -> None:
        emitted = []

        stdlib_http.dispatch(
            lambda: (_ for _ in ()).throw(RuntimeError("private detail")),
            classify=lambda _exc: None,
            emit=emitted.append,
            unexpected_message="internal error",
        )

        self.assertEqual(
            emitted,
            [
                stdlib_http.HttpFailure(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "internal error",
                    "RuntimeError",
                    "error",
                )
            ],
        )
        self.assertNotIn("private detail", repr(emitted))


if __name__ == "__main__":
    unittest.main()
