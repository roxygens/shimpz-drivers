from __future__ import annotations

import ast
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

APP_SOURCE = Path(__file__).resolve().parents[1] / "app.py"


def _handler_method(name: str) -> ast.FunctionDef:
    tree = ast.parse(APP_SOURCE.read_text(encoding="utf-8"))
    handler = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Handler")
    return next(node for node in handler.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _load_app():
    apps_dir = APP_SOURCE.parent
    sys.path.insert(0, str(apps_dir))
    import docker
    import egress_lock
    import manifests
    import token_store

    spec = importlib.util.spec_from_file_location("app_http_security_subject", APP_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load app driver")
    module = importlib.util.module_from_spec(spec)
    with (
        mock.patch.object(egress_lock, "require_enabled"),
        mock.patch.object(docker, "from_env", return_value=object()),
        mock.patch.object(token_store, "ensure_token", return_value="test-token"),
        mock.patch.object(manifests, "resolve_host_projects_root", return_value=Path("/workspace/projects")),
    ):
        spec.loader.exec_module(module)
    return module


class HttpSecurityStaticTests(unittest.TestCase):
    def test_bearer_authorization_uses_constant_time_comparison(self) -> None:
        authorized = _handler_method("_authed")
        returned = next(node for node in authorized.body if isinstance(node, ast.Return))

        self.assertEqual(ast.unparse(returned.value), "hmac.compare_digest(auth, f'Bearer {_token}')")

    def test_unexpected_exception_text_is_redacted_from_http_500(self) -> None:
        dispatch = _handler_method("_dispatch")
        unexpected = next(
            node
            for node in ast.walk(dispatch)
            if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Name) and node.type.id == "Exception"
        )
        boundary = "\n".join(ast.unparse(statement) for statement in unexpected.body)

        self.assertIn("reason=type(exc).__name__", boundary)
        self.assertIn("{'error': 'internal error'}", boundary)
        self.assertNotIn("str(exc)", boundary)


class HttpBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _load_app()

    def test_oversized_body_is_rejected_before_it_is_read(self) -> None:
        handler = object.__new__(self.app.Handler)
        handler.headers = {"Content-Length": str(self.app.MAX_BODY_BYTES + 1)}
        handler.rfile = mock.Mock()

        with self.assertRaises(self.app.ApiError) as caught:
            handler._body()

        self.assertEqual(caught.exception.status, self.app.HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        handler.rfile.read.assert_not_called()

    def test_invalid_and_negative_content_lengths_fail_closed(self) -> None:
        for raw_length, expected_status in (
            ("invalid", self.app.HTTPStatus.BAD_REQUEST),
            ("-1", self.app.HTTPStatus.REQUEST_ENTITY_TOO_LARGE),
        ):
            with self.subTest(raw_length=raw_length):
                handler = object.__new__(self.app.Handler)
                handler.headers = {"Content-Length": raw_length}
                handler.rfile = io.BytesIO(b"")
                with self.assertRaises(self.app.ApiError) as caught:
                    handler._body()
                self.assertEqual(caught.exception.status, expected_status)


if __name__ == "__main__":
    unittest.main()
