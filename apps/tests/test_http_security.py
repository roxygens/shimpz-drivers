from __future__ import annotations

import ast
import unittest
from pathlib import Path

APP_SOURCE = Path(__file__).resolve().parents[1] / "app.py"


def _handler_method(name: str) -> ast.FunctionDef:
    tree = ast.parse(APP_SOURCE.read_text(encoding="utf-8"))
    handler = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Handler")
    return next(node for node in handler.body if isinstance(node, ast.FunctionDef) and node.name == name)


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


if __name__ == "__main__":
    unittest.main()
