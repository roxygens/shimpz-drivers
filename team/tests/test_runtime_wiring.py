from __future__ import annotations

import ast
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app.py"
TREE = ast.parse(APP.read_text(encoding="utf-8"))


def _function(name: str) -> ast.FunctionDef:
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _calls(function: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        current = node.func
        parts: list[str] = []
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        if parts:
            names.add(".".join(reversed(parts)))
    return names


class RuntimeWiringTests(unittest.TestCase):
    def test_controller_contains_no_provider_cli_or_oauth_bridge(self) -> None:
        source = APP.read_text(encoding="utf-8").lower()
        for legacy in ("claude", "codex", "oauth", "shimpz-chat-exec", "shimpz-chat-stop"):
            with self.subTest(legacy=legacy):
                self.assertNotIn(legacy, source)

    def test_chat_uses_runtime_and_controller_owned_power_execution(self) -> None:
        calls = _calls(_function("_run_hosted_chat_segment"))
        drive_calls = _calls(_function("_drive_hosted_chat"))
        setup_calls = _calls(_function("_hosted_chat_setup"))
        self.assertIn("chat_orchestrator.run_until_pause", drive_calls)
        self.assertIn("chat_orchestrator.continue_after_pause", drive_calls)
        self.assertIn("_invoke_assistant_power", calls)
        self.assertIn("_inference_store.load", setup_calls)
        for legacy in ("_run_brain_once", "_brain_exec"):
            self.assertNotIn(legacy, calls)
            self.assertNotIn(legacy, drive_calls)

    def test_model_selection_does_not_replace_the_team(self) -> None:
        calls = _calls(_function("_create"))
        self.assertIn("inference_config.normalize", calls)
        self.assertIn("_inference_store.save", calls)
        self.assertNotIn("_replace_brain", calls)
        self.assertNotIn("brain_credentials_client.resolve", calls)

    def test_stop_marks_the_controller_turn_without_execing_provider_cli(self) -> None:
        calls = _calls(_function("_stop_chat"))
        self.assertIn("_stop_active_power", calls)
        self.assertNotIn("_request_chat_stop", calls)
        self.assertNotIn("_terminate_chat_token", calls)

    def test_controller_owns_runtime_token_bootstrap(self) -> None:
        calls = _calls(_function("main"))
        self.assertIn("brain_runtime_token_store.ensure", calls)

    def test_inference_configuration_is_metadata_only(self) -> None:
        calls = _calls(_function("_configure_inference"))
        self.assertIn("inference_config.normalize", calls)
        self.assertIn("_inference_store.save", calls)
        self.assertIn("_require_current_authorization", calls)
        self.assertNotIn("brain_credentials_client.resolve", calls)
        self.assertNotIn("_replace_brain", calls)


if __name__ == "__main__":
    unittest.main()
