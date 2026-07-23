"""Behavior contracts for the shared deny-by-default egress proxies."""

from __future__ import annotations

import base64
import importlib.util
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path

DRIVERS = Path(__file__).resolve().parents[2]


def _load_driver(relative: str, name: str):
    """Load one standalone driver without leaking its sibling ``audit`` module."""
    source = DRIVERS / relative / "app.py"
    previous_audit = sys.modules.pop("audit", None)
    sys.path.insert(0, str(source.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)
        sys.modules.pop("audit", None)
        if previous_audit is not None:
            sys.modules["audit"] = previous_audit


APP_EGRESS = _load_driver("app-egress", "app_egress_test_target")
BRAIN_EGRESS = _load_driver("egress", "brain_egress_test_target")


class AppPolicyTest(unittest.TestCase):
    def test_policy_is_exact_case_insensitive_and_deny_by_default(self) -> None:
        policy = {"tok-shop": frozenset({"api.example.com", "pay.shimpz.com"})}

        self.assertTrue(APP_EGRESS.permitted("tok-shop", "API.EXAMPLE.COM", 443, policy))
        self.assertFalse(APP_EGRESS.permitted("tok-shop", "evil.com", 443, policy))
        self.assertFalse(APP_EGRESS.permitted("tok-shop", "api.example.com", 80, policy))
        self.assertFalse(APP_EGRESS.permitted("tok-unknown", "api.example.com", 443, policy))
        self.assertFalse(APP_EGRESS.permitted("", "api.example.com", 443, policy))
        self.assertFalse(APP_EGRESS.permitted("tok-empty", "api.example.com", 443, {"tok-empty": frozenset()}))

    def test_policy_loader_normalizes_valid_files_and_skips_invalid_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tok-a.json").write_text('["graph.facebook.com", "API.X.com."]', encoding="utf-8")
            (root / "bad.json").write_text("{not json", encoding="utf-8")
            (root / "object.json").write_text('{"host":"open.example"}', encoding="utf-8")

            policy = APP_EGRESS.load_policy(root)

        self.assertEqual(policy, {"tok-a": frozenset({"graph.facebook.com", "api.x.com"})})
        self.assertEqual(APP_EGRESS.load_policy(Path(directory)), {})

    def test_proxy_authorization_accepts_only_a_basic_username_token(self) -> None:
        identity = "opaque-test-identity"
        encoded = base64.b64encode(f"{identity}:".encode()).decode()

        self.assertEqual(
            APP_EGRESS.extract_token(f"CONNECT x:443 HTTP/1.1\r\nProxy-Authorization: Basic {encoded}"),
            identity,
        )
        self.assertIsNone(APP_EGRESS.extract_token("CONNECT x:443 HTTP/1.1\r\nHost: x"))
        self.assertIsNone(APP_EGRESS.extract_token("CONNECT x:443\r\nProxy-Authorization: Bearer xyz"))
        self.assertIsNone(APP_EGRESS.extract_token("CONNECT x:443\r\nProxy-Authorization: Basic !!!notb64"))

    def test_resolution_rejects_private_and_metadata_addresses(self) -> None:
        self.assertIsNone(APP_EGRESS.resolve_public("127.0.0.1", 443))
        self.assertIsNone(APP_EGRESS.resolve_public("10.0.0.1", 443))
        self.assertIsNone(APP_EGRESS.resolve_public("169.254.169.254", 443))
        resolved = APP_EGRESS.resolve_public("1.1.1.1", 443)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved[1][0], "1.1.1.1")


class ProxyHandlerTest(unittest.TestCase):
    def _request(self, payload: bytes) -> bytes:
        server = APP_EGRESS.Server(("127.0.0.1", 0), APP_EGRESS.Handler, bind_and_activate=False)
        accepted, peer = socket.socketpair()
        peer.settimeout(2)
        previous_audit_path = APP_EGRESS.audit.AUDIT_PATH
        with tempfile.TemporaryDirectory() as directory:
            APP_EGRESS.audit.AUDIT_PATH = Path(directory) / "audit.jsonl"
            try:
                peer.sendall(payload)
                server.process_request(accepted, ("127.0.0.1", 12345))
                return peer.recv(512)
            finally:
                APP_EGRESS.audit.AUDIT_PATH = previous_audit_path
                peer.close()
                server.server_close()

    def test_handler_rejects_non_connect_requests(self) -> None:
        response = self._request(b"GET https://example.com HTTP/1.1\r\nHost: example.com\r\n\r\n")

        self.assertTrue(response.startswith(b"HTTP/1.1 405"))

    def test_handler_requires_proxy_authentication(self) -> None:
        response = self._request(b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com\r\n\r\n")

        self.assertTrue(response.startswith(b"HTTP/1.1 407"))
        self.assertIn(b"Proxy-Authenticate: Basic", response)


class ProxyCapacityTest(unittest.TestCase):
    @staticmethod
    def _permit_released(server) -> bool:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if server._request_slots.acquire(blocking=False):
                server._request_slots.release()
                return True
            time.sleep(0.01)
        return False

    def _exercise_bounded_server(self, module) -> None:
        server = module.Server(
            ("127.0.0.1", 0),
            module.Handler,
            bind_and_activate=False,
            max_concurrency=1,
            max_source_concurrency=1,
        )
        rejected = rejected_peer = accepted = accepted_peer = None
        try:
            self.assertEqual(server.request_queue_size, 16)
            self.assertTrue(server._request_slots.acquire(blocking=False))
            rejected, rejected_peer = socket.socketpair()
            server.process_request(rejected, ("local", 1))
            self.assertTrue(rejected_peer.recv(256).startswith(b"HTTP/1.1 503"))
            self.assertFalse(server._request_slots.acquire(blocking=False))
            server._request_slots.release()

            accepted, accepted_peer = socket.socketpair()
            server.process_request(accepted, ("local", 2))
            accepted_peer.close()
            accepted_peer = None
            self.assertTrue(self._permit_released(server))
            self.assertEqual(server._source_counts, {})
        finally:
            for stream in (rejected, rejected_peer, accepted, accepted_peer):
                if stream is not None:
                    stream.close()
            server.server_close()

    def test_both_proxies_bound_workers_before_thread_creation(self) -> None:
        for module in (APP_EGRESS, BRAIN_EGRESS):
            with self.subTest(module=module.__name__):
                self._exercise_bounded_server(module)

    def test_both_proxies_preserve_capacity_between_sources(self) -> None:
        for module in (APP_EGRESS, BRAIN_EGRESS):
            with self.subTest(module=module.__name__):
                server = module.Server(
                    ("127.0.0.1", 0),
                    module.Handler,
                    bind_and_activate=False,
                    max_concurrency=2,
                    max_source_concurrency=1,
                )
                source_a = ("10.0.0.2", 1000)
                source_b = ("10.0.0.3", 1001)
                try:
                    self.assertTrue(server._acquire_request_slot(source_a))
                    self.assertFalse(server._acquire_request_slot(source_a))
                    self.assertTrue(server._acquire_request_slot(source_b))
                    self.assertFalse(server._acquire_request_slot(("10.0.0.4", 1002)))
                    server._release_request_slot(source_a)
                    self.assertTrue(server._acquire_request_slot(source_a))
                    server._release_request_slot(source_a)
                    server._release_request_slot(source_b)
                    self.assertEqual(server._source_counts, {})
                finally:
                    server.server_close()


if __name__ == "__main__":
    unittest.main()
