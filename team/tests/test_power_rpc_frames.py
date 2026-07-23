"""Adversarial frame contracts for hosted and local Assistant Power RPC."""

from __future__ import annotations

import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import local_app
from test_hosted_app import _patched, app


def _frame(stream_id: int, payload: bytes) -> bytes:
    return struct.pack(">BxxxL", stream_id, len(payload)) + payload


@contextmanager
def _socket_bytes(payload: bytes, *, pieces: tuple[int, ...] = ()):
    reader, writer = socket.socketpair()

    def send() -> None:
        offset = 0
        for size in pieces:
            writer.sendall(payload[offset : offset + size])
            offset += size
            time.sleep(0.005)
        writer.sendall(payload[offset:])
        writer.shutdown(socket.SHUT_WR)

    sender = threading.Thread(target=send, daemon=True)
    sender.start()
    try:
        yield reader
    finally:
        sender.join(timeout=1)
        reader.close()
        writer.close()


class PowerRpcFrameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.local = object.__new__(local_app.LocalController)

    def test_split_stdout_and_stderr_frames_are_read_exactly(self) -> None:
        payload = _frame(1, b'{"ok":') + _frame(2, b"warning") + _frame(1, b"true}")
        with _socket_bytes(payload, pieces=(1, 2, 5, 3, 7)) as hosted_socket:
            stdout, stderr = app._read_rpc_frames(hosted_socket, time.monotonic() + 1)
        with _socket_bytes(payload, pieces=(4, 1, 6, 2)) as local_socket:
            local_stdout, local_stderr = self.local._read_rpc_frames(local_socket, time.monotonic() + 1)

        self.assertEqual(stdout, b'{"ok":true}')
        self.assertEqual(stderr, b"warning")
        self.assertEqual(local_stdout, stdout)
        self.assertEqual(local_stderr, stderr)

    def test_malformed_frames_fail_closed_in_both_readers(self) -> None:
        oversized = struct.pack(">BxxxL", 1, max(app.MAX_ASSISTANT_RPC_OUTPUT_BYTES, local_app.MAX_RESPONSE_BYTES) + 2)
        cases = (
            b"\x01\x00\x00",
            _frame(1, b"payload")[:-2],
            oversized,
            b"garbage!",
        )

        for payload in cases:
            with self.subTest(payload=payload):
                with _socket_bytes(payload) as hosted_socket, self.assertRaises(ValueError):
                    app._read_rpc_frames(hosted_socket, time.monotonic() + 1)
                with _socket_bytes(payload) as local_socket, self.assertRaises(ValueError):
                    self.local._read_rpc_frames(local_socket, time.monotonic() + 1)

    def test_clean_eof_is_the_only_empty_success(self) -> None:
        with _socket_bytes(b"") as hosted_socket:
            self.assertEqual(app._read_rpc_frames(hosted_socket, time.monotonic() + 1), (b"", b""))
        with _socket_bytes(b"") as local_socket:
            self.assertEqual(self.local._read_rpc_frames(local_socket, time.monotonic() + 1), (b"", b""))

    def test_both_controller_bindings_reject_the_same_generation_drift(self) -> None:
        request = app.brain_runtime_client.PowerRequest(
            "interrupt-1", "assistant", "lookup", {"query": "safe"}, "none"
        )
        generation = [1]
        execute = mock.Mock(return_value={"ok": True})
        image = "example.invalid/assistant@sha256:" + "a" * 64
        bindings = (
            (
                SimpleNamespace(container=SimpleNamespace(id="container-1"), image=image),
                lambda item: (item.container.id, item.image),
            ),
            (
                SimpleNamespace(container_id="container-1", spec=SimpleNamespace(image=image)),
                lambda item: (item.container_id, item.spec.image),
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            for index, (binding, identity) in enumerate(bindings):
                with self.subTest(adapter=index):
                    journal = app.power_journal.PowerJournal(Path(directory) / f"journal-{index}.sqlite3")
                    self.addCleanup(journal.close)
                    batch = app.power_execution.PowerBatch(
                        journal,
                        "generation-1",
                        "thread-1",
                        {"assistant": binding},
                        identity,
                        execute,
                        lambda _request: None,
                        lambda _request: (("secret", generation[0]),),
                    )
                    generation[0] = 1
                    batch.prepare((request,))
                    generation[0] = 2
                    with self.assertRaisesRegex(
                        app.power_journal.PowerJournalConflictError,
                        "Power credential generation changed",
                    ):
                        batch.invoke(request)
        execute.assert_not_called()

    def test_hosted_exchange_fail_stops_on_malformed_frame(self) -> None:
        with _socket_bytes(b"truncated") as raw_socket:
            stream = SimpleNamespace(_sock=raw_socket, close=lambda: None)
            create = mock.Mock(return_value={"Id": "exec-1"})
            api = SimpleNamespace(
                exec_create=create,
                exec_start=lambda *_args, **_kwargs: stream,
            )
            fail_stop = mock.Mock()
            container = SimpleNamespace(id="assistant-container")
            with (
                _patched(_docker=SimpleNamespace(api=api), _fail_stop_power=fail_stop),
                mock.patch.object(app.assistant_secret_flow, "encode_private_rpc_envelope", return_value=b"request"),
                self.assertRaises(app.ApiError) as caught,
            ):
                app._assistant_rpc_exchange(
                    "team_1",
                    container,
                    "/app/rpc",
                    "POST",
                    "/v1/powers/test",
                    {},
                    token=None,
                    operation="Assistant Power",
                )

        self.assertEqual(caught.exception.status, HTTPStatus.BAD_GATEWAY)
        fail_stop.assert_called_once_with("team_1", container)
        self.assertEqual(create.call_args.kwargs["workdir"], app.manifests.CONTAINER_TMP)
        self.assertEqual(create.call_args.kwargs["environment"], {})

    def test_local_exchange_fail_stops_on_malformed_frame(self) -> None:
        with _socket_bytes(b"truncated") as raw_socket:
            stream = SimpleNamespace(_sock=raw_socket, close=lambda: None)
            create = mock.Mock(return_value={"Id": "exec-1"})
            api = SimpleNamespace(
                exec_create=create,
                exec_start=lambda *_args, **_kwargs: stream,
            )
            controller = object.__new__(local_app.LocalController)
            controller.client = SimpleNamespace(api=api)
            controller._fail_stop_power = mock.Mock()
            spec = SimpleNamespace(rpc_command="/app/rpc")
            with (
                mock.patch.object(
                    local_app.assistant_secret_flow,
                    "encode_private_rpc_envelope",
                    return_value=b"request",
                ),
                self.assertRaises(local_app.ApiProblem) as caught,
            ):
                controller._rpc(SimpleNamespace(id="assistant-container"), spec, "POST", "/v1/powers/test", {})

        self.assertEqual(caught.exception.status, HTTPStatus.BAD_GATEWAY)
        controller._fail_stop_power.assert_called_once()
        self.assertEqual(create.call_args.kwargs["workdir"], local_app.ASSISTANT_WORKDIR)
        self.assertEqual(create.call_args.kwargs["environment"], {})


if __name__ == "__main__":
    unittest.main()
