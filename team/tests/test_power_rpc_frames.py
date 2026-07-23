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
from hosted_app_fixture import _patched, app
from local_support import assistant_rpc as local_assistant_rpc


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

    def test_rpc_response_distinguishes_results_and_suspensions(self) -> None:
        self.assertEqual(
            app.power_execution.decode_rpc_response(b'{"result":{"ok":true}}'),
            {"ok": True},
        )
        suspension = app.power_execution.decode_rpc_response(b'{"suspend":{"ordinal":0,"kind":"request"}}')
        self.assertIsInstance(suspension, app.power_execution.RpcSuspension)
        self.assertEqual(suspension.payload, {"ordinal": 0, "kind": "request"})
        for invalid in (
            b'{"ok":true}',
            b'{"result":{},"suspend":{}}',
            b'{"suspend":"invalid"}',
        ):
            with self.subTest(invalid=invalid), self.assertRaises(app.power_execution.RpcExchangeError):
                app.power_execution.decode_rpc_response(invalid)

    def test_rpc_failure_kinds_share_one_http_status_table(self) -> None:
        self.assertEqual(
            {
                kind: app.power_execution.rpc_failure_status(kind)
                for kind in ("timeout", "ambiguous", "invalid-result", "failed")
            },
            {
                "timeout": HTTPStatus.GATEWAY_TIMEOUT,
                "ambiguous": HTTPStatus.BAD_GATEWAY,
                "invalid-result": HTTPStatus.BAD_GATEWAY,
                "failed": HTTPStatus.BAD_GATEWAY,
            },
        )
        with self.assertRaisesRegex(AssertionError, "unknown RPC failure"):
            app.power_execution.rpc_failure_status("unknown")

    def test_private_generation_helpers_apply_one_power_contract(self) -> None:
        powers = {
            "lookup": SimpleNamespace(secrets=("api-key",), accounts=("cloud",)),
        }
        secret_metadata = mock.Mock(
            return_value=(SimpleNamespace(id="api-key", configured=True, generation=3),),
        )
        account_metadata = mock.Mock(
            return_value=(SimpleNamespace(id="cloud", status="connected", generation=5),),
        )

        self.assertEqual(
            app.power_execution.secret_generations(powers, "lookup", secret_metadata),
            (("api-key", 3),),
        )
        self.assertEqual(
            app.power_execution.account_generations(
                powers,
                {"cloud": "declaration"},
                "lookup",
                account_metadata,
            ),
            (("cloud", 5),),
        )
        secret_metadata.assert_called_once_with(("api-key",))
        account_metadata.assert_called_once_with({"cloud": "declaration"})
        with self.assertRaisesRegex(app.power_journal.PowerJournalConflictError, "account contract"):
            app.power_execution.account_generations(powers, {}, "lookup", account_metadata)

    def test_rpc_result_projection_rejects_private_and_invalid_outputs(self) -> None:
        projected = app.power_execution.project_rpc_result(
            {"ok": True},
            {"secret": "private"},
            {},
            (),
            lambda value: value,
        )
        self.assertEqual(projected, app.power_execution.RpcInvocationResult({"ok": True}, False))

        suspended = app.power_execution.project_rpc_result(
            app.power_execution.RpcSuspension({"kind": "request"}),
            {},
            {},
            (),
            lambda _value: self.fail("suspension reached output validation"),
        )
        self.assertEqual(suspended, app.power_execution.RpcInvocationResult({"kind": "request"}, True))

        with self.assertRaises(app.power_execution.RpcSecretExposureError):
            app.power_execution.project_rpc_result(
                {"echo": "private"},
                {"secret": "private"},
                {},
                (),
                lambda value: value,
            )
        with self.assertRaises(app.power_execution.RpcInvalidResultError):
            app.power_execution.project_rpc_result(
                {"invalid": True},
                {},
                {},
                (),
                lambda _value: (_ for _ in ()).throw(ValueError("invalid")),
            )

    def test_malformed_frames_fail_closed_in_both_readers(self) -> None:
        oversized = struct.pack(
            ">BxxxL",
            1,
            max(app.MAX_ASSISTANT_RPC_OUTPUT_BYTES, local_assistant_rpc.MAX_RESPONSE_BYTES) + 2,
        )
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
        request = app.brain_runtime_client.PowerRequest("interrupt-1", "assistant", "lookup", {"query": "safe"})
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
                        app.power_execution.PowerBatchStrategy(
                            identity,
                            execute,
                            lambda _request: None,
                            lambda _request: (("secret", generation[0]),),
                        ),
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

    def test_power_batch_replays_an_explicit_rpc_suspension(self) -> None:
        request = app.brain_runtime_client.PowerRequest(
            "interrupt-1",
            "assistant",
            "lookup",
            {"query": "safe"},
        )
        suspension = app.power_execution.RpcSuspension({"ordinal": 0, "kind": "request"})
        execute = mock.Mock(side_effect=(suspension, {"ok": True}))
        binding = SimpleNamespace(container=SimpleNamespace(id="container-1"), image="image@sha256:" + "a" * 64)

        with tempfile.TemporaryDirectory() as directory:
            journal = app.power_journal.PowerJournal(Path(directory) / "journal.sqlite3")
            self.addCleanup(journal.close)
            batch = app.power_execution.PowerBatch(
                journal,
                "generation-1",
                "thread-1",
                {"assistant": binding},
                app.power_execution.PowerBatchStrategy(
                    lambda item: (item.container.id, item.image),
                    execute,
                    lambda _request: None,
                ),
            )
            batch.prepare((request,))

            self.assertIs(batch.invoke(request), suspension)
            self.assertEqual(batch.invoke(request), {"ok": True})

        self.assertEqual(execute.call_count, 2)

    def test_power_resolution_failures_have_identical_statuses(self) -> None:
        hosted_contract = SimpleNamespace(powers={})
        local_spec = SimpleNamespace(assistant_id="assistant", name="Assistant", powers={}, accounts={})

        with self.assertRaises(app.ApiError) as hosted_secret:
            app._resolve_power_secrets("team_1", "assistant", hosted_contract, "missing")
        with self.assertRaises(local_app.ApiProblem) as local_secret:
            self.local._resolve_power_secrets("team_1", local_spec, "missing")
        self.assertEqual(
            hosted_secret.exception.status,
            local_secret.exception.status,
            app.power_execution.UNDECLARED_POWER_STATUS,
        )

        hosted_active = SimpleNamespace(
            assistant_id="assistant",
            contract=SimpleNamespace(powers={}, secrets={}, accounts={}),
        )
        self.local.assistant_accounts = object()
        with self.assertRaises(app.ApiError) as hosted_account:
            app._resolve_power_accounts("team_1", hosted_active, "missing")
        with self.assertRaises(local_app.ApiProblem) as local_account:
            self.local._resolve_power_accounts("team_1", local_spec, "missing")
        self.assertEqual(
            hosted_account.exception.status,
            local_account.exception.status,
            app.power_execution.ACCOUNT_PRECONDITION_STATUS,
        )

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
                    app.AssistantRpcRequest(
                        team_id="team_1",
                        container=container,
                        command="/app/rpc",
                        method="POST",
                        path="/v1/powers/test",
                        payload={},
                        token=None,
                        operation="Assistant Power",
                    )
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
                    local_assistant_rpc.assistant_secret_flow,
                    "encode_private_rpc_envelope",
                    return_value=b"request",
                ),
                self.assertRaises(local_app.ApiProblem) as caught,
            ):
                controller._rpc(SimpleNamespace(id="assistant-container"), spec, "POST", "/v1/powers/test", {})

        self.assertEqual(caught.exception.status, HTTPStatus.BAD_GATEWAY)
        controller._fail_stop_power.assert_called_once()
        self.assertEqual(create.call_args.kwargs["workdir"], local_assistant_rpc.ASSISTANT_WORKDIR)
        self.assertEqual(create.call_args.kwargs["environment"], {})


if __name__ == "__main__":
    unittest.main()
