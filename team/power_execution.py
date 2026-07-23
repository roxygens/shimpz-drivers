"""Shared fail-closed Power execution primitives for hosted and local Controllers."""

from __future__ import annotations

import hashlib
import json
import select
import socket
import struct
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from http import HTTPStatus

import assistant_secret_flow
import power_journal

# A missing manifest Power is a missing resource; an unavailable connected account is an unmet
# request precondition. Both Controllers use these statuses so their public contracts cannot drift.
UNDECLARED_POWER_STATUS = HTTPStatus.NOT_FOUND
ACCOUNT_PRECONDITION_STATUS = HTTPStatus.PRECONDITION_REQUIRED
RPC_FAILURE_STATUSES = {
    "timeout": HTTPStatus.GATEWAY_TIMEOUT,
    "ambiguous": HTTPStatus.BAD_GATEWAY,
    "invalid-result": HTTPStatus.BAD_GATEWAY,
    "failed": HTTPStatus.BAD_GATEWAY,
}


def rpc_failure_status(kind: str) -> HTTPStatus:
    """Map every non-routing RPC failure kind to its shared HTTP status."""
    try:
        return RPC_FAILURE_STATUSES[kind]
    except KeyError:
        raise AssertionError(f"unknown RPC failure: {kind}") from None


def power_operation(
    request: object,
    assistant_container_id: object,
    assistant_image: object,
    secret_generations: tuple[tuple[str, int], ...] = (),
    account_generations: tuple[tuple[str, int], ...] = (),
) -> power_journal.Operation:
    """Fingerprint one normalized request and every immutable private-state generation."""
    if not isinstance(assistant_container_id, str) or not assistant_container_id:
        raise power_journal.PowerJournalConflictError("Assistant generation is invalid")
    if not isinstance(assistant_image, str) or not assistant_image:
        raise power_journal.PowerJournalConflictError("Assistant generation is invalid")
    try:
        encoded = json.dumps(
            {
                "assistant_container_id": assistant_container_id,
                "assistant_id": request.assistant_id,
                "assistant_image": assistant_image,
                "account_generations": account_generations,
                "input": request.input,
                "power": request.power,
                "secret_generations": secret_generations,
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        raise power_journal.PowerJournalConflictError("Power request cannot be fingerprinted") from exc
    return power_journal.Operation(request.interrupt_id, hashlib.sha256(encoded).hexdigest())


@dataclass(frozen=True, slots=True)
class PowerBatchStrategy:
    binding_identity: Callable[[object], tuple[object, object]]
    execute: Callable[[object], object]
    preflight: Callable[[object], None]
    secret_generations: Callable[[object], tuple[tuple[str, int], ...]] = lambda _request: ()
    account_generations: Callable[[object], tuple[tuple[str, int], ...]] = lambda _request: ()


class PowerBatch:
    """Bind a Brain suspension to one durable journal batch and immutable workload identities."""

    def __init__(
        self,
        journal: power_journal.PowerJournal | Callable[[], power_journal.PowerJournal],
        generation: str,
        thread_id: str,
        bindings: Mapping[str, object],
        strategy: PowerBatchStrategy,
    ) -> None:
        self._journal_source = journal
        self._journal = journal if isinstance(journal, power_journal.PowerJournal) else None
        self._generation = generation
        self._thread_id = thread_id
        self._bindings = bindings
        self._strategy = strategy
        self._batch: power_journal.Batch | None = None
        self._operations: dict[str, power_journal.Operation] = {}

    def _operation(self, request: object) -> power_journal.Operation:
        active = self._bindings.get(request.assistant_id)
        if active is None:
            raise power_journal.PowerJournalConflictError("Power Assistant is unavailable")
        self._strategy.preflight(request)
        container_id, image = self._strategy.binding_identity(active)
        return power_operation(
            request,
            container_id,
            image,
            self._strategy.secret_generations(request),
            self._strategy.account_generations(request),
        )

    def prepare(self, requests: tuple[object, ...]) -> None:
        if self._batch is not None:
            raise power_journal.PowerJournalConflictError("Power batch is already prepared")
        operations = tuple(self._operation(request) for request in requests)
        if self._journal is None:
            self._journal = self._journal_source()
        self._batch = self._journal.prepare_batch(self._generation, self._thread_id, operations)
        self._operations = {operation.interrupt_id: operation for operation in operations}

    def invoke(self, request: object) -> object:
        if self._journal is None or self._batch is None:
            raise power_journal.PowerJournalConflictError("Power batch is not prepared")
        operation = self._operations.get(request.interrupt_id)
        if operation is None:
            raise power_journal.PowerJournalConflictError("Power operation is not prepared")
        if self._operation(request) != operation:
            raise power_journal.PowerJournalConflictError("Power credential generation changed")
        decision = self._journal.begin(self._batch, operation)
        if not decision.execute:
            return decision.result
        result = self._strategy.execute(request)
        if isinstance(result, RpcSuspension):
            self._journal.suspend(self._batch, operation)
            return result
        self._journal.complete(self._batch, operation, result)
        return result

    def delivered(self, requests: tuple[object, ...]) -> None:
        if self._journal is None or self._batch is None:
            raise power_journal.PowerJournalConflictError("Power batch is not prepared")
        expected = tuple(operation.interrupt_id for operation in self._batch.operations)
        if tuple(request.interrupt_id for request in requests) != expected:
            raise power_journal.PowerJournalConflictError("Power delivery batch changed")
        self._journal.delivered(self._batch)
        self._batch = None
        self._operations = {}
        if callable(self._journal_source):
            self._journal = None


class RpcExchangeError(RuntimeError):
    """One stable failure kind translated into each Controller's public error shape."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


@dataclass(frozen=True, slots=True)
class RpcSuspension:
    """One SDK-requested deterministic replay suspension."""

    payload: dict[str, object]


class RpcSecretExposureError(ValueError):
    """An Assistant returned a literal private value."""


class RpcInvalidResultError(ValueError):
    """An Assistant result failed its reviewed Power schema."""


@dataclass(frozen=True, slots=True)
class RpcInvocationResult:
    value: object
    suspended: bool


def project_rpc_result(
    raw_result: object,
    secrets_by_id: Mapping[str, str],
    accounts_by_id: Mapping[str, Mapping[str, object]],
    answers: tuple[object, ...],
    validate: Callable[[object], object],
) -> RpcInvocationResult:
    """Reject private echoes, retain suspensions, and validate one terminal Power result."""
    private_values = protected_rpc_values(secrets_by_id, accounts_by_id, answers)
    inspected = raw_result.payload if isinstance(raw_result, RpcSuspension) else raw_result
    if contains_secret(inspected, private_values):
        raise RpcSecretExposureError
    if isinstance(raw_result, RpcSuspension):
        return RpcInvocationResult(raw_result.payload, True)
    try:
        return RpcInvocationResult(validate(raw_result), False)
    except ValueError as exc:
        raise RpcInvalidResultError from exc


def decode_rpc_response(raw: bytes) -> object:
    """Decode the closed result-or-suspend stdout protocol."""
    try:
        response = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RpcExchangeError("invalid-result") from exc
    if not isinstance(response, dict) or len(response) != 1:
        raise RpcExchangeError("invalid-result")
    if set(response) == {"result"}:
        return response["result"]
    if set(response) == {"suspend"} and isinstance(response["suspend"], dict):
        return RpcSuspension(response["suspend"])
    raise RpcExchangeError("invalid-result")


@dataclass(frozen=True, slots=True)
class RpcExchangeStrategy:
    api: object
    user: str
    workdir: str
    timeout: float
    maximum: int
    transport_errors: tuple[type[BaseException], ...]
    fail_stop: Callable[[], None]
    cancelled: Callable[[BaseException | None], None]
    close_stream: Callable[[object], None]


def rpc_exchange(
    container_id: str,
    argv: list[str],
    encoded: bytes,
    strategy: RpcExchangeStrategy,
    *,
    detect_unsupported_path: bool = False,
) -> object:
    """Execute one bounded Docker RPC with shared fail-stop and framing decisions."""
    transport_errors = strategy.transport_errors
    stream = None
    try:
        try:
            created = strategy.api.exec_create(
                container_id,
                argv,
                stdin=True,
                stdout=True,
                stderr=True,
                privileged=False,
                user=strategy.user,
                workdir=strategy.workdir,
                environment={},
            )
            exec_id = created["Id"]
            stream = strategy.api.exec_start(exec_id, socket=True)
            raw_socket = getattr(stream, "_sock", None)
            if raw_socket is None:
                raise OSError("Docker attach socket cannot half-close stdin")
            raw_socket.sendall(encoded)
            raw_socket.shutdown(socket.SHUT_WR)
            stdout, stderr = read_rpc_frames(
                raw_socket,
                time.monotonic() + strategy.timeout,
                strategy.maximum,
            )
        except TimeoutError as exc:
            strategy.fail_stop()
            strategy.cancelled(exc)
            raise RpcExchangeError("timeout") from exc
        except (*transport_errors, OSError, ValueError, KeyError) as exc:
            strategy.fail_stop()
            strategy.cancelled(exc)
            raise RpcExchangeError("failed") from exc
    finally:
        if stream is not None:
            strategy.close_stream(stream)

    try:
        details = strategy.api.exec_inspect(exec_id)
    except transport_errors as exc:
        strategy.fail_stop()
        strategy.cancelled(exc)
        raise RpcExchangeError("ambiguous") from exc
    exit_code = details.get("ExitCode")
    if not isinstance(exit_code, int):
        strategy.fail_stop()
        strategy.cancelled(None)
        raise RpcExchangeError("ambiguous")
    if exit_code != 0 or stderr:
        if detect_unsupported_path and exit_code == 2 and not stdout and not stderr:
            raise RpcExchangeError("unsupported-path")
        strategy.cancelled(None)
        raise RpcExchangeError("failed")
    return decode_rpc_response(bytes(stdout))


def private_generations(metadata: tuple[object, ...], *, connected: bool) -> tuple[tuple[str, int], ...]:
    """Project only usable positive generations from secret or account metadata."""
    if connected:
        valid = all(getattr(item, "status", None) == "connected" for item in metadata)
    else:
        valid = all(getattr(item, "configured", False) is True for item in metadata)
    generations = tuple(getattr(item, "generation", None) for item in metadata)
    if not valid or any(type(generation) is not int or generation < 1 for generation in generations):
        kind = "account" if connected else "secret"
        raise power_journal.PowerJournalConflictError(f"Power {kind} generation is unavailable")
    return tuple((item.id, generation) for item, generation in zip(metadata, generations, strict=True))


def secret_generations(
    powers: Mapping[str, object],
    power_id: str,
    metadata: Callable[[tuple[str, ...]], tuple[object, ...]],
) -> tuple[tuple[str, int], ...]:
    """Read one declared Power's configured secret generations."""
    power = powers.get(power_id)
    if power is None:
        raise power_journal.PowerJournalConflictError("Power secret contract is unavailable")
    return private_generations(tuple(metadata(tuple(getattr(power, "secrets", ())))), connected=False)


def account_generations(
    powers: Mapping[str, object],
    accounts: Mapping[str, object],
    power_id: str,
    metadata: Callable[[dict[str, object]], tuple[object, ...]],
) -> tuple[tuple[str, int], ...]:
    """Read one declared Power's connected account generations."""
    power = powers.get(power_id)
    if power is None:
        raise power_journal.PowerJournalConflictError("Power account contract is unavailable")
    account_ids = tuple(getattr(power, "accounts", ()))
    declarations = {account_id: accounts[account_id] for account_id in account_ids if account_id in accounts}
    if len(declarations) != len(account_ids):
        raise power_journal.PowerJournalConflictError("Power account contract is unavailable")
    return private_generations(tuple(metadata(declarations)), connected=True)


def require_rpc_envelope(
    active: object,
    request: object,
    resolve_secrets: Callable[[object, str], Mapping[str, str]],
    resolve_accounts: Callable[[object, str], Mapping[str, Mapping[str, object]]],
    answers: tuple[object, ...] = (),
) -> None:
    """Resolve and size-check one complete private RPC envelope before journaling."""
    assistant_secret_flow.require_power_rpc_envelope(
        request.input,
        resolve_secrets(active, request.power),
        resolve_accounts(active, request.power),
        answers,
    )


def contains_secret(value: object, secrets_by_id: Mapping[str, str]) -> bool:
    """Fail closed on literal secret echoes or inputs nested beyond the inspection bound."""
    secret_values = tuple(secret for secret in secrets_by_id.values() if secret)

    def visit(item: object, depth: int = 0) -> bool:
        if depth > 32:
            return True
        if isinstance(item, str):
            return any(secret in item for secret in secret_values)
        if isinstance(item, list | tuple):
            return any(visit(child, depth + 1) for child in item)
        if isinstance(item, dict):
            return any(visit(key, depth + 1) or visit(child, depth + 1) for key, child in item.items())
        return False

    return bool(secret_values) and visit(value)


def protected_rpc_values(
    secrets_by_id: Mapping[str, str],
    accounts_by_id: Mapping[str, Mapping[str, object]],
    answers: Iterable[object],
) -> dict[str, str]:
    """Collect literal private strings that an Assistant must not return."""
    return {
        **secrets_by_id,
        **{
            f"account:{account_id}": access_token
            for account_id, envelope in accounts_by_id.items()
            if isinstance((access_token := envelope.get("access_token")), str)
        },
        **{f"answer:{ordinal}": answer for ordinal, answer in enumerate(answers) if isinstance(answer, str)},
    }


def _read_exact(raw_socket: socket.socket, amount: int, deadline: float) -> bytes:
    output = bytearray()
    while len(output) < amount:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([raw_socket], [], [], remaining)[0]:
            raise TimeoutError
        chunk = raw_socket.recv(amount - len(output))
        if not chunk:
            raise EOFError
        output.extend(chunk)
    return bytes(output)


def read_rpc_frames(raw_socket: socket.socket, deadline: float, maximum: int) -> tuple[bytes, bytes]:
    """Read Docker's multiplexed exec frames with one shared bounded parser."""
    stdout = bytearray()
    stderr = bytearray()
    while True:
        try:
            first = _read_exact(raw_socket, 1, deadline)
        except EOFError:
            break
        try:
            header = first + _read_exact(raw_socket, 7, deadline)
        except EOFError as exc:
            raise ValueError("truncated Assistant RPC frame header") from exc
        stream_id, length = struct.unpack(">BxxxL", header)
        if stream_id not in {1, 2}:
            raise ValueError("invalid Assistant RPC stream")
        if length > maximum + 1:
            raise ValueError("oversized Assistant RPC frame")
        try:
            chunk = _read_exact(raw_socket, length, deadline)
        except EOFError as exc:
            raise ValueError("truncated Assistant RPC frame payload") from exc
        target = stdout if stream_id == 1 else stderr
        target.extend(chunk)
        if len(stdout) + len(stderr) > maximum:
            raise ValueError("oversized Assistant RPC response")
    return bytes(stdout), bytes(stderr)


def close_exec_stream(stream: object) -> None:
    """Close docker-py's owning HTTP response before its raw socket."""
    response = getattr(stream, "_response", None)
    if response is not None:
        response.close()
    else:
        stream.close()
