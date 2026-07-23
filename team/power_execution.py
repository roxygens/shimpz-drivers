"""Shared fail-closed Power execution primitives for hosted and local Controllers."""

from __future__ import annotations

import hashlib
import json
import select
import socket
import struct
import time
from collections.abc import Callable, Mapping
from http import HTTPStatus

import power_journal

# A missing manifest Power is a missing resource; an unavailable connected account is an unmet
# request precondition. Both Controllers use these statuses so their public contracts cannot drift.
UNDECLARED_POWER_STATUS = HTTPStatus.NOT_FOUND
ACCOUNT_PRECONDITION_STATUS = HTTPStatus.PRECONDITION_REQUIRED


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
                "approval": request.approval,
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


class PowerBatch:
    """Bind a Brain suspension to one durable journal batch and immutable workload identities."""

    def __init__(
        self,
        journal: power_journal.PowerJournal | Callable[[], power_journal.PowerJournal],
        generation: str,
        thread_id: str,
        bindings: Mapping[str, object],
        binding_identity: Callable[[object], tuple[object, object]],
        execute: Callable[[object], object],
        preflight: Callable[[object], None],
        secret_generations: Callable[[object], tuple[tuple[str, int], ...]] = lambda _request: (),
        account_generations: Callable[[object], tuple[tuple[str, int], ...]] = lambda _request: (),
    ) -> None:
        self._journal_source = journal
        self._journal = journal if isinstance(journal, power_journal.PowerJournal) else None
        self._generation = generation
        self._thread_id = thread_id
        self._bindings = bindings
        self._binding_identity = binding_identity
        self._execute = execute
        self._preflight = preflight
        self._secret_generations = secret_generations
        self._account_generations = account_generations
        self._batch: power_journal.Batch | None = None
        self._operations: dict[str, power_journal.Operation] = {}

    def _operation(self, request: object) -> power_journal.Operation:
        active = self._bindings.get(request.assistant_id)
        if active is None:
            raise power_journal.PowerJournalConflictError("Power Assistant is unavailable")
        self._preflight(request)
        container_id, image = self._binding_identity(active)
        return power_operation(
            request,
            container_id,
            image,
            self._secret_generations(request),
            self._account_generations(request),
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
        result = self._execute(request)
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
