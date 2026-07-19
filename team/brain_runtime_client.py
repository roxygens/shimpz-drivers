"""Narrow Team Controller client for the isolated LangGraph Brain runtime."""

from __future__ import annotations

import http.client
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

RUNTIME_URL = os.environ.get("SHIMPZ_BRAIN_RUNTIME_URL", "http://brain-runtime:8080")
TOKEN_FILE = Path(os.environ.get("SHIMPZ_BRAIN_RUNTIME_TOKEN_FILE", "/run/shimpz-brain-runtime/token"))
MAX_RESPONSE_BYTES = 256 * 1024
MAX_REPLY_CHARS = 64 * 1024
MAX_POWER_REQUESTS = 64
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
POWER_ID_RE = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*\Z")
APPROVALS = frozenset({"none", "once", "each-run"})


class BrainRuntimeError(RuntimeError):
    """The private runtime was unavailable or violated its closed response contract."""


@dataclass(frozen=True, slots=True)
class RuntimePower:
    id: str
    summary: str
    input_schema: Mapping[str, Any]
    approval: Literal["none", "once", "each-run"]


@dataclass(frozen=True, slots=True)
class RuntimeAssistant:
    id: str
    genesis: str
    powers: tuple[RuntimePower, ...]


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    thread_id: str
    team_name: str
    assistants: tuple[RuntimeAssistant, ...]
    provider: Literal["anthropic", "openai"]
    model: str
    api_key: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class PowerRequest:
    interrupt_id: str
    assistant_id: str
    power: str
    input: Mapping[str, Any]
    approval: Literal["none", "once", "each-run"]


@dataclass(frozen=True, slots=True)
class RuntimeTurn:
    status: Literal["completed", "power-required"]
    reply: str
    powers: tuple[PowerRequest, ...]


ConnectionFactory = Callable[[str, int, float], http.client.HTTPConnection]


def _connection(host: str, port: int, timeout: float) -> http.client.HTTPConnection:
    return http.client.HTTPConnection(host, port, timeout=timeout)


class BrainRuntimeClient:
    def __init__(
        self,
        *,
        base_url: str = RUNTIME_URL,
        token_file: Path = TOKEN_FILE,
        connection_factory: ConnectionFactory = _connection,
    ) -> None:
        parsed = urlparse(base_url)
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise BrainRuntimeError("Brain runtime URL is invalid")
        self._host = parsed.hostname
        self._port = parsed.port or 80
        self._token_file = token_file
        self._connection_factory = connection_factory

    def _token(self) -> str:
        try:
            token = self._token_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise BrainRuntimeError("Brain runtime authentication is unavailable") from exc
        if not token or len(token) > 4 * 1024 or "\0" in token:
            raise BrainRuntimeError("Brain runtime authentication is unavailable")
        return token

    @staticmethod
    def _context(context: RuntimeContext) -> dict[str, object]:
        return {
            "thread_id": context.thread_id,
            "team_name": context.team_name,
            "assistants": [
                {
                    "id": assistant.id,
                    "genesis": assistant.genesis,
                    "powers": [
                        {
                            "id": power.id,
                            "summary": power.summary,
                            "input_schema": dict(power.input_schema),
                            "approval": power.approval,
                        }
                        for power in assistant.powers
                    ],
                }
                for assistant in context.assistants
            ],
            "provider": {
                "provider": context.provider,
                "model": context.model,
                "api_key": context.api_key,
            },
        }

    def _post(self, path: str, payload: Mapping[str, object]) -> object:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
        connection = self._connection_factory(self._host, self._port, 65.0)
        try:
            connection.request(
                "POST",
                path,
                body,
                {
                    "Authorization": f"Bearer {self._token()}",
                    "Content-Type": "application/json",
                },
            )
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        except OSError as exc:
            raise BrainRuntimeError("Brain runtime is unavailable") from exc
        finally:
            connection.close()
        if len(raw) > MAX_RESPONSE_BYTES:
            raise BrainRuntimeError("Brain runtime returned an invalid response")
        if response.status != 200:
            raise BrainRuntimeError("Brain runtime request failed")
        try:
            decoded = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BrainRuntimeError("Brain runtime returned an invalid response") from exc
        return decoded

    @staticmethod
    def _parse_turn(value: object) -> RuntimeTurn:
        if not isinstance(value, dict) or set(value) != {"status", "reply", "powers"}:
            raise BrainRuntimeError("Brain runtime returned an invalid response")
        status = value["status"]
        reply = value["reply"]
        raw_powers = value["powers"]
        if (
            status not in {"completed", "power-required"}
            or not isinstance(reply, str)
            or not isinstance(raw_powers, list)
        ):
            raise BrainRuntimeError("Brain runtime returned an invalid response")
        if len(reply) > MAX_REPLY_CHARS or len(raw_powers) > MAX_POWER_REQUESTS:
            raise BrainRuntimeError("Brain runtime returned an invalid response")
        powers: list[PowerRequest] = []
        for raw in raw_powers:
            if not isinstance(raw, dict) or set(raw) != {
                "interrupt_id",
                "assistant_id",
                "power",
                "input",
                "approval",
            }:
                raise BrainRuntimeError("Brain runtime returned an invalid response")
            interrupt_id = raw["interrupt_id"]
            assistant_id = raw["assistant_id"]
            power = raw["power"]
            power_input = raw["input"]
            approval = raw["approval"]
            if (
                not isinstance(interrupt_id, str)
                or SAFE_ID_RE.fullmatch(interrupt_id) is None
                or not isinstance(assistant_id, str)
                or POWER_ID_RE.fullmatch(assistant_id) is None
                or not isinstance(power, str)
                or POWER_ID_RE.fullmatch(power) is None
                or not isinstance(power_input, dict)
                or approval not in APPROVALS
            ):
                raise BrainRuntimeError("Brain runtime returned an invalid response")
            powers.append(
                PowerRequest(
                    interrupt_id=interrupt_id,
                    assistant_id=assistant_id,
                    power=power,
                    input=power_input,
                    approval=approval,
                )
            )
        if status == "completed" and (not reply.strip() or powers):
            raise BrainRuntimeError("Brain runtime returned an invalid response")
        if status == "power-required" and (reply or not powers):
            raise BrainRuntimeError("Brain runtime returned an invalid response")
        return RuntimeTurn(status=status, reply=reply, powers=tuple(powers))

    def start(self, context: RuntimeContext, message: str) -> RuntimeTurn:
        payload = self._context(context)
        payload["message"] = message
        return self._parse_turn(self._post("/v1/turns", payload))

    def resume(self, context: RuntimeContext, results: Mapping[str, object]) -> RuntimeTurn:
        payload = self._context(context)
        payload["results"] = dict(results)
        return self._parse_turn(self._post("/v1/turns/resume", payload))

    def delete_thread(self, thread_id: str) -> None:
        if not isinstance(thread_id, str) or SAFE_ID_RE.fullmatch(thread_id) is None:
            raise BrainRuntimeError("Brain runtime thread ID is invalid")
        response = self._post("/v1/threads/delete", {"thread_id": thread_id})
        if not isinstance(response, dict) or response != {"status": "deleted"}:
            raise BrainRuntimeError("Brain runtime returned an invalid response")
