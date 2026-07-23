"""Closed JSON codec for encrypted local Team chat continuations."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

import assistant_account_challenges
import assistant_secret_challenges
import brain_runtime_client
import chat_orchestrator
import inference_config
import local_chat_continuation_store
from assistant_human import approval_challenges, input_challenges

SCHEMA_VERSION = 1
MAX_ANSWER_LOGS = 64
MAX_ANSWERS_PER_POWER = 64
MAX_JSON_DEPTH = 16
MAX_JSON_NODES = 4096
MAX_INVOKED_POWERS = 512
MAX_IDENTITY_ASSISTANTS = 16
MAX_IDENTITY_FILES = 8
_FILE_ID = re.compile(r"[0-9a-f]{32}\Z")
_IMAGE = re.compile(r"[^\s\x00-\x1f\x7f]{1,512}@sha256:[0-9a-f]{64}\Z")
_NETWORK_ID = re.compile(r"[^\s\x00-\x1f\x7f]{1,256}\Z")
_CONTAINER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")


class ContinuationCodecError(RuntimeError):
    """A decrypted local continuation violated the closed runtime contract."""


@dataclass(frozen=True, slots=True)
class PendingLocalChat:
    """Secret-free state required to replay one paused local Team turn."""

    continuation: chat_orchestrator.ChatContinuation
    assistant_ids: tuple[str, ...]
    file_ids: tuple[str, ...]
    provider: str
    identity: tuple[object, ...]
    answer_logs: tuple[tuple[str, tuple[object, ...]], ...] = ()


@dataclass(frozen=True, slots=True)
class DecodedContinuation:
    kind: str
    requirements: tuple[object, ...]
    pending: PendingLocalChat


def _mapping(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ContinuationCodecError(f"{label} is malformed")
    return value


def _sequence(value: object, maximum: int, label: str) -> list[object]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ContinuationCodecError(f"{label} is malformed")
    return value


def _text(
    value: object,
    maximum: int,
    label: str,
    *,
    optional: bool = False,
) -> str | None:
    if optional and value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or not value.isprintable()
    ):
        raise ContinuationCodecError(f"{label} is malformed")
    return value


def _component_id(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) > 80 or brain_runtime_client.POWER_ID_RE.fullmatch(value) is None:
        raise ContinuationCodecError(f"{label} is malformed")
    return value


def _interrupt_id(value: object) -> str:
    if not isinstance(value, str) or brain_runtime_client.SAFE_ID_RE.fullmatch(value) is None:
        raise ContinuationCodecError("continuation interrupt is malformed")
    return value


def _json_value(value: object) -> object:
    budget = [MAX_JSON_NODES]

    def visit(item: object, depth: int) -> object:
        budget[0] -= 1
        if budget[0] < 0 or depth > MAX_JSON_DEPTH:
            raise ContinuationCodecError("continuation JSON exceeds its structure limit")
        if item is None or isinstance(item, bool | str):
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ContinuationCodecError("continuation JSON contains a non-finite number")
            return item
        if isinstance(item, list | tuple):
            return [visit(nested, depth + 1) for nested in item]
        if isinstance(item, dict):
            result: dict[str, object] = {}
            for key, nested in item.items():
                if not isinstance(key, str) or len(key) > 128 or key in result:
                    raise ContinuationCodecError("continuation JSON object is malformed")
                result[key] = visit(nested, depth + 1)
            return result
        raise ContinuationCodecError("continuation contains a non-JSON value")

    return visit(value, 0)


def _turn_payload(turn: brain_runtime_client.RuntimeTurn) -> dict[str, object]:
    return {
        "status": turn.status,
        "reply": turn.reply,
        "powers": [
            {
                "interrupt_id": request.interrupt_id,
                "assistant_id": request.assistant_id,
                "power": request.power,
                "input": _json_value(dict(request.input)),
            }
            for request in turn.powers
        ],
    }


def _pending_payload(pending: PendingLocalChat) -> dict[str, object]:
    if not isinstance(pending, PendingLocalChat):
        raise ContinuationCodecError("pending continuation is malformed")
    identity = _identity_payload(pending.identity)
    return {
        "continuation": {
            "turn": _turn_payload(pending.continuation.turn),
            "seen_interrupts": list(pending.continuation.seen_interrupts),
            "invoked": [asdict(item) for item in pending.continuation.invoked],
            "round_index": pending.continuation.round_index,
        },
        "assistant_ids": list(pending.assistant_ids),
        "file_ids": list(pending.file_ids),
        "provider": pending.provider,
        "identity": identity,
        "answer_logs": [
            {
                "interrupt_id": interrupt_id,
                "answers": [_json_value(answer) for answer in answers],
            }
            for interrupt_id, answers in pending.answer_logs
        ],
    }


def _identity_payload(identity: tuple[object, ...]) -> dict[str, object]:
    if not isinstance(identity, tuple) or len(identity) != 5:
        raise ContinuationCodecError("continuation Team identity is malformed")
    team_name, network_id, assistants, files, config = identity
    if not isinstance(config, inference_config.InferenceConfig):
        raise ContinuationCodecError("continuation inference identity is malformed")
    if not isinstance(assistants, tuple) or not isinstance(files, list):
        raise ContinuationCodecError("continuation Team identity is malformed")
    return {
        "team_name": team_name,
        "network_id": network_id,
        "assistants": [list(item) if isinstance(item, tuple) else item for item in assistants],
        "files": _json_value(files),
        "inference": {"provider": config.provider, "model": config.model},
    }


def _requirements_payload(kind: str, requirements: tuple[object, ...]) -> list[dict[str, object]]:
    expected = {
        "accounts": assistant_account_challenges.AccountRequirement,
        "secrets": assistant_secret_challenges.SecretRequirement,
        "input": input_challenges.InputRequirement,
        "approval": approval_challenges.ApprovalRequirement,
    }.get(kind)
    if expected is None or not requirements or any(not isinstance(item, expected) for item in requirements):
        raise ContinuationCodecError("continuation requirements are malformed")
    return [_json_value(asdict(item)) for item in requirements]  # type: ignore[list-item]


def _release_images(pending: PendingLocalChat) -> dict[str, str]:
    identity = _identity_payload(pending.identity)
    images: dict[str, str] = {}
    for raw in identity["assistants"]:
        if not isinstance(raw, list) or len(raw) != 3:
            raise ContinuationCodecError("continuation Assistant identity is malformed")
        assistant = _component_id(raw[0], "continuation Assistant identity")
        image = raw[1]
        if not isinstance(image, str) or _IMAGE.fullmatch(image) is None:
            raise ContinuationCodecError("continuation Assistant release is malformed")
        images[assistant] = image
    return images


def _bindings(kind: str, requirements: tuple[object, ...], pending: PendingLocalChat) -> tuple[str, ...]:
    images = _release_images(pending)
    bindings: set[str] = set()
    if kind in {"input", "approval"}:
        for requirement in requirements:
            assistant = _component_id(requirement.assistant_id, "continuation binding Assistant")
            power = _component_id(requirement.power_id, "continuation binding Power")
            image = requirement.assistant_image
            ordinal = requirement.ordinal
            if image != images.get(assistant) or type(ordinal) is not int or not 0 <= ordinal <= 63:
                raise ContinuationCodecError("continuation release binding is malformed")
            bindings.add(f"{assistant}/{power}/{image}/{ordinal}")
    else:
        for requirement in requirements:
            assistant = _component_id(requirement.assistant_id, "continuation binding Assistant")
            image = images.get(assistant)
            if image is None:
                raise ContinuationCodecError("continuation release binding is malformed")
            for power_id in requirement.power_ids:
                power = _component_id(power_id, "continuation binding Power")
                bindings.add(f"{assistant}/{power}/{image}/-")
    return tuple(sorted(bindings))


def encode(
    kind: str,
    requirements: tuple[object, ...],
    pending: PendingLocalChat,
) -> tuple[tuple[str, ...], bytes]:
    """Encode one authenticated plaintext payload and its AAD release bindings."""
    body = {
        "schema": SCHEMA_VERSION,
        "kind": kind,
        "requirements": _requirements_payload(kind, requirements),
        "pending": _pending_payload(pending),
    }
    try:
        payload = json.dumps(
            body,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ContinuationCodecError("continuation could not be encoded") from exc
    if not 1 <= len(payload) <= local_chat_continuation_store.MAX_PLAINTEXT_BYTES:
        raise ContinuationCodecError("continuation exceeds its fixed byte limit")
    return _bindings(kind, requirements, pending), payload


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContinuationCodecError("continuation contains duplicate fields")
        result[key] = value
    return result


def _decode_payload(payload: bytes) -> dict[str, object]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ContinuationCodecError("continuation contains a non-finite number")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinuationCodecError("continuation is not valid JSON") from exc
    return _mapping(value, {"schema", "kind", "requirements", "pending"}, "continuation")


def _power_request(value: object) -> brain_runtime_client.PowerRequest:
    raw = _mapping(
        value,
        {"interrupt_id", "assistant_id", "power", "input"},
        "continuation Power request",
    )
    power_input = _json_value(raw["input"])
    if not isinstance(power_input, dict):
        raise ContinuationCodecError("continuation Power input is malformed")
    return brain_runtime_client.PowerRequest(
        interrupt_id=_interrupt_id(raw["interrupt_id"]),
        assistant_id=_component_id(raw["assistant_id"], "continuation Power Assistant"),
        power=_component_id(raw["power"], "continuation Power"),
        input=power_input,
    )


def _continuation(value: object) -> chat_orchestrator.ChatContinuation:
    raw = _mapping(
        value,
        {"turn", "seen_interrupts", "invoked", "round_index"},
        "Brain continuation",
    )
    turn_value = _mapping(raw["turn"], {"status", "reply", "powers"}, "Brain turn")
    turn = brain_runtime_client.BrainRuntimeClient._parse_turn(
        {
            "status": turn_value["status"],
            "reply": turn_value["reply"],
            "powers": [
                {
                    "interrupt_id": item.interrupt_id,
                    "assistant_id": item.assistant_id,
                    "power": item.power,
                    "input": dict(item.input),
                }
                for item in (
                    _power_request(power)
                    for power in _sequence(
                        turn_value["powers"],
                        brain_runtime_client.MAX_POWER_REQUESTS,
                        "Brain turn Powers",
                    )
                )
            ],
        }
    )
    seen = tuple(
        _interrupt_id(item)
        for item in _sequence(
            raw["seen_interrupts"],
            brain_runtime_client.MAX_POWER_REQUESTS * 8,
            "seen Brain interrupts",
        )
    )
    if len(seen) != len(set(seen)):
        raise ContinuationCodecError("seen Brain interrupts are malformed")
    invoked: list[chat_orchestrator.InvokedPower] = []
    for item in _sequence(raw["invoked"], MAX_INVOKED_POWERS, "invoked Powers"):
        entry = _mapping(item, {"assistant_id", "power"}, "invoked Power")
        invoked.append(
            chat_orchestrator.InvokedPower(
                _component_id(entry["assistant_id"], "invoked Power Assistant"),
                _component_id(entry["power"], "invoked Power"),
            )
        )
    round_index = raw["round_index"]
    if type(round_index) is not int or not 0 <= round_index < chat_orchestrator.MAX_POWER_ROUNDS:
        raise ContinuationCodecError("continuation round is malformed")
    return chat_orchestrator.ChatContinuation(turn, seen, tuple(invoked), round_index)


def _identity(value: object) -> tuple[object, ...]:
    raw = _mapping(
        value,
        {"team_name", "network_id", "assistants", "files", "inference"},
        "continuation Team identity",
    )
    team_name = _text(raw["team_name"], 80, "continuation Team name")
    network_id = raw["network_id"]
    if not isinstance(network_id, str) or _NETWORK_ID.fullmatch(network_id) is None:
        raise ContinuationCodecError("continuation network identity is malformed")
    assistants: list[tuple[str, str, str]] = []
    for item in _sequence(raw["assistants"], MAX_IDENTITY_ASSISTANTS, "continuation Assistants"):
        if not isinstance(item, list) or len(item) != 3:
            raise ContinuationCodecError("continuation Assistant identity is malformed")
        assistant = _component_id(item[0], "continuation Assistant identity")
        image = item[1]
        container = item[2]
        if (
            not isinstance(image, str)
            or _IMAGE.fullmatch(image) is None
            or not isinstance(container, str)
            or _CONTAINER_ID.fullmatch(container) is None
        ):
            raise ContinuationCodecError("continuation Assistant identity is malformed")
        assistants.append((assistant, image, container))
    if len({item[0] for item in assistants}) != len(assistants):
        raise ContinuationCodecError("continuation Assistant identity is malformed")
    files: list[dict[str, object]] = []
    for item in _sequence(raw["files"], MAX_IDENTITY_FILES, "continuation files"):
        entry = _mapping(item, {"id", "name", "media_type", "size"}, "continuation file")
        if (
            not isinstance(entry["id"], str)
            or _FILE_ID.fullmatch(entry["id"]) is None
            or _text(entry["name"], 255, "continuation filename") in {".", ".."}
            or not isinstance(entry["media_type"], str)
            or not 1 <= len(entry["media_type"]) <= 127
            or type(entry["size"]) is not int
            or not 0 <= entry["size"] <= 2**53 - 1
        ):
            raise ContinuationCodecError("continuation file is malformed")
        files.append(dict(entry))
    if len({item["id"] for item in files}) != len(files):
        raise ContinuationCodecError("continuation files are malformed")
    inference = _mapping(raw["inference"], {"provider", "model"}, "continuation inference")
    try:
        config = inference_config.normalize(inference["provider"], inference["model"])
    except inference_config.InferenceConfigError as exc:
        raise ContinuationCodecError("continuation inference is malformed") from exc
    return team_name, network_id, tuple(assistants), files, config


def _pending(value: object) -> PendingLocalChat:
    raw = _mapping(
        value,
        {
            "continuation",
            "assistant_ids",
            "file_ids",
            "provider",
            "identity",
            "answer_logs",
        },
        "pending continuation",
    )
    assistant_ids = tuple(
        _component_id(item, "pending Assistant") for item in _sequence(raw["assistant_ids"], 16, "pending Assistants")
    )
    if len(assistant_ids) != len(set(assistant_ids)) or tuple(sorted(assistant_ids)) != assistant_ids:
        raise ContinuationCodecError("pending Assistants are malformed")
    file_ids = tuple(
        item
        for item in _sequence(raw["file_ids"], MAX_IDENTITY_FILES, "pending files")
        if isinstance(item, str) and _FILE_ID.fullmatch(item) is not None
    )
    if len(file_ids) != len(raw["file_ids"]) or len(file_ids) != len(set(file_ids)):
        raise ContinuationCodecError("pending files are malformed")
    provider = raw["provider"]
    if not isinstance(provider, str) or provider not in inference_config.PROVIDERS:
        raise ContinuationCodecError("pending provider is malformed")
    answer_logs: list[tuple[str, tuple[object, ...]]] = []
    for item in _sequence(raw["answer_logs"], MAX_ANSWER_LOGS, "human answer logs"):
        entry = _mapping(item, {"interrupt_id", "answers"}, "human answer log")
        answers = tuple(
            _json_value(answer)
            for answer in _sequence(
                entry["answers"],
                MAX_ANSWERS_PER_POWER,
                "human answers",
            )
        )
        answer_logs.append((_interrupt_id(entry["interrupt_id"]), answers))
    if len({item[0] for item in answer_logs}) != len(answer_logs) or tuple(sorted(answer_logs)) != tuple(answer_logs):
        raise ContinuationCodecError("human answer logs are malformed")
    identity = _identity(raw["identity"])
    if identity[4].provider != provider:
        raise ContinuationCodecError("pending provider binding is malformed")
    return PendingLocalChat(
        continuation=_continuation(raw["continuation"]),
        assistant_ids=assistant_ids,
        file_ids=file_ids,
        provider=provider,
        identity=identity,
        answer_logs=tuple(answer_logs),
    )


def _tuple_text(value: object, maximum: int, label: str) -> tuple[str, ...]:
    result = tuple(str(_text(item, maximum, label)) for item in _sequence(value, 128, label))
    if not result or len(result) != len(set(result)) or tuple(sorted(result)) != result:
        raise ContinuationCodecError(f"{label} is malformed")
    return result


def _account_requirement(value: object) -> assistant_account_challenges.AccountRequirement:
    raw = _mapping(
        value,
        {"assistant_id", "assistant_name", "power_ids", "accounts"},
        "account requirement",
    )
    accounts: list[tuple[str, str, tuple[str, ...]]] = []
    for item in _sequence(raw["accounts"], 16, "account requirement accounts"):
        if not isinstance(item, list) or len(item) != 3:
            raise ContinuationCodecError("account requirement is malformed")
        accounts.append(
            (
                _component_id(item[0], "account id"),
                _component_id(item[1], "account provider"),
                _tuple_text(item[2], 128, "account scopes"),
            )
        )
    if not accounts:
        raise ContinuationCodecError("account requirement is malformed")
    return assistant_account_challenges.AccountRequirement(
        _component_id(raw["assistant_id"], "account Assistant"),
        str(_text(raw["assistant_name"], 80, "account Assistant name")),
        _tuple_text(raw["power_ids"], 80, "account Powers"),
        tuple(accounts),
    )


def _secret_requirement(value: object) -> assistant_secret_challenges.SecretRequirement:
    raw = _mapping(
        value,
        {"assistant_id", "assistant_name", "power_ids", "secrets"},
        "secret requirement",
    )
    secrets: list[tuple[str, str, str]] = []
    for item in _sequence(raw["secrets"], 64, "secret requirements"):
        if not isinstance(item, list) or len(item) != 3:
            raise ContinuationCodecError("secret requirement is malformed")
        secrets.append(
            (
                _component_id(item[0], "secret id"),
                str(_text(item[1], 80, "secret name")),
                str(_text(item[2], 160, "secret summary")),
            )
        )
    if not secrets:
        raise ContinuationCodecError("secret requirement is malformed")
    return assistant_secret_challenges.SecretRequirement(
        _component_id(raw["assistant_id"], "secret Assistant"),
        str(_text(raw["assistant_name"], 80, "secret Assistant name")),
        _tuple_text(raw["power_ids"], 80, "secret Powers"),
        tuple(secrets),
    )


def _input_requirement(value: object) -> input_challenges.InputRequirement:
    raw = _mapping(
        value,
        {
            "interrupt_id",
            "assistant_id",
            "power_id",
            "assistant_image",
            "ordinal",
            "request_type",
            "title",
            "summary",
            "docs",
            "options",
        },
        "input requirement",
    )
    ordinal = raw["ordinal"]
    request_type = raw["request_type"]
    if (
        type(ordinal) is not int
        or not 0 <= ordinal <= 63
        or request_type not in {"str", "int", "float", "bool", "choice", "choices"}
        or not isinstance(raw["assistant_image"], str)
        or _IMAGE.fullmatch(raw["assistant_image"]) is None
    ):
        raise ContinuationCodecError("input requirement is malformed")
    options = tuple(str(_text(item, 200, "input option")) for item in _sequence(raw["options"], 64, "input options"))
    if len(options) != len(set(options)):
        raise ContinuationCodecError("input options are malformed")
    if (request_type in {"choice", "choices"}) != bool(options):
        raise ContinuationCodecError("input options are malformed")
    return input_challenges.InputRequirement(
        _interrupt_id(raw["interrupt_id"]),
        _component_id(raw["assistant_id"], "input Assistant"),
        _component_id(raw["power_id"], "input Power"),
        raw["assistant_image"],
        ordinal,
        request_type,
        str(_text(raw["title"], 80, "input title")),
        str(_text(raw["summary"], 240, "input summary")),
        _text(raw["docs"], 2048, "input docs", optional=True),
        options,
    )


def _approval_requirement(value: object) -> approval_challenges.ApprovalRequirement:
    raw = _mapping(
        value,
        {
            "interrupt_id",
            "assistant_id",
            "assistant_name",
            "power_id",
            "assistant_image",
            "ordinal",
            "title",
            "summary",
            "docs",
            "runs",
        },
        "approval requirement",
    )
    ordinal = raw["ordinal"]
    if (
        type(ordinal) is not int
        or not 0 <= ordinal <= 63
        or raw["runs"] not in {"always", "once"}
        or not isinstance(raw["assistant_image"], str)
        or _IMAGE.fullmatch(raw["assistant_image"]) is None
    ):
        raise ContinuationCodecError("approval requirement is malformed")
    return approval_challenges.ApprovalRequirement(
        _interrupt_id(raw["interrupt_id"]),
        _component_id(raw["assistant_id"], "approval Assistant"),
        str(_text(raw["assistant_name"], 80, "approval Assistant name")),
        _component_id(raw["power_id"], "approval Power"),
        raw["assistant_image"],
        ordinal,
        str(_text(raw["title"], 80, "approval title")),
        str(_text(raw["summary"], 240, "approval summary")),
        _text(raw["docs"], 2048, "approval docs", optional=True),
        raw["runs"],
    )


def decode(
    stored: local_chat_continuation_store.StoredContinuation,
) -> DecodedContinuation:
    """Authenticate structural bindings again after decrypting one record."""
    if not isinstance(stored, local_chat_continuation_store.StoredContinuation):
        raise ContinuationCodecError("stored continuation is malformed")
    body = _decode_payload(stored.payload)
    if body["schema"] != SCHEMA_VERSION or body["kind"] != stored.kind:
        raise ContinuationCodecError("stored continuation contract changed")
    constructors = {
        "accounts": _account_requirement,
        "secrets": _secret_requirement,
        "input": _input_requirement,
        "approval": _approval_requirement,
    }
    constructor = constructors.get(stored.kind)
    if constructor is None:
        raise ContinuationCodecError("stored continuation kind is malformed")
    requirements = tuple(constructor(item) for item in _sequence(body["requirements"], 64, "continuation requirements"))
    if not requirements:
        raise ContinuationCodecError("continuation requirements are malformed")
    pending = _pending(body["pending"])
    if _bindings(stored.kind, requirements, pending) != stored.bindings:
        raise ContinuationCodecError("stored continuation release binding changed")
    return DecodedContinuation(stored.kind, requirements, pending)
