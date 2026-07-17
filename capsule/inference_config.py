"""Small provider/model registry owned by the Capsule Controller; never stores secrets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(os.environ.get("SHIMPZ_CAPSULE_INFERENCE_DIR", "/var/lib/capsule-driver/inference"))
SCHEMA = 1
PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {"title": "OpenAI", "default_model": "gpt-5.5"},
    "anthropic": {"title": "Anthropic", "default_model": "claude-sonnet-5"},
}
DEFAULT_PROVIDER = "openai"
MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")
CAPSULE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


class InferenceConfigError(ValueError):
    """Inference metadata is invalid or its private store failed closed."""


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    provider: Literal["anthropic", "openai"]
    model: str


def normalize(provider: object = None, model: object = None) -> InferenceConfig:
    selected = str(provider or DEFAULT_PROVIDER).strip().lower()
    if selected not in PROVIDERS:
        raise InferenceConfigError(f"provider must be one of {sorted(PROVIDERS)}")
    selected_model = str(model or PROVIDERS[selected]["default_model"]).strip()
    if MODEL_RE.fullmatch(selected_model) is None:
        raise InferenceConfigError("model must be a safe identifier of at most 128 characters")
    return InferenceConfig(provider=selected, model=selected_model)


def _capsule_id(value: object) -> str:
    capsule_id = str(value or "")
    if CAPSULE_ID_RE.fullmatch(capsule_id) is None:
        raise InferenceConfigError("invalid Capsule id")
    return capsule_id


class InferenceConfigStore:
    def __init__(self, root: Path = ROOT) -> None:
        self.root = root

    def _prepare(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)

    def _path(self, capsule_id: str) -> Path:
        digest = hashlib.sha256(capsule_id.encode()).hexdigest()
        return self.root / f"{digest}.json"

    def save(self, capsule_id: object, config: InferenceConfig) -> InferenceConfig:
        cid = _capsule_id(capsule_id)
        validated = normalize(config.provider, config.model)
        self._prepare()
        target = self._path(cid)
        temporary = self.root / f".{target.name}.{secrets.token_hex(8)}.tmp"
        payload = json.dumps(
            {"schema": SCHEMA, "capsule": cid, **asdict(validated)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
            target.chmod(0o600)
        finally:
            temporary.unlink(missing_ok=True)
        return validated

    def load(self, capsule_id: object) -> InferenceConfig:
        cid = _capsule_id(capsule_id)
        try:
            raw = self._path(cid).read_bytes()
        except OSError as exc:
            raise InferenceConfigError("Capsule inference configuration is unavailable") from exc
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InferenceConfigError("Capsule inference configuration is invalid") from exc
        if not isinstance(value, dict) or set(value) != {"schema", "capsule", "provider", "model"}:
            raise InferenceConfigError("Capsule inference configuration is invalid")
        if value["schema"] != SCHEMA or value["capsule"] != cid:
            raise InferenceConfigError("Capsule inference configuration is invalid")
        return normalize(value["provider"], value["model"])

    def delete(self, capsule_id: object) -> None:
        cid = _capsule_id(capsule_id)
        try:
            self._path(cid).unlink(missing_ok=True)
        except OSError as exc:
            raise InferenceConfigError("Capsule inference configuration could not be removed") from exc
