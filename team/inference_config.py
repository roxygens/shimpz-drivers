"""Small provider/model registry owned by the Team Controller; never stores secrets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

ROOT = Path(os.environ.get("SHIMPZ_TEAM_INFERENCE_DIR", "/var/lib/team-driver/inference"))
SCHEMA = 1


class ProviderDefinition(TypedDict):
    title: str
    default_model: str
    models: frozenset[str]


_MODEL_CATALOG = json.loads(Path(__file__).with_name("model_catalog.json").read_text(encoding="utf-8"))
PROVIDERS: dict[str, ProviderDefinition] = {
    provider["id"]: {
        "title": provider["title"],
        "default_model": provider["default_model"],
        "models": frozenset(model["id"] for model in provider["models"]),
    }
    for provider in _MODEL_CATALOG["providers"]
}
DEFAULT_PROVIDER = _MODEL_CATALOG["default_provider"]
MODEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")
TEAM_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


class InferenceConfigError(ValueError):
    """Inference metadata is invalid or its private store failed closed."""


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    provider: str
    model: str


def normalize(provider: object = None, model: object = None) -> InferenceConfig:
    selected = str(provider or DEFAULT_PROVIDER).strip().lower()
    if selected not in PROVIDERS:
        raise InferenceConfigError(f"provider must be one of {sorted(PROVIDERS)}")
    selected_model = str(model or PROVIDERS[selected]["default_model"]).strip()
    if MODEL_RE.fullmatch(selected_model) is None or selected_model not in PROVIDERS[selected]["models"]:
        raise InferenceConfigError("model is not supported by the selected provider")
    return InferenceConfig(provider=selected, model=selected_model)


def _team_id(value: object) -> str:
    team_id = str(value or "")
    if TEAM_ID_RE.fullmatch(team_id) is None:
        raise InferenceConfigError("invalid Team id")
    return team_id


class InferenceConfigStore:
    def __init__(self, root: Path = ROOT) -> None:
        self.root = root

    def _prepare(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)

    def _path(self, team_id: str) -> Path:
        digest = hashlib.sha256(team_id.encode()).hexdigest()
        return self.root / f"{digest}.json"

    def save(self, team_id: object, config: InferenceConfig) -> InferenceConfig:
        team_id = _team_id(team_id)
        validated = normalize(config.provider, config.model)
        self._prepare()
        target = self._path(team_id)
        temporary = self.root / f".{target.name}.{secrets.token_hex(8)}.tmp"
        payload = json.dumps(
            {"schema": SCHEMA, "team_id": team_id, **asdict(validated)},
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

    def load(self, team_id: object) -> InferenceConfig:
        team_id = _team_id(team_id)
        try:
            raw = self._path(team_id).read_bytes()
        except OSError as exc:
            raise InferenceConfigError("Team inference configuration is unavailable") from exc
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InferenceConfigError("Team inference configuration is invalid") from exc
        if not isinstance(value, dict) or set(value) != {"schema", "team_id", "provider", "model"}:
            raise InferenceConfigError("Team inference configuration is invalid")
        if value["schema"] != SCHEMA or value["team_id"] != team_id:
            raise InferenceConfigError("Team inference configuration is invalid")
        return normalize(value["provider"], value["model"])

    def delete(self, team_id: object) -> None:
        team_id = _team_id(team_id)
        try:
            self._path(team_id).unlink(missing_ok=True)
        except OSError as exc:
            raise InferenceConfigError("Team inference configuration could not be removed") from exc
