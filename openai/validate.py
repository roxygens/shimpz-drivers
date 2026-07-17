"""Allowlist validation for openai-driver — runs BEFORE any OpenAI call.

Nothing here touches the OpenAI SDK; it only decides yes/no and returns validated values the caller
(app.py) turns into oa_client.py calls. Same shape as every other driver's validate.py — the
actual security boundary. Models are an explicit allowlist so an authenticated caller cannot redirect
spend onto an arbitrary (e.g. far more expensive) model through these endpoints.
"""

from __future__ import annotations

# Exactly the models admitted by the three audited media operations — an allowlist, not a passthrough.
IMAGE_MODELS = frozenset({"gpt-image-2", "gpt-image-1"})
STT_MODELS = frozenset({"gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"})
TTS_MODELS = frozenset({"gpt-4o-mini-tts", "tts-1", "tts-1-hd"})
IMAGE_SIZES = frozenset({"1024x1024", "1536x1024", "1024x1536", "auto"})
IMAGE_QUALITIES = frozenset({"low", "medium", "high", "auto"})
TTS_VOICES = frozenset({"alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"})
TTS_FORMATS = frozenset({"opus", "mp3", "aac", "flac", "wav", "pcm"})

PROMPT_MAX = 32_000
TTS_TEXT_MAX = 4_096
FILENAME_MAX = 200
# Whisper/gpt-4o-transcribe cap uploads at 25 MB; bound here so a caller can't stream an unbounded body.
AUDIO_MAX_BYTES = 25 * 1024 * 1024


class ValidationError(Exception):
    """An openai-driver request failed the allowlist — nothing was sent to OpenAI."""


def _one_of(value: object, allowed: frozenset, field: str) -> str:
    if value not in allowed:
        raise ValidationError(f"{field} must be one of {sorted(allowed)}: {value!r}")
    return value  # type: ignore[return-value]


def validate_prompt(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("prompt must be a non-empty string")
    if len(value) > PROMPT_MAX:
        raise ValidationError(f"prompt length {len(value)} exceeds {PROMPT_MAX}")
    return value


def validate_tts_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("text must be a non-empty string")
    if len(value) > TTS_TEXT_MAX:
        raise ValidationError(f"text length {len(value)} exceeds {TTS_TEXT_MAX}")
    return value


def validate_filename(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > FILENAME_MAX or "/" in value or "\x00" in value:
        raise ValidationError(f"filename must be a simple name up to {FILENAME_MAX} chars: {value!r}")
    return value


def validate_image_model(value: object) -> str:
    return _one_of(value, IMAGE_MODELS, "model")


def validate_stt_model(value: object) -> str:
    return _one_of(value, STT_MODELS, "model")


def validate_tts_model(value: object) -> str:
    return _one_of(value, TTS_MODELS, "model")


def validate_size(value: object) -> str:
    return _one_of(value, IMAGE_SIZES, "size")


def validate_quality(value: object) -> str:
    return _one_of(value, IMAGE_QUALITIES, "quality")


def validate_voice(value: object) -> str:
    return _one_of(value, TTS_VOICES, "voice")


def validate_format(value: object) -> str:
    return _one_of(value, TTS_FORMATS, "response_format")


def validate_audio_size(byte_count: int) -> int:
    if byte_count <= 0 or byte_count > AUDIO_MAX_BYTES:
        raise ValidationError(f"audio size {byte_count} outside 1-{AUDIO_MAX_BYTES}")
    return byte_count
