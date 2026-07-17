"""The ONLY place the OpenAI API key is ever read or sent.

A thin wrapper over the `openai` SDK, called ONLY by app.py's already-allowlisted (validate.py)
endpoint handlers. Never exposes a generic "any OpenAI call" passthrough — every function here is one
SPECIFIC operation (image / transcribe / speech) with a fixed shape. The credential stays inside
this module and cannot be spent on arbitrary endpoints such as fine-tuning or the Assistants API.
No current Assistant Power calls these functions; they remain private building blocks for a future
Controller integration.
"""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

from openai import OpenAI, OpenAIError

_KEY = (os.environ.get("VOICE_TOOLS_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()


class OAError(Exception):
    """An OpenAI call failed — the SDK's error message, surfaced (never a silent empty result)."""


_client = OpenAI(api_key=_KEY) if _KEY else None


def _require() -> OpenAI:
    if _client is None:
        raise OAError("openai-driver has no API key configured")
    return _client


def image(prompt: str, size: str, quality: str, model: str) -> bytes:
    """Generate one image; returns the raw PNG bytes (decoded from the API's b64_json)."""
    try:
        resp = _require().images.generate(model=model, prompt=prompt, size=size, quality=quality, n=1)
    except OpenAIError as exc:
        raise OAError(str(exc)) from exc
    b64 = resp.data[0].b64_json
    if not b64:
        raise OAError("image response carried no b64_json")
    return base64.b64decode(b64)


def transcribe(audio: bytes, filename: str, model: str) -> str:
    """Transcribe audio bytes to text. `filename` gives the SDK the format hint (extension)."""
    fd, tmp_str = tempfile.mkstemp(prefix="oa-stt-", suffix=f"-{filename}", dir="/tmp")
    tmp = Path(tmp_str)
    os.close(fd)
    try:
        tmp.write_bytes(audio)
        with tmp.open("rb") as fh:
            try:
                return _require().audio.transcriptions.create(model=model, file=fh).text.strip()
            except OpenAIError as exc:
                raise OAError(str(exc)) from exc
    finally:
        tmp.unlink(missing_ok=True)


def speech(text: str, model: str, voice: str, response_format: str) -> bytes:
    """Synthesize `text` to speech; returns the raw audio bytes in `response_format`."""
    try:
        resp = _require().audio.speech.create(model=model, voice=voice, input=text, response_format=response_format)
    except OpenAIError as exc:
        raise OAError(str(exc)) from exc
    return resp.content
