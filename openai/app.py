#!/usr/local/bin/python3
"""OpenAI media credential boundary with three audited operations.

This private sidecar implements image generation, audio transcription, and speech synthesis with
closed request shapes and allowlisted models. These operations are not yet exposed as Assistant
Powers, so no Assistant currently consumes this API. It deliberately provides no generic OpenAI
passthrough.

Mandatory controls (same contract as the other sidecars):
  - Auth fail-closed on EVERY endpoint: `Authorization: Bearer <token>` required; no anonymous route.
  - No CORS, ever: this API is reserved for authenticated internal control-plane callers.
  - No execution endpoint: only the three named OpenAI operations, each with an allowlisted model.
  - Redacted audit: only model/size/duration/byte counts — never prompts, transcripts, audio/image
    bytes, or the key.

Endpoints (all require `Authorization: Bearer <token>`):
  POST /v1/openai/image       {prompt, size?, quality?, model?}        -> <PNG bytes>
  POST /v1/openai/transcribe  body=<audio bytes>  headers: X-Filename, X-Model? -> {text}
  POST /v1/openai/speech      {text, model?, voice?, response_format?} -> <audio bytes>
"""

from __future__ import annotations

import ipaddress
import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import audit
import oa_client
import token_store
import validate

LISTEN_PORT = int(os.environ.get("SHIMPZ_OPENAIDRIVER_PORT", "7076"))
_CHUNK = 1024 * 1024

_token = token_store.ensure_token()


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class Handler(BaseHTTPRequestHandler):
    server_version = "openai-driver/1.0"

    def _authed(self) -> bool:
        return self.headers.get("Authorization", "") == f"Bearer {_token}"

    def _send_bytes(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # NEVER an Access-Control-Allow-Origin header — this API is not browser-callable.
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        self._send_bytes(status, "application/json", json.dumps(payload).encode())

    def _json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"invalid JSON body: {exc}") from exc

    def _read_body(self) -> bytes:
        """Read the raw request body (audio upload) in bounded chunks, enforcing the size cap."""
        remaining = int(self.headers.get("Content-Length", "0") or "0")
        validate.validate_audio_size(remaining)
        buf = bytearray()
        while remaining > 0:
            chunk = self.rfile.read(min(_CHUNK, remaining))
            if not chunk:
                break
            buf.extend(chunk)
            remaining -= len(chunk)
        return bytes(buf)

    def _dispatch(self, method: str) -> None:
        if not self._authed():
            # 127.0.0.1 = this container's own Docker HEALTHCHECK proving the 403 gate is live
            # (an unauthenticated probe every 30s BY DESIGN) — keep the audit line but at info,
            # so warn/error carries only real denials, never a heartbeat.
            if self.client_address[0] == "127.0.0.1":
                audit.log("auth", self.path, result="denied", level="info", source="loopback-probe")
            else:
                audit.log("auth", self.path, result="denied")
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "invalid or missing bearer token"})
            return
        try:
            self._route(method)
        except ApiError as exc:
            audit.log(method.lower(), self.path, result="denied", reason=exc.message)
            self._send_json(exc.status, {"error": exc.message})
        except validate.ValidationError as exc:
            audit.log(method.lower(), self.path, result="denied", reason=str(exc))
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except oa_client.OAError as exc:
            audit.log(method.lower(), self.path, result="error", reason=str(exc))
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
        except Exception as exc:
            audit.log(method.lower(), self.path, result="error", reason=str(exc))
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _route(self, method: str) -> None:
        if method == "POST" and self.path == "/v1/openai/image":
            self._image()
            return
        if method == "POST" and self.path == "/v1/openai/transcribe":
            self._transcribe()
            return
        if method == "POST" and self.path == "/v1/openai/speech":
            self._speech()
            return
        raise ApiError(HTTPStatus.NOT_FOUND, f"no route for {method} {self.path}")

    def _image(self) -> None:
        body = self._json_body()
        prompt = validate.validate_prompt(body.get("prompt"))
        size = validate.validate_size(body.get("size", "1024x1024"))
        quality = validate.validate_quality(body.get("quality", "low"))
        model = validate.validate_image_model(body.get("model", "gpt-image-2"))
        png = oa_client.image(prompt, size, quality, model)
        audit.log("image", model, result="ok", size=size, quality=quality, byte_count=len(png))
        self._send_bytes(HTTPStatus.OK, "image/png", png)

    def _transcribe(self) -> None:
        filename = validate.validate_filename(self.headers.get("X-Filename", "audio.ogg"))
        model = validate.validate_stt_model(self.headers.get("X-Model", "gpt-4o-transcribe"))
        audio = self._read_body()
        if not audio:
            raise ApiError(HTTPStatus.BAD_REQUEST, "empty audio body")
        text = oa_client.transcribe(audio, filename, model)
        audit.log("transcribe", model, result="ok", audio_bytes=len(audio), text_len=len(text))
        self._send_json(HTTPStatus.OK, {"text": text})

    def _speech(self) -> None:
        body = self._json_body()
        text = validate.validate_tts_text(body.get("text"))
        model = validate.validate_tts_model(body.get("model", "gpt-4o-mini-tts"))
        voice = validate.validate_voice(body.get("voice", "onyx"))
        fmt = validate.validate_format(body.get("response_format", "opus"))
        audio = oa_client.speech(text, model, voice, fmt)
        audit.log("speech", model, result="ok", voice=voice, fmt=fmt, byte_count=len(audio))
        self._send_bytes(HTTPStatus.OK, "application/octet-stream", audio)

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_GET(self) -> None:
        self._dispatch("GET")

    def log_message(self, fmt: str, *args: object) -> None:
        # Suppress BaseHTTPRequestHandler's default stderr access log — audit.log() is the single
        # source of truth, in the schema logq expects.
        pass


def main() -> None:
    server = ThreadingHTTPServer((str(ipaddress.IPv4Address(0)), LISTEN_PORT), Handler)
    print(f"openai-driver listening on :{LISTEN_PORT}", file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
