"""Real tiny Assistant used only by the Docker integration contract."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer

MAX_BODY = 16 * 1024


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args) -> None:
        return

    def _send(self, status: HTTPStatus, payload: dict[str, str]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._send(HTTPStatus.OK, {"status": "ok"})
            return
        self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/operations/hello":
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if not 2 <= length <= MAX_BODY or self.headers.get("Content-Type") != "application/json":
                raise ValueError
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or set(payload) != {"name"}:
                raise ValueError
            name = payload["name"]
            if not isinstance(name, str) or not 1 <= len(name) <= 80:
                raise ValueError
        except ValueError, json.JSONDecodeError:
            self._send(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "invalid input"})
            return
        self._send(HTTPStatus.OK, {"message": f"Hello, {name}. Your Capsule is alive."})


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8080), Handler).serve_forever()
