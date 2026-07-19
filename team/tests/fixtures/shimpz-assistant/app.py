"""Deterministic Shimpz Assistant used only by the Docker integration contract."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer

MAX_BODY = 16 * 1024


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args) -> None:
        return

    def _send(self, status: HTTPStatus, payload: dict[str, object]) -> None:
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
        if self.path == "/v1/help":
            self._send(
                HTTPStatus.OK,
                {"markdown": "# Shimpz Assistant\n\nAsk me to find Lisbon or check its weather."},
            )
            return
        self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path not in {
            "/v1/powers/search-location",
            "/v1/powers/current-weather",
            "/v1/powers/daily-forecast",
        }:
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if not 2 <= length <= MAX_BODY or self.headers.get("Content-Type") != "application/json":
                raise ValueError
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError
        except ValueError, json.JSONDecodeError:
            self._send(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "invalid input"})
            return
        if self.path == "/v1/powers/search-location":
            self._send(
                HTTPStatus.OK,
                {
                    "locations": [
                        {
                            "name": "Lisbon",
                            "country": "Portugal",
                            "latitude": 38.72,
                            "longitude": -9.14,
                            "timezone": "Europe/Lisbon",
                        }
                    ]
                },
            )
        elif self.path == "/v1/powers/current-weather":
            self._send(
                HTTPStatus.OK,
                {
                    "observed_at": "2026-07-18T10:00",
                    "temperature_c": 22.5,
                    "apparent_temperature_c": 22.0,
                    "wind_speed_kmh": 11.2,
                    "weather_code": 1,
                    "timezone": "Europe/Lisbon",
                },
            )
        else:
            self._send(
                HTTPStatus.OK,
                {
                    "timezone": "Europe/Lisbon",
                    "days": [
                        {
                            "date": "2026-07-19",
                            "temperature_min_c": 17.0,
                            "temperature_max_c": 27.0,
                            "precipitation_probability_max": 10,
                            "weather_code": 1,
                        }
                    ],
                },
            )


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8080), Handler).serve_forever()
