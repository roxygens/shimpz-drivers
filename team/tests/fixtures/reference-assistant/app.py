"""Deterministic Cloudflare Assistant used by the Docker integration contract."""

from __future__ import annotations

import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer

MAX_BODY = 16 * 1024
HELP_PATHS = {"/v1/help", *(f"/v1/help/{locale}" for locale in ("en", "pt", "es", "zh", "fr", "de", "ja", "ar"))}
POWERS = {"list-zones", "list-dns-records"}
ZONE_ID = re.compile(r"[0-9a-f]{32}\Z")


def _power_input(payload: object, power: str) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != {"input", "secrets", "accounts"}:
        raise ValueError
    power_input = payload["input"]
    if not isinstance(power_input, dict) or payload["secrets"] != {}:
        raise ValueError
    accounts = payload["accounts"]
    if not isinstance(accounts, dict) or set(accounts) != {"cloudflare"}:
        raise ValueError
    account = accounts["cloudflare"]
    if not isinstance(account, dict) or set(account) != {"type", "access_token"}:
        raise ValueError
    token = account["access_token"]
    if account["type"] != "oauth2-bearer" or not isinstance(token, str) or not 16 <= len(token) <= 16 * 1024:
        raise ValueError
    expected = {"page", "per_page"} | ({"zone_id"} if power == "list-dns-records" else set())
    if set(power_input) != expected:
        raise ValueError
    page = power_input["page"]
    per_page = power_input["per_page"]
    if type(page) is not int or type(per_page) is not int or page < 1 or not 1 <= per_page <= 100:
        raise ValueError
    if power == "list-dns-records" and (
        not isinstance(power_input["zone_id"], str) or ZONE_ID.fullmatch(power_input["zone_id"]) is None
    ):
        raise ValueError
    return power_input


def _power_result(power: str, power_input: dict[str, object]) -> dict[str, object]:
    pagination = {
        "page": power_input["page"],
        "per_page": power_input["per_page"],
        "count": 0,
        "total_count": 0,
        "total_pages": 0,
    }
    return {"zones": [], "pagination": pagination} if power == "list-zones" else {
        "records": [],
        "pagination": pagination,
    }


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
        if self.path in HELP_PATHS:
            self._send(HTTPStatus.OK, {"markdown": "# Shimpz Cloudflare\n\nList zones and inspect DNS records."})
            return
        self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        power = self.path.removeprefix("/v1/powers/")
        if power not in POWERS or self.path != f"/v1/powers/{power}":
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if not 2 <= length <= MAX_BODY or self.headers.get("Content-Type") != "application/json":
                raise ValueError
            payload = json.loads(self.rfile.read(length))
            result = _power_result(power, _power_input(payload, power))
        except (ValueError, UnicodeError, json.JSONDecodeError):
            self._send(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": "invalid input"})
            return
        self._send(HTTPStatus.OK, result)


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8080), Handler).serve_forever()
