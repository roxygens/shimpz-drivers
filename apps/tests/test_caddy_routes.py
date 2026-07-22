from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

APPS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APPS))

import caddy_routes
from validate import RouteRequest


def request(**overrides: object) -> RouteRequest:
    values: dict[str, object] = {
        "fqdn": "app.example.com",
        "web_target": "app_web",
        "web_port": 3100,
        "api_target": None,
        "api_port": None,
        "ws_target": None,
        "ws_port": None,
    }
    values.update(overrides)
    return RouteRequest(**values)


class CaddyRouteTests(unittest.TestCase):
    def test_web_api_and_websocket_keep_distinct_targets(self) -> None:
        rendered = caddy_routes.render(
            request(api_target="app_api", api_port=3101, ws_target="app_ws", ws_port=3102),
            "172.19.0.0/16",
        )

        self.assertIn("reverse_proxy app_web:3100", rendered)
        self.assertIn("reverse_proxy @api app_api:3101", rendered)
        self.assertIn("reverse_proxy @ws app_ws:3102", rendered)
        self.assertIn("uri @api strip_prefix /api", rendered)

    def test_optional_routes_are_not_rendered_without_a_real_port(self) -> None:
        web_only = caddy_routes.render(request(), "172.19.0.0/16")
        zero_api = caddy_routes.render(request(api_target="app_api", api_port=0), "172.19.0.0/16")

        self.assertNotIn("@api", web_only)
        self.assertNotIn("@ws", web_only)
        self.assertNotIn("@api", zero_api)

    def test_lateral_access_gate_precedes_every_upstream(self) -> None:
        rendered = caddy_routes.render(
            request(api_target="app_api", api_port=3101, ws_target="app_ws", ws_port=3102),
            "172.19.0.0/16 172.31.0.0/16",
        )

        self.assertIn("route {", rendered)
        self.assertIn("@not_edge not remote_ip 172.19.0.0/16 172.31.0.0/16", rendered)
        gate = rendered.index("respond @not_edge 403")
        for upstream in ("reverse_proxy @ws", "reverse_proxy @api", "reverse_proxy app_web"):
            with self.subTest(upstream=upstream):
                self.assertGreater(rendered.index(upstream), gate)

    def test_browser_security_policy_is_complete_and_eval_free(self) -> None:
        rendered = caddy_routes.render(request(), "172.19.0.0/16")
        headers = dict(caddy_routes.SECURITY_HEADERS)

        for name, value in headers.items():
            with self.subTest(header=name):
                self.assertEqual(rendered.count(f'{name} "{value}"'), 1)
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertNotIn("'unsafe-eval'", headers["Content-Security-Policy"])
        self.assertEqual(headers["X-Frame-Options"], "DENY")

    def test_allowed_sources_come_from_current_network_metadata(self) -> None:
        class Network:
            def __init__(self, subnet: str) -> None:
                self.attrs = {"IPAM": {"Config": [{"Subnet": subnet}]}}

        class Networks:
            def get(self, name: str) -> Network:
                self.assert_name(name)
                return Network("172.19.0.0/16")

            def list(self, names: list[str]) -> list[Network]:
                return [Network("172.31.0.0/16")] if names == [caddy_routes.CADDY_NETWORK] else []

            @staticmethod
            def assert_name(name: str) -> None:
                if name != caddy_routes.EDGE_NETWORK:
                    raise AssertionError(name)

        class Client:
            networks = Networks()

        self.assertEqual(caddy_routes._allowed_source_cidrs(Client()), "172.19.0.0/16 172.31.0.0/16")

    def test_route_filenames_and_inventory_cannot_escape_the_site_directory(self) -> None:
        self.assertNotIn("/", caddy_routes._safe_filename("../../etc/passwd"))
        self.assertNotIn(" ", caddy_routes._safe_filename("app.example.com {"))

        original = caddy_routes.SITES_DIR
        with tempfile.TemporaryDirectory(prefix="caddy-routes-test-") as temporary:
            caddy_routes.SITES_DIR = Path(temporary)
            (caddy_routes.SITES_DIR / "b.example.com.caddy").write_text("test", encoding="utf-8")
            (caddy_routes.SITES_DIR / "a.example.com.caddy").write_text("test", encoding="utf-8")
            (caddy_routes.SITES_DIR / "ignored.txt").write_text("test", encoding="utf-8")
            try:
                self.assertEqual(caddy_routes.list_routes(), ["a.example.com", "b.example.com"])
            finally:
                caddy_routes.SITES_DIR = original


if __name__ == "__main__":
    unittest.main()
