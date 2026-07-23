from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import assistant_manifest
import local_registry
import marketplace


class LocalRegistryAccountTests(unittest.TestCase):
    def test_cloudflare_contract_matches_hosted_and_local_registries(self) -> None:
        digest = "127.0.0.1:5000/shimpz/shimpz-cloudflare@sha256:" + "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(
                json.dumps({"schema": 2, "images": {"shimpz-cloudflare": digest}}),
                encoding="utf-8",
            )
            spec = local_registry.load_registry(path)["shimpz-cloudflare"]

        self.assertEqual(spec.secrets, {})
        self.assertEqual(spec.accounts["cloudflare"].provider, "cloudflare")
        self.assertEqual(spec.accounts["cloudflare"].scopes, ("dns.read", "offline_access", "zone.read"))
        self.assertEqual(spec.allowed_hosts, ("api.cloudflare.com",))
        self.assertTrue(all(power.accounts == ("cloudflare",) for power in spec.powers.values()))

        hosted = marketplace.APPS["shimpz-cloudflare"]
        assert hosted.assistant is not None
        self.assertEqual(
            assistant_manifest.reviewed_manifest_contract(
                allowed_hosts=spec.allowed_hosts,
                accounts=spec.accounts,
            ),
            assistant_manifest.reviewed_manifest_contract(
                allowed_hosts=hosted.allowed_hosts,
                accounts=hosted.assistant.accounts,
            ),
        )


if __name__ == "__main__":
    unittest.main()
