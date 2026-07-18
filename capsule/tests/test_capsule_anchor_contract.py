from __future__ import annotations

import sys
import unittest
from pathlib import Path

CAPSULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAPSULE))

import manifests
import network_policy


class CapsuleAnchorContractTests(unittest.TestCase):
    def test_anchor_has_no_model_runtime_or_secret_bearing_filesystem(self) -> None:
        kwargs = manifests.build_capsule_kwargs(
            "capsule_1",
            "Capsule 1",
            database_url="postgresql://must-not-enter-anchor",
        )

        self.assertEqual(set(manifests.BRAINS), {"runtime"})
        self.assertIn("registry.k8s.io/pause:3.10.1@sha256:", kwargs["image"])
        self.assertEqual(kwargs["environment"], {"SHIMPZ_CAPSULE_ID": "capsule_1", "SHIMPZ_CAPSULE_NAME": "Capsule 1"})
        self.assertNotIn("postgresql://", repr(kwargs))
        self.assertTrue(kwargs["read_only"])
        self.assertEqual(kwargs["cap_drop"], ["ALL"])
        self.assertEqual(kwargs["cap_add"], [])
        self.assertEqual(kwargs["mounts"], [])
        self.assertNotIn("healthcheck", kwargs)
        self.assertEqual(kwargs["network"], network_policy.network_name("capsule_1", network_policy.CORE_KIND))
        self.assertEqual(network_policy.NETWORK_KINDS, {network_policy.CORE_KIND})

    def test_anchor_reserves_only_a_small_idle_envelope(self) -> None:
        self.assertEqual(manifests.MEM_LIMIT_BYTES, 64 * 1024 * 1024)
        self.assertEqual(manifests.NANO_CPUS, 100_000_000)
        self.assertEqual(manifests.PIDS_LIMIT, 128)


if __name__ == "__main__":
    unittest.main()
