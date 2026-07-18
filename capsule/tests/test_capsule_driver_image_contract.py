from __future__ import annotations

import unittest
from pathlib import Path


class CapsuleDriverImageContractTests(unittest.TestCase):
    def test_image_packages_brain_runtime_client_with_a_narrow_token_group(self) -> None:
        dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(encoding="utf-8")

        for module in (
            "brain_runtime_client.py",
            "brain_runtime_token_store.py",
            "chat_orchestrator.py",
            "inference_config.py",
            "power_journal.py",
        ):
            with self.subTest(module=module):
                self.assertIn(module, dockerfile)
        self.assertIn("ARG SHIMPZ_BRAIN_RUNTIME_TOKEN_GID=10016", dockerfile)
        self.assertIn(
            'groupadd -g "${SHIMPZ_BRAIN_RUNTIME_TOKEN_GID}" shimpzbrain-runtime-token',
            dockerfile,
        )
        self.assertIn("shimpzr2provisioner-token,shimpzbrain-runtime-token", dockerfile)
        self.assertIn(
            "chown capsuledriver:shimpzbrain-runtime-token /run/shimpz-brain-runtime",
            dockerfile,
        )
        self.assertIn("chmod 0750 /run/shimpz-brain-runtime", dockerfile)
        self.assertIn("/var/lib/capsule-driver/power-journal", dockerfile)
        self.assertIn(
            "/var/lib/capsule-driver/cleanup \\\n        /var/lib/capsule-driver/power-journal \\",
            dockerfile,
        )


if __name__ == "__main__":
    unittest.main()
