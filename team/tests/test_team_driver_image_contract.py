from __future__ import annotations

import unittest
from pathlib import Path


class TeamDriverImageContractTests(unittest.TestCase):
    def test_image_packages_brain_runtime_client_with_a_narrow_token_group(self) -> None:
        dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(encoding="utf-8")

        for module in (
            "brain_runtime_client.py",
            "brain_runtime_token_store.py",
            "chat_orchestrator.py",
            "inference_config.py",
            "power_journal.py",
            "assistant_secret_challenges.py",
            "assistant_secret_flow.py",
            "assistant_secret_store.py",
            "assistant_account_challenges.py",
            "assistant_account_flow.py",
            "cloudflare_assistant_contract.py",
            "oauth_account_store.py",
            "oauth_account_service.py",
            "oauth_http_client.py",
            "oauth_pkce_challenges.py",
            "oauth_providers.py",
            "local_registry.py",
        ):
            with self.subTest(module=module):
                self.assertIn(module, dockerfile)
        self.assertIn("ARG SHIMPZ_BRAIN_RUNTIME_TOKEN_GID=10016", dockerfile)
        self.assertIn(
            'groupadd -g "${SHIMPZ_BRAIN_RUNTIME_TOKEN_GID}" shimpzbrain-runtime-token',
            dockerfile,
        )
        self.assertNotIn("r2", dockerfile.lower())
        self.assertIn(
            "chown teamdriver:shimpzbrain-runtime-token /run/shimpz-brain-runtime",
            dockerfile,
        )
        self.assertIn("chmod 0750 /run/shimpz-brain-runtime", dockerfile)
        self.assertIn("/var/lib/team-driver/inference", dockerfile)
        self.assertIn("/var/lib/team-driver/power-journal", dockerfile)
        self.assertIn("/var/lib/team-driver/assistant-secrets/state", dockerfile)
        self.assertIn("/var/lib/team-driver/assistant-secrets/key", dockerfile)
        self.assertIn("/var/lib/team-driver/assistant-accounts/state", dockerfile)
        self.assertIn("/var/lib/team-driver/assistant-accounts/key", dockerfile)
        self.assertIn(
            "/var/lib/team-driver/cleanup \\\n"
            "        /var/lib/team-driver/inference \\\n"
            "        /var/lib/team-driver/power-journal \\",
            dockerfile,
        )


if __name__ == "__main__":
    unittest.main()
