from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import inference_config


class InferenceConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "inference"
        self.store = inference_config.InferenceConfigStore(self.root)

    def test_defaults_are_provider_metadata_not_a_capsule_image(self):
        config = inference_config.normalize()

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.model, "gpt-5.5")
        self.assertNotIn("image", inference_config.PROVIDERS[config.provider])

    def test_save_and_load_preserve_only_provider_and_model(self):
        expected = inference_config.normalize("anthropic", "claude-sonnet-5")

        self.store.save("capsule_1", expected)
        actual = self.store.load("capsule_1")

        self.assertEqual(actual, expected)
        files = list(self.root.glob("*.json"))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.root.stat().st_mode & 0o777, 0o700)
        self.assertNotIn(b"api_key", files[0].read_bytes())

    def test_replace_is_atomic_and_delete_is_idempotent(self):
        self.store.save("capsule_1", inference_config.normalize("openai", "gpt-5.5"))
        self.store.save("capsule_1", inference_config.normalize("anthropic", "claude-sonnet-5"))

        self.assertEqual(self.store.load("capsule_1").provider, "anthropic")
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        self.store.delete("capsule_1")
        self.store.delete("capsule_1")
        with self.assertRaises(inference_config.InferenceConfigError):
            self.store.load("capsule_1")

    def test_unknown_provider_model_and_capsule_fail_closed(self):
        with self.assertRaises(inference_config.InferenceConfigError):
            inference_config.normalize("codex", "gpt-test")
        with self.assertRaises(inference_config.InferenceConfigError):
            inference_config.normalize("openai", "../../model")
        with self.assertRaises(inference_config.InferenceConfigError):
            self.store.save("../capsule", inference_config.normalize())


if __name__ == "__main__":
    unittest.main()
