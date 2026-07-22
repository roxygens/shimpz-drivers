from __future__ import annotations

import sys
import unittest
from pathlib import Path

APPS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APPS))

import manifests
import validate


class ManifestTests(unittest.TestCase):
    def request(self, *, persist: bool = False) -> validate.DeployRequest:
        return validate.DeployRequest(
            name="website",
            image="runtime@sha256:test",
            entrypoint=["python3", "main.py"],
            port=3100,
            env={"SECRET_KEY": "application-secret"},
            persist=persist,
            run_subpath="website",
            working_dir="/app",
            worker=False,
            egress=[],
        )

    def test_container_kwargs_enforce_the_runtime_sandbox(self) -> None:
        request = self.request()
        kwargs = manifests.build_container_kwargs(request, "/workspace/projects")

        self.assertFalse(kwargs["privileged"])
        self.assertTrue(kwargs["read_only"])
        self.assertEqual(kwargs["user"], "10001:10001")
        self.assertEqual(kwargs["cap_drop"], ["ALL"])
        self.assertEqual(kwargs["security_opt"], ["no-new-privileges:true"])
        self.assertEqual(kwargs["network"], "net_app_website")
        self.assertNotIn("ports", kwargs)
        self.assertNotIn("devices", kwargs)
        self.assertNotIn("pid_mode", kwargs)

        mounts = kwargs["mounts"]
        self.assertEqual(len(mounts), 1)
        self.assertEqual(mounts[0]["Source"], "/workspace/projects/website")
        self.assertEqual(mounts[0]["Target"], "/app")
        self.assertTrue(mounts[0]["ReadOnly"])

    def test_persistence_adds_only_the_apps_named_volume(self) -> None:
        kwargs = manifests.build_container_kwargs(self.request(persist=True), "/workspace/projects")
        mounts = kwargs["mounts"]

        self.assertEqual(len(mounts), 2)
        persistent = mounts[1]
        self.assertEqual(persistent["Type"], "volume")
        self.assertEqual(persistent["Source"], "app_website_data")
        self.assertEqual(persistent["Target"], "/data")
        self.assertFalse(persistent["ReadOnly"])


if __name__ == "__main__":
    unittest.main()
