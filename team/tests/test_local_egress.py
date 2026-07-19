from __future__ import annotations

import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import local_app


class _Proxy:
    def __init__(self, space_id: str) -> None:
        self.name = "local-egress-proxy"
        self.status = "running"
        self.attrs = {
            "Config": {
                "User": "10005:10005",
                "Labels": {
                    local_app.MANAGED_LABEL: "1",
                    local_app.PROFILE_LABEL: local_app.PROFILE,
                    local_app.SPACE_LABEL: space_id,
                    local_app.KIND_LABEL: local_app.APP_EGRESS_PROXY_KIND,
                },
            },
            "HostConfig": {
                "ReadonlyRootfs": True,
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges:true"],
                "Privileged": False,
                "PortBindings": {},
            },
            "Mounts": [{"Destination": "/policy", "RW": False}],
            "NetworkSettings": {"Networks": {"outbound": {"Aliases": ["local-egress-proxy"]}}},
        }

    def reload(self) -> None:
        return


class _Network:
    name = "team-network"

    def __init__(self, proxy: _Proxy) -> None:
        self.proxy = proxy
        self.attrs = {"Containers": {}}

    def reload(self) -> None:
        networks = self.proxy.attrs["NetworkSettings"]["Networks"]
        self.attrs["Containers"] = {"proxy": {"Name": self.proxy.name}} if self.name in networks else {}

    def connect(self, proxy: _Proxy, *, aliases: list[str]) -> None:
        proxy.attrs["NetworkSettings"]["Networks"][self.name] = {"Aliases": aliases}
        self.reload()

    def disconnect(self, proxy: _Proxy) -> None:
        proxy.attrs["NetworkSettings"]["Networks"].pop(self.name, None)
        self.reload()


class _Containers:
    def __init__(self, proxy: _Proxy) -> None:
        self.proxy = proxy
        self.installed: list[object] = []

    def get(self, name: str):
        if name != self.proxy.name:
            raise AssertionError(f"unexpected container {name}")
        return self.proxy

    def list(self, **_kwargs):
        return self.installed


class LocalAssistantEgressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.policy_root = Path(self.directory.name) / "policy"
        self.policy_root.mkdir(mode=0o770)
        self.policy_root.chmod(0o770)
        self.proxy = _Proxy("local-space")
        self.network = _Network(self.proxy)
        self.controller = object.__new__(local_app.LocalController)
        self.controller.space_id = "local-space"
        self.controller.client = types.SimpleNamespace(containers=_Containers(self.proxy))
        self.spec = types.SimpleNamespace(
            assistant_id="shimpz-assistant",
            egress=("api.open-meteo.com", "geocoding-api.open-meteo.com"),
        )
        self.controller.registry = {self.spec.assistant_id: self.spec}
        self.patches = (
            mock.patch.object(local_app, "APP_EGRESS_POLICY_DIR", self.policy_root),
            mock.patch.object(local_app, "APP_EGRESS_POLICY_GID", os.getgid()),
            mock.patch.object(local_app, "APP_EGRESS_PROXY_CONTAINER", self.proxy.name),
        )
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_policy_is_private_stable_exact_and_proxy_attachment_is_dynamic(self) -> None:
        environment = self.controller._prepare_assistant_egress(
            "team_1",
            self.spec,
            self.network,
        )

        token = environment["HTTPS_PROXY"].split("@", 1)[0].rsplit("/", 1)[-1]
        self.assertRegex(token, r"^[0-9a-f]{32}$")
        self.assertEqual(environment["HTTPS_PROXY"], environment["https_proxy"])
        self.assertEqual(environment["NO_PROXY"], "127.0.0.1,localhost")
        self.assertIn(self.network.name, self.proxy.attrs["NetworkSettings"]["Networks"])
        self.assertIn(
            local_app.APP_EGRESS_PROXY_ALIAS,
            self.proxy.attrs["NetworkSettings"]["Networks"][self.network.name]["Aliases"],
        )
        policy = self.policy_root / f"{token}.json"
        self.assertEqual(json.loads(policy.read_text(encoding="ascii")), sorted(self.spec.egress))
        self.assertEqual(policy.stat().st_mode & 0o777, 0o640)
        token_files = list((self.policy_root / ".tokens").glob("*.token"))
        self.assertEqual(len(token_files), 1)
        self.assertEqual(token_files[0].stat().st_mode & 0o777, 0o600)

        repeated = self.controller._prepare_assistant_egress("team_1", self.spec, self.network)
        self.assertEqual(repeated, environment)
        self.assertEqual(self.controller._validate_egress_policy("team_1", self.spec), environment)

    def test_last_uninstall_removes_policy_and_detaches_proxy(self) -> None:
        environment = self.controller._prepare_assistant_egress("team_1", self.spec, self.network)
        token = environment["HTTPS_PROXY"].split("@", 1)[0].rsplit("/", 1)[-1]

        self.controller._release_assistant_egress("team_1", self.spec.assistant_id, self.network)

        self.assertFalse((self.policy_root / f"{token}.json").exists())
        self.assertEqual(list((self.policy_root / ".tokens").glob("*.token")), [])
        self.assertNotIn(self.network.name, self.proxy.attrs["NetworkSettings"]["Networks"])

    def test_policy_tampering_fails_closed(self) -> None:
        environment = self.controller._prepare_assistant_egress("team_1", self.spec, self.network)
        token = environment["HTTPS_PROXY"].split("@", 1)[0].rsplit("/", 1)[-1]
        policy = self.policy_root / f"{token}.json"
        policy.write_text('["evil.example"]', encoding="ascii")

        with self.assertRaises(local_app.ApiProblem) as caught:
            self.controller._validate_egress_policy("team_1", self.spec)

        self.assertEqual(caught.exception.code, "egress-policy-drift")


if __name__ == "__main__":
    unittest.main()
