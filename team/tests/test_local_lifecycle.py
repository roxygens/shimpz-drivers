from __future__ import annotations

import copy
import sys
from dataclasses import replace
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM))
import local_app
from assistant_human import approval_grants as assistant_approval_grants
from local_controller_harness import LocalContractCase
from local_support.egress import APP_EGRESS_PROXY_ALIAS

LOOKUP_INPUT = {"page": 1, "per_page": 25}
LOOKUP_RESULT = {
    "zones": [],
    "pagination": {"page": 1, "per_page": 25, "count": 0, "total_count": 0, "total_pages": 0},
}
DNS_INPUT = {"zone_id": "a" * 32, "page": 1, "per_page": 25}
DNS_RESULT = {
    "records": [],
    "pagination": {"page": 1, "per_page": 25, "count": 0, "total_count": 0, "total_pages": 0},
}
TEST_SECRET_VALUES = {
    "service-token": "service-test-credential-123456789",
    "client-key": "client-key-test-credential-123456789",
    "client-secret": "client-secret-test-credential-123456789",
    "session-token": "session-token-test-credential-123456789",
    "session-secret": "session-secret-test-credential-123456789",
}
TEST_ACCOUNT_ACCESS_TOKEN = "-".join(("oauth", "access", "test", "token", "123456789"))
TEST_ACCOUNT_REFRESH_TOKEN = "-".join(("oauth", "refresh", "test", "token", "123456789"))
CURRENT_ASSISTANT_IMAGE = "ghcr.io/theshimpz/shimpz-space@sha256:" + "b" * 64
OUTDATED_ASSISTANT_IMAGE = "ghcr.io/theshimpz/shimpz-space@sha256:" + "a" * 64


class LocalLifecycleTests(LocalContractCase):
    def test_assistant_lifecycle_is_rejected_before_mutation_during_an_active_chat(self) -> None:
        controller, _container, events = self._lifecycle_controller()
        chat_lock = controller._chat_lock("team_1")
        self.assertTrue(chat_lock.acquire(blocking=False))
        try:
            operations = (controller.install_assistant, controller.uninstall_assistant)
            for operation in operations:
                with self.subTest(operation=operation.__name__), self.assertRaises(local_app.ApiProblem) as caught:
                    operation("team_1", "shimpz-cloudflare")
                self.assertEqual((caught.exception.status, caught.exception.code), (HTTPStatus.CONFLICT, "chat-active"))
        finally:
            chat_lock.release()

        self.assertEqual(events, [])

    def test_install_replaces_an_outdated_release_after_current_contract_admission(self) -> None:
        controller, container, events = self._lifecycle_controller()
        controller.assistant_accounts.put(
            "team_1",
            "shimpz-cloudflare",
            "obsolete-account",
            "cloudflare",
            ("zone.read",),
            SimpleNamespace(
                access_token=TEST_ACCOUNT_ACCESS_TOKEN,
                refresh_token=TEST_ACCOUNT_REFRESH_TOKEN,
                scopes=("zone.read",),
                expires_in=3600,
            ),
        )
        trusted_image = object()
        controller._trusted_image = lambda _spec: events.append("trusted") or trusted_image
        controller._create_assistant_container = lambda _team_id, _spec, _network, image: events.append(
            ("create", image)
        )

        result = controller.install_assistant("team_1", "shimpz-cloudflare")

        self.assertEqual(result, {"assistant": "shimpz-cloudflare", "installed": False})
        self.assertEqual(events, ["reload", "trusted", "reload", ("remove", True), ("create", trusted_image)])
        self.assertEqual(container.attrs["Config"]["Image"], OUTDATED_ASSISTANT_IMAGE)
        self.assertFalse(controller.assistant_accounts.delete_assistant("team_1", "shimpz-cloudflare"))

    def test_release_update_is_generic_for_future_assistants(self) -> None:
        controller, container, events = self._lifecycle_controller()
        spec = controller.registry.pop("shimpz-cloudflare")
        spec.assistant_id = "future-assistant"
        controller.registry[spec.assistant_id] = spec
        labels = container.attrs["Config"]["Labels"]
        labels[local_app.ASSISTANT_LABEL] = spec.assistant_id
        container.name = controller._container_name("team_1", spec.assistant_id)
        controller._trusted_image = lambda _spec: events.append("trusted") or object()
        controller._create_assistant_container = lambda *_args: events.append("create")

        self.assertEqual(
            controller.list_assistants("team_1"),
            {"assistants": [{"assistant": "future-assistant", "status": "outdated"}]},
        )
        self.assertEqual(
            controller.install_assistant("team_1", "future-assistant"),
            {"assistant": "future-assistant", "installed": False},
        )
        self.assertEqual(events, ["reload", "reload", "trusted", "reload", ("remove", True), "create"])

    def test_listing_fetches_the_egress_proxy_once_for_multiple_assistants(self) -> None:
        controller, first, _events = self._lifecycle_controller()
        first_spec = controller.registry["shimpz-cloudflare"]
        first_spec.allowed_hosts = ("api.example.com",)
        second_spec = copy.copy(first_spec)
        second_spec.assistant_id = "future-assistant"
        controller.registry[second_spec.assistant_id] = second_spec
        second = copy.deepcopy(first)
        second.labels[local_app.ASSISTANT_LABEL] = second_spec.assistant_id
        second.attrs["Config"]["Labels"][local_app.ASSISTANT_LABEL] = second_spec.assistant_id
        second.name = controller._container_name("team_1", second_spec.assistant_id)
        proxy_environment = {"HTTPS_PROXY": "http://app-egress-proxy:8889"}
        for container in (first, second):
            container.attrs["Config"]["Env"] = [f"{key}={value}" for key, value in proxy_environment.items()]
        network_name = controller._network_name("team_1")
        proxy = SimpleNamespace(
            attrs={
                "NetworkSettings": {
                    "Networks": {
                        network_name: {
                            "Aliases": [APP_EGRESS_PROXY_ALIAS],
                        }
                    }
                }
            }
        )
        controller.client.containers.list = lambda **_kwargs: [first, second]
        controller._validate_egress_policy = lambda *_args: proxy_environment
        controller._egress_proxy = mock.Mock(return_value=proxy)

        result = controller.list_assistants("team_1")

        self.assertEqual(
            tuple(item["assistant"] for item in result["assistants"]),
            ("future-assistant", "shimpz-cloudflare"),
        )
        controller._egress_proxy.assert_called_once_with()

    def test_release_update_rejects_a_previous_security_contract(self) -> None:
        controller, _container, events = self._lifecycle_controller()
        controller.registry["shimpz-cloudflare"].allowed_hosts = ("api.example.com",)
        controller._trusted_image = lambda _spec: self.fail("contract drift reached image resolution")

        with self.assertRaises(local_app.ApiProblem) as caught:
            controller.install_assistant("team_1", "shimpz-cloudflare")

        self.assertEqual(caught.exception.code, "egress-policy-drift")
        self.assertEqual(events, ["reload"])

    def test_container_profile_rejects_duplicate_or_malformed_environment_entries(self) -> None:
        invalid_environments = (
            ["SHIMPZ_TEAM_ID=team_1", "SHIMPZ_TEAM_ID=other"],
            ["HTTPS_PROXY=http://safe", "HTTPS_PROXY=http://evil"],
            ["missing-separator"],
        )
        for environment in invalid_environments:
            with self.subTest(environment=environment):
                controller, container, events = self._lifecycle_controller()
                container.attrs["Config"]["Env"] = environment

                with self.assertRaises(local_app.ApiProblem) as caught:
                    controller.list_assistants("team_1")

                self.assertEqual(caught.exception.code, "assistant-isolation-drift")
                self.assertEqual(events, ["reload"])

    def test_unready_same_release_recovery_preserves_once_approval(self) -> None:
        controller, container, _events = self._lifecycle_controller()
        spec = controller.registry["shimpz-cloudflare"]
        controller.approval_grants.grant_many(
            (
                assistant_approval_grants.Grant(
                    "team_1",
                    "shimpz-cloudflare",
                    "list-zones",
                    CURRENT_ASSISTANT_IMAGE,
                    0,
                ),
            )
        )
        controller._trusted_image = lambda _spec: object()
        controller._validate_container = lambda *_args: None
        controller._create_assistant_container = lambda *_args: None

        controller._replace_unready_assistant("team_1", spec, SimpleNamespace(name="team-network"), container)

        self.assertTrue(
            controller.approval_grants.is_granted(
                "team_1",
                "shimpz-cloudflare",
                "list-zones",
                CURRENT_ASSISTANT_IMAGE,
                0,
            )
        )

    def test_new_assistant_is_admitted_before_egress_and_start(self) -> None:
        events: list[object] = []
        controller = object.__new__(local_app.LocalController)
        controller.space_id = "local-space"
        controller.cpuset_cpus = "0"
        controller._assistant_genesis_cache = local_app.assistant_genesis.GenesisCache()
        controller._assistant_allowed_hosts_cache = local_app.assistant_manifest.ManifestContractCache()
        controller._assistant_machine_contract_cache = local_app.assistant_manifest.MachineContractCache()
        controller._blocked_power_workloads = set()
        spec = SimpleNamespace(
            assistant_id="shimpz-cloudflare",
            image=CURRENT_ASSISTANT_IMAGE,
            allowed_hosts=("api.open-meteo.com", "geocoding-api.open-meteo.com"),
        )
        network = SimpleNamespace(name=controller._network_name("team_1"))
        image = SimpleNamespace(id="sha256:" + "d" * 64)
        container = SimpleNamespace(
            id="assistant-generation",
            attrs={"Image": image.id},
            reload=lambda: events.append("reload"),
            start=lambda: events.append("start"),
            remove=lambda *, force: events.append(("remove", force)),
        )
        controller.client = SimpleNamespace(
            containers=SimpleNamespace(
                create=lambda **_kwargs: events.append("create") or container,
            )
        )
        controller._egress_token = lambda *_args, **_kwargs: events.append("token") or "a" * 32
        controller._admit_assistant_allowed_hosts = lambda _container, _spec: (
            events.append("admit") or tuple(sorted(_spec.allowed_hosts))
        )
        controller._activate_assistant_egress = lambda *_args: events.append("activate-egress")
        controller._validate_container = lambda *_args: events.append("validate")
        controller._wait_ready = lambda *_args: events.append("ready")
        controller._active_assistant_genesis = lambda *_args: events.append("genesis") or "Genesis"

        controller._create_assistant_container("team_1", spec, network, image)

        self.assertLess(events.index("admit"), events.index("activate-egress"))
        self.assertLess(events.index("admit"), events.index("start"))
        self.assertEqual(events[-4:], ["start", "validate", "ready", "genesis"])

    def test_local_admission_reviews_hosts_and_accounts(self) -> None:
        controller = object.__new__(local_app.LocalController)
        reviewed_contracts: list[local_app.assistant_manifest.ManifestContract] = []

        def admit(_container, reviewed):
            reviewed_contracts.append(reviewed)
            return reviewed

        controller._assistant_allowed_hosts_cache = SimpleNamespace(get=admit)
        controller._assistant_machine_contract_cache = SimpleNamespace(
            get=lambda _container, _accounts, reviewed: reviewed
        )
        spec = self._registry(CURRENT_ASSISTANT_IMAGE)["shimpz-cloudflare"]

        allowed_hosts = controller._admit_assistant_allowed_hosts(SimpleNamespace(id="generation"), spec)

        self.assertEqual(allowed_hosts, tuple(sorted(spec.allowed_hosts)))
        self.assertEqual(len(reviewed_contracts), 1)
        self.assertEqual(
            {account.id: (account.provider, account.scopes) for account in reviewed_contracts[0].accounts},
            {
                account_id: (account.provider, tuple(sorted(account.scopes)))
                for account_id, account in spec.accounts.items()
            },
        )
        exact = reviewed_contracts[0]
        account = exact.accounts[0]
        drifted = (
            replace(exact, accounts=(replace(account, provider="other"),)),
            replace(exact, accounts=(replace(account, scopes=("tweet.read",)),)),
        )
        controller._assistant_allowed_hosts_cache = local_app.assistant_manifest.ManifestContractCache()
        with mock.patch.object(
            local_app.assistant_manifest,
            "read_container_manifest_contract",
            return_value=exact,
        ):
            self.assertEqual(
                controller._admit_assistant_allowed_hosts(SimpleNamespace(id="exact-generation"), spec),
                exact.allowed_hosts,
            )
        for index, declared in enumerate(drifted):
            controller._assistant_allowed_hosts_cache = local_app.assistant_manifest.ManifestContractCache()
            with (
                self.subTest(declared=declared),
                mock.patch.object(
                    local_app.assistant_manifest,
                    "read_container_manifest_contract",
                    return_value=declared,
                ),
                self.assertRaises(local_app.ApiProblem) as drift,
            ):
                controller._admit_assistant_allowed_hosts(SimpleNamespace(id=f"drift-generation-{index}"), spec)
            self.assertEqual(drift.exception.code, "assistant-manifest-invalid")
