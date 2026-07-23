from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM))

import assistant_account_challenges
import assistant_secret_challenges
import assistant_secret_store
import inference_config
import local_app
import local_chat_continuation_store
import local_registry
import oauth_account_store
import oauth_pkce_challenges
from assistant_human import approval_challenges as assistant_approval_challenges
from assistant_human import approval_grants as assistant_approval_grants
from assistant_human import input_challenges as assistant_input_challenges
from local_support import assistant_lifecycle
from local_support.assistant_rpc import ASSISTANT_UID
from local_support.chat_types import ActiveAssistant

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


class LocalContractCase(unittest.TestCase):
    def _registry(
        self,
        image: str,
        *,
        with_test_secrets: bool = False,
    ) -> dict[str, local_registry.AssistantSpec]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": 2,
                        "images": {"shimpz-cloudflare": image},
                    }
                ),
                encoding="utf-8",
            )
            registry = local_registry.load_registry(path)
        if not with_test_secrets:
            return registry
        spec = registry["shimpz-cloudflare"]
        test_secrets = {
            secret_id: local_registry.SecretSpec(
                name=secret_id.replace("-", " ").title(),
                summary="Test-only credential used to exercise the generic secret boundary.",
            )
            for secret_id in TEST_SECRET_VALUES
        }
        test_powers = {
            power_id: replace(
                power,
                secrets=tuple(TEST_SECRET_VALUES),
                accounts=(),
            )
            for power_id, power in spec.powers.items()
        }
        return {
            spec.assistant_id: replace(
                spec,
                powers=test_powers,
                secrets=test_secrets,
                accounts={},
            )
        }

    def _chat_controller(
        self,
        directory: str,
        runtime,
        *,
        configure_secrets: bool | None = None,
    ) -> local_app.LocalController:
        image = "127.0.0.1:5000/shimpz/shimpz-cloudflare@sha256:" + "a" * 64
        controller = object.__new__(local_app.LocalController)
        controller.space_id = "local-space"
        controller.registry = self._registry(
            image,
            with_test_secrets=configure_secrets is not None,
        )
        controller.storage = SimpleNamespace(metadata=lambda _team_id, _files: [])
        controller.inference_store = inference_config.InferenceConfigStore(Path(directory) / "inference")
        controller.inference_store.save(
            "team_1",
            inference_config.normalize("openai", "gpt-5.5"),
        )
        controller.brain_runtime = runtime
        controller.power_state = local_app.power_journal.PowerJournal(
            Path(directory) / "power-journal" / "journal.sqlite3"
        )
        self.addCleanup(controller.power_state.close)
        controller.assistant_secrets = assistant_secret_store.AssistantSecretStore(
            Path(directory) / "assistant-secrets" / "state" / "secrets.json",
            Path(directory) / "assistant-secrets" / "key" / "aes256.key",
        )
        controller.secret_challenges = assistant_secret_challenges.SecretChallengeStore()
        controller.assistant_accounts = oauth_account_store.OAuthAccountStore(
            Path(directory) / "assistant-accounts" / "state" / "accounts.json",
            Path(directory) / "assistant-accounts" / "key" / "aes256.key",
        )
        controller.account_challenges = assistant_account_challenges.AccountChallengeStore()
        controller.oauth_pkce = oauth_pkce_challenges.OAuthPKCEChallengeStore()
        controller.approval_challenges = assistant_approval_challenges.ApprovalChallengeStore()
        controller.input_challenges = assistant_input_challenges.InputChallengeStore()
        controller.chat_continuations = local_chat_continuation_store.EncryptedContinuationStore(
            Path(directory) / "chat-continuations" / "state" / "continuations.json",
            Path(directory) / "chat-continuations" / "key" / "aes256.key",
        )
        controller.approval_grants = assistant_approval_grants.ApprovalGrantStore(
            Path(directory) / "assistant-approvals" / "grants.sqlite3"
        )
        self.addCleanup(controller.approval_grants.close)
        if configure_secrets is True:
            controller.assistant_secrets.put_many(
                "team_1",
                "shimpz-cloudflare",
                TEST_SECRET_VALUES,
            )
        if configure_secrets is None:
            account = controller.registry["shimpz-cloudflare"].accounts["cloudflare"]
            existing = controller.assistant_accounts.metadata(
                "team_1",
                "shimpz-cloudflare",
                {"cloudflare": account},
            )
            if existing[0].status == "missing":
                controller.assistant_accounts.put(
                    "team_1",
                    "shimpz-cloudflare",
                    "cloudflare",
                    account.provider,
                    account.scopes,
                    SimpleNamespace(
                        access_token=TEST_ACCOUNT_ACCESS_TOKEN,
                        refresh_token=TEST_ACCOUNT_REFRESH_TOKEN,
                        scopes=account.scopes,
                        expires_in=3600,
                    ),
                )
        controller._blocked_power_workloads = set()
        controller._locks = tuple(threading.RLock() for _ in range(64))
        controller._active_chat_guard = threading.Lock()
        controller._chat_locks = {}
        controller._active_chat_tokens = {}
        controller._active_power_containers = {}
        controller._cancelled_chat_tokens = set()
        controller._assistant_genesis_cache = local_app.assistant_genesis.GenesisCache()
        controller._assistant_allowed_hosts_cache = local_app.assistant_manifest.ManifestContractCache()
        controller._assistant_machine_contract_cache = local_app.assistant_manifest.MachineContractCache()
        controller._admit_assistant_allowed_hosts = lambda _container, spec: tuple(sorted(spec.allowed_hosts))
        container = SimpleNamespace(id="assistant-container", status="running", reload=lambda: None)
        network = SimpleNamespace(id="a" * 64, name="team-network")
        controller._network = lambda _team_id: network
        controller._validate_network = lambda _network, _team_id: "Marketing"
        controller._assistant_container = lambda _team_id, _assistant: container
        controller._validate_container = lambda *_args: None
        controller._active_chat_assistants = lambda _team_id, _network: (
            ActiveAssistant(controller.registry["shimpz-cloudflare"], container.id, container),
        )
        controller._active_assistant_genesis = lambda _active: "Use only the declared Cloudflare Powers."
        controller._restore_all_chat_continuations()
        return controller

    @staticmethod
    def _secret_submission(challenge: dict[str, object]) -> dict[str, object]:
        return {
            "challenge_id": challenge["challenge_id"],
            "values": [
                {
                    "assistant_id": requirement["assistant_id"],
                    "secret_id": secret["id"],
                    "value": TEST_SECRET_VALUES[secret["id"]],
                }
                for requirement in challenge["requirements"]
                for secret in requirement["secrets"]
            ],
        }

    def _lifecycle_controller(self) -> tuple[local_app.LocalController, SimpleNamespace, list[object]]:
        events: list[object] = []
        controller = object.__new__(local_app.LocalController)
        controller.space_id = "local-space"
        controller.cpuset_cpus = "0"
        controller._locks = tuple(threading.RLock() for _ in range(64))
        controller._active_chat_guard = threading.Lock()
        controller._chat_locks = {}
        controller._blocked_power_workloads = set()
        controller._assistant_genesis_cache = local_app.assistant_genesis.GenesisCache()
        controller._assistant_allowed_hosts_cache = local_app.assistant_manifest.ManifestContractCache()
        controller._assistant_machine_contract_cache = local_app.assistant_manifest.MachineContractCache()
        secret_directory = tempfile.TemporaryDirectory()
        self.addCleanup(secret_directory.cleanup)
        controller.assistant_secrets = assistant_secret_store.AssistantSecretStore(
            Path(secret_directory.name) / "state" / "secrets.json",
            Path(secret_directory.name) / "key" / "aes256.key",
        )
        controller.secret_challenges = assistant_secret_challenges.SecretChallengeStore()
        controller.assistant_accounts = oauth_account_store.OAuthAccountStore(
            Path(secret_directory.name) / "assistant-accounts" / "state" / "accounts.json",
            Path(secret_directory.name) / "assistant-accounts" / "key" / "aes256.key",
        )
        controller.account_challenges = assistant_account_challenges.AccountChallengeStore()
        controller.oauth_pkce = oauth_pkce_challenges.OAuthPKCEChallengeStore()
        controller.approval_challenges = assistant_approval_challenges.ApprovalChallengeStore()
        controller.input_challenges = assistant_input_challenges.InputChallengeStore()
        controller.approval_grants = assistant_approval_grants.ApprovalGrantStore(
            Path(secret_directory.name) / "assistant-approvals" / "grants.sqlite3"
        )
        self.addCleanup(controller.approval_grants.close)
        controller._admit_assistant_allowed_hosts = lambda _container, spec: tuple(sorted(spec.allowed_hosts))
        controller._read_admitted_egress_policy = lambda *_args: None
        spec = SimpleNamespace(
            assistant_id="shimpz-cloudflare",
            image=CURRENT_ASSISTANT_IMAGE,
            allowed_hosts=(),
            secrets={},
            accounts={},
        )
        controller.registry = {spec.assistant_id: spec}
        network_name = controller._network_name("team_1")
        network = SimpleNamespace(name=network_name)
        controller._network = lambda _team_id: network
        labels = controller._assistant_labels("team_1", spec)
        labels[local_app.IMAGE_LABEL] = OUTDATED_ASSISTANT_IMAGE
        container = SimpleNamespace(
            id="assistant-container",
            name=controller._container_name("team_1", spec.assistant_id),
            status="running",
            labels=labels,
            attrs={
                "Config": {
                    "Labels": labels,
                    "Image": OUTDATED_ASSISTANT_IMAGE,
                    "User": ASSISTANT_UID,
                    "Env": [],
                },
                "HostConfig": {
                    "ReadonlyRootfs": True,
                    "CapDrop": ["ALL"],
                    "SecurityOpt": ["no-new-privileges:true"],
                    "Privileged": False,
                    "NetworkMode": network_name,
                    "Memory": assistant_lifecycle.ASSISTANT_MEMORY,
                    "MemorySwap": assistant_lifecycle.ASSISTANT_MEMORY,
                    "NanoCpus": assistant_lifecycle.ASSISTANT_NANO_CPUS,
                    "CpusetCpus": controller.cpuset_cpus,
                    "PidsLimit": assistant_lifecycle.ASSISTANT_PIDS,
                    "IpcMode": "private",
                    "CgroupnsMode": "private",
                    "Tmpfs": None,
                    "AutoRemove": False,
                    "RestartPolicy": {"Name": "no"},
                    "LogConfig": {
                        "Type": "json-file",
                        "Config": {"max-file": "2", "max-size": "1m"},
                    },
                    "PortBindings": None,
                    "Binds": None,
                    "Devices": None,
                    "DeviceRequests": None,
                },
                "Mounts": [],
                "NetworkSettings": {"Networks": {network_name: {}}},
            },
        )
        container.reload = lambda: events.append("reload")
        container.remove = lambda *, force: events.append(("remove", force))
        controller._assistant_container = lambda *_args, **_kwargs: container
        controller.client = SimpleNamespace(containers=SimpleNamespace(list=lambda **_kwargs: [container]))
        return controller, container, events
