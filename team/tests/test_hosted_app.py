from __future__ import annotations

import contextlib
import importlib.util
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from unittest import mock

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM))
_MODULES_BEFORE_APP_LOAD = dict(sys.modules)


class _DockerError(Exception):
    pass


class _NotFoundError(_DockerError):
    pass


class _APIError(_DockerError):
    pass


class _Passthru:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


class _LogConfig(_Passthru):
    types = types.SimpleNamespace(JSON="json-file")


class _EmptyCollection:
    @staticmethod
    def get(_identity):
        raise _NotFoundError

    @staticmethod
    def list(**_kwargs):
        return []


_engine = types.SimpleNamespace(
    containers=_EmptyCollection(),
    networks=_EmptyCollection(),
    volumes=_EmptyCollection(),
    images=_EmptyCollection(),
)
_docker_types = types.ModuleType("docker.types")
_docker_types.Mount = _Passthru
_docker_types.Ulimit = _Passthru
_docker_types.Healthcheck = _Passthru
_docker_types.LogConfig = _LogConfig
_docker_errors = types.ModuleType("docker.errors")
_docker_errors.DockerException = _DockerError
_docker_errors.NotFound = _NotFoundError
_docker_errors.APIError = _APIError
_docker_errors.ImageNotFound = _NotFoundError
_docker_socket = types.ModuleType("docker.utils.socket")
_docker_utils = types.ModuleType("docker.utils")
_docker_utils.socket = _docker_socket
_docker = types.ModuleType("docker")
_docker.from_env = lambda: _engine
_docker.types = _docker_types
_docker.errors = _docker_errors
_docker.utils = _docker_utils
sys.modules.update(
    {
        "docker": _docker,
        "docker.types": _docker_types,
        "docker.errors": _docker_errors,
        "docker.utils": _docker_utils,
        "docker.utils.socket": _docker_socket,
    }
)


def _stub(name: str, **members):
    module = types.ModuleType(name)
    for key, value in members.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _BrainCredentialError(Exception):
    pass


class _PgDriverError(Exception):
    pass


_stub("accounts_client", verify=lambda _token: None)
_stub("audit", log=lambda *_args, **_kwargs: "trace")
_stub(
    "brain_credentials_client",
    BrainCredentialError=_BrainCredentialError,
    resolve=lambda *_args: None,
    generation_is_current=lambda *_args: True,
)
_stub(
    "pgdriver_client",
    PgDriverError=_PgDriverError,
    provision_team=lambda _team_id: {"database_url": "postgres://scoped"},
    create_app_db=lambda *_args: {},
    drop_app_db=lambda *_args: {},
    drop_team=lambda *_args: {},
    finalize_team_drop=lambda *_args: {},
)
_stub("token_store", ensure_token=lambda: "operator-token")

spec = importlib.util.spec_from_file_location("team_app_hosted_test", TEAM / "app.py")
app = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = app
spec.loader.exec_module(app)

# The loaded app keeps direct references to its fakes. Restore the process import table so discovery
# order can never make unrelated tests import a partial Docker/client module.
for module_name, module in tuple(sys.modules.items()):
    source = getattr(module, "__file__", None)
    if source is None:
        continue
    try:
        belongs_to_team = Path(source).resolve().is_relative_to(TEAM)
    except OSError, RuntimeError, ValueError:
        belongs_to_team = False
    if belongs_to_team and module_name not in {__name__, spec.name}:
        previous = _MODULES_BEFORE_APP_LOAD.get(module_name)
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
for module_name in (
    "docker",
    "docker.types",
    "docker.errors",
    "docker.utils",
    "docker.utils.socket",
    "accounts_client",
    "audit",
    "brain_credentials_client",
    "pgdriver_client",
    "token_store",
):
    previous = _MODULES_BEFORE_APP_LOAD.get(module_name)
    if previous is None:
        sys.modules.pop(module_name, None)
    else:
        sys.modules[module_name] = previous

ANCHOR_ID = "a" * 64


@contextlib.contextmanager
def _patched(**replacements):
    originals = {name: getattr(app, name) for name in replacements}
    try:
        for name, replacement in replacements.items():
            setattr(app, name, replacement)
        yield
    finally:
        for name, original in originals.items():
            setattr(app, name, original)


class _RouteHarness:
    def __init__(self, body: dict | None = None) -> None:
        self.body = body
        self.read_count = 0
        self.sent: list[tuple[HTTPStatus, dict]] = []

    def _read_driver_body(self, keys: set[str]) -> dict:
        self.read_count += 1
        if self.body is None or set(self.body) != keys:
            raise AssertionError("unexpected body contract")
        return self.body

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        self.sent.append((status, payload))


class HostedAllowedHostsAdmissionTests(unittest.TestCase):
    @staticmethod
    def _container_with_environment(environment: dict[str, str]):
        return types.SimpleNamespace(
            attrs={"Config": {"Env": [f"{key}={value}" for key, value in environment.items()]}},
        )

    def test_manifest_must_match_reviewed_hosts_before_admission(self) -> None:
        spec = app.marketplace.APPS["shimpz-cloudflare"]
        container = types.SimpleNamespace(id="assistant-generation")
        reviewed_contracts: list[app.assistant_manifest.ManifestContract] = []

        def admit(_container, reviewed):
            reviewed_contracts.append(reviewed)
            return reviewed

        cache = types.SimpleNamespace(
            get=admit,
        )
        with _patched(
            _assistant_allowed_hosts_cache=cache,
            _require_assistant_genesis=lambda _container: "Use reviewed Powers.",
        ):
            self.assertEqual(app._admit_app_contract(spec, container), tuple(sorted(spec.allowed_hosts)))
        self.assertEqual(len(reviewed_contracts), 1)
        self.assertEqual(
            {secret.id for secret in reviewed_contracts[0].secrets},
            set(spec.assistant.secrets),
        )
        self.assertEqual(
            dict(reviewed_contracts[0].power_secrets),
            {power_id: tuple(sorted(power.secrets)) for power_id, power in spec.assistant.powers.items()},
        )
        self.assertEqual(
            {account.id: (account.provider, account.scopes) for account in reviewed_contracts[0].accounts},
            {
                account_id: (account.provider, tuple(sorted(account.scopes)))
                for account_id, account in spec.assistant.accounts.items()
            },
        )
        self.assertEqual(
            dict(reviewed_contracts[0].power_accounts),
            {power_id: tuple(sorted(power.accounts)) for power_id, power in spec.assistant.powers.items()},
        )

        exact = reviewed_contracts[0]
        account = exact.accounts[0]
        first_power, _first_refs = next(item for item in exact.power_accounts if item[1])
        drifted = (
            replace(exact, accounts=(replace(account, provider="other"),)),
            replace(exact, accounts=(replace(account, scopes=("tweet.read",)),)),
            replace(
                exact,
                power_accounts=tuple(
                    (power_id, ()) if power_id == first_power else (power_id, refs)
                    for power_id, refs in exact.power_accounts
                ),
            ),
        )
        with (
            _patched(_assistant_allowed_hosts_cache=app.assistant_manifest.ManifestContractCache()),
            mock.patch.object(app.assistant_manifest, "read_container_manifest_contract", return_value=exact),
        ):
            self.assertEqual(app._require_assistant_allowed_hosts(spec, container), exact.allowed_hosts)
        for declared in drifted:
            with (
                self.subTest(declared=declared),
                _patched(_assistant_allowed_hosts_cache=app.assistant_manifest.ManifestContractCache()),
                mock.patch.object(
                    app.assistant_manifest,
                    "read_container_manifest_contract",
                    return_value=declared,
                ),
                self.assertRaises(app.ApiError) as drift,
            ):
                app._require_assistant_allowed_hosts(spec, container)
            self.assertEqual(drift.exception.status, HTTPStatus.CONFLICT)

        def reject(_container, _reviewed):
            raise app.assistant_manifest.ManifestError("mismatch")

        with (
            _patched(_assistant_allowed_hosts_cache=types.SimpleNamespace(get=reject)),
            self.assertRaises(app.ApiError) as caught,
        ):
            app._admit_app_contract(spec, container)
        self.assertEqual(caught.exception.status, HTTPStatus.CONFLICT)

    def test_manifest_mismatch_rolls_back_before_policy_proxy_or_start(self) -> None:
        events: list[object] = []
        spec = app.marketplace.APPS["shimpz-cloudflare"]
        state = {"created": False}
        container = types.SimpleNamespace(
            id="assistant-generation",
            attrs={},
            labels={"team.app.db": "0"},
            reload=lambda: None,
        )
        network = types.SimpleNamespace(
            disconnect=lambda target: events.append(("disconnect", target.id)),
            connect=lambda target, *, aliases: events.append(("connect-app", target.id, tuple(aliases))),
        )

        def create(**_kwargs):
            state["created"] = True
            events.append("create")
            return container

        engine = types.SimpleNamespace(containers=types.SimpleNamespace(create=create))

        def reject(_spec, _container):
            events.append("admit")
            raise app.ApiError(HTTPStatus.CONFLICT, "allowed_hosts mismatch")

        with tempfile.TemporaryDirectory() as directory:
            with (
                _patched(
                    _lock_for=lambda _team_id: contextlib.nullcontext(),
                    _require_current_authorization=lambda *_args, **_kwargs: types.SimpleNamespace(
                        labels={"team.name": "Marketing"}
                    ),
                    _prepare_marketplace_image=lambda _spec: None,
                    _get_container=lambda _name: container if state["created"] else None,
                    _team_app_containers=lambda _team_id: [],
                    _reserve_capacity=lambda *_args, **_kwargs: contextlib.nullcontext(),
                    _require_team_runtime=lambda: None,
                    _ensure_team_network=lambda _team_id: network,
                    _docker=engine,
                    _admit_app_contract=reject,
                    _write_egress_policy=lambda *_args: events.append("write-policy"),
                    _safe_connect=lambda *_args, **_kwargs: events.append("connect-proxy"),
                    _start_team_with_isolation=lambda _container: events.append("start"),
                    _remove_team_container=lambda target: events.append(("remove-container", target.id)) or True,
                    APP_EGRESS_POLICY_DIR=Path(directory),
                ),
                mock.patch.object(app.manifests, "build_team_app_kwargs", return_value={}),
                mock.patch.object(app.network_policy, "app_identity_valid", return_value=True),
                self.assertRaises(app.ApiError) as caught,
            ):
                app._install_app(
                    "team_1",
                    "shimpz-cloudflare",
                    spec,
                    "account_1",
                    types.SimpleNamespace(owner="account_1"),
                )
            self.assertEqual(list(Path(directory).rglob("*")), [Path(directory) / ".tokens"])

        self.assertEqual(caught.exception.status, HTTPStatus.CONFLICT)
        self.assertEqual(
            events,
            [
                "create",
                ("disconnect", "assistant-generation"),
                ("connect-app", "assistant-generation", ("shimpz-cloudflare", "shimpz-cloudflare.team")),
                "admit",
                ("remove-container", "assistant-generation"),
            ],
        )

    def test_existing_policy_bytes_must_match_the_admitted_hosts(self) -> None:
        hosts = ("api.open-meteo.com", "geocoding-api.open-meteo.com")
        with tempfile.TemporaryDirectory() as directory, _patched(APP_EGRESS_POLICY_DIR=Path(directory)):
            token = app._app_egress_token("team_1", "shimpz-cloudflare")
            assert token is not None
            app._write_egress_policy(token, hosts)
            self.assertEqual(
                app._validate_egress_policy("team_1", "shimpz-cloudflare", hosts),
                token,
            )

            (Path(directory) / f"{token}.json").write_text('["evil.example"]', encoding="ascii")
            with self.assertRaises(app.ApiError) as caught:
                app._validate_egress_policy("team_1", "shimpz-cloudflare", hosts)
        self.assertEqual(caught.exception.status, HTTPStatus.CONFLICT)

    def test_nonempty_hosts_require_the_exact_admitted_proxy_token(self) -> None:
        token = "a" * 32
        hosts = ("api.open-meteo.com",)
        expected = app._egress_proxy_environment(token)
        app._validate_assistant_proxy_environment(self._container_with_environment(expected), token, hosts)

        drifted_environments = {
            "wrong-token": {**expected, "HTTPS_PROXY": expected["HTTPS_PROXY"].replace(token, "b" * 32)},
            "missing-lowercase": {key: value for key, value in expected.items() if key != "https_proxy"},
            "http-proxy": {**expected, "HTTP_PROXY": "http://app-egress-proxy:8889"},
            "all-proxy": {**expected, "all_proxy": "http://app-egress-proxy:8889"},
        }
        for name, environment in drifted_environments.items():
            with self.subTest(name=name), self.assertRaises(app.ApiError) as caught:
                app._validate_assistant_proxy_environment(
                    self._container_with_environment(environment),
                    token,
                    hosts,
                )
            self.assertEqual(caught.exception.status, HTTPStatus.CONFLICT)

    def test_empty_hosts_forbid_every_proxy_environment_variable(self) -> None:
        app._validate_assistant_proxy_environment(
            self._container_with_environment({"SHIMPZ_TEAM_ID": "team_1"}),
            None,
            (),
        )

        for key in ("HTTPS_PROXY", "http_proxy", "ALL_PROXY", "no_proxy", "FTP_PROXY", "custom_proxy"):
            with self.subTest(key=key), self.assertRaises(app.ApiError) as caught:
                app._validate_assistant_proxy_environment(
                    self._container_with_environment({key: "unexpected"}),
                    None,
                    (),
                )
            self.assertEqual(caught.exception.status, HTTPStatus.CONFLICT)

    def test_empty_hosts_build_no_proxy_environment(self) -> None:
        spec = app.marketplace.APPS["shimpz-cloudflare"]
        kwargs = app.manifests.build_team_app_kwargs("team_1", "shimpz-cloudflare", spec)
        environment = kwargs["environment"]

        self.assertFalse({key for key in environment if key.upper().endswith("_PROXY")})


class HostedCredentialLeaseTests(unittest.TestCase):
    def setUp(self) -> None:
        """Keep pending private-input state isolated from every hosted test."""
        original_accounts = app._assistant_account_challenges
        original_secrets = app._assistant_secret_challenges
        app._assistant_account_challenges = app.assistant_account_challenges.AccountChallengeStore()
        app._assistant_secret_challenges = app.assistant_secret_challenges.SecretChallengeStore()
        self.addCleanup(setattr, app, "_assistant_account_challenges", original_accounts)
        self.addCleanup(setattr, app, "_assistant_secret_challenges", original_secrets)

    def _journal_chat_environment(self, journal, runtime, rpc):
        contract = app.marketplace.APPS["shimpz-cloudflare"].assistant
        assert contract is not None
        assistant = app._ActiveAssistant(
            "shimpz-cloudflare",
            contract,
            types.SimpleNamespace(id="b" * 64),
        )
        anchor = types.SimpleNamespace(
            id=ANCHOR_ID,
            labels={"team.name": "Marketing", "team.owner": "account_1"},
        )
        config = types.SimpleNamespace(provider="openai", model="gpt-test")
        secret_store = types.SimpleNamespace(
            metadata=lambda _team_id, _assistant_id, secret_ids: tuple(
                types.SimpleNamespace(id=secret_id, configured=True, generation=1) for secret_id in secret_ids
            ),
            resolve_many=lambda _team_id, _assistant_id, secret_ids: dict.fromkeys(
                secret_ids,
                "configured-test-secret",
            ),
        )
        account_store = app.oauth_account_store.OAuthAccountStore(
            journal.path.parent / "oauth-state" / "accounts.json",
            journal.path.parent / "oauth-key" / "aes256.key",
        )
        for account_id, declaration in contract.accounts.items():
            account_store.put(
                "team_1",
                "shimpz-cloudflare",
                account_id,
                declaration.provider,
                declaration.scopes,
                app.oauth_http_client.OAuthTokenSet(
                    f"synthetic-hosted-access-token-{account_id}",
                    f"synthetic-hosted-refresh-token-{account_id}",
                    declaration.scopes,
                    3600,
                ),
            )
        return anchor, _patched(
            _active_team_assistants=lambda _team_id: (assistant,),
            _require_assistant_genesis=lambda _container: "Use only the declared Cloudflare Powers.",
            _chat_file_metadata=lambda _team_id, _files: [],
            _inference_store=types.SimpleNamespace(load=lambda _team_id: config),
            _model_credential=lambda _owner, _provider: ("secret-in-memory", 7),
            _require_model_credential_current=lambda *_args: None,
            _current_team_anchor=lambda *_args: anchor,
            _brain_runtime=runtime,
            _power_execution_journal=lambda: journal,
            _assistant_secrets=secret_store,
            _assistant_accounts=account_store,
            _invoke_assistant_power=rpc,
            _commit_chat_terminal=lambda _team_id, _token: True,
        )

    def test_hosted_thread_identity_is_generation_scoped_and_closed(self) -> None:
        first = app._brain_thread_id("team_1", ANCHOR_ID)
        second = app._brain_thread_id("team_1", "b" * 64)

        self.assertEqual(first, f"hosted:team_1:{ANCHOR_ID}:default")
        self.assertNotEqual(first, second)
        for team_id, anchor_id in (("bad team", ANCHOR_ID), ("team_1", "not-a-container")):
            with self.subTest(team_id=team_id, anchor_id=anchor_id), self.assertRaises(app.ApiError) as caught:
                app._brain_thread_id(team_id, anchor_id)
            self.assertEqual(caught.exception.status, HTTPStatus.CONFLICT)

    def test_team_name_contract_rejects_padding_controls_and_oversize_values(self) -> None:
        self.assertEqual(app._validated_team_name("Marketing"), "Marketing")
        for invalid in ("", " Marketing", "Marketing ", "Marketing\n", "x" * 81, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                app._validated_team_name(invalid)

    def test_hosted_lifecycle_rejects_an_active_chat_before_any_mutation(self) -> None:
        spec = app.marketplace.APPS["shimpz-cloudflare"]
        lease = types.SimpleNamespace(owner="account_1")
        operations = (
            lambda: app._install_app("team_1", "shimpz-cloudflare", spec, "account_1", lease),
            lambda: app._uninstall_app("team_1", "shimpz-cloudflare", lease),
            lambda: app._lifecycle("team_1", "restart", lease),
        )
        chat_lock = app._chat_lock_for("team_1")
        self.assertTrue(chat_lock.acquire(blocking=False))
        try:
            with _patched(_lock_for=lambda _team_id: self.fail("lifecycle mutation acquired its inner lock")):
                for operation in operations:
                    with self.subTest(operation=operation), self.assertRaises(app.ApiError) as caught:
                        operation()
                    self.assertEqual(caught.exception.status, HTTPStatus.CONFLICT)
        finally:
            chat_lock.release()

    def test_hosted_stream_emits_the_exact_v2_done_shape(self) -> None:
        class StreamHarness:
            def __init__(self) -> None:
                self.status = None
                self.headers: list[tuple[str, str]] = []
                self.wfile = BytesIO()

            def send_response(self, status) -> None:
                self.status = status

            def send_header(self, name: str, value: str) -> None:
                self.headers.append((name, value))

            def end_headers(self) -> None:
                pass

        @contextlib.contextmanager
        def exclusive_turn(_team_id, _lease):
            yield "turn-token", types.SimpleNamespace(id=ANCHOR_ID)

        stream = StreamHarness()
        with _patched(
            _exclusive_chat_turn=exclusive_turn,
            _chat_in_turn=lambda *_args: {
                "team_id": "team_1",
                "team_name": "Marketing",
                "reply": "Campaign ready.",
            },
        ):
            app.Handler._stream_chat(
                stream,
                "team_1",
                "Prepare the campaign",
                [],
                ("shimpz-cloudflare",),
                types.SimpleNamespace(owner="account_1"),
            )

        size_line, chunked = stream.wfile.getvalue().split(b"\r\n", 1)
        size = int(size_line, 16)
        encoded_event = chunked[:size]
        self.assertEqual(stream.status, HTTPStatus.OK)
        self.assertIn(("Content-Type", "application/x-ndjson"), stream.headers)
        self.assertIn(("Cache-Control", "no-store"), stream.headers)
        self.assertEqual(chunked[size:], b"\r\n0\r\n\r\n")
        self.assertEqual(
            app.json.loads(encoded_event),
            {
                "type": "done",
                "team_id": "team_1",
                "team_name": "Marketing",
                "reply": "Campaign ready.",
            },
        )

    def test_hosted_chat_scope_is_explicit_bounded_and_selects_only_requested_assistants(self) -> None:
        contract = types.SimpleNamespace(powers={})
        places = app._ActiveAssistant("places", contract, types.SimpleNamespace(id="places-container"))
        weather = app._ActiveAssistant("weather", contract, types.SimpleNamespace(id="weather-container"))

        self.assertEqual(app._chat_assistant_ids([]), ())
        self.assertEqual(app._chat_assistant_ids(["weather", "places"]), ("places", "weather"))
        self.assertEqual(
            app._select_team_assistants((places, weather), ("weather",)),
            (weather,),
        )

        for invalid in (
            ["weather", "weather"],
            ["bad_assistant"],
            [f"helper-{index}" for index in range(app.MAX_CHAT_ASSISTANTS + 1)],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(app.ApiError) as caught:
                app._chat_assistant_ids(invalid)
            self.assertEqual(caught.exception.status, HTTPStatus.UNPROCESSABLE_ENTITY)

        with self.assertRaises(app.ApiError) as unavailable:
            app._select_team_assistants((places,), ("weather",))
        self.assertEqual(unavailable.exception.status, HTTPStatus.CONFLICT)
        self.assertEqual(unavailable.exception.message, "a selected Assistant is unavailable")

    def test_hosted_empty_scope_reaches_the_brain_without_assistant_tools(self) -> None:
        class Runtime:
            context = None

            def start(self, context, _message):
                self.context = context
                return app.brain_runtime_client.RuntimeTurn("completed", "Brain only.", ())

            def resume(self, _context, _results):
                raise AssertionError("a Brain-only reply must not resume")

        runtime = Runtime()
        with tempfile.TemporaryDirectory() as directory:
            journal = app.power_journal.PowerJournal(Path(directory) / "journal.sqlite3")
            self.addCleanup(journal.close)
            anchor, environment = self._journal_chat_environment(journal, runtime, mock.Mock())
            with environment:
                result = app._chat_in_turn(
                    "team_1",
                    "Hello",
                    [],
                    (),
                    "turn-token",
                    anchor,
                    "account_1",
                )

        self.assertEqual(runtime.context.assistants, ())
        self.assertEqual(result["reply"], "Brain only.")

    def test_revoked_generation_during_turn_cannot_commit_reply(self) -> None:
        checks: list[tuple[str, str, int]] = []
        commit = mock.Mock(return_value=True)
        contract = types.SimpleNamespace(powers={})
        assistant_container = types.SimpleNamespace(id="assistant-container")
        anchor = types.SimpleNamespace(
            id=ANCHOR_ID,
            labels={"team.name": "Marketing", "team.owner": "account_1"},
        )
        store = types.SimpleNamespace(load=lambda _team_id: types.SimpleNamespace(provider="openai", model="gpt-5.5"))

        def require_current(owner: str, provider: str, generation: int) -> None:
            checks.append((owner, provider, generation))
            if len(checks) == 2:
                raise app.ApiError(HTTPStatus.CONFLICT, "model credential changed or was revoked; retry")

        with (
            _patched(
                _active_team_assistants=lambda _team_id: (
                    app._ActiveAssistant(
                        "hello-pulse",
                        contract,
                        assistant_container,
                    ),
                ),
                _require_assistant_genesis=lambda _container: "Use only declared Powers.",
                _chat_file_metadata=lambda _team_id, _files: [],
                _inference_store=store,
                _model_credential=lambda _owner, _provider: ("secret-in-memory", 7),
                _require_model_credential_current=require_current,
                _brain_runtime=object(),
                _commit_chat_terminal=commit,
            ),
            mock.patch.object(
                app.chat_orchestrator,
                "run_until_pause",
                return_value=app.chat_orchestrator.ChatOutcome(reply="late reply", powers=()),
            ),
            self.assertRaises(app.ApiError) as caught,
        ):
            app._chat_in_turn(
                "team_1",
                "hello",
                [],
                ("hello-pulse",),
                "turn-token",
                anchor,
                "account_1",
            )

        self.assertEqual(caught.exception.status, HTTPStatus.CONFLICT)
        self.assertEqual(checks, [("account_1", "openai", 7), ("account_1", "openai", 7)])
        commit.assert_not_called()

    def test_hosted_team_context_contains_and_routes_two_active_assistants(self) -> None:
        place_power = types.SimpleNamespace(summary="Find a place.", input_schema={"type": "object"}, approval="none")
        weather_power = types.SimpleNamespace(
            summary="Read current weather.",
            input_schema={"type": "object"},
            approval="none",
        )
        place_contract = types.SimpleNamespace(powers={"search": place_power})
        weather_contract = types.SimpleNamespace(powers={"current": weather_power})
        place_container = types.SimpleNamespace(id="places-container")
        weather_container = types.SimpleNamespace(id="weather-container")
        anchor = types.SimpleNamespace(
            id=ANCHOR_ID,
            labels={"team.name": "Marketing", "team.owner": "account_1"},
        )
        store = types.SimpleNamespace(load=lambda _team_id: types.SimpleNamespace(provider="openai", model="gpt-test"))
        invoked: list[tuple[str, str, object]] = []

        def run(_runtime, context, _prompt, validate_power, invoke_power, **hooks):
            self.assertEqual([assistant.id for assistant in context.assistants], ["places", "weather"])
            self.assertEqual(
                [assistant.genesis for assistant in context.assistants],
                ["Compose Powers for places-container.", "Compose Powers for weather-container."],
            )
            self.assertEqual(context.thread_id, app._brain_thread_id("team_1", ANCHOR_ID))
            self.assertTrue(callable(validate_power))
            requests = (
                app.brain_runtime_client.PowerRequest("place-1", "places", "search", {"name": "Berlin"}, "none"),
                app.brain_runtime_client.PowerRequest(
                    "weather-1",
                    "weather",
                    "current",
                    {"latitude": 52.52, "longitude": 13.41},
                    "none",
                ),
            )
            hooks["prepare_batch"](requests)
            for request in requests:
                invoke_power(request)
            hooks["batch_delivered"](requests)
            return app.chat_orchestrator.ChatOutcome(
                reply="Berlin weather is ready.",
                powers=(
                    app.chat_orchestrator.InvokedPower("places", "search"),
                    app.chat_orchestrator.InvokedPower("weather", "current"),
                ),
            )

        def invoke(_team_id, _token, assistant_id, _contract, _container, power, payload):
            invoked.append((assistant_id, power, payload))
            return {"result": {"ok": True}}

        with tempfile.TemporaryDirectory() as directory:
            journal = app.power_journal.PowerJournal(Path(directory) / "journal.sqlite3")
            self.addCleanup(journal.close)
            with (
                _patched(
                    _active_team_assistants=lambda _team_id: (
                        app._ActiveAssistant("places", place_contract, place_container),
                        app._ActiveAssistant("weather", weather_contract, weather_container),
                    ),
                    _require_assistant_genesis=lambda container: f"Compose Powers for {container.id}.",
                    _chat_file_metadata=lambda _team_id, _files: [],
                    _inference_store=store,
                    _model_credential=lambda _owner, _provider: ("secret-in-memory", 7),
                    _require_model_credential_current=lambda *_args: None,
                    _brain_runtime=object(),
                    _power_execution_journal=lambda: journal,
                    _invoke_assistant_power=invoke,
                    _commit_chat_terminal=lambda _team_id, _token: True,
                ),
                mock.patch.object(app.chat_orchestrator, "run_until_pause", side_effect=run),
            ):
                result = app._chat_in_turn(
                    "team_1",
                    "Find Berlin weather",
                    [],
                    ("places", "weather"),
                    "turn-token",
                    anchor,
                    "account_1",
                )

        self.assertEqual([item[:2] for item in invoked], [("places", "search"), ("weather", "current")])
        self.assertEqual(result, {"team_id": "team_1", "team_name": "Marketing", "reply": "Berlin weather is ready."})

    def test_completed_power_is_cached_until_a_successful_brain_resume(self) -> None:
        request = app.brain_runtime_client.PowerRequest(
            "power-1",
            "shimpz-cloudflare",
            "list-zones",
            {"page": 1, "per_page": 25},
            "none",
        )

        class Runtime:
            def __init__(self) -> None:
                self.resume_calls = 0
                self.results: list[dict[str, object]] = []

            def start(self, _context, _message):
                return app.brain_runtime_client.RuntimeTurn("power-required", "", (request,))

            def resume(self, _context, results):
                self.resume_calls += 1
                self.results.append(results)
                if self.resume_calls == 1:
                    raise app.brain_runtime_client.BrainRuntimeError("private-provider-response")
                return app.brain_runtime_client.RuntimeTurn("completed", "Cached reply", ())

        runtime = Runtime()
        power_result = {"zones": [], "page": 1, "per_page": 25, "total_pages": 0}
        rpc = mock.Mock(return_value={"result": power_result})
        with tempfile.TemporaryDirectory() as directory:
            journal = app.power_journal.PowerJournal(Path(directory) / "journal.sqlite3")
            self.addCleanup(journal.close)
            anchor, environment = self._journal_chat_environment(journal, runtime, rpc)
            with mock.patch.object(journal, "delivered", wraps=journal.delivered) as delivered, environment:
                with self.assertRaises(app.ApiError) as failed:
                    app._chat_in_turn(
                        "team_1",
                        "Greet me",
                        [],
                        ("shimpz-cloudflare",),
                        "first-turn",
                        anchor,
                        "account_1",
                    )
                self.assertEqual(failed.exception.status, HTTPStatus.BAD_GATEWAY)
                self.assertNotIn("private-provider-response", str(failed.exception))
                delivered.assert_not_called()

                result = app._chat_in_turn(
                    "team_1",
                    "Greet me",
                    [],
                    ("shimpz-cloudflare",),
                    "retry-turn",
                    anchor,
                    "account_1",
                )

        self.assertEqual(rpc.call_count, 1)
        self.assertEqual(
            runtime.results,
            [
                {"power-1": power_result},
                {"power-1": power_result},
            ],
        )
        delivered.assert_called_once()
        self.assertEqual(result["reply"], "Cached reply")

    def test_uncertain_power_fails_closed_before_a_second_rpc(self) -> None:
        normalized = app.brain_runtime_client.PowerRequest(
            "power-1",
            "shimpz-cloudflare",
            "list-zones",
            {"page": 1, "per_page": 25},
            "none",
        )
        thread_id = app._brain_thread_id("team_1", ANCHOR_ID)

        class Runtime:
            @staticmethod
            def start(_context, _message):
                raw = app.brain_runtime_client.PowerRequest(
                    "power-1",
                    "shimpz-cloudflare",
                    "list-zones",
                    {"page": 1, "per_page": 25},
                    "none",
                )
                return app.brain_runtime_client.RuntimeTurn("power-required", "", (raw,))

            @staticmethod
            def resume(_context, _results):
                raise AssertionError("an uncertain Power must not reach Brain resume")

        runtime = Runtime()
        rpc = mock.Mock(side_effect=AssertionError("an uncertain Power must not execute"))
        with tempfile.TemporaryDirectory() as directory:
            journal = app.power_journal.PowerJournal(Path(directory) / "journal.sqlite3")
            self.addCleanup(journal.close)
            operation = app._power_operation(
                normalized,
                "b" * 64,
                account_generations=(("cloudflare", 1),),
            )
            batch = journal.prepare_batch(ANCHOR_ID, thread_id, (operation,))
            journal.begin(batch, operation)
            anchor, environment = self._journal_chat_environment(journal, runtime, rpc)

            with environment, self.assertRaises(app.ApiError) as failed:
                app._chat_in_turn(
                    "team_1",
                    "Greet me",
                    [],
                    ("shimpz-cloudflare",),
                    "retry-turn",
                    anchor,
                    "account_1",
                )

        self.assertEqual(failed.exception.status, HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertEqual(failed.exception.message, "Team Power execution state is unavailable")
        self.assertNotIn("uncertain", str(failed.exception).lower())
        rpc.assert_not_called()

    def test_power_journal_uses_the_injected_path_lazily(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private" / "journal.sqlite3"
            with _patched(POWER_JOURNAL_PATH=path, _power_journal_instance=None):
                self.assertFalse(path.exists())
                journal = app._power_execution_journal()
                self.addCleanup(journal.close)
                self.assertTrue(path.exists())
                self.assertIs(app._power_execution_journal(), journal)

    def test_hosted_approval_error_does_not_expose_the_power_id(self) -> None:
        private_power_id = "private-campaign-export"
        request = app.brain_runtime_client.PowerRequest(
            interrupt_id="approval-1",
            assistant_id="salesnator",
            power=private_power_id,
            input={},
            approval="each-run",
        )
        contract = types.SimpleNamespace(powers={})
        anchor = types.SimpleNamespace(
            id=ANCHOR_ID,
            labels={"team.name": "Marketing", "team.owner": "account_1"},
        )
        store = types.SimpleNamespace(load=lambda _team_id: types.SimpleNamespace(provider="openai", model="gpt-test"))

        with (
            _patched(
                _active_team_assistants=lambda _team_id: (
                    app._ActiveAssistant("salesnator", contract, types.SimpleNamespace(id="assistant-container")),
                ),
                _require_assistant_genesis=lambda _container: "Use only campaign Powers.",
                _chat_file_metadata=lambda _team_id, _files: [],
                _inference_store=store,
                _model_credential=lambda _owner, _provider: ("secret-in-memory", 7),
                _require_model_credential_current=lambda *_args: None,
                _brain_runtime=object(),
            ),
            mock.patch.object(
                app.chat_orchestrator,
                "run_until_pause",
                side_effect=app.chat_orchestrator.ApprovalRequiredError(request),
            ),
            self.assertRaises(app.ApiError) as caught,
        ):
            app._chat_in_turn(
                "team_1",
                "Export the campaign",
                [],
                ("salesnator",),
                "turn-token",
                anchor,
                "account_1",
            )

        self.assertEqual(caught.exception.status, HTTPStatus.CONFLICT)
        self.assertEqual(caught.exception.message, "Assistant Power requires Captain approval")
        self.assertNotIn(private_power_id, caught.exception.message)

    def test_destroy_deletes_generation_after_chat_drain_before_teardown(self) -> None:
        events: list[object] = []
        expected_thread = app._brain_thread_id("team_1", ANCHOR_ID)
        lease = app._AuthorizationLease(
            team_id="team_1",
            container_id=ANCHOR_ID,
            owner="account_1",
            principal=("account", "account_1"),
            cleanup_nonce="retry-nonce",
        )

        class ChatLock:
            def acquire(self, *, timeout: int) -> bool:
                self.assert_timeout = timeout
                events.append("chat-drained")
                return True

            def release(self) -> None:
                events.append("chat-released")

        chat_lock = ChatLock()

        def delete_thread(thread_id: str) -> None:
            events.append(("thread-deleted", thread_id))

        def teardown(team_id: str, *, owner: str, brain_id: str):
            events.append(("teardown", team_id, owner, brain_id))
            return app._CleanupResult(True, True)

        journal = types.SimpleNamespace(purge=lambda generation: events.append(("journal-purged", generation)))

        with _patched(
            _lock_for=lambda _team_id: contextlib.nullcontext(),
            _require_cleanup_authorization=lambda _team_id, _lease: events.append("authorized"),
            _chat_lock_for=lambda _team_id: chat_lock,
            _brain_runtime=types.SimpleNamespace(delete_thread=delete_thread),
            _power_execution_journal=lambda: journal,
            _teardown=teardown,
            _clear_team_id_runtime_state=lambda _team_id: events.append("runtime-cleared"),
        ):
            result = app._destroy("team_1", lease)

        self.assertEqual(
            events,
            [
                "authorized",
                "chat-drained",
                ("thread-deleted", expected_thread),
                ("journal-purged", ANCHOR_ID),
                ("teardown", "team_1", "account_1", ANCHOR_ID),
                "runtime-cleared",
                "chat-released",
            ],
        )
        self.assertEqual(result, {"team_id": "team_1", "destroyed": True, "db_dropped": True})

    def test_destroy_retries_thread_delete_without_teardown_after_redacted_failure(self) -> None:
        delete_calls: list[str] = []
        teardown = mock.Mock(return_value=app._CleanupResult(True, True))
        clear = mock.Mock()
        lease = app._AuthorizationLease(
            team_id="team_1",
            container_id=ANCHOR_ID,
            owner="account_1",
            principal=("account", "account_1"),
            cleanup_nonce="retry-nonce",
        )

        class ChatLock:
            @staticmethod
            def acquire(*, timeout: int) -> bool:
                return timeout == 30

            @staticmethod
            def release() -> None:
                return None

        def delete_thread(thread_id: str) -> None:
            delete_calls.append(thread_id)
            if len(delete_calls) == 1:
                raise app.brain_runtime_client.BrainRuntimeError("persisted-private-data")

        purge_calls: list[str] = []
        journal = types.SimpleNamespace(purge=lambda generation: purge_calls.append(generation))

        with _patched(
            _lock_for=lambda _team_id: contextlib.nullcontext(),
            _require_cleanup_authorization=lambda _team_id, _lease: object(),
            _chat_lock_for=lambda _team_id: ChatLock(),
            _brain_runtime=types.SimpleNamespace(delete_thread=delete_thread),
            _power_execution_journal=lambda: journal,
            _teardown=teardown,
            _clear_team_id_runtime_state=clear,
        ):
            with self.assertRaises(app.ApiError) as caught:
                app._destroy("team_1", lease)
            self.assertEqual(caught.exception.status, HTTPStatus.SERVICE_UNAVAILABLE)
            self.assertEqual(caught.exception.message, "Team conversation state could not be deleted")
            self.assertNotIn("persisted-private-data", str(caught.exception))
            teardown.assert_not_called()
            clear.assert_not_called()

            result = app._destroy("team_1", lease)

        expected_thread = app._brain_thread_id("team_1", ANCHOR_ID)
        self.assertEqual(delete_calls, [expected_thread, expected_thread])
        self.assertEqual(purge_calls, [ANCHOR_ID])
        teardown.assert_called_once_with("team_1", owner="account_1", brain_id=ANCHOR_ID)
        clear.assert_called_once_with("team_1")
        self.assertTrue(result["destroyed"])

    def test_destroy_journal_failure_is_redacted_before_teardown(self) -> None:
        teardown = mock.Mock(return_value=app._CleanupResult(True, True))
        clear = mock.Mock()
        lease = app._AuthorizationLease(
            team_id="team_1",
            container_id=ANCHOR_ID,
            owner="account_1",
            principal=("account", "account_1"),
            cleanup_nonce="retry-nonce",
        )

        class ChatLock:
            released = False

            @staticmethod
            def acquire(*, timeout: int) -> bool:
                return timeout == 30

            @classmethod
            def release(cls) -> None:
                cls.released = True

        def fail_purge(_generation: str) -> None:
            raise app.power_journal.PowerJournalError("private-journal-state")

        with (
            _patched(
                _lock_for=lambda _team_id: contextlib.nullcontext(),
                _require_cleanup_authorization=lambda _team_id, _lease: object(),
                _chat_lock_for=lambda _team_id: ChatLock(),
                _brain_runtime=types.SimpleNamespace(delete_thread=lambda _thread: None),
                _power_execution_journal=lambda: types.SimpleNamespace(purge=fail_purge),
                _teardown=teardown,
                _clear_team_id_runtime_state=clear,
            ),
            self.assertRaises(app.ApiError) as failed,
        ):
            app._destroy("team_1", lease)

        self.assertEqual(failed.exception.status, HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertEqual(failed.exception.message, "Team Power execution state could not be deleted")
        self.assertNotIn("private-journal-state", str(failed.exception))
        teardown.assert_not_called()
        clear.assert_not_called()
        self.assertTrue(ChatLock.released)


if __name__ == "__main__":
    unittest.main()
