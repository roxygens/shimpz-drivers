from __future__ import annotations

import contextlib
import importlib.util
import sys
import tempfile
import types
import unittest
from http import HTTPStatus
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

spec = importlib.util.spec_from_file_location("team_app_r2_bridge_test", TEAM / "app.py")
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


class HostedCredentialLeaseTests(unittest.TestCase):
    def _journal_chat_environment(self, journal, runtime, rpc):
        contract = app.marketplace.APPS["shimpz-assistant"].assistant
        assert contract is not None
        assistant = app._ActiveAssistant(
            "shimpz-assistant",
            contract,
            types.SimpleNamespace(id="b" * 64),
        )
        anchor = types.SimpleNamespace(
            id=ANCHOR_ID,
            labels={"team.name": "Marketing", "team.owner": "account_1"},
        )
        config = types.SimpleNamespace(provider="openai", model="gpt-test")
        return anchor, _patched(
            _active_team_assistants=lambda _team_id: (assistant,),
            _chat_file_metadata=lambda _team_id, _files: [],
            _inference_store=types.SimpleNamespace(load=lambda _team_id: config),
            _model_credential=lambda _owner, _provider: ("secret-in-memory", 7),
            _require_model_credential_current=lambda *_args: None,
            _current_team_anchor=lambda *_args: anchor,
            _brain_runtime=runtime,
            _power_execution_journal=lambda: journal,
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

    def test_hosted_chat_scope_is_explicit_bounded_and_selects_only_requested_assistants(self) -> None:
        contract = types.SimpleNamespace(rules="Use declared Powers.", powers={})
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
        contract = types.SimpleNamespace(rules="Use only declared Powers.", powers={})
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
                _chat_file_metadata=lambda _team_id, _files: [],
                _inference_store=store,
                _model_credential=lambda _owner, _provider: ("secret-in-memory", 7),
                _require_model_credential_current=require_current,
                _brain_runtime=object(),
                _commit_chat_terminal=commit,
            ),
            mock.patch.object(
                app.chat_orchestrator,
                "run",
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
        place_contract = types.SimpleNamespace(rules="Resolve place names.", powers={"search": place_power})
        weather_contract = types.SimpleNamespace(rules="Read weather data.", powers={"current": weather_power})
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
                    _chat_file_metadata=lambda _team_id, _files: [],
                    _inference_store=store,
                    _model_credential=lambda _owner, _provider: ("secret-in-memory", 7),
                    _require_model_credential_current=lambda *_args: None,
                    _brain_runtime=object(),
                    _power_execution_journal=lambda: journal,
                    _invoke_assistant_power=invoke,
                    _commit_chat_terminal=lambda _team_id, _token: True,
                ),
                mock.patch.object(app.chat_orchestrator, "run", side_effect=run),
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
            "shimpz-assistant",
            "search-location",
            {"query": "Lisbon"},
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
        power_result = {
            "locations": [
                {
                    "name": "Lisbon",
                    "country": "Portugal",
                    "latitude": 38.72,
                    "longitude": -9.14,
                    "timezone": "Europe/Lisbon",
                }
            ]
        }
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
                        ("shimpz-assistant",),
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
                    ("shimpz-assistant",),
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
            "shimpz-assistant",
            "search-location",
            {"query": "Lisbon", "limit": 5},
            "none",
        )
        thread_id = app._brain_thread_id("team_1", ANCHOR_ID)

        class Runtime:
            @staticmethod
            def start(_context, _message):
                raw = app.brain_runtime_client.PowerRequest(
                    "power-1",
                    "shimpz-assistant",
                    "search-location",
                    {"query": "Lisbon"},
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
            operation = app._power_operation(normalized, "b" * 64)
            batch = journal.prepare_batch(ANCHOR_ID, thread_id, (operation,))
            journal.begin(batch, operation)
            anchor, environment = self._journal_chat_environment(journal, runtime, rpc)

            with environment, self.assertRaises(app.ApiError) as failed:
                app._chat_in_turn(
                    "team_1",
                    "Greet me",
                    [],
                    ("shimpz-assistant",),
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
        contract = types.SimpleNamespace(rules="Manage campaigns.", powers={})
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
                _chat_file_metadata=lambda _team_id, _files: [],
                _inference_store=store,
                _model_credential=lambda _owner, _provider: ("secret-in-memory", 7),
                _require_model_credential_current=lambda *_args: None,
                _brain_runtime=object(),
            ),
            mock.patch.object(
                app.chat_orchestrator,
                "run",
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


class R2BridgeTests(unittest.TestCase):
    def test_driver_operation_rechecks_owner_inside_lock_before_lazy_provision(self) -> None:
        events: list[str] = []
        lease = object()

        @contextlib.contextmanager
        def locked(_team_id):
            events.append("lock")
            yield

        def recheck(team_id, supplied):
            self.assertEqual((team_id, supplied), ("team_1", lease))
            events.append("recheck")
            return object()

        with (
            _patched(_lock_for=locked, _require_current_authorization=recheck),
            mock.patch.object(
                app.r2driver_client,
                "ensure_provisioned",
                side_effect=lambda _team_id: events.append("provision"),
            ),
        ):
            result = app._r2_driver_operation(
                "team_1",
                lease,
                lambda: events.append("operation") or {"ok": True},
            )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(events, ["lock", "recheck", "provision", "operation"])

    def test_stale_or_cross_tenant_lease_cannot_reach_r2_after_body_parse(self) -> None:
        harness = _RouteHarness(
            {
                "profile_id": "s3-access-key",
                "label": "tenant-a",
                "values": {"secret_access_key": "never-forwarded"},
                "expected_generation": 1,
            }
        )

        @contextlib.contextmanager
        def locked(_team_id):
            yield

        def reject(_team_id, _lease):
            raise app.ApiError(HTTPStatus.NOT_FOUND, "team not found")

        with (
            _patched(_lock_for=locked, _require_current_authorization=reject),
            mock.patch.object(app.r2driver_client, "ensure_provisioned") as provision,
            mock.patch.object(app.r2driver_client, "rotate_credential") as rotate,
            self.assertRaises(app.ApiError) as caught,
        ):
            app.Handler._route_driver(
                harness,
                "PUT",
                ["v1", "teams", "team_b", "drivers", "r2", "credentials", "primary"],
                "team_b",
                object(),
            )
        self.assertEqual(caught.exception.status, HTTPStatus.NOT_FOUND)
        self.assertEqual(harness.read_count, 1)
        provision.assert_not_called()
        rotate.assert_not_called()

    def test_route_allows_only_r2_and_forwards_closed_put(self) -> None:
        harness = _RouteHarness(
            {
                "profile_id": "s3-access-key",
                "label": "primary",
                "values": {"secret_access_key": "never-returned"},
                "expected_generation": 2,
            }
        )
        lease = object()
        calls: list[tuple] = []
        metadata = {
            "id": "primary",
            "profile_id": "s3-access-key",
            "label": "primary",
            "generation": 3,
            "status": "active",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:01Z",
        }

        def operation(team_id, supplied_lease, callback):
            calls.append((team_id, supplied_lease))
            return callback()

        with (
            _patched(_r2_driver_operation=operation),
            mock.patch.object(
                app.r2driver_client,
                "rotate_credential",
                side_effect=lambda *args: calls.append(args) or metadata,
            ),
        ):
            app.Handler._route_driver(
                harness,
                "PUT",
                ["v1", "teams", "team_1", "drivers", "r2", "credentials", "primary"],
                "team_1",
                lease,
            )
        self.assertEqual(harness.sent, [(HTTPStatus.OK, metadata)])
        self.assertEqual(calls[0], ("team_1", lease))
        self.assertEqual(calls[1][0:2], ("team_1", "primary"))
        self.assertNotIn("values", harness.sent[0][1])

        with self.assertRaises(app.ApiError) as caught:
            app.Handler._route_driver(
                harness,
                "GET",
                ["v1", "teams", "team_1", "drivers", "postgresql"],
                "team_1",
                lease,
            )
        self.assertEqual(caught.exception.status, HTTPStatus.NOT_FOUND)

    def test_create_provisions_r2_after_postgres_and_uses_common_rollback(self) -> None:
        events: list[str] = []

        @contextlib.contextmanager
        def reserve(*_args, **_kwargs):
            yield

        def pg_provision(_team_id):
            events.append("postgres-provision")
            return {"database_url": "postgres://scoped"}

        def r2_provision(_team_id):
            events.append("r2-provision")
            raise app.r2driver_client.R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver is unavailable", category="test")

        def rollback(*_args, **_kwargs):
            events.append("rollback")
            return app._CleanupResult(True, True)

        with (
            _patched(
                _cleanup_record=lambda _team_id: None,
                _get_container=lambda _name: None,
                _reserve_capacity=reserve,
                _require_team_runtime=lambda: None,
                _teardown=rollback,
            ),
            mock.patch.object(app.brain_credentials_client, "resolve", return_value=None),
            mock.patch.object(app.pgdriver_client, "provision_team", side_effect=pg_provision),
            mock.patch.object(app.r2driver_client, "provision_team", side_effect=r2_provision),
            self.assertRaises(app.ApiError) as caught,
        ):
            app._create("team_1", {}, "owner-1")
        self.assertEqual(caught.exception.status, HTTPStatus.BAD_GATEWAY)
        self.assertEqual(events, ["postgres-provision", "r2-provision", "rollback"])

    def test_retire_failure_stops_teardown_before_artifact_removal(self) -> None:
        events: list[str] = []
        record = types.SimpleNamespace(db_dropped=False)
        brain = object()

        def stop(_brain):
            events.append("brain-stop")
            return True

        def retire(_team_id):
            events.append("r2-retire")
            raise app.r2driver_client.R2DriverError(HTTPStatus.BAD_GATEWAY, "R2 Driver is unavailable", category="test")

        with (
            _patched(
                _owned_teardown_brain=lambda *_args: (True, brain),
                _stop_teardown_brain=stop,
            ),
            mock.patch.object(app.cleanup_state, "begin", return_value=record),
            mock.patch.object(app.cleanup_state, "finish") as finish,
            mock.patch.object(app.r2driver_client, "retire_team", side_effect=retire),
        ):
            result = app._teardown("team_1", owner="owner-1", brain_id="brain-id")
        self.assertFalse(result.complete)
        self.assertEqual(events, ["brain-stop", "r2-retire"])
        finish.assert_not_called()

    def test_finalization_orders_r2_before_postgres_and_cleanup_record(self) -> None:
        events: list[str] = []
        record = object()
        with (
            mock.patch.object(
                app.r2driver_client,
                "finalize_team_drop",
                side_effect=lambda _team_id: events.append("r2-finalize"),
            ),
            mock.patch.object(
                app.pgdriver_client,
                "finalize_team_drop",
                side_effect=lambda _team_id: events.append("postgres-finalize"),
            ),
            mock.patch.object(app.cleanup_state, "finish", side_effect=lambda _record: events.append("cleanup-finish")),
        ):
            self.assertTrue(app._finalize_teardown("team_1", record))
        self.assertEqual(events, ["r2-finalize", "postgres-finalize", "cleanup-finish"])


if __name__ == "__main__":
    unittest.main()
