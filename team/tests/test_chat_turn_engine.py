"""Characterize the shared hosted/local chat-segment decision engine."""

from __future__ import annotations

import contextlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

TEAM = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TEAM))
sys.path.insert(0, str(TESTS))

import brain_runtime_client
import chat_orchestrator
import chat_turn_engine
import hosted_app_fixture as hosted_harness
import inference_config
import local_app
import local_registry
import power_execution
from local_support import chat_segment as local_chat_segment
from local_support.chat_segment import SegmentRequest
from local_support.chat_types import ActiveAssistant

hosted_app = hosted_harness.app


def _context() -> brain_runtime_client.RuntimeContext:
    return brain_runtime_client.RuntimeContext(
        thread_id="thread-1",
        team_name="Team",
        assistants=(
            brain_runtime_client.RuntimeAssistant(
                id="assistant",
                genesis="Use the declared Power.",
                powers=(
                    brain_runtime_client.RuntimePower(
                        id="lookup",
                        summary="Look up one value.",
                        input_schema={"type": "object"},
                    ),
                ),
            ),
        ),
        provider="openai",
        model="gpt-test",
        api_key="test-key",
    )


class _Runtime:
    @staticmethod
    def start(_context, _message):
        return brain_runtime_client.RuntimeTurn(
            status="power-required",
            reply="",
            powers=(
                brain_runtime_client.PowerRequest(
                    interrupt_id="interrupt-1",
                    assistant_id="assistant",
                    power="lookup",
                    input={"query": "Ada"},
                ),
            ),
        )


class _Batch:
    @staticmethod
    def prepare(_requests) -> None:
        raise AssertionError("a suspended batch must not be prepared")

    @staticmethod
    def invoke(_request):
        raise AssertionError("a suspended Power must not be invoked")

    @staticmethod
    def delivered(_requests) -> None:
        raise AssertionError("a suspended batch must not be delivered")


class _InteractionBatch:
    @staticmethod
    def prepare(_requests) -> None:
        return None

    @staticmethod
    def invoke(_request):
        return power_execution.RpcSuspension({"ordinal": 0, "kind": "request", "request_type": "str"})

    @staticmethod
    def delivered(_requests) -> None:
        raise AssertionError("an interactive batch must not be delivered")


def _local_controller(local_active, config, events: list[str], fail):
    controller = object.__new__(local_app.LocalController)
    controller.space_id = "local-space"
    controller.brain_runtime = SimpleNamespace()
    controller.power_state = SimpleNamespace()
    controller._lock = lambda _team_id: contextlib.nullcontext()
    controller._network = lambda _team_id: SimpleNamespace(id="a" * 64, name="team-network")
    controller._validate_network = lambda _network, _team_id, **_kwargs: "Team"
    controller._active_chat_assistants = lambda _team_id, _network_name: (local_active,)
    controller.storage = SimpleNamespace(
        metadata=lambda _team_id, _files, _connection=None: [],
        metadata_connection=lambda _team_id, _files: contextlib.nullcontext(None),
    )
    controller.inference_store = SimpleNamespace(load=lambda _team_id: config)
    controller._active_assistant_genesis = lambda _active: "Use the declared Power."

    def local_private_inputs(_team_id, _bindings, _requests, requirements) -> bool:
        requirements.accounts = ("account-required",)
        return True

    controller._require_chat_private_inputs = local_private_inputs
    controller._require_power_rpc_envelope = lambda *_args: events.append("preflight")
    controller._power_secret_generations = lambda *_args: events.append("secrets") or ()
    controller._power_account_generations = lambda *_args: events.append("accounts") or ()
    controller._chat_cancelled = lambda _token: False
    controller._validate_chat_context = lambda *_args: None
    controller._raise_chat_problem = lambda reason, _exc: fail(reason)
    controller.approval_grants = SimpleNamespace()
    return controller


def _context_contract(prepared) -> tuple[object, ...]:
    context = prepared.context
    assistants = tuple(
        (
            assistant.id,
            assistant.genesis,
            tuple((power.id, power.summary, power.input_schema) for power in assistant.powers),
        )
        for assistant in context.assistants
    )
    return context.team_name, assistants, context.provider, context.model, context.api_key, prepared.files


class SharedChatTurnEngineTest(unittest.TestCase):
    def _strategy(self, *, decisions: list[str]) -> chat_turn_engine.SegmentStrategy:
        def private_inputs(_requests, requirements) -> bool:
            requirements.accounts = ("account-required",)
            decisions.append("accounts")
            return True

        def raise_problem(reason: str, _exc: BaseException | None) -> None:
            raise AssertionError(reason)

        return chat_turn_engine.SegmentStrategy(
            runtime=_Runtime(),
            prepare=lambda: chat_turn_engine.PreparedSegment(
                "Team",
                ("identity",),
                _context(),
                [],
                _Batch(),
            ),
            validate_power=lambda _assistant, _power, payload: payload,
            pause_for_private_inputs=private_inputs,
            cancelled=lambda: False,
            validate_context=lambda: None,
            raise_problem=raise_problem,
        )

    def test_hosted_and_local_strategies_make_the_same_real_suspension_decision(self) -> None:
        decisions: dict[str, list[str]] = {"hosted": [], "local": []}

        hosted = chat_turn_engine.run_segment(
            self._strategy(decisions=decisions["hosted"]),
            message="look this up",
            continuation=None,
            expected_identity=("identity",),
        )
        local = chat_turn_engine.run_segment(
            self._strategy(decisions=decisions["local"]),
            message="look this up",
            continuation=None,
            expected_identity=("identity",),
        )

        self.assertIsInstance(hosted[2], chat_orchestrator.ChatSuspension)
        self.assertIsInstance(local[2], chat_orchestrator.ChatSuspension)
        self.assertEqual(hosted[2], local[2])
        self.assertEqual(hosted[3].accounts, local[3].accounts)
        self.assertEqual(decisions, {"hosted": ["accounts"], "local": ["accounts"]})

    def test_human_input_is_one_distinct_suspension_category(self) -> None:
        requirements = chat_turn_engine.SegmentRequirements(inputs=("input-required",))

        self.assertEqual(
            requirements.groups(),
            ((), (), ("input-required",), ()),
        )
        self.assertEqual(
            chat_turn_engine.suspension_gate_count(*requirements.groups()),
            1,
        )

        suspension = chat_orchestrator.ChatSuspension(
            continuation=SimpleNamespace(),
            requests=(),
        )
        dispatched = chat_turn_engine.dispatch(
            suspension,
            requirements.groups(),
            lambda _outcome: "pending",
            (
                lambda *_args: "accounts",
                lambda *_args: "secrets",
                lambda *_args: "inputs",
                lambda *_args: "approvals",
            ),
            lambda _outcome: "complete",
        )
        self.assertEqual(dispatched, "inputs")

    def test_segment_with_two_populated_gates_fails_closed(self) -> None:
        base = self._strategy(decisions=[])

        def conflicting_requirements(_requests, requirements) -> bool:
            requirements.accounts = ("account-required",)
            requirements.inputs = ("input-required",)
            return True

        strategy = chat_turn_engine.SegmentStrategy(
            runtime=base.runtime,
            prepare=base.prepare,
            validate_power=base.validate_power,
            pause_for_private_inputs=conflicting_requirements,
            cancelled=base.cancelled,
            validate_context=base.validate_context,
            raise_problem=base.raise_problem,
        )

        with self.assertRaisesRegex(AssertionError, "invalid-suspension"):
            chat_turn_engine.run_segment(
                strategy,
                message="look this up",
                continuation=None,
                expected_identity=("identity",),
            )

    def test_matching_suspension_commits_without_rollback(self) -> None:
        decisions: list[str] = []

        chat_turn_engine.commit_suspension(
            "continuation",
            "continuation",
            lambda: decisions.append("commit") is None,
            lambda: decisions.append("cancel"),
            lambda: RuntimeError("stopped"),
            lambda: decisions.append("cleanup"),
        )

        self.assertEqual(decisions, ["commit"])

    def test_stale_or_failed_suspension_rolls_back_before_error(self) -> None:
        def assert_rollback(continuation: str, commit_result: bool, expected: list[str]) -> None:
            decisions: list[str] = []

            with self.assertRaisesRegex(RuntimeError, "stopped"):
                chat_turn_engine.commit_suspension(
                    continuation,
                    "continuation",
                    lambda: decisions.append("commit") is None and commit_result,
                    lambda: decisions.append("cancel"),
                    lambda: RuntimeError("stopped"),
                    lambda: decisions.append("cleanup"),
                )

            self.assertEqual(decisions, expected)

        assert_rollback("stale", True, ["cancel", "cleanup"])
        assert_rollback("continuation", False, ["commit", "cancel", "cleanup"])

    def test_rpc_request_suspension_populates_the_input_category(self) -> None:
        strategy = self._strategy(decisions=[])
        strategy = chat_turn_engine.SegmentStrategy(
            runtime=strategy.runtime,
            prepare=lambda: chat_turn_engine.PreparedSegment(
                "Team",
                ("identity",),
                _context(),
                [],
                _InteractionBatch(),
            ),
            validate_power=strategy.validate_power,
            pause_for_private_inputs=lambda _requests, _requirements: False,
            cancelled=strategy.cancelled,
            validate_context=strategy.validate_context,
            raise_problem=strategy.raise_problem,
        )

        _, _, outcome, requirements = chat_turn_engine.run_segment(
            strategy,
            message="ask",
            continuation=None,
            expected_identity=("identity",),
        )

        self.assertIsInstance(outcome, chat_orchestrator.ChatSuspension)
        self.assertEqual(requirements.inputs, (outcome.interaction,))
        self.assertEqual(requirements.approvals, ())

    def test_hosted_context_validation_reuses_the_prepared_inventory(self) -> None:
        anchor = SimpleNamespace(id="a" * 64, labels={"team.name": "Team"})
        config = inference_config.InferenceConfig("openai", "gpt-test")
        turn_token = "turn-token"

        def run_with_validation(strategy, **_kwargs):
            prepared = strategy.prepare()
            strategy.validate_context()
            return (
                prepared.team_name,
                prepared.identity,
                chat_orchestrator.ChatOutcome("done", ()),
                chat_turn_engine.SegmentRequirements(),
            )

        with (
            mock.patch.multiple(
                hosted_harness.hosted_assistants,
                _active_team_assistants=lambda _team_id: (),
                _model_credential=lambda _owner, _provider: ("test-key", 7),
                _require_model_credential_current=lambda *_args: None,
            ),
            mock.patch.object(
                hosted_harness.runtime_state,
                "_inference_store",
                SimpleNamespace(load=lambda _team_id: config),
            ),
            mock.patch.object(
                hosted_harness.hosted_chat_segment,
                "_current_team_anchor",
                side_effect=lambda *_args: anchor,
            ),
            mock.patch.object(
                hosted_harness.runtime_state,
                "_storage",
                side_effect=lambda: SimpleNamespace(metadata=lambda _team_id, _files: []),
            ),
            mock.patch.object(
                hosted_harness.hosted_chat_segment,
                "_hosted_chat_setup",
                wraps=hosted_harness.hosted_chat_segment._hosted_chat_setup,
            ) as setup,
            mock.patch.object(
                hosted_harness.hosted_chat_segment.chat_turn_engine,
                "run_segment",
                side_effect=run_with_validation,
            ),
        ):
            result = hosted_app._run_hosted_chat_segment(
                hosted_app.HostedChatSegmentRequest(
                    team_id="team_1",
                    file_ids=[],
                    assistant_ids=(),
                    token=turn_token,
                    container=anchor,
                    owner="owner",
                    message="Hello",
                )
            )

        self.assertEqual(result.outcome.reply, "done")
        setup.assert_called_once_with("team_1", [], (), anchor, "owner")

    def test_hosted_and_local_controllers_build_equivalent_real_segment_strategies(self) -> None:
        assistant_id = "shimpz-cloudflare"
        declared_contract = hosted_app.marketplace.APPS[assistant_id].assistant
        if declared_contract is None:
            self.fail("the hosted Assistant contract is unavailable")
        declared_power = declared_contract.powers["list-zones"]
        hosted_power = hosted_app.marketplace.PowerSpec(
            declared_power.method,
            declared_power.path,
            declared_power.summary,
            declared_power.input_schema,
            declared_power.output_schema,
        )
        hosted_contract = hosted_app.marketplace.AssistantContract(
            "assistant-rpc",
            {"list-zones": hosted_power},
            {},
        )
        assistant_container = SimpleNamespace(id="b" * 64)
        hosted_active = hosted_app._ActiveAssistant(assistant_id, hosted_contract, assistant_container)

        local_power = local_registry.PowerSpec(
            declared_power.method,
            declared_power.path,
            declared_power.summary,
            dict(declared_power.input_schema),
            dict(declared_power.output_schema),
            (),
        )
        local_spec = local_registry.AssistantSpec(
            assistant_id=assistant_id,
            name="Assistant",
            summary="Test Assistant",
            image="example.invalid/assistant@sha256:" + ("c" * 64),
            rpc_command="assistant-rpc",
            health_path="/healthz",
            powers={"list-zones": local_power},
            secrets={},
            allowed_hosts=(),
        )
        local_active = ActiveAssistant(local_spec, assistant_container.id)
        request = SimpleNamespace(
            interrupt_id="interrupt-1",
            assistant_id=assistant_id,
            power="list-zones",
            input={"page": 1, "per_page": 25},
        )
        config = inference_config.InferenceConfig("openai", "gpt-test")
        captures: dict[str, tuple[object, object, chat_turn_engine.SegmentRequirements, bool]] = {}

        def capture(label: str):
            def run(strategy, **_kwargs):
                prepared = strategy.prepare()
                requirements = chat_turn_engine.SegmentRequirements()
                paused = strategy.pause_for_private_inputs((request,), requirements)
                captures[label] = (strategy, prepared, requirements, paused)
                return prepared.team_name, prepared.identity, SimpleNamespace(), requirements

            return run

        hosted_events: list[str] = []
        hosted_anchor = SimpleNamespace(id="a" * 64, labels={"team.name": "Team"})
        hosted_token = "turn-token"
        with (
            mock.patch.multiple(
                hosted_harness.hosted_assistants,
                _active_team_assistants=lambda _team_id: (hosted_active,),
                _model_credential=lambda _owner, _provider: ("test-key", 7),
                _require_model_credential_current=lambda *_args: hosted_events.append("model"),
                _require_hosted_power_rpc_envelope=lambda *_args: hosted_events.append("preflight"),
                _hosted_power_identity=lambda _active: (assistant_container.id, local_spec.image),
                _power_secret_generations=lambda *_args: hosted_events.append("secrets") or (),
                _power_account_generations=lambda *_args: hosted_events.append("accounts") or (),
            ),
            mock.patch.object(
                hosted_harness.hosted_apps,
                "_require_assistant_genesis",
                return_value="Use the declared Power.",
            ),
            mock.patch.object(
                hosted_harness.hosted_chat_segment,
                "_hosted_private_requirements",
                return_value=(("account-required",), ()),
            ),
            mock.patch.object(
                hosted_harness.runtime_state,
                "_inference_store",
                SimpleNamespace(load=lambda _team_id: config),
            ),
            mock.patch.object(
                hosted_harness.runtime_state,
                "_storage",
                side_effect=lambda: SimpleNamespace(metadata=lambda _team_id, _files: []),
            ),
            mock.patch.object(
                hosted_harness.hosted_chat_segment.chat_turn_engine,
                "run_segment",
                side_effect=capture("hosted"),
            ),
        ):
            hosted_app._run_hosted_chat_segment(
                hosted_app.HostedChatSegmentRequest(
                    team_id="team_1",
                    file_ids=[],
                    assistant_ids=(assistant_id,),
                    token=hosted_token,
                    container=hosted_anchor,
                    owner="owner",
                    message="Look this up",
                )
            )
            hosted_strategy, hosted_prepared, _, _ = captures["hosted"]
            hosted_validation_result = hosted_strategy.validate_power(assistant_id, "list-zones", request.input)
            hosted_prepared.durable_batch._operation(request)
            hosted_strategy.finalize()

        local_events: list[str] = []
        controller = _local_controller(local_active, config, local_events, self.fail)
        turn_token = "turn-token"

        with mock.patch.object(local_chat_segment.chat_turn_engine, "run_segment", side_effect=capture("local")):
            controller._run_chat_segment(
                SegmentRequest(
                    team_id="team_1",
                    file_ids=[],
                    assistant_ids=(assistant_id,),
                    provider="openai",
                    api_key="test-key",
                    token=turn_token,
                    message="Look this up",
                )
            )

        hosted_strategy, hosted_prepared, hosted_requirements, hosted_paused = captures["hosted"]
        local_strategy, local_prepared, local_requirements, local_paused = captures["local"]

        self.assertEqual(_context_contract(hosted_prepared), _context_contract(local_prepared))
        self.assertEqual(hosted_requirements.groups(), local_requirements.groups())
        self.assertTrue(hosted_paused)
        self.assertTrue(local_paused)
        self.assertEqual(
            hosted_validation_result,
            local_strategy.validate_power(assistant_id, "list-zones", request.input),
        )

        local_prepared.durable_batch._operation(request)
        local_strategy.finalize()
        self.assertEqual(hosted_events, ["model", "preflight", "secrets", "accounts", "model"])
        self.assertEqual(local_events, ["preflight", "secrets", "accounts"])


if __name__ == "__main__":
    unittest.main()
