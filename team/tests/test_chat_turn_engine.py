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
import inference_config
import local_app
import local_registry
import test_hosted_app as hosted_harness

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
                        approval="none",
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
                    approval="none",
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


class SharedChatTurnEngineTest(unittest.TestCase):
    def _strategy(self, *, local: bool, decisions: list[str]) -> chat_turn_engine.SegmentStrategy:
        def private_inputs(_requests, requirements) -> bool:
            requirements.accounts = ("account-required",)
            decisions.append("accounts")
            return True

        def approval(_requests, _requirements) -> bool:
            decisions.append("approval")
            return False

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
            pause_for_approval=approval if local else None,
            approval_granted=(lambda _request: False) if local else None,
        )

    def test_hosted_and_local_strategies_make_the_same_real_suspension_decision(self) -> None:
        decisions: dict[str, list[str]] = {"hosted": [], "local": []}

        hosted = chat_turn_engine.run_segment(
            self._strategy(local=False, decisions=decisions["hosted"]),
            message="look this up",
            continuation=None,
            expected_identity=("identity",),
        )
        local = chat_turn_engine.run_segment(
            self._strategy(local=True, decisions=decisions["local"]),
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
            requirements.groups(approvals=True),
            ((), (), ("input-required",), ()),
        )
        self.assertEqual(
            chat_turn_engine.suspension_gate_count(*requirements.groups(approvals=True)),
            1,
        )

        suspension = chat_orchestrator.ChatSuspension(
            continuation=SimpleNamespace(),
            requests=(),
        )
        dispatched = chat_turn_engine.dispatch(
            suspension,
            requirements.groups(approvals=True),
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
            "none",
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
        local_active = local_app._ActiveAssistant(local_spec, assistant_container.id)
        request = SimpleNamespace(
            interrupt_id="interrupt-1",
            assistant_id=assistant_id,
            power="list-zones",
            input={"page": 1, "per_page": 25},
            approval="none",
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
        with (
            hosted_harness._patched(
                _active_team_assistants=lambda _team_id: (hosted_active,),
                _storage=lambda: SimpleNamespace(metadata=lambda _team_id, _files: []),
                _inference_store=SimpleNamespace(load=lambda _team_id: config),
                _model_credential=lambda _owner, _provider: ("test-key", 7),
                _require_model_credential_current=lambda *_args: hosted_events.append("model"),
                _require_assistant_genesis=lambda _container: "Use the declared Power.",
                _hosted_private_requirements=lambda *_args: (("account-required",), ()),
                _require_hosted_power_rpc_envelope=lambda *_args: hosted_events.append("preflight"),
                _hosted_power_identity=lambda _active: (assistant_container.id, local_spec.image),
                _power_secret_generations=lambda *_args: hosted_events.append("secrets") or (),
                _power_account_generations=lambda *_args: hosted_events.append("accounts") or (),
            ),
            mock.patch.object(hosted_app.chat_turn_engine, "run_segment", side_effect=capture("hosted")),
        ):
            hosted_app._run_hosted_chat_segment(
                "team_1",
                [],
                (assistant_id,),
                "turn-token",
                hosted_anchor,
                "owner",
                message="Look this up",
            )
            hosted_strategy, hosted_prepared, _, _ = captures["hosted"]
            hosted_validation_result = hosted_strategy.validate_power(assistant_id, "list-zones", request.input)
            hosted_prepared.durable_batch._operation(request)
            hosted_strategy.finalize()

        local_events: list[str] = []
        controller = object.__new__(local_app.LocalController)
        controller.space_id = "local-space"
        controller.brain_runtime = SimpleNamespace()
        controller.power_state = SimpleNamespace()
        controller._lock = lambda _team_id: contextlib.nullcontext()
        controller._network = lambda _team_id: SimpleNamespace(id="a" * 64, name="team-network")
        controller._validate_network = lambda _network, _team_id: "Team"
        controller._active_chat_assistants = lambda _team_id, _network_name: (local_active,)
        controller.storage = SimpleNamespace(metadata=lambda _team_id, _files: [])
        controller.inference_store = SimpleNamespace(load=lambda _team_id: config)
        controller._active_assistant_genesis = lambda _active: "Use the declared Power."

        def local_private_inputs(_team_id, _bindings, _requests, requirements) -> bool:
            requirements.accounts = ("account-required",)
            return True

        def local_approval(_bindings, _requests, requirements) -> bool:
            requirements.approvals = ("approval-required",)
            return True

        controller._require_chat_private_inputs = local_private_inputs
        controller._require_chat_approval = local_approval
        controller._require_power_rpc_envelope = lambda *_args: local_events.append("preflight")
        controller._power_secret_generations = lambda *_args: local_events.append("secrets") or ()
        controller._power_account_generations = lambda *_args: local_events.append("accounts") or ()
        controller._chat_cancelled = lambda _token: False
        controller._validate_chat_context = lambda *_args: None
        controller._raise_chat_problem = lambda reason, _exc: self.fail(reason)
        controller._chat_approval_granted = lambda *_args: False

        with mock.patch.object(local_app.chat_turn_engine, "run_segment", side_effect=capture("local")):
            controller._run_chat_segment(
                "team_1",
                [],
                (assistant_id,),
                "openai",
                "test-key",
                "turn-token",
                message="Look this up",
            )

        hosted_strategy, hosted_prepared, hosted_requirements, hosted_paused = captures["hosted"]
        local_strategy, local_prepared, local_requirements, local_paused = captures["local"]

        def context_contract(prepared) -> tuple[object, ...]:
            context = prepared.context
            assistants = tuple(
                (
                    assistant.id,
                    assistant.genesis,
                    tuple((power.id, power.summary, power.input_schema, power.approval) for power in assistant.powers),
                )
                for assistant in context.assistants
            )
            return context.team_name, assistants, context.provider, context.model, context.api_key, prepared.files

        self.assertEqual(context_contract(hosted_prepared), context_contract(local_prepared))
        self.assertEqual(hosted_requirements.groups(approvals=False), local_requirements.groups(approvals=False))
        self.assertTrue(hosted_paused)
        self.assertTrue(local_paused)
        self.assertIsNone(hosted_strategy.pause_for_approval)
        self.assertIsNotNone(local_strategy.pause_for_approval)
        local_approval_requirements = chat_turn_engine.SegmentRequirements()
        self.assertTrue(local_strategy.pause_for_approval((request,), local_approval_requirements))
        self.assertEqual(local_approval_requirements.groups(approvals=True)[-1], ("approval-required",))

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
