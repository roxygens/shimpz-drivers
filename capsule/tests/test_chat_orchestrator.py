from __future__ import annotations

import unittest

import brain_runtime_client
import chat_orchestrator


def context(*powers: brain_runtime_client.RuntimePower) -> brain_runtime_client.RuntimeContext:
    return brain_runtime_client.RuntimeContext(
        thread_id="capsule:assistant:conversation",
        team_name="Marketing",
        assistants=(
            brain_runtime_client.RuntimeAssistant(
                id="hello-pulse",
                rules="Use only declared Powers.",
                powers=powers
                or (
                    brain_runtime_client.RuntimePower(
                        id="hello",
                        summary="Return a greeting.",
                        input_schema={"type": "object", "additionalProperties": False},
                        approval="none",
                    ),
                ),
            ),
        ),
        provider="openai",
        model="gpt-test",
        api_key="not-a-real-key",
    )


def completed(reply: str = "Done") -> brain_runtime_client.RuntimeTurn:
    return brain_runtime_client.RuntimeTurn(status="completed", reply=reply, powers=())


def suspended(
    power: str = "hello",
    *,
    assistant_id: str = "hello-pulse",
    interrupt_id: str = "interrupt-1",
    approval: str = "none",
) -> brain_runtime_client.RuntimeTurn:
    return brain_runtime_client.RuntimeTurn(
        status="power-required",
        reply="",
        powers=(
            brain_runtime_client.PowerRequest(
                interrupt_id=interrupt_id,
                assistant_id=assistant_id,
                power=power,
                input={"name": "Ada"},
                approval=approval,
            ),
        ),
    )


class FakeRuntime:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.resumes = []

    def start(self, _context, _message):
        return next(self.turns)

    def resume(self, _context, results):
        self.resumes.append(results)
        return next(self.turns)


class ChatOrchestratorTests(unittest.TestCase):
    def test_direct_reply_never_invokes_a_power(self):
        invoked = []

        outcome = chat_orchestrator.run(
            FakeRuntime([completed("Hello")]),
            context(),
            "Hello",
            lambda assistant, power, payload: invoked.append((assistant, power, payload)),
        )

        self.assertEqual(outcome.reply, "Hello")
        self.assertEqual(outcome.powers, ())
        self.assertEqual(invoked, [])

    def test_power_result_is_returned_to_the_model_before_the_final_reply(self):
        runtime = FakeRuntime([suspended(), completed("Hello, Ada.")])
        invoked = []

        outcome = chat_orchestrator.run(
            runtime,
            context(),
            "Greet Ada",
            lambda assistant, power, payload: invoked.append((assistant, power, payload))
            or {"message": "Hello, Ada."},
        )

        self.assertEqual(invoked, [("hello-pulse", "hello", {"name": "Ada"})])
        self.assertEqual(runtime.resumes, [{"interrupt-1": {"message": "Hello, Ada."}}])
        self.assertEqual(outcome.reply, "Hello, Ada.")
        self.assertEqual(
            outcome.powers,
            (chat_orchestrator.InvokedPower(assistant_id="hello-pulse", power="hello"),),
        )

    def test_multiple_power_rounds_remain_bounded_and_controller_brokered(self):
        runtime = FakeRuntime(
            [
                suspended(interrupt_id="one"),
                suspended(interrupt_id="two"),
                completed(),
            ]
        )

        outcome = chat_orchestrator.run(
            runtime,
            context(),
            "Run twice",
            lambda _assistant, _power, _payload: {"message": "ok"},
        )

        self.assertEqual([item.power for item in outcome.powers], ["hello", "hello"])
        self.assertEqual(len(runtime.resumes), 2)

    def test_undeclared_power_or_changed_approval_fails_before_invocation(self):
        invoked = []
        for turn in (suspended("shell"), suspended(approval="each-run")):
            with self.subTest(turn=turn), self.assertRaises(chat_orchestrator.ChatOrchestrationError):
                chat_orchestrator.run(
                    FakeRuntime([turn]),
                    context(),
                    "Do it",
                    lambda assistant, power, payload: invoked.append((assistant, power, payload)),
                )
        self.assertEqual(invoked, [])

    def test_approval_policy_fails_closed_until_the_controller_has_a_grant(self):
        protected = brain_runtime_client.RuntimePower(
            id="hello",
            summary="Return a greeting.",
            input_schema={"type": "object"},
            approval="each-run",
        )
        invoked = []

        with self.assertRaises(chat_orchestrator.ApprovalRequiredError) as raised:
            chat_orchestrator.run(
                FakeRuntime([suspended(approval="each-run")]),
                context(protected),
                "Do it",
                lambda assistant, power, payload: invoked.append((assistant, power, payload)),
            )

        self.assertEqual(raised.exception.request.power, "hello")
        self.assertEqual(invoked, [])

    def test_cancelled_turn_never_starts_or_resumes_work(self):
        runtime = FakeRuntime([completed()])

        with self.assertRaises(chat_orchestrator.ChatStoppedError):
            chat_orchestrator.run(
                runtime,
                context(),
                "Stop",
                lambda _assistant, _power, _payload: {},
                cancelled=lambda: True,
            )
        self.assertEqual(runtime.resumes, [])

    def test_power_round_limit_stops_an_unbounded_model_loop(self):
        turns = [
            suspended(interrupt_id=f"interrupt-{index}") for index in range(chat_orchestrator.MAX_POWER_ROUNDS + 1)
        ]

        with self.assertRaisesRegex(chat_orchestrator.ChatOrchestrationError, "round limit"):
            chat_orchestrator.run(
                FakeRuntime(turns),
                context(),
                "Loop",
                lambda _assistant, _power, _payload: {"message": "ok"},
            )

    def test_two_assistants_can_own_the_same_local_power_id(self):
        shared = brain_runtime_client.RuntimePower(
            id="lookup",
            summary="Look up data.",
            input_schema={"type": "object"},
            approval="none",
        )
        base = context()
        team = brain_runtime_client.RuntimeContext(
            thread_id=base.thread_id,
            team_name=base.team_name,
            assistants=(
                brain_runtime_client.RuntimeAssistant("places", "Find places.", (shared,)),
                brain_runtime_client.RuntimeAssistant("weather", "Find weather.", (shared,)),
            ),
            provider=base.provider,
            model=base.model,
            api_key=base.api_key,
        )
        runtime = FakeRuntime(
            [
                suspended("lookup", assistant_id="places", interrupt_id="place-1"),
                suspended("lookup", assistant_id="weather", interrupt_id="weather-1"),
                completed("Integrated."),
            ]
        )
        invoked = []

        outcome = chat_orchestrator.run(
            runtime,
            team,
            "Find Berlin's weather",
            lambda assistant, power, payload: invoked.append((assistant, power, payload)) or {"ok": True},
        )

        self.assertEqual([item.assistant_id for item in outcome.powers], ["places", "weather"])
        self.assertEqual([item[:2] for item in invoked], [("places", "lookup"), ("weather", "lookup")])

    def test_context_is_revalidated_around_every_side_effect_and_resume(self):
        runtime = FakeRuntime([suspended(), completed()])
        validations = []

        chat_orchestrator.run(
            runtime,
            context(),
            "Greet Ada",
            lambda _assistant, _power, _payload: {"message": "ok"},
            validate_context=lambda: validations.append("valid"),
        )

        self.assertEqual(len(validations), 4)


if __name__ == "__main__":
    unittest.main()
