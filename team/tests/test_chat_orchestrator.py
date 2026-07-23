from __future__ import annotations

import unittest

import brain_runtime_client
import chat_orchestrator


def context(*powers: brain_runtime_client.RuntimePower) -> brain_runtime_client.RuntimeContext:
    return brain_runtime_client.RuntimeContext(
        thread_id="team:assistant:conversation",
        team_name="Marketing",
        assistants=(
            brain_runtime_client.RuntimeAssistant(
                id="hello-pulse",
                genesis="Compose the declared greeting Power into one bounded welcome.",
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


def suspension(*requests: brain_runtime_client.PowerRequest) -> brain_runtime_client.RuntimeTurn:
    return brain_runtime_client.RuntimeTurn(status="power-required", reply="", powers=requests)


def accept_input(_assistant: str, _power: str, payload):
    return payload


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
            accept_input,
            lambda request: invoked.append((request.assistant_id, request.power, request.input)),
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
            accept_input,
            lambda request: (
                invoked.append((request.assistant_id, request.power, request.input)) or {"message": "Hello, Ada."}
            ),
        )

        self.assertEqual(invoked, [("hello-pulse", "hello", {"name": "Ada"})])
        self.assertEqual(runtime.resumes, [{"interrupt-1": {"message": "Hello, Ada."}}])
        self.assertEqual(outcome.reply, "Hello, Ada.")
        self.assertEqual(
            outcome.powers,
            (chat_orchestrator.InvokedPower(assistant_id="hello-pulse", power="hello"),),
        )

    def test_power_human_interaction_pauses_and_replays_the_whole_batch(self):
        runtime = FakeRuntime([suspended(), completed("Answered.")])
        interactions = iter(
            (
                chat_orchestrator.power_execution.RpcSuspension(
                    {"ordinal": 0, "kind": "request", "request_type": "str"}
                ),
                {"answer": "Ada"},
            )
        )
        prepared = []

        paused = chat_orchestrator.run_until_pause(
            runtime,
            context(),
            "Ask first",
            accept_input,
            lambda _request: next(interactions),
            prepare_batch=lambda batch: prepared.append(batch),
        )

        self.assertIsInstance(paused, chat_orchestrator.ChatSuspension)
        self.assertEqual(paused.interaction.request.interrupt_id, "interrupt-1")
        self.assertEqual(paused.interaction.payload["kind"], "request")
        self.assertEqual(runtime.resumes, [])

        outcome = chat_orchestrator.continue_after_pause(
            runtime,
            context(),
            paused.continuation,
            accept_input,
            lambda _request: next(interactions),
            prepare_batch=lambda batch: prepared.append(batch),
        )
        self.assertIsInstance(outcome, chat_orchestrator.ChatOutcome)
        self.assertEqual(outcome.reply, "Answered.")
        self.assertEqual(len(prepared), 2)
        self.assertEqual(len(outcome.powers), 1)

    def test_pause_happens_after_full_validation_and_before_any_side_effect(self):
        requests = (
            suspended(interrupt_id="first").powers[0],
            suspended(interrupt_id="second").powers[0],
        )
        runtime = FakeRuntime([suspension(*requests), completed("Finished")])
        events = []

        progress = chat_orchestrator.run_until_pause(
            runtime,
            context(),
            "Use the Powers",
            lambda assistant, power, payload: events.append(("validate", assistant, power)) or payload,
            lambda request: events.append(("invoke", request.interrupt_id)) or {"ok": True},
            prepare_batch=lambda batch: events.append(("prepare", len(batch))),
            pause_before_batch=lambda batch: events.append(("pause", len(batch))) or True,
        )

        self.assertIsInstance(progress, chat_orchestrator.ChatSuspension)
        self.assertEqual(
            events,
            [
                ("validate", "hello-pulse", "hello"),
                ("validate", "hello-pulse", "hello"),
                ("pause", 2),
            ],
        )
        self.assertEqual(runtime.resumes, [])

        resumed = chat_orchestrator.continue_after_pause(
            runtime,
            context(),
            progress.continuation,
            accept_input,
            lambda request: events.append(("invoke", request.interrupt_id)) or {"ok": True},
            prepare_batch=lambda batch: events.append(("prepare", len(batch))),
        )

        self.assertIsInstance(resumed, chat_orchestrator.ChatOutcome)
        self.assertEqual(resumed.reply, "Finished")
        self.assertEqual(events[-3:], [("prepare", 2), ("invoke", "first"), ("invoke", "second")])
        self.assertEqual(set(runtime.resumes[0]), {"first", "second"})

    def test_paused_batch_revalidates_and_rejects_context_drift_before_invocation(self):
        runtime = FakeRuntime([suspended(), completed()])
        progress = chat_orchestrator.run_until_pause(
            runtime,
            context(),
            "Use one Power",
            accept_input,
            lambda _request: self.fail("Power must not run before secrets are available"),
            pause_before_batch=lambda _batch: True,
        )
        self.assertIsInstance(progress, chat_orchestrator.ChatSuspension)

        with self.assertRaises(chat_orchestrator.ChatOrchestrationError):
            chat_orchestrator.continue_after_pause(
                runtime,
                context(),
                progress.continuation,
                lambda _assistant, _power, _payload: (_ for _ in ()).throw(
                    chat_orchestrator.ChatOrchestrationError("context changed")
                ),
                lambda _request: self.fail("drifted Power must not run"),
            )
        self.assertEqual(runtime.resumes, [])

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
            accept_input,
            lambda _request: {"message": "ok"},
        )

        self.assertEqual([item.power for item in outcome.powers], ["hello", "hello"])
        self.assertEqual(len(runtime.resumes), 2)

    def test_repeated_suspension_cannot_replay_a_completed_power(self):
        repeated = suspended(interrupt_id="same-interrupt")
        runtime = FakeRuntime([repeated, repeated])
        invoked = []

        with self.assertRaisesRegex(chat_orchestrator.ChatOrchestrationError, "across rounds"):
            chat_orchestrator.run(
                runtime,
                context(),
                "Run once",
                accept_input,
                lambda request: (
                    invoked.append((request.assistant_id, request.power, request.input)) or {"message": "ok"}
                ),
            )

        self.assertEqual(invoked, [("hello-pulse", "hello", {"name": "Ada"})])
        self.assertEqual(len(runtime.resumes), 1)

    def test_undeclared_power_or_changed_approval_fails_before_invocation(self):
        invoked = []
        for turn in (suspended("shell"), suspended(approval="each-run")):
            with self.subTest(turn=turn), self.assertRaises(chat_orchestrator.ChatOrchestrationError):
                chat_orchestrator.run(
                    FakeRuntime([turn]),
                    context(),
                    "Do it",
                    accept_input,
                    lambda request: invoked.append((request.assistant_id, request.power, request.input)),
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
                accept_input,
                lambda request: invoked.append((request.assistant_id, request.power, request.input)),
            )

        self.assertEqual(raised.exception.request.power, "hello")
        self.assertEqual(invoked, [])

    def test_explicit_approval_pauses_the_whole_batch_then_grants_only_that_interrupt(self):
        protected = brain_runtime_client.RuntimePower(
            id="hello",
            summary="Return a greeting.",
            input_schema={"type": "object"},
            approval="each-run",
        )
        runtime = FakeRuntime([suspended(approval="each-run"), completed("Approved.")])
        invoked = []

        paused = chat_orchestrator.run_until_pause(
            runtime,
            context(protected),
            "Do it",
            accept_input,
            lambda request: invoked.append(request.interrupt_id) or {"ok": True},
            pause_for_approval=lambda requests: (
                self.assertEqual(
                    [request.interrupt_id for request in requests],
                    ["interrupt-1"],
                )
                is None
            ),
        )

        self.assertIsInstance(paused, chat_orchestrator.ChatSuspension)
        self.assertEqual(invoked, [])
        outcome = chat_orchestrator.continue_after_pause(
            runtime,
            context(protected),
            paused.continuation,
            accept_input,
            lambda request: invoked.append(request.interrupt_id) or {"ok": True},
            approval_granted=lambda request: request.interrupt_id == "interrupt-1",
        )
        self.assertIsInstance(outcome, chat_orchestrator.ChatOutcome)
        self.assertEqual(invoked, ["interrupt-1"])

    def test_approval_grant_cannot_authorize_another_interrupt(self):
        protected = brain_runtime_client.RuntimePower(
            id="hello",
            summary="Return a greeting.",
            input_schema={"type": "object"},
            approval="each-run",
        )
        with self.assertRaises(chat_orchestrator.ApprovalRequiredError):
            chat_orchestrator.run_until_pause(
                FakeRuntime([suspended(approval="each-run", interrupt_id="other")]),
                context(protected),
                "Do it",
                accept_input,
                lambda _request: self.fail("unbound grant must not invoke a Power"),
                approval_granted=lambda request: request.interrupt_id == "interrupt-1",
            )

    def test_cancelled_turn_never_starts_or_resumes_work(self):
        runtime = FakeRuntime([completed()])

        with self.assertRaises(chat_orchestrator.ChatStoppedError):
            chat_orchestrator.run(
                runtime,
                context(),
                "Stop",
                accept_input,
                lambda _request: {},
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
                accept_input,
                lambda _request: {"message": "ok"},
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
                brain_runtime_client.RuntimeAssistant("places", "Resolve places first.", (shared,)),
                brain_runtime_client.RuntimeAssistant("weather", "Use resolved places for weather.", (shared,)),
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
            accept_input,
            lambda request: invoked.append((request.assistant_id, request.power, request.input)) or {"ok": True},
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
            accept_input,
            lambda _request: {"message": "ok"},
            validate_context=lambda: validations.append("valid"),
        )

        self.assertEqual(len(validations), 5)

    def test_invalid_later_request_prevents_every_batch_side_effect(self):
        first = suspended(interrupt_id="first").powers[0]
        invalid = suspended("shell", interrupt_id="second").powers[0]
        invoked = []

        with self.assertRaisesRegex(chat_orchestrator.ChatOrchestrationError, "undeclared"):
            chat_orchestrator.run(
                FakeRuntime([suspension(first, invalid)]),
                context(),
                "Run the batch",
                accept_input,
                lambda request: invoked.append((request.assistant_id, request.power, request.input)),
            )

        self.assertEqual(invoked, [])

    def test_every_batch_input_is_validated_before_the_first_side_effect(self):
        hello = context().assistants[0].powers[0]
        lookup = brain_runtime_client.RuntimePower(
            id="lookup",
            summary="Look up data.",
            input_schema={"type": "object"},
            approval="none",
        )
        first = suspended(interrupt_id="first").powers[0]
        second = suspended("lookup", interrupt_id="second").powers[0]
        validated = []
        invoked = []

        def validate(assistant, power, payload):
            validated.append((assistant, power, payload))
            if power == "lookup":
                raise ValueError("invalid lookup input")
            return payload

        with self.assertRaisesRegex(ValueError, "invalid lookup input"):
            chat_orchestrator.run(
                FakeRuntime([suspension(first, second)]),
                context(hello, lookup),
                "Run the batch",
                validate,
                lambda request: invoked.append((request.assistant_id, request.power, request.input)),
            )

        self.assertEqual([item[1] for item in validated], ["hello", "lookup"])
        self.assertEqual(invoked, [])

    def test_batch_hooks_receive_normalized_requests_around_resume(self):
        events = []

        class Runtime(FakeRuntime):
            def resume(self, context, results):
                events.append(("resume", results))
                return super().resume(context, results)

        def normalize(_assistant, _power, _payload):
            return {"name": "Normalized"}

        def prepare(batch):
            events.append(("prepare", batch[0].interrupt_id, batch[0].input))

        def invoke(request):
            events.append(("invoke", request.interrupt_id, request.input))
            return {"message": "ok"}

        runtime = Runtime([suspended(), completed()])
        chat_orchestrator.run(
            runtime,
            context(),
            "Run safely",
            normalize,
            invoke,
            prepare_batch=prepare,
            batch_delivered=lambda batch: events.append(("delivered", batch[0].interrupt_id)),
        )

        self.assertEqual(
            events,
            [
                ("prepare", "interrupt-1", {"name": "Normalized"}),
                ("invoke", "interrupt-1", {"name": "Normalized"}),
                ("resume", {"interrupt-1": {"message": "ok"}}),
                ("delivered", "interrupt-1"),
            ],
        )

    def test_prepare_failure_stops_before_invoke_resume_or_delivery(self):
        runtime = FakeRuntime([suspended(), completed()])
        invoked = []
        delivered = []

        def fail_prepare(_batch):
            raise RuntimeError("journal unavailable")

        with self.assertRaisesRegex(RuntimeError, "journal unavailable"):
            chat_orchestrator.run(
                runtime,
                context(),
                "Run safely",
                accept_input,
                lambda request: invoked.append(request),
                prepare_batch=fail_prepare,
                batch_delivered=lambda batch: delivered.append(batch),
            )

        self.assertEqual(invoked, [])
        self.assertEqual(runtime.resumes, [])
        self.assertEqual(delivered, [])

    def test_resume_failure_never_marks_the_batch_delivered(self):
        delivered = []

        class Runtime(FakeRuntime):
            def resume(self, _context, _results):
                raise RuntimeError("runtime unavailable")

        with self.assertRaisesRegex(RuntimeError, "runtime unavailable"):
            chat_orchestrator.run(
                Runtime([suspended()]),
                context(),
                "Run safely",
                accept_input,
                lambda _request: {"message": "ok"},
                batch_delivered=lambda batch: delivered.append(batch),
            )

        self.assertEqual(delivered, [])


if __name__ == "__main__":
    unittest.main()
