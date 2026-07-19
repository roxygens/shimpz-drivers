from __future__ import annotations

import json
import secrets
import tempfile
import unittest
from pathlib import Path

import brain_runtime_client


class _Response:
    def __init__(self, payload: object, *, status: int = 200, raw: bytes | None = None) -> None:
        self.status = status
        self._raw = raw if raw is not None else json.dumps(payload).encode()

    def read(self, _maximum: int) -> bytes:
        return self._raw


class _Connection:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.requests = []
        self.closed = False

    def request(self, *request) -> None:
        self.requests.append(request)

    def getresponse(self) -> _Response:
        return self.response

    def close(self) -> None:
        self.closed = True


def context(secret: str) -> brain_runtime_client.RuntimeContext:
    return brain_runtime_client.RuntimeContext(
        thread_id="team:hello-pulse:conversation-1",
        team_name="Marketing",
        assistants=(
            brain_runtime_client.RuntimeAssistant(
                id="hello-pulse",
                genesis="Combine the declared greeting Powers into one bounded welcome.",
                powers=(
                    brain_runtime_client.RuntimePower(
                        id="hello",
                        summary="Return a greeting.",
                        input_schema={
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "additionalProperties": False,
                        },
                        approval="none",
                    ),
                ),
            ),
        ),
        provider="openai",
        model="gpt-test",
        api_key=secret,
    )


class BrainRuntimeClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.token = secrets.token_hex(32)
        self.secret = secrets.token_urlsafe(32)
        self.token_file = Path(self.directory.name) / "token"
        self.token_file.write_text(self.token, encoding="utf-8")

    def client(self, response: _Response):
        connection = _Connection(response)
        client = brain_runtime_client.BrainRuntimeClient(
            base_url="http://brain-runtime:8080",
            token_file=self.token_file,
            connection_factory=lambda _host, _port, _timeout: connection,
        )
        return client, connection

    def test_start_uses_only_the_fixed_runtime_endpoint_and_private_token(self):
        client, connection = self.client(_Response({"status": "completed", "reply": "Hello.", "powers": []}))

        result = client.start(context(self.secret), "Hello")

        self.assertEqual(result.status, "completed")
        method, path, raw_body, headers = connection.requests[0]
        self.assertEqual((method, path), ("POST", "/v1/turns"))
        self.assertEqual(headers["Authorization"], f"Bearer {self.token}")
        payload = json.loads(raw_body)
        self.assertEqual(payload["provider"]["api_key"], self.secret)
        self.assertEqual(payload["team_name"], "Marketing")
        self.assertEqual(payload["assistants"][0]["id"], "hello-pulse")
        self.assertEqual(
            payload["assistants"][0]["genesis"],
            "Combine the declared greeting Powers into one bounded welcome.",
        )
        self.assertTrue(connection.closed)

    def test_power_suspension_is_parsed_without_gaining_execution_authority(self):
        client, _connection = self.client(
            _Response(
                {
                    "status": "power-required",
                    "reply": "",
                    "powers": [
                        {
                            "interrupt_id": "interrupt-1",
                            "assistant_id": "hello-pulse",
                            "power": "hello",
                            "input": {"name": "Ada"},
                            "approval": "each-run",
                        }
                    ],
                }
            )
        )

        result = client.start(context(self.secret), "Greet Ada")

        self.assertEqual(result.powers[0].power, "hello")
        self.assertEqual(result.powers[0].assistant_id, "hello-pulse")
        self.assertEqual(result.powers[0].input, {"name": "Ada"})
        self.assertEqual(result.powers[0].approval, "each-run")

    def test_resume_sends_only_interrupt_results(self):
        client, connection = self.client(_Response({"status": "completed", "reply": "Done.", "powers": []}))

        client.resume(context(self.secret), {"interrupt-1": {"message": "Hello, Ada."}})

        _method, path, raw_body, _headers = connection.requests[0]
        self.assertEqual(path, "/v1/turns/resume")
        self.assertEqual(json.loads(raw_body)["results"], {"interrupt-1": {"message": "Hello, Ada."}})

    def test_delete_thread_uses_the_closed_runtime_endpoint(self):
        client, connection = self.client(_Response({"status": "deleted"}))

        result = client.delete_thread("team:hello-pulse:conversation-1")

        self.assertIsNone(result)
        method, path, raw_body, headers = connection.requests[0]
        self.assertEqual((method, path), ("POST", "/v1/threads/delete"))
        self.assertEqual(headers["Authorization"], f"Bearer {self.token}")
        self.assertEqual(
            json.loads(raw_body),
            {"thread_id": "team:hello-pulse:conversation-1"},
        )
        self.assertTrue(connection.closed)

    def test_delete_thread_rejects_invalid_ids_before_connecting(self):
        for thread_id in ("", "bad thread", "a" * 257, None):
            with self.subTest(thread_id=thread_id):
                client, connection = self.client(_Response({"status": "deleted"}))

                with self.assertRaises(brain_runtime_client.BrainRuntimeError):
                    client.delete_thread(thread_id)

                self.assertEqual(connection.requests, [])

    def test_delete_thread_response_must_match_the_closed_contract(self):
        for payload in (
            {},
            {"status": "ok"},
            {"status": "deleted", "thread_id": "conversation-1"},
            ["deleted"],
        ):
            with self.subTest(payload=payload):
                client, _connection = self.client(_Response(payload))

                with self.assertRaises(brain_runtime_client.BrainRuntimeError):
                    client.delete_thread("team:hello-pulse:conversation-1")

    def test_malformed_runtime_responses_fail_closed(self):
        invalid = (
            {"status": "completed", "reply": "", "powers": []},
            {"status": "completed", "reply": "ok", "powers": [{"power": "hello"}]},
            {"status": "power-required", "reply": "unexpected", "powers": []},
            {"status": "unknown", "reply": "ok", "powers": []},
        )
        for payload in invalid:
            with self.subTest(payload=payload):
                client, _connection = self.client(_Response(payload))
                with self.assertRaises(brain_runtime_client.BrainRuntimeError):
                    client.start(context(self.secret), "Hello")

    def test_provider_or_transport_errors_never_echo_the_api_key(self):
        for response in (
            _Response({}, status=502, raw=self.secret.encode()),
            _Response({}, raw=b"not-json" + self.secret.encode()),
        ):
            with self.subTest(status=response.status):
                client, _connection = self.client(response)
                with self.assertRaises(brain_runtime_client.BrainRuntimeError) as raised:
                    client.start(context(self.secret), "Hello")
                self.assertNotIn(self.secret, str(raised.exception))

    def test_runtime_url_cannot_carry_credentials_paths_or_queries(self):
        for url in (
            "https://brain-runtime:8080",
            "http://user:secret@brain-runtime:8080",
            "http://brain-runtime:8080/other",
            "http://brain-runtime:8080?redirect=evil",
        ):
            with self.subTest(url=url), self.assertRaises(brain_runtime_client.BrainRuntimeError):
                brain_runtime_client.BrainRuntimeClient(base_url=url, token_file=self.token_file)


if __name__ == "__main__":
    unittest.main()
