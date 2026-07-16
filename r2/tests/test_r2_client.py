from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import r2_client


def credentials(*, account: str = "a", access: str = "A", private: str = "b", bucket: str = "bucket-one"):
    return r2_client.R2Credentials(
        account_id=account * 32,
        access_key_id=access * 24,
        secret_access_key=private * 64,
        bucket=bucket,
    )


class FakeProcess:
    def __init__(self, command, response, kwargs) -> None:
        self.args = command
        self.pid = 999_999
        self.returncode, self.stdout_data, self.stderr_data = response
        self.kwargs = kwargs

    def communicate(self, timeout=None):
        del timeout
        return self.stdout_data, self.stderr_data


class R2ClientTests(unittest.TestCase):
    def popen(self, responses, calls):
        def create(command, **kwargs):
            process = FakeProcess(command, responses.pop(0), kwargs)
            calls.append(process)
            return process

        return create

    def test_each_call_has_an_explicit_isolated_environment(self) -> None:
        first = credentials()
        second = credentials(account="c", access="D", private="e", bucket="bucket-two")
        calls = []
        responses = [(0, "", ""), (0, "", "")]
        original = dict(os.environ)
        injected = {
            "AWS_SECRET_ACCESS_KEY": "must-not-leak",
            "CLOUDFLARE_API_TOKEN": "must-not-leak",
            "RCLONE_CONFIG_OTHER_SECRET": "must-not-leak",
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": "managed-must-not-leak",
            "HTTPS_PROXY": "https://proxy-user:proxy-secret@proxy.invalid",
            "SSL_CERT_FILE": "/run/untrusted-ca.pem",
        }
        with mock.patch.dict(os.environ, injected, clear=False):
            snapshot = dict(os.environ)
            with mock.patch.object(r2_client.subprocess, "Popen", self.popen(responses, calls)):
                r2_client._run(["version"], credentials=first)
                r2_client._run(["version"], credentials=second)
            self.assertEqual(dict(os.environ), snapshot)
        self.assertEqual(dict(os.environ), original)

        first_environment = calls[0].kwargs["env"]
        second_environment = calls[1].kwargs["env"]
        self.assertEqual(first_environment["RCLONE_CONFIG_R2_ACCESS_KEY_ID"], first.access_key_id)
        self.assertEqual(second_environment["RCLONE_CONFIG_R2_ACCESS_KEY_ID"], second.access_key_id)
        self.assertNotEqual(first_environment, second_environment)
        for name in (
            "AWS_SECRET_ACCESS_KEY",
            "CLOUDFLARE_API_TOKEN",
            "RCLONE_CONFIG_OTHER_SECRET",
            "HTTPS_PROXY",
            "SSL_CERT_FILE",
        ):
            self.assertNotIn(name, first_environment)
            self.assertNotIn(name, second_environment)
        self.assertEqual(first_environment["RCLONE_CONFIG_R2_ACL"], "private")
        self.assertEqual(first_environment["RCLONE_CONFIG_R2_NO_CHECK_BUCKET"], "true")

    def test_only_the_explicit_bandwidth_limit_global_option_is_inherited(self) -> None:
        calls = []
        responses = [(0, "", ""), (0, "", "")]
        injected = {
            "RCLONE_BWLIMIT": "64k",
            "RCLONE_ARBITRARY_OPTION": "must-not-pass",
        }
        with (
            mock.patch.dict(os.environ, injected, clear=False),
            mock.patch.object(r2_client.subprocess, "Popen", self.popen(responses, calls)),
        ):
            r2_client._run(["version"])
            r2_client._run(["version"], credentials=credentials())
        for call in calls:
            environment = call.kwargs["env"]
            self.assertEqual(environment["RCLONE_BWLIMIT"], "64k")
            self.assertNotIn("RCLONE_ARBITRARY_OPTION", environment)

    def test_managed_fallback_only_inherits_the_r2_remote(self) -> None:
        calls = []
        responses = [(0, "", "")]
        injected = {
            "RCLONE_CONFIG_R2_TYPE": "s3",
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": "managed-secret",
            "RCLONE_CONFIG_OTHER_SECRET": "other-secret",
            "AWS_ACCESS_KEY_ID": "aws-secret",
        }
        with (
            mock.patch.dict(os.environ, injected, clear=False),
            mock.patch.object(r2_client.subprocess, "Popen", self.popen(responses, calls)),
        ):
            r2_client._run(["version"])
        environment = calls[0].kwargs["env"]
        self.assertEqual(environment["RCLONE_CONFIG_R2_SECRET_ACCESS_KEY"], "managed-secret")
        self.assertNotIn("RCLONE_CONFIG_OTHER_SECRET", environment)
        self.assertNotIn("AWS_ACCESS_KEY_ID", environment)

    def test_raw_stderr_and_credentials_never_escape_errors(self) -> None:
        selected = credentials()
        calls = []
        stderr_marker = "RAW_SECRET InvalidAccessKeyId"
        responses = [(1, "", stderr_marker)]
        with (
            tempfile.NamedTemporaryFile() as source,
            mock.patch.object(r2_client.subprocess, "Popen", self.popen(responses, calls)),
            self.assertRaises(r2_client.R2Error) as raised,
        ):
            r2_client.upload(source.name, "object", credentials=selected)
        self.assertEqual(raised.exception.category, "authentication")
        public_error = str(raised.exception)
        self.assertNotIn("RAW_SECRET", public_error)
        self.assertNotIn(selected.access_key_id, public_error)
        self.assertNotIn(selected.secret_access_key, public_error)

    def test_probe_is_a_read_only_bucket_stat(self) -> None:
        selected = credentials()
        calls = []
        responses = [(0, '{"IsDir":true}', "")]
        with mock.patch.object(r2_client.subprocess, "Popen", self.popen(responses, calls)):
            self.assertTrue(r2_client.probe(credentials=selected))
        self.assertEqual(calls[0].args[1:3], ["lsjson", "--stat"])
        self.assertIn("R2:bucket-one/", calls[0].args[3])
        self.assertNotIn(calls[0].args[1], {"copy", "copyto", "delete", "purge", "sync"})


if __name__ == "__main__":
    unittest.main()
