from __future__ import annotations

import io
import json
import tarfile
import unittest

import assistant_manifest


def manifest(
    *,
    allowed_hosts: tuple[str, ...] = ("api.example.com",),
    secrets: dict[str, tuple[str, str]] | None = None,
    powers: dict[str, tuple[str, ...]] | None = None,
    connections: dict[str, tuple[str, tuple[str, ...]]] | None = None,
    power_connections: dict[str, tuple[str, ...]] | None = None,
) -> bytes:
    declarations = secrets if secrets is not None else {"api-token": ("API Token", "Token for the public API.")}
    bindings = powers if powers is not None else {"lookup": tuple(declarations)}
    connection_declarations = connections if connections is not None else {}
    connection_bindings = power_connections if power_connections is not None else dict.fromkeys(bindings, ())
    lines = [
        "schema_version = 2",
        'name = "Fixture Assistant"',
        'summary = "Exercise immutable admission."',
        'creators = ["@fixture"]',
        f"allowed_hosts = {json.dumps(list(allowed_hosts))}",
    ]
    for secret_id, (name, summary) in declarations.items():
        lines.extend(
            (
                f"[secrets.{secret_id}]",
                f"name = {json.dumps(name)}",
                f"summary = {json.dumps(summary)}",
            )
        )
    for connection_id, (provider, scopes) in connection_declarations.items():
        lines.extend(
            (
                f"[connections.{connection_id}]",
                f"provider = {json.dumps(provider)}",
                f"scopes = {json.dumps(list(scopes))}",
            )
        )
    for power_id, refs in bindings.items():
        lines.extend(
            (
                f"[powers.{power_id}]",
                f"summary = {json.dumps(f'Run {power_id}.')}",
                'approval = "never"',
                f"secrets = {json.dumps(list(refs))}",
                f"connections = {json.dumps(list(connection_bindings.get(power_id, ())))}",
            )
        )
    return ("\n".join(lines) + "\n").encode()


def archive(
    content: bytes,
    *,
    name: str = "shimpz.assistant.toml",
    member_type: bytes | None = None,
    mode: int = 0o444,
) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as bundle:
        member = tarfile.TarInfo(name)
        member.size = len(content)
        member.mode = mode
        if member_type is not None:
            member.type = member_type
        bundle.addfile(member, io.BytesIO(content))
    return output.getvalue()


class Container:
    def __init__(self, container_id: str, content: bytes) -> None:
        self.id = container_id
        self.content = content
        self.reads = 0

    def get_archive(self, path: str):
        self.reads += 1
        if path != assistant_manifest.MANIFEST_PATH:
            raise AssertionError(f"unexpected archive path: {path}")
        payload = archive(self.content)
        return (
            iter((payload[:113], payload[113:])),
            {"name": "shimpz.assistant.toml", "size": len(self.content), "mode": 0o444},
        )


class AssistantManifestTests(unittest.TestCase):
    def test_manifest_limit_matches_the_public_sdk_contract(self):
        self.assertEqual(assistant_manifest.MAX_MANIFEST_BYTES, 256 * 1024)
        self.assertEqual(assistant_manifest.MAX_SECRET_ID_LENGTH, 64)

    def test_secret_ids_share_the_encrypted_store_bound(self):
        accepted = "s" + ("a" * 63)
        rejected = "s" + ("a" * 64)

        contract = assistant_manifest.parse_manifest_contract(
            manifest(secrets={accepted: ("Bounded", "Exactly sixty-four characters.")})
        )
        self.assertEqual(contract.secrets[0].id, accepted)

        for content in (
            manifest(secrets={rejected: ("Too long", "Exceeds the store bound.")}),
            manifest(secrets={accepted: ("Bounded", "Exactly sixty-four characters.")}).replace(
                f'secrets = ["{accepted}"]'.encode(),
                f'secrets = ["{rejected}"]'.encode(),
            ),
        ):
            with self.subTest(content=content), self.assertRaises(assistant_manifest.ManifestError):
                assistant_manifest.parse_manifest_contract(content)

    def test_reads_and_canonicalizes_complete_security_contract(self):
        content = manifest(
            allowed_hosts=("cdn.example.com", "api.example.com"),
            secrets={
                "write-key": ("Write Key", "Authorizes reviewed writes."),
                "read-key": ("Read Key", "Authorizes reviewed reads."),
            },
            powers={"write": ("write-key",), "read": ("read-key",)},
            connections={"social": ("x", ("users.read", "tweet.read"))},
            power_connections={"write": ("social",), "read": ("social",)},
        )

        contract = assistant_manifest.read_container_manifest_contract(Container("container-one", content))

        self.assertEqual(contract.allowed_hosts, ("api.example.com", "cdn.example.com"))
        self.assertEqual(
            contract.secrets,
            (
                assistant_manifest.SecretDeclaration("read-key", "Read Key", "Authorizes reviewed reads."),
                assistant_manifest.SecretDeclaration("write-key", "Write Key", "Authorizes reviewed writes."),
            ),
        )
        self.assertEqual(
            contract.connections,
            (
                assistant_manifest.ConnectionDeclaration(
                    "social",
                    "x",
                    ("tweet.read", "users.read"),
                ),
            ),
        )
        self.assertEqual(contract.power_secrets, (("read", ("read-key",)), ("write", ("write-key",))))
        self.assertEqual(contract.power_connections, (("read", ("social",)), ("write", ("social",))))

    def test_empty_lists_are_valid_and_security_intent_is_required(self):
        empty = manifest(allowed_hosts=(), secrets={}, powers={"hello": ()})
        self.assertEqual(assistant_manifest.parse_manifest_contract(empty).allowed_hosts, ())
        self.assertEqual(assistant_manifest.parse_allowed_hosts(empty), ())
        for content in (b'name = "No intent"\n', b"schema_version = 2\nallowed_hosts = 1\n"):
            with self.subTest(content=content), self.assertRaises(assistant_manifest.ManifestError):
                assistant_manifest.parse_manifest_contract(content)

    def test_unsafe_hosts_fail_closed(self):
        unsafe = (
            "*.example.com",
            "https://example.com",
            "example.com:443",
            "127.0.0.1",
            "localhost",
            "Example.com",
            "example.com.",
            "example..com",
            "tést.example",
            "api.example.test",
            "api.example.123",
        )
        for host in unsafe:
            with self.subTest(host=host), self.assertRaises(assistant_manifest.ManifestError):
                assistant_manifest.canonical_allowed_hosts([host])
        with self.assertRaises(assistant_manifest.ManifestError):
            assistant_manifest.canonical_allowed_hosts(["api.example.com", "api.example.com"])

    def test_invalid_text_toml_and_size_fail_closed(self):
        invalid = (
            b"",
            manifest() + b"\x00",
            b"\xff",
            b'schema_version = 2\nallowed_hosts = ["example.com"',
            b"x" * (assistant_manifest.MAX_MANIFEST_BYTES + 1),
        )
        for content in invalid:
            with self.subTest(size=len(content)), self.assertRaises(assistant_manifest.ManifestError):
                assistant_manifest.parse_manifest_contract(content)

    def test_secret_declarations_and_power_references_fail_closed(self):
        invalid = (
            manifest(secrets={"unused": ("Unused", "Never exposed.")}, powers={"hello": ()}),
            manifest(secrets={}, powers={"hello": ("missing",)}),
            manifest(powers={"hello": ("api-token", "api-token")}),
            manifest().replace(
                b'summary = "Token for the public API."',
                b'summary = "Token for the public API."\nvalue = "must-not-exist"',
            ),
            manifest() + b'api_key = "sk-examplecredentialmaterial123"\n',
        )
        for content in invalid:
            with self.subTest(content=content), self.assertRaises(assistant_manifest.ManifestError):
                assistant_manifest.parse_manifest_contract(content)

    def test_connection_declarations_and_power_references_fail_closed(self):
        valid = manifest(
            secrets={},
            powers={"lookup": ()},
            connections={"x": ("x", ("tweet.read", "users.read"))},
            power_connections={"lookup": ("x",)},
        )
        contract = assistant_manifest.parse_manifest_contract(valid)
        self.assertEqual(contract.connections[0].provider, "x")
        self.assertEqual(contract.connections[0].scopes, ("tweet.read", "users.read"))

        invalid = (
            manifest(
                secrets={},
                powers={"lookup": ()},
                connections={"x": ("x", ("tweet.read",))},
                power_connections={"lookup": ()},
            ),
            manifest(secrets={}, powers={"lookup": ()}, power_connections={"lookup": ("missing",)}),
            manifest(
                secrets={},
                powers={"lookup": ()},
                connections={"x": ("x", ("tweet.read", "tweet.read"))},
                power_connections={"lookup": ("x",)},
            ),
            valid.replace(b'scopes = ["tweet.read", "users.read"]', b"scopes = []"),
            valid.replace(b'provider = "x"', b'provider = "X"'),
            valid.replace(b'scopes = ["tweet.read", "users.read"]', b'scopes = ["tweet/read"]'),
            valid.replace(
                b'scopes = ["tweet.read", "users.read"]',
                b'scopes = ["tweet.read"]\ntoken_url = "https://evil.example"',
            ),
        )
        for content in invalid:
            with self.subTest(content=content), self.assertRaises(assistant_manifest.ManifestError):
                assistant_manifest.parse_manifest_contract(content)

    def test_archive_shape_and_metadata_fail_closed(self):
        valid = manifest()
        invalid_cases = (
            (archive(valid, name="other.toml"), {"name": "shimpz.assistant.toml", "size": len(valid), "mode": 0o444}),
            (
                archive(valid, member_type=tarfile.SYMTYPE),
                {"name": "shimpz.assistant.toml", "size": len(valid), "mode": 0o444},
            ),
            (archive(valid, mode=0o644), {"name": "shimpz.assistant.toml", "size": len(valid), "mode": 0o444}),
            (archive(valid), {"name": "shimpz.assistant.toml", "size": len(valid), "mode": 0o100444}),
            (archive(valid), {"name": "shimpz.assistant.toml", "size": len(valid), "mode": 0o644}),
            (archive(valid), {"name": "shimpz.assistant.toml", "size": len(valid) + 1, "mode": 0o444}),
        )
        for payload, metadata in invalid_cases:
            with self.subTest(metadata=metadata), self.assertRaises(assistant_manifest.ManifestError):
                assistant_manifest.read_container_manifest_contract(
                    type(
                        "InvalidContainer",
                        (),
                        {
                            "get_archive": lambda _self, _path, value=(payload, metadata): (
                                iter((value[0],)),
                                value[1],
                            )
                        },
                    )()
                )

    def test_cache_compares_every_reviewed_field_and_rejects_drift(self):
        content = manifest(
            allowed_hosts=("api.example.com", "cdn.example.com"),
            secrets={"api-token": ("API Token", "Token for the public API.")},
            powers={"lookup": ("api-token",)},
            connections={"social": ("x", ("tweet.read", "users.read"))},
            power_connections={"lookup": ("social",)},
        )
        container = Container("container-one", content)
        cache = assistant_manifest.ManifestContractCache(max_entries=1)

        expected = assistant_manifest.canonical_manifest_contract(
            allowed_hosts=("cdn.example.com", "api.example.com"),
            secret_declarations={"api-token": ("API Token", "Token for the public API.")},
            power_secret_refs={"lookup": ("api-token",)},
            connection_declarations={"social": ("x", ("users.read", "tweet.read"))},
            power_connection_refs={"lookup": ("social",)},
        )
        self.assertEqual(cache.get(container, expected), expected)
        self.assertEqual(cache.get(container, expected), expected)
        self.assertEqual(container.reads, 1)
        drifted = (
            assistant_manifest.canonical_manifest_contract(
                allowed_hosts=("api.example.com", "evil.example.com"),
                secret_declarations={"api-token": ("API Token", "Token for the public API.")},
                power_secret_refs={"lookup": ("api-token",)},
                connection_declarations={"social": ("x", ("tweet.read", "users.read"))},
                power_connection_refs={"lookup": ("social",)},
            ),
            assistant_manifest.canonical_manifest_contract(
                allowed_hosts=("api.example.com", "cdn.example.com"),
                secret_declarations={"api-token": ("Different Name", "Token for the public API.")},
                power_secret_refs={"lookup": ("api-token",)},
                connection_declarations={"social": ("x", ("tweet.read", "users.read"))},
                power_connection_refs={"lookup": ("social",)},
            ),
            assistant_manifest.canonical_manifest_contract(
                allowed_hosts=("api.example.com", "cdn.example.com"),
                secret_declarations={"api-token": ("API Token", "Token for the public API.")},
                power_secret_refs={"other-power": ("api-token",)},
                connection_declarations={"social": ("x", ("tweet.read", "users.read"))},
                power_connection_refs={"other-power": ("social",)},
            ),
            assistant_manifest.canonical_manifest_contract(
                allowed_hosts=("api.example.com", "cdn.example.com"),
                secret_declarations={"api-token": ("API Token", "Token for the public API.")},
                power_secret_refs={"lookup": ("api-token",)},
                connection_declarations={"social": ("other", ("tweet.read", "users.read"))},
                power_connection_refs={"lookup": ("social",)},
            ),
            assistant_manifest.canonical_manifest_contract(
                allowed_hosts=("api.example.com", "cdn.example.com"),
                secret_declarations={"api-token": ("API Token", "Token for the public API.")},
                power_secret_refs={"lookup": ("api-token",)},
            ),
        )
        for reviewed in drifted:
            with self.subTest(reviewed=reviewed), self.assertRaises(assistant_manifest.ManifestError):
                cache.get(container, reviewed)
        cache.discard(container.id)
        cache.get(container, expected)
        self.assertEqual(container.reads, 2)


if __name__ == "__main__":
    unittest.main()
