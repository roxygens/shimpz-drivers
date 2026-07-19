from __future__ import annotations

import io
import tarfile
import unittest

import assistant_manifest


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
    def test_reads_and_canonicalizes_exact_hosts(self):
        content = b'allowed_hosts = ["geocoding-api.open-meteo.com", "api.open-meteo.com"]\n'

        hosts = assistant_manifest.read_container_allowed_hosts(Container("container-one", content))

        self.assertEqual(hosts, ("api.open-meteo.com", "geocoding-api.open-meteo.com"))

    def test_empty_list_is_valid_and_network_intent_is_required(self):
        self.assertEqual(assistant_manifest.parse_allowed_hosts(b"allowed_hosts = []\n"), ())
        for content in (b'name = "No intent"\n', b"allowed_hosts = 1\n"):
            with self.subTest(content=content), self.assertRaises(assistant_manifest.ManifestError):
                assistant_manifest.parse_allowed_hosts(content)

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
            b'allowed_hosts = ["example.com"]\x00',
            b"\xff",
            b'allowed_hosts = ["example.com"',
            b"x" * (assistant_manifest.MAX_MANIFEST_BYTES + 1),
        )
        for content in invalid:
            with self.subTest(size=len(content)), self.assertRaises(assistant_manifest.ManifestError):
                assistant_manifest.parse_allowed_hosts(content)

    def test_archive_shape_and_metadata_fail_closed(self):
        valid = b'allowed_hosts = ["api.example.com"]\n'
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
                assistant_manifest.read_container_allowed_hosts(
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

    def test_cache_compares_canonical_membership_and_rejects_drift(self):
        content = b'allowed_hosts = ["api.example.com", "cdn.example.com"]\n'
        container = Container("container-one", content)
        cache = assistant_manifest.AllowedHostsCache(max_entries=1)

        expected = ("api.example.com", "cdn.example.com")
        self.assertEqual(cache.get(container, ("cdn.example.com", "api.example.com")), expected)
        self.assertEqual(cache.get(container, ("api.example.com", "cdn.example.com")), expected)
        self.assertEqual(container.reads, 1)
        with self.assertRaises(assistant_manifest.ManifestError):
            cache.get(container, ("api.example.com", "evil.example.com"))
        cache.discard(container.id)
        cache.get(container, ("api.example.com", "cdn.example.com"))
        self.assertEqual(container.reads, 2)


if __name__ == "__main__":
    unittest.main()
