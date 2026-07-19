from __future__ import annotations

import io
import tarfile
import unittest

import assistant_genesis


def archive(
    content: bytes,
    *,
    name: str = "GENESIS.md",
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
    def __init__(self, container_id: str, content: bytes = b"# Purpose\n\nUse declared Powers safely.\n") -> None:
        self.id = container_id
        self.content = content
        self.reads = 0

    def get_archive(self, path: str):
        self.reads += 1
        if path != assistant_genesis.GENESIS_PATH:
            raise AssertionError(f"unexpected archive path: {path}")
        return (
            iter((archive(self.content)[:113], archive(self.content)[113:])),
            {
                "name": "GENESIS.md",
                "size": len(self.content),
                "mode": 0o444,
            },
        )


class AssistantGenesisTests(unittest.TestCase):
    def test_reads_and_canonicalizes_the_fixed_regular_file(self):
        container = Container("container-one", b"  # Purpose\r\n\r\nCompose declared Powers.\r\n  ")

        genesis = assistant_genesis.read_container_genesis(container)

        self.assertEqual(genesis, "# Purpose\n\nCompose declared Powers.")
        self.assertEqual(container.reads, 1)

    def test_cache_reads_once_per_container_generation(self):
        cache = assistant_genesis.GenesisCache(max_entries=2)
        first = Container("container-one")
        replacement = Container("container-two", b"Replacement generation")

        self.assertEqual(cache.get(first), cache.get(first))
        self.assertEqual(first.reads, 1)
        self.assertEqual(cache.get(replacement), "Replacement generation")
        self.assertEqual(replacement.reads, 1)

    def test_cache_is_bounded_and_discardable(self):
        cache = assistant_genesis.GenesisCache(max_entries=1)
        first = Container("container-one")
        second = Container("container-two")

        cache.get(first)
        cache.get(second)
        cache.get(first)
        self.assertEqual(first.reads, 2)
        cache.discard(first.id)
        cache.get(first)
        self.assertEqual(first.reads, 3)

    def test_invalid_text_and_size_fail_closed(self):
        invalid = (
            b"",
            b"hidden\x00directive",
            b"\xff",
            b"x" * (assistant_genesis.MAX_GENESIS_BYTES + 1),
        )
        for content in invalid:
            with self.subTest(size=len(content)), self.assertRaises(assistant_genesis.GenesisError):
                assistant_genesis.canonical_genesis(content)

    def test_archive_shape_and_metadata_fail_closed(self):
        valid = b"Safe purpose"
        invalid_cases = (
            (
                iter((archive(valid, name="other.md"),)),
                {"name": "GENESIS.md", "size": len(valid), "mode": 0o444},
            ),
            (
                iter((archive(valid, member_type=tarfile.SYMTYPE),)),
                {"name": "GENESIS.md", "size": len(valid), "mode": 0o444},
            ),
            (
                iter((archive(valid, mode=0o644),)),
                {"name": "GENESIS.md", "size": len(valid), "mode": 0o444},
            ),
            (
                iter((archive(valid),)),
                {"name": "GENESIS.md", "size": len(valid), "mode": 0o100444},
            ),
            (
                iter((archive(valid),)),
                {"name": "GENESIS.md", "size": len(valid), "mode": 0o644},
            ),
            (
                iter((archive(valid),)),
                {"name": "GENESIS.md", "size": len(valid) + 1, "mode": 0o444},
            ),
        )
        for chunks, metadata in invalid_cases:
            with self.subTest(metadata=metadata), self.assertRaises(assistant_genesis.GenesisError):
                assistant_genesis.read_container_genesis(
                    type(
                        "InvalidContainer",
                        (),
                        {"get_archive": lambda _self, _path, value=(chunks, metadata): value},
                    )()
                )

    def test_invalid_container_identity_never_reads_the_artifact(self):
        cache = assistant_genesis.GenesisCache()
        container = Container("bad/container")

        with self.assertRaises(assistant_genesis.GenesisError):
            cache.get(container)

        self.assertEqual(container.reads, 0)


if __name__ == "__main__":
    unittest.main()
