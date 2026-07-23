from __future__ import annotations

import io
import json
import tarfile
import unittest
from pathlib import Path

import assistant_manifest

FIXTURE_MANIFEST = Path(__file__).resolve().parent / "fixtures" / "reference-assistant" / "shimpz.toml"


def manifest(
    *,
    allowed_hosts: tuple[str, ...] = ("api.example.com",),
    accounts: str = "",
    name: str = "Fixture Assistant",
    summary: str = "Exercise immutable admission.",
    creators: str = '["@fixture"]',
    github: str = "https://github.com/TheShimpz/fixture-assistant",
) -> bytes:
    hosts = ", ".join(f'"{host}"' for host in allowed_hosts)
    return (
        f'name = "{name}"\n'
        f'summary = "{summary}"\n'
        f"creators = {creators}\n"
        f'github = "{github}"\n'
        f"allowed_hosts = [{hosts}]\n"
        f"{accounts}"
    ).encode()


def archive(
    content: bytes,
    *,
    name: str = "shimpz.toml",
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
            {"name": "shimpz.toml", "size": len(self.content), "mode": 0o444},
        )


class ContractContainer:
    def __init__(self, container_id: str, content: bytes) -> None:
        self.id = container_id
        self.content = content
        self.reads = 0

    def get_archive(self, path: str):
        self.reads += 1
        if path != assistant_manifest.CONTRACT_PATH:
            raise AssertionError(f"unexpected archive path: {path}")
        payload = archive(self.content, name="shimpz.contract.json")
        return (
            iter((payload,)),
            {"name": "shimpz.contract.json", "size": len(self.content), "mode": 0o444},
        )


class AssistantManifestTests(unittest.TestCase):
    def test_reads_the_sdk_baked_v3_manifest_path(self) -> None:
        self.assertEqual(assistant_manifest.MANIFEST_PATH, "/opt/shimpz/shimpz.toml")

    def test_reference_fixture_matches_the_reviewed_cloudflare_security_intent(self) -> None:
        declared = assistant_manifest.parse_manifest_contract(FIXTURE_MANIFEST.read_bytes())
        reviewed_assistant = assistant_manifest.load_reviewed_catalog()["shimpz-cloudflare"]
        reviewed = assistant_manifest.reviewed_manifest_contract(
            allowed_hosts=reviewed_assistant.allowed_hosts,
            accounts={account.id: account for account in reviewed_assistant.accounts},
        )

        self.assertEqual(declared, reviewed)

    def test_reads_reduced_manifest_and_derives_provider_from_account_id(self) -> None:
        content = manifest(
            allowed_hosts=("api.cloudflare.com",),
            accounts='[accounts.cloudflare]\nscopes = ["zone.read", "dns.read", "offline_access"]\n',
        )

        contract = assistant_manifest.read_container_manifest_contract(Container("container-one", content))

        self.assertEqual(contract.allowed_hosts, ("api.cloudflare.com",))
        self.assertEqual(
            contract.accounts,
            (
                assistant_manifest.AccountDeclaration(
                    "cloudflare",
                    "cloudflare",
                    ("dns.read", "offline_access", "zone.read"),
                ),
            ),
        )

    def test_accounts_are_optional(self) -> None:
        contract = assistant_manifest.parse_manifest_contract(manifest(allowed_hosts=()))

        self.assertEqual(contract.allowed_hosts, ())
        self.assertEqual(contract.accounts, ())

    def test_v2_manifest_fields_fail_closed(self) -> None:
        obsolete = (
            b"schema_version = 2\n",
            b'[powers.lookup]\nsummary = "Lookup."\n',
            b'[secrets.token]\nname = "Token"\nsummary = "Old."\n',
            b'[accounts.cloudflare]\nprovider = "cloudflare"\nscopes = ["zone.read"]\n',
        )

        for addition in obsolete:
            with self.subTest(addition=addition), self.assertRaises(assistant_manifest.ManifestError):
                assistant_manifest.parse_manifest_contract(manifest() + addition)

    def test_unknown_provider_and_unreviewed_scopes_fail_closed(self) -> None:
        invalid = (
            '[accounts.github]\nscopes = ["repo.read"]\n',
            '[accounts.cloudflare]\nscopes = ["zone.write"]\n',
            '[accounts.cloudflare]\nscopes = ["zone.read", "zone.read"]\n',
            "[accounts.cloudflare]\nscopes = []\n",
        )

        for accounts in invalid:
            with self.subTest(accounts=accounts), self.assertRaises(assistant_manifest.ManifestError):
                assistant_manifest.parse_manifest_contract(manifest(accounts=accounts))

    def test_public_metadata_is_required_and_bounded(self) -> None:
        invalid = (
            b'name = "Only a name"\n',
            manifest(name=" Leading"),
            manifest(summary="line\nbreak"),
            manifest(creators="[]"),
            manifest(creators='["fixture"]'),
            manifest(github="http://github.com/TheShimpz/fixture"),
            manifest() + b'homepage = "https://example.com"\n',
        )

        for content in invalid:
            with self.subTest(content=content), self.assertRaises(assistant_manifest.ManifestError):
                assistant_manifest.parse_manifest_contract(content)

    def test_unsafe_hosts_fail_closed(self) -> None:
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
        )
        for host in unsafe:
            with self.subTest(host=host), self.assertRaises(assistant_manifest.ManifestError):
                assistant_manifest.parse_manifest_contract(manifest(allowed_hosts=(host,)))

    def test_invalid_text_toml_size_and_credential_material_fail_closed(self) -> None:
        invalid = (
            b"",
            manifest() + b"\x00",
            b"\xff",
            b'name = "invalid',
            b"cloudflare" * (assistant_manifest.MAX_MANIFEST_BYTES + 1),
            manifest() + b'access_token = "credential-value-123456"\n',
        )
        for content in invalid:
            with self.subTest(size=len(content)), self.assertRaises(assistant_manifest.ManifestError):
                assistant_manifest.parse_manifest_contract(content)

    def test_archive_shape_and_metadata_fail_closed(self) -> None:
        valid = manifest()
        invalid_cases = (
            (archive(valid, name="other.toml"), {"name": "shimpz.toml", "size": len(valid), "mode": 0o444}),
            (
                archive(valid, member_type=tarfile.SYMTYPE),
                {"name": "shimpz.toml", "size": len(valid), "mode": 0o444},
            ),
            (archive(valid, mode=0o644), {"name": "shimpz.toml", "size": len(valid), "mode": 0o444}),
            (archive(valid), {"name": "shimpz.toml", "size": len(valid), "mode": 0o100444}),
            (archive(valid), {"name": "shimpz.toml", "size": len(valid), "mode": 0o644}),
            (archive(valid), {"name": "shimpz.toml", "size": len(valid) + 1, "mode": 0o444}),
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

    def test_cache_compares_reviewed_hosts_and_accounts_and_rejects_drift(self) -> None:
        content = manifest(
            allowed_hosts=("api.cloudflare.com",),
            accounts='[accounts.cloudflare]\nscopes = ["dns.read", "zone.read"]\n',
        )
        container = Container("container-one", content)
        cache = assistant_manifest.ManifestContractCache(max_entries=1)
        expected = assistant_manifest.canonical_manifest_contract(
            allowed_hosts=("api.cloudflare.com",),
            account_declarations={"cloudflare": ("zone.read", "dns.read")},
        )

        self.assertEqual(cache.get(container, expected), expected)
        self.assertEqual(cache.get(container, expected), expected)
        self.assertEqual(container.reads, 1)

        drifted = (
            assistant_manifest.canonical_manifest_contract(
                allowed_hosts=("api.github.com",),
                account_declarations={"cloudflare": ("zone.read", "dns.read")},
            ),
            assistant_manifest.canonical_manifest_contract(
                allowed_hosts=("api.cloudflare.com",),
                account_declarations={"cloudflare": ("zone.read",)},
            ),
        )
        for reviewed in drifted:
            with self.subTest(reviewed=reviewed), self.assertRaises(assistant_manifest.ManifestError):
                cache.get(container, reviewed)

    def test_machine_contract_loader_accepts_reviewed_artifact_and_rejects_foreign_accounts(self) -> None:
        reviewed = assistant_manifest.load_reviewed_catalog()["shimpz-cloudflare"]
        raw = json.dumps(reviewed.machine_contract, separators=(",", ":")).encode()

        self.assertEqual(
            assistant_manifest.parse_machine_contract(raw, reviewed.accounts),
            reviewed.machine_contract,
        )

        foreign = json.loads(raw)
        foreign["powers"][0]["accounts"] = ["github"]
        with self.assertRaises(assistant_manifest.ManifestError):
            assistant_manifest.parse_machine_contract(json.dumps(foreign).encode(), reviewed.accounts)

    def test_machine_contract_loader_rejects_malformed_schema_and_oversized_artifact(self) -> None:
        reviewed = assistant_manifest.load_reviewed_catalog()["shimpz-cloudflare"]
        malformed = json.loads(json.dumps(reviewed.machine_contract))
        malformed["powers"][0]["input_schema"] = {"type": "not-a-json-schema-type"}

        for raw in (
            json.dumps(malformed).encode(),
            b'{"version":1,"version":1,"powers":[]}',
            b"x" * (assistant_manifest.MAX_CONTRACT_BYTES + 1),
        ):
            with self.subTest(size=len(raw)), self.assertRaises(assistant_manifest.ManifestError):
                assistant_manifest.parse_machine_contract(raw, reviewed.accounts)

    def test_machine_contract_loader_rejects_open_top_level_and_nested_schemas(self) -> None:
        reviewed = assistant_manifest.load_reviewed_catalog()["shimpz-cloudflare"]
        open_contracts = []
        for schema_name in ("input_schema", "output_schema"):
            contract = json.loads(json.dumps(reviewed.machine_contract))
            contract["powers"][0][schema_name].pop("additionalProperties")
            open_contracts.append((schema_name, contract))
        nested = json.loads(json.dumps(reviewed.machine_contract))
        nested["powers"][0]["output_schema"]["properties"]["pagination"].pop("additionalProperties")
        open_contracts.append(("nested output schema", nested))

        for label, contract in open_contracts:
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(
                    assistant_manifest.ManifestError,
                    "must close every object",
                ),
            ):
                assistant_manifest.parse_machine_contract(json.dumps(contract).encode(), reviewed.accounts)

    def test_machine_contract_cache_reads_once_and_requires_exact_review(self) -> None:
        reviewed = assistant_manifest.load_reviewed_catalog()["shimpz-cloudflare"]
        raw = json.dumps(reviewed.machine_contract, separators=(",", ":")).encode()
        container = ContractContainer("machine-generation", raw)
        cache = assistant_manifest.MachineContractCache()

        self.assertEqual(
            cache.get(container, reviewed.accounts, reviewed.machine_contract),
            reviewed.machine_contract,
        )
        self.assertEqual(
            cache.get(container, reviewed.accounts, reviewed.machine_contract),
            reviewed.machine_contract,
        )
        self.assertEqual(container.reads, 1)

        drifted = json.loads(raw)
        drifted["powers"][0]["path"] = "/v1/powers/other"
        with self.assertRaises(assistant_manifest.ManifestError):
            cache.get(container, reviewed.accounts, drifted)


if __name__ == "__main__":
    unittest.main()
