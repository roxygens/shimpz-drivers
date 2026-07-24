"""The marketplace registry — the TRUSTED map from a catalog app id to its deployable artifact.

The store forwards ONLY an app id; this table (baked into the socket-holding driver, never
caller-suppliable) decides what image actually runs, on which port, and with which needs. An app id
missing here is not installable — the storefront catalog may advertise more than the Space can deploy,
never the reverse. Every image is a reviewed pinned tag or digest: an artifact change is a code change
here, rebuilt like any other. Packaging follows the reviewed Assistant manifest contract.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import assistant_manifest
from container_policy import network as network_policy

# Also bounds derived names: the per-app DB project "team_<sha10>_<app>" stays within pg-driver's
# 58-char cap at this id length (see manifests.team_app_db_project).
APP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,38}[a-z0-9]$")
DIGEST_IMAGE_RE = re.compile(r"^[a-z0-9.-]+(?::[0-9]{1,5})?/[a-z0-9]+(?:[._/-][a-z0-9]+)*@sha256:[0-9a-f]{64}$")
RESERVED_APP_IDS = network_policy.RESERVED_SERVICE_ALIASES
SHIMPZ_CLOUDFLARE_ASSISTANT_IMAGE = (
    "ghcr.io/theshimpz/shimpz-space@sha256:39d19e65fc0e3f36b0fccd8dc5eb1c60ee84ead7c3e9e84558fe428af038ef18"
)
_REVIEWED_ASSISTANTS = assistant_manifest.load_reviewed_catalog()
_CLOUDFLARE = _REVIEWED_ASSISTANTS["shimpz-cloudflare"]


class MarketplaceError(Exception):
    """The requested app id is malformed or not in this Space's registry — nothing was touched."""


@dataclass(frozen=True, slots=True)
class PowerSpec:
    method: str
    path: str
    summary: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    secrets: tuple[str, ...] = ()
    accounts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SecretSpec:
    name: str
    summary: str


@dataclass(frozen=True, slots=True)
class AccountSpec:
    provider: str
    scopes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AssistantContract:
    rpc_command: str
    powers: dict[str, PowerSpec]
    secrets: dict[str, SecretSpec]
    accounts: dict[str, AccountSpec] = field(default_factory=dict)
    machine_contract: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AppSpec:
    image: str  # the pinned artifact — an image this Space can resolve locally or pull
    port: int  # where the app answers HTTP inside the team's own network
    health_path: str = "/health"  # exact endpoint that must answer 200 before install commits
    db: bool = True  # provision a scoped per-(team, app) Postgres DB via pg-driver
    allowed_hosts: tuple[str, ...] = ()  # reviewed maximum; packaged intent must match before proxy admission
    first_party: bool = True  # False = a marketplace app → the install REQUIRES a verified Shimpz account
    archs: tuple[str, ...] = ("amd64", "arm64")  # CPU archs the image supports; an amd64-only Shimpz
    # (e.g. the Chrome browser) can't deploy onto an arm64 Team — mirrors the storefront's `archs`.

    required_image_labels: tuple[tuple[str, str], ...] = ()  # Proven after an exact digest get/pull.
    assistant: AssistantContract | None = None


APPS: dict[str, AppSpec] = {
    # v0 of the catalog's Notification Center (sdk/examples/notification-center): the per-Team
    # notifications/approvals inbox, backed by its own scoped DB, reachable inside the team net
    # as http://notification-center:8080.
    "notification-center": AppSpec(
        image="shimpz-marketapp-notification-center:v1",
        port=8080,
        health_path="/health",
    ),
    # First closed Genesis/Powers adapter for the hosted Team controller. The browser supplies only
    # this ID; the controller owns the digest, runtime envelope and identity labels below.
    _CLOUDFLARE.assistant_id: AppSpec(
        image=SHIMPZ_CLOUDFLARE_ASSISTANT_IMAGE,
        port=8080,
        health_path=_CLOUDFLARE.health_path,
        db=False,
        allowed_hosts=_CLOUDFLARE.allowed_hosts,
        first_party=True,
        required_image_labels=(
            ("org.shimpz.assistant.id", _CLOUDFLARE.assistant_id),
            ("org.shimpz.assistant.api", "1"),
        ),
        assistant=AssistantContract(
            rpc_command=_CLOUDFLARE.rpc_command,
            powers={
                power_id: PowerSpec(
                    method=power["method"],
                    path=power["path"],
                    summary=power_id.replace("-", " ").capitalize(),
                    input_schema=power["input_schema"],
                    output_schema=power["output_schema"],
                    secrets=(),
                    accounts=tuple(power["accounts"]),
                )
                for power_id, power in _CLOUDFLARE.powers.items()
            },
            secrets={},
            accounts={
                account.id: AccountSpec(provider=account.provider, scopes=account.scopes)
                for account in _CLOUDFLARE.accounts
            },
            machine_contract=_CLOUDFLARE.machine_contract,
        ),
    ),
}
if RESERVED_APP_IDS & set(APPS):
    raise ValueError("marketplace App ids cannot impersonate reserved Team service aliases")


def health_response_ok(status: object) -> bool:
    """Only the registry-declared health endpoint's exact success contract commits an install."""
    return isinstance(status, int) and not isinstance(status, bool) and status == 200


def is_digest_image(image: object) -> bool:
    """True only for a complete registry/repository OCI sha256 reference."""
    return isinstance(image, str) and DIGEST_IMAGE_RE.fullmatch(image) is not None


def validate_app_id(app_id: object) -> str:
    if not isinstance(app_id, str) or not APP_ID_RE.match(app_id):
        raise MarketplaceError(f"app id must match {APP_ID_RE.pattern}: {app_id!r}")
    return app_id


def resolve(app_id: object) -> tuple[str, AppSpec]:
    """(app_id, spec) for a deployable app; MarketplaceError (→ 404) for anything else."""
    aid = validate_app_id(app_id)
    if aid in RESERVED_APP_IDS:
        raise MarketplaceError(f"app id {aid!r} is reserved for Team infrastructure")
    spec = APPS.get(aid)
    if spec is None:
        raise MarketplaceError(f"app {aid!r} is not deployable from this Space's registry")
    return aid, spec


def validate_power_input(assistant_id: str, power: str, payload: object) -> dict[str, object]:
    try:
        validator = _REVIEWED_ASSISTANTS[assistant_id].power_validators[power]["input"]
    except KeyError as exc:
        raise ValueError("the Power has no declared input contract") from exc
    return assistant_manifest.validate_schema_payload(validator, payload)


def validate_power_output(assistant_id: str, power: str, payload: object) -> dict[str, object]:
    try:
        validator = _REVIEWED_ASSISTANTS[assistant_id].power_validators[power]["output"]
    except KeyError as exc:
        raise ValueError("the Power has no declared output contract") from exc
    return assistant_manifest.validate_schema_payload(validator, payload)
