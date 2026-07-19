"""The marketplace registry — the TRUSTED map from a catalog app id to its deployable artifact.

The store forwards ONLY an app id; this table (baked into the socket-holding driver, never
caller-suppliable) decides what image actually runs, on which port, and with which needs. An app id
missing here is not installable — the storefront catalog may advertise more than the Space can deploy,
never the reverse. Every image is a reviewed pinned tag or digest: an artifact change is a code change
here, rebuilt like any other. Packaging contract: sdk/docs/build-a-shimpz-app.md ("Package for the marketplace").
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import assistant_contract
import network_policy

# Also bounds derived names: the per-app DB project "team_<sha10>_<app>" stays within pg-driver's
# 58-char cap at this id length (see manifests.team_app_db_project).
APP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,38}[a-z0-9]$")
DIGEST_IMAGE_RE = re.compile(r"^[a-z0-9.-]+(?::[0-9]{1,5})?/[a-z0-9]+(?:[._/-][a-z0-9]+)*@sha256:[0-9a-f]{64}$")
RESERVED_APP_IDS = network_policy.RESERVED_SERVICE_ALIASES
SHIMPZ_ASSISTANT_IMAGE = (
    "ghcr.io/roxygens/shimpz-space@sha256:c68a7d055c8dbe0cf350d00975caa259741dec7a072ca30efa077a688e5b41b4"
)


class MarketplaceError(Exception):
    """The requested app id is malformed or not in this Space's registry — nothing was touched."""


@dataclass(frozen=True, slots=True)
class PowerSpec:
    method: str
    path: str
    summary: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    approval: Literal["none", "once", "each-run"] = "none"


@dataclass(frozen=True, slots=True)
class AssistantContract:
    rules: str
    rpc_command: str
    powers: dict[str, PowerSpec]


@dataclass(frozen=True)
class AppSpec:
    image: str  # the pinned artifact — an image this Space can resolve locally or pull
    port: int  # where the app answers HTTP inside the team's own network
    health_path: str = "/health"  # exact endpoint that must answer 200 before install commits
    db: bool = True  # provision a scoped per-(team, app) Postgres DB via pg-driver
    egress: tuple[str, ...] = ()  # external HTTPS hosts, reached ONLY via the token-gated app-egress-proxy
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
    # First closed Rules/Powers adapter for the hosted Team controller. The browser supplies only
    # this ID; the controller owns the digest, runtime envelope and identity labels below.
    assistant_contract.ASSISTANT_ID: AppSpec(
        image=SHIMPZ_ASSISTANT_IMAGE,
        port=8080,
        health_path="/health",
        db=False,
        egress=assistant_contract.ASSISTANT_EGRESS,
        first_party=True,
        required_image_labels=(
            ("org.shimpz.assistant.id", assistant_contract.ASSISTANT_ID),
            ("org.shimpz.assistant.api", "1"),
        ),
        assistant=AssistantContract(
            rules=assistant_contract.ASSISTANT_RULES,
            rpc_command=assistant_contract.ASSISTANT_RPC_COMMAND,
            powers={
                power_id: PowerSpec(**contract) for power_id, contract in assistant_contract.power_contracts().items()
            },
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
    return assistant_contract.validate_power_input(assistant_id, power, payload)


def validate_power_output(assistant_id: str, power: str, payload: object) -> dict[str, object]:
    return assistant_contract.validate_power_output(assistant_id, power, payload)
