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

import network_policy

# Also bounds derived names: the per-app DB project "cap_<sha10>_<app>" stays within pg-driver's
# 58-char cap at this id length (see manifests.capsule_app_db_project).
APP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,38}[a-z0-9]$")
DIGEST_IMAGE_RE = re.compile(r"^[a-z0-9.-]+(?::[0-9]{1,5})?/[a-z0-9]+(?:[._/-][a-z0-9]+)*@sha256:[0-9a-f]{64}$")
RESERVED_APP_IDS = network_policy.RESERVED_SERVICE_ALIASES
HELLO_PULSE_IMAGE = (
    "ghcr.io/roxygens/shimpz-space@sha256:cf907cf814ebeeb8bd2d01d927583b071592405b1597d7ad04fbfdb4afd04855"
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
    port: int  # where the app answers HTTP inside the capsule's own network
    health_path: str = "/health"  # exact endpoint that must answer 200 before install commits
    db: bool = True  # provision a scoped per-(capsule, app) Postgres DB via pg-driver
    egress: tuple[str, ...] = ()  # external HTTPS hosts, reached ONLY via the token-gated app-egress-proxy
    first_party: bool = True  # False = a marketplace app → the install REQUIRES a verified Shimpz account
    archs: tuple[str, ...] = ("amd64", "arm64")  # CPU archs the image supports; an amd64-only Shimpz
    # (e.g. the Chrome browser) can't deploy onto an arm64 Capsule — mirrors the storefront's `archs`.

    required_image_labels: tuple[tuple[str, str], ...] = ()  # Proven after an exact digest get/pull.
    assistant: AssistantContract | None = None


APPS: dict[str, AppSpec] = {
    # v0 of the catalog's Notification Center (sdk/examples/notification-center): the per-Capsule
    # notifications/approvals inbox, backed by its own scoped DB, reachable inside the capsule net
    # as http://notification-center:8080.
    "notification-center": AppSpec(
        image="shimpz-marketapp-notification-center:v1",
        port=8080,
        health_path="/health",
    ),
    # First closed Rules/Powers adapter for the hosted Capsule controller. The browser supplies only
    # this ID; the controller owns the digest, runtime envelope and identity labels below.
    "hello-pulse": AppSpec(
        image=HELLO_PULSE_IMAGE,
        port=8080,
        health_path="/health",
        db=False,
        egress=(),
        first_party=True,
        required_image_labels=(
            ("org.shimpz.assistant.id", "hello-pulse"),
            ("org.shimpz.assistant.api", "1"),
        ),
        assistant=AssistantContract(
            rules=(
                "Return a friendly greeting when useful. Use only the declared hello Power. "
                "Never infer additional authority, install dependencies, access files, or send data "
                "outside the Capsule."
            ),
            rpc_command="/usr/local/bin/shimpz-assistant-rpc",
            powers={
                "hello": PowerSpec(
                    method="POST",
                    path="/v1/powers/hello",
                    summary="Return a friendly greeting for an optional name of 1 to 80 characters.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 80,
                            }
                        },
                        "additionalProperties": False,
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 256,
                            }
                        },
                        "required": ["message"],
                        "additionalProperties": False,
                    },
                    approval="none",
                )
            },
        ),
    ),
}
if RESERVED_APP_IDS & set(APPS):
    raise ValueError("marketplace App ids cannot impersonate reserved Capsule service aliases")


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
        raise MarketplaceError(f"app id {aid!r} is reserved for Capsule infrastructure")
    spec = APPS.get(aid)
    if spec is None:
        raise MarketplaceError(f"app {aid!r} is not deployable from this Space's registry")
    return aid, spec


def validate_power_input(assistant_id: str, power: str, payload: object) -> dict[str, str]:
    if assistant_id != "hello-pulse" or power != "hello":
        raise ValueError("the Power has no declared input contract")
    if not isinstance(payload, dict) or not set(payload).issubset({"name"}):
        raise ValueError("hello accepts only an optional name")
    name = payload.get("name", "Shimpz")
    if (
        not isinstance(name, str)
        or not 1 <= len(name) <= 80
        or name.strip() != name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise ValueError("name must contain 1 to 80 trimmed characters")
    return {"name": name}


def validate_power_output(assistant_id: str, power: str, payload: object) -> dict[str, str]:
    if assistant_id != "hello-pulse" or power != "hello":
        raise ValueError("the Power has no declared output contract")
    if not isinstance(payload, dict) or set(payload) != {"message"}:
        raise ValueError("the Assistant returned an invalid result")
    message = payload["message"]
    if (
        not isinstance(message, str)
        or not 1 <= len(message) <= 256
        or any(ord(character) < 32 and character not in "\t\n" for character in message)
    ):
        raise ValueError("the Assistant returned an invalid result")
    return {"message": message}
