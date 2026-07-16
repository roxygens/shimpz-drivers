"""The marketplace registry — the TRUSTED map from a catalog app id to its deployable artifact.

The store forwards ONLY an app id; this table (baked into the socket-holding driver, never
caller-suppliable) decides what image actually runs, on which port, and with which needs. An app id
missing here is not installable — the storefront catalog may advertise more than the Space can deploy,
never the reverse. Every image is a reviewed pinned tag or digest: an artifact change is a code change
here, rebuilt like any other. Packaging contract: sdk/docs/build-a-shimpz-app.md ("Package for the marketplace").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import network_policy

# Also bounds derived names: the per-app DB project "cap_<sha10>_<app>" stays within pg-driver's
# 58-char cap at this id length (see manifests.capsule_app_db_project).
APP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,38}[a-z0-9]$")
DIGEST_IMAGE_RE = re.compile(r"^[a-z0-9.-]+(?::[0-9]{1,5})?/[a-z0-9]+(?:[._/-][a-z0-9]+)*@sha256:[0-9a-f]{64}$")
RESERVED_APP_IDS = network_policy.RESERVED_SERVICE_ALIASES
HELLO_PULSE_IMAGE = (
    "ghcr.io/roxygens/shimpz-space@sha256:2b051d2db0b83ec4689901033f8ca38459265a38c7a3e84f53271a3f786471b5"
)


class MarketplaceError(Exception):
    """The requested app id is malformed or not in this Space's registry — nothing was touched."""


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


APPS: dict[str, AppSpec] = {
    # v0 of the catalog's Notification Center (sdk/examples/notification-center): the per-Capsule
    # notifications/approvals inbox, backed by its own scoped DB, reachable inside the capsule net
    # as http://notification-center:8080.
    "notification-center": AppSpec(
        image="shimpz-marketapp-notification-center:v1",
        port=8080,
        health_path="/health",
    ),
    # First Assistant Spec v1 adapter for the hosted Capsule controller. The browser supplies only this
    # ID; the controller owns the digest, runtime envelope and identity labels below.
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
