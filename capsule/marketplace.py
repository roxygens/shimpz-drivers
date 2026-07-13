"""The marketplace registry — the TRUSTED map from a catalog app id to its deployable artifact.

The store forwards ONLY an app id; this table (baked into the socket-holding driver, never
caller-suppliable) decides what image actually runs, on which port, and with which needs. An app id
missing here is not installable — the storefront catalog may advertise more than the Space can deploy,
never the reverse. Pinned tags only (no :latest): an image change is a code change here, reviewed and
rebuilt like any other. Packaging contract: sdk/docs/build-a-shimpz-app.md ("Package for the marketplace").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Also bounds derived names: the per-app DB project "cap_<sha10>_<app>" stays within pg-driver's
# 58-char cap at this id length (see manifests.capsule_app_db_project).
APP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,38}[a-z0-9]$")


class MarketplaceError(Exception):
    """The requested app id is malformed or not in this Space's registry — nothing was touched."""


@dataclass(frozen=True)
class AppSpec:
    image: str  # the pinned artifact — an image this Space can resolve locally or pull
    port: int  # where the app answers HTTP inside the capsule's own network
    db: bool = True  # provision a scoped per-(capsule, app) Postgres DB via pg-driver
    egress: tuple[str, ...] = ()  # external HTTPS hosts, reached ONLY via the token-gated app-egress-proxy
    first_party: bool = True  # False = a marketplace app → the install REQUIRES a verified Shimpz account
    archs: tuple[str, ...] = ("amd64", "arm64")  # CPU archs the image supports; an amd64-only Shimpz
    # (e.g. the Chrome browser) can't deploy onto an arm64 Capsule — mirrors the storefront's `archs`.


APPS: dict[str, AppSpec] = {
    # v0 of the catalog's Notification Center (sdk/examples/notification-center): the per-Capsule
    # notifications/approvals inbox, backed by its own scoped DB, reachable inside the capsule net
    # as http://notification-center:8080.
    "notification-center": AppSpec(image="shimpz-marketapp-notification-center:v1", port=8080),
}


def validate_app_id(app_id: object) -> str:
    if not isinstance(app_id, str) or not APP_ID_RE.match(app_id):
        raise MarketplaceError(f"app id must match {APP_ID_RE.pattern}: {app_id!r}")
    return app_id


def resolve(app_id: object) -> tuple[str, AppSpec]:
    """(app_id, spec) for a deployable app; MarketplaceError (→ 404) for anything else."""
    aid = validate_app_id(app_id)
    spec = APPS.get(aid)
    if spec is None:
        raise MarketplaceError(f"app {aid!r} is not deployable from this Space's registry")
    return aid, spec
