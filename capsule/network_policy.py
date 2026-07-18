"""Pure Capsule network identity, membership, and workload-posture policy.

This module deliberately has no Docker SDK dependency.  The lifecycle driver applies it to SDK
inspect dictionaries and the healthcheck applies it to raw Engine API dictionaries, so admission and
continuous readiness cannot drift into two different definitions of an isolated Capsule.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation

SUFFIX = os.environ.get("SHIMPZ_SUFFIX", "")
CAPSULE_PREFIX = f"capsule{SUFFIX}_"
CORE_NETWORK_PREFIX = f"net_capsule{SUFFIX}_"
APP_WORKLOAD_DELIMITER = ".app."
DOCKER_RESOURCE_NAME_MAX = 255
DOCKER_NETWORK_NAME_MAX = DOCKER_RESOURCE_NAME_MAX
TMPFS_MOUNT_PATH = f"{os.sep}tmp"

CORE_KIND = "core"
NETWORK_KINDS = frozenset({CORE_KIND})

# The hostile-tenant runtime is a name-to-binary binding, not merely a Docker registry key. Keep the
# path in the SDK-free policy module so lifecycle admission and the raw-Engine healthcheck evaluate
# the same exact registration after the host install proof has completed.
CAPSULE_RUNTIME_PATH = "/usr/local/bin/runsc"

NETWORK_MANAGED_LABEL = "shimpz.capsule.network"
NETWORK_CID_LABEL = "shimpz.capsule.cid"
NETWORK_KIND_LABEL = "shimpz.capsule.network.kind"
VOLUME_MANAGED_LABEL = "shimpz.capsule.volume"
VOLUME_CID_LABEL = "shimpz.capsule.cid"
VOLUME_KIND_LABEL = "shimpz.capsule.volume.kind"

CONFIG_VOLUME_KIND = "config"
WORKSPACE_VOLUME_KIND = "workspace"
VOLUME_KINDS = frozenset({CONFIG_VOLUME_KIND, WORKSPACE_VOLUME_KIND})

POSTGRES_CONTAINER = os.environ.get("SHIMPZ_POSTGRES_CONTAINER", f"shimpz-postgres{SUFFIX}")
APP_EGRESS_CONTAINER = os.environ.get("SHIMPZ_APP_EGRESS_PROXY_CONTAINER", f"app-egress-proxy{SUFFIX}")

APP_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,38}[a-z0-9])?$")
EXPECTED_BRAIN_CAP_ADD = frozenset()
SHARED_MANAGED_LABEL = "shimpz.capsule.shared"
SHARED_ROLE_LABEL = "shimpz.capsule.shared.role"
POSTGRES_ROLE = "postgres"
APP_EGRESS_ROLE = "app-egress"
SHARED_ROLES = frozenset({POSTGRES_ROLE, APP_EGRESS_ROLE})
RESERVED_SERVICE_ALIASES = frozenset({"postgres", "app-egress-proxy"})


def _normalized_capabilities(values: object) -> set[str]:
    """Normalize Engine inspect capability names across API versions."""
    if not isinstance(values, list):
        return set()
    return {str(value).upper().removeprefix("CAP_") for value in values}


def _memory_bytes(value: str, setting: str) -> int:
    match = re.fullmatch(
        r"(?P<number>[0-9]+(?:\.[0-9]+)?)(?P<unit>[kmgtp]?)(?:i?b)?",
        value.strip(),
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"{setting} must be a positive Docker memory size")
    try:
        parsed = Decimal(match.group("number")) * Decimal(1024 ** "bkmgtp".index(match.group("unit").lower() or "b"))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{setting} must be a positive Docker memory size") from exc
    if parsed <= 0 or parsed != parsed.to_integral_value():
        raise ValueError(f"{setting} must be a positive Docker memory size")
    return int(parsed)


BRAIN_MEMORY_BYTES = _memory_bytes(os.environ.get("SHIMPZ_CAPSULE_MEM_LIMIT", "64m"), "SHIMPZ_CAPSULE_MEM_LIMIT")
BRAIN_MEMORY_RESERVATION_BYTES = _memory_bytes(
    os.environ.get("SHIMPZ_CAPSULE_MEM_RESERVATION", "16m"),
    "SHIMPZ_CAPSULE_MEM_RESERVATION",
)
BRAIN_NANO_CPUS = int(os.environ.get("SHIMPZ_CAPSULE_NANO_CPUS", str(100_000_000)))
BRAIN_PIDS_LIMIT = int(os.environ.get("SHIMPZ_CAPSULE_PIDS_LIMIT", "128"))
APP_MEMORY_BYTES = _memory_bytes(os.environ.get("SHIMPZ_CAPSULE_APP_MEM_LIMIT", "1g"), "SHIMPZ_CAPSULE_APP_MEM_LIMIT")
APP_NANO_CPUS = int(os.environ.get("SHIMPZ_CAPSULE_APP_NANO_CPUS", str(500_000_000)))
APP_PIDS_LIMIT = int(os.environ.get("SHIMPZ_CAPSULE_APP_PIDS_LIMIT", "256"))
CAP_LOG_MAX_SIZE = "5m"
CAP_LOG_MAX_FILE = "2"
if min(BRAIN_NANO_CPUS, BRAIN_PIDS_LIMIT, APP_NANO_CPUS, APP_PIDS_LIMIT) < 1:
    raise ValueError("Capsule CPU and PID limits must be positive")


def _bounded_resource_name(name: str, resource: str) -> str:
    if len(name.encode()) > DOCKER_RESOURCE_NAME_MAX:
        raise ValueError(f"derived Capsule {resource} name exceeds Docker's 255-byte resource-name limit")
    return name


def capsule_container_name(cid: str) -> str:
    return _bounded_resource_name(f"{CAPSULE_PREFIX}{cid}", "Brain container")


def capsule_app_container_name(cid: str, app_id: str) -> str:
    if APP_ID_RE.fullmatch(app_id) is None:
        raise ValueError(f"invalid Capsule App id: {app_id!r}")
    # Dot is outside the CID alphabet, so no valid Brain CID can impersonate an App workload. Keeping
    # the App id verbatim also makes (CID, app id) -> Docker name injective.
    name = f"{CAPSULE_PREFIX}{cid}{APP_WORKLOAD_DELIMITER}{app_id}"
    return _bounded_resource_name(name, "App container")


def volume_name(cid: str, kind: str) -> str:
    if kind == CONFIG_VOLUME_KIND:
        name = f"{CAPSULE_PREFIX}{cid}_config"
    elif kind == WORKSPACE_VOLUME_KIND:
        name = f"{CAPSULE_PREFIX}{cid}_workspace"
    else:
        raise ValueError(f"unknown Capsule volume kind: {kind!r}")
    return _bounded_resource_name(name, "volume")


def volume_labels(cid: str, kind: str) -> dict[str, str]:
    if kind not in VOLUME_KINDS:
        raise ValueError(f"unknown Capsule volume kind: {kind!r}")
    return {
        VOLUME_MANAGED_LABEL: "1",
        VOLUME_CID_LABEL: cid,
        VOLUME_KIND_LABEL: kind,
    }


def volume_identity_valid(metadata: Mapping, cid: str, kind: str) -> bool:
    """Require an exact locally managed per-Capsule volume before reuse or deletion."""
    labels = _mapping(metadata.get("Labels"))
    options = metadata.get("Options")
    expected_labels = volume_labels(cid, kind)
    return (
        metadata.get("Name") == volume_name(cid, kind)
        and metadata.get("Driver") == "local"
        and metadata.get("Scope") == "local"
        # The driver creates ordinary Docker-local volumes with no driver options. A same-name,
        # correctly labeled local volume backed by a host bind/NFS device must never be mounted into
        # tenant code or claimed during cleanup.
        and (options is None or (isinstance(options, Mapping) and not options))
        and all(labels.get(key) == value for key, value in expected_labels.items())
    )


def network_name(cid: str, kind: str) -> str:
    if kind != CORE_KIND:
        raise ValueError(f"unknown Capsule network kind: {kind!r}")
    return _bounded_resource_name(f"{CORE_NETWORK_PREFIX}{cid}", "network")


def network_labels(cid: str, kind: str) -> dict[str, str]:
    if kind not in NETWORK_KINDS:
        raise ValueError(f"unknown Capsule network kind: {kind!r}")
    return {
        NETWORK_MANAGED_LABEL: "1",
        NETWORK_CID_LABEL: cid,
        NETWORK_KIND_LABEL: kind,
    }


def _mapping(value: object) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _container_name(metadata: Mapping) -> str:
    name = metadata.get("Name")
    if isinstance(name, str) and name:
        return name.removeprefix("/")
    names = metadata.get("Names")
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], str):
        return names[0].removeprefix("/")
    return ""


def _container_labels(metadata: Mapping) -> Mapping:
    config_labels = _mapping(_mapping(metadata.get("Config")).get("Labels"))
    return config_labels or _mapping(metadata.get("Labels"))


def _container_id(metadata: Mapping) -> str:
    container_id = metadata.get("Id")
    return container_id if isinstance(container_id, str) else ""


def network_identity_valid(metadata: Mapping, cid: str, kind: str) -> bool:
    """Require a locally managed internal bridge with an unambiguous Capsule identity."""
    labels = _mapping(metadata.get("Labels"))
    expected_labels = network_labels(cid, kind)
    return (
        metadata.get("Name") == network_name(cid, kind)
        and metadata.get("Driver") == "bridge"
        and metadata.get("Scope") == "local"
        and metadata.get("Internal") is True
        and not bool(metadata.get("Attachable"))
        and not bool(metadata.get("Ingress"))
        and not bool(metadata.get("ConfigOnly"))
        and all(labels.get(key) == value for key, value in expected_labels.items())
    )


def _workload_role(metadata: Mapping, cid: str) -> tuple[str, str] | None:
    labels = _container_labels(metadata)
    name = _container_name(metadata)
    if labels.get("capsule.driver") == "1" and labels.get("capsule.id") == cid and name == capsule_container_name(cid):
        return "brain", ""
    app_id = labels.get("capsule.app")
    if (
        labels.get("capsule.app.driver") == "1"
        and labels.get("capsule.id") == cid
        and isinstance(app_id, str)
        and APP_ID_RE.fullmatch(app_id) is not None
        and app_id not in RESERVED_SERVICE_ALIASES
        and name == capsule_app_container_name(cid, app_id)
    ):
        return "app", app_id
    return None


def brain_identity_valid(metadata: Mapping, cid: str) -> bool:
    """Require both the deterministic Docker name and exact Brain ownership labels."""
    return _workload_role(metadata, cid) == ("brain", "")


def app_identity_valid(metadata: Mapping, cid: str, app_id: str) -> bool:
    """Require both the injective Docker name and exact App ownership labels."""
    return _workload_role(metadata, cid) == ("app", app_id)


def workload_network_kinds(metadata: Mapping, cid: str) -> frozenset[str] | None:
    """Return the single core network this trusted workload identity must join."""
    role = _workload_role(metadata, cid)
    if role is None:
        return None
    return frozenset({CORE_KIND})


def workload_live_membership_valid(network: Mapping, metadata: Mapping, cid: str, kind: str) -> bool:
    """Require this exact running workload ID in one of its expected network inventories."""
    container_id = _container_id(metadata)
    members = _mapping(network.get("Containers"))
    return workload_endpoint_valid(network, metadata, cid, kind) and bool(container_id) and container_id in members


def shared_service_labels(role: str) -> dict[str, str]:
    if role not in SHARED_ROLES:
        raise ValueError(f"unknown shared Capsule service role: {role!r}")
    return {SHARED_MANAGED_LABEL: "1", SHARED_ROLE_LABEL: role}


def _shared_role_for_name(name: str) -> str | None:
    return {
        POSTGRES_CONTAINER: POSTGRES_ROLE,
        APP_EGRESS_CONTAINER: APP_EGRESS_ROLE,
    }.get(name)


def shared_service_identity_valid(metadata: Mapping, expected_role: str | None = None) -> bool:
    """Bind a shared plane member to its exact configured name and release-owned role labels."""
    name = _container_name(metadata)
    role = _shared_role_for_name(name)
    if role is None or (expected_role is not None and role != expected_role):
        return False
    labels = _container_labels(metadata)
    expected = shared_service_labels(role)
    return all(labels.get(key) == value for key, value in expected.items())


def shared_service_role_for_name(name: str) -> str | None:
    """Return the configured role for one exact suffix-aware shared service name."""
    return _shared_role_for_name(name)


def _member_role(metadata: Mapping, cid: str, kind: str) -> tuple[str, str] | None:
    if kind != CORE_KIND:
        return None
    name = _container_name(metadata)
    if name == POSTGRES_CONTAINER and shared_service_identity_valid(metadata, POSTGRES_ROLE):
        return POSTGRES_ROLE, ""
    if name == APP_EGRESS_CONTAINER and shared_service_identity_valid(metadata, APP_EGRESS_ROLE):
        return APP_EGRESS_ROLE, ""
    return _workload_role(metadata, cid)


def network_member_managed(metadata: Mapping, cid: str, kind: str) -> bool:
    """Whether teardown may disconnect this exact member from this Capsule network.

    Cleanup removes only exact configured shared services or CID-bound Capsule workloads. Unknown
    identities are never disconnected.
    """
    if kind != CORE_KIND:
        return False
    if shared_service_identity_valid(metadata):
        return True
    return _workload_role(metadata, cid) is not None


def _required_aliases(role: tuple[str, str]) -> frozenset[str]:
    name, value = role
    if name == "postgres":
        return frozenset({"postgres"})
    if name == "app-egress":
        return frozenset({"app-egress-proxy"})
    if name == "app":
        return frozenset({value, f"{value}.capsule"})
    return frozenset()


def _network_endpoint(metadata: Mapping, name: str) -> Mapping:
    settings = _mapping(metadata.get("NetworkSettings"))
    return _mapping(_mapping(settings.get("Networks")).get(name))


def _automatic_aliases(metadata: Mapping) -> frozenset[str]:
    """Aliases Docker may add independently of the requested network-scoped aliases."""
    container_id = _container_id(metadata)
    config = _mapping(metadata.get("Config"))
    candidates = {
        _container_name(metadata),
        container_id,
        container_id[:12] if container_id else "",
        config.get("Hostname"),
    }
    return frozenset(value for value in candidates if isinstance(value, str) and value)


def _normalized_aliases(values: object) -> frozenset[str] | None:
    """Normalize Engine's null spelling for no submitted aliases."""
    if values is None:
        return frozenset()
    if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
        return None
    return frozenset(values)


def _aliases_valid(metadata: Mapping, role: tuple[str, str], endpoint: Mapping) -> bool:
    actual = _normalized_aliases(endpoint.get("Aliases"))
    if actual is None:
        return False
    required = _required_aliases(role)
    automatic = _automatic_aliases(metadata)
    # An automatic name is still security-relevant DNS. In particular, a Brain whose hostname/CID is
    # ``postgres`` must not become a second claimant for the database role merely because Docker added
    # that hostname itself.
    if (automatic & RESERVED_SERVICE_ALIASES) - required:
        return False
    allowed = required | automatic
    if not (required <= actual and actual <= allowed and not ((actual & RESERVED_SERVICE_ALIASES) - required)):
        return False
    # API 1.45+ exposes all effective embedded-DNS names separately. Older Engines folded automatic
    # names into Aliases, so accept the field being absent/null while applying the same exact allowlist
    # whenever it is reported.
    dns_names = _normalized_aliases(endpoint.get("DNSNames"))
    return dns_names is not None and dns_names <= allowed and not ((dns_names & RESERVED_SERVICE_ALIASES) - required)


def _endpoint_network_binding_valid(network: Mapping, metadata: Mapping, endpoint: Mapping) -> bool:
    """Accept an exact live binding or Engine 29's strictly empty stopped-endpoint placeholder.

    Engine 29 retains the requested network-name entries in container inspect for created/stopped
    containers, but clears their NetworkID/EndpointID/address material until start.  That placeholder
    is useful pre-start proof only when the workload is explicitly not running, the network inventory
    also omits its ID, and every field that could encode a different attachment is empty.  The caller
    still proves the independently inspected network identity; post-start admission requires the full
    ID binding and live membership, so this allowance can never make a running workload admissible.
    """
    network_id = network.get("Id")
    if endpoint.get("NetworkID") == network_id:
        return isinstance(network_id, str) and bool(network_id)
    state = _mapping(metadata.get("State"))
    members = _mapping(network.get("Containers"))
    empty_or_missing = (None, {})
    known_fields = {
        "IPAMConfig",
        "Links",
        "Aliases",
        "DriverOpts",
        "GwPriority",
        "NetworkID",
        "EndpointID",
        "Gateway",
        "IPAddress",
        "MacAddress",
        "IPPrefixLen",
        "IPv6Gateway",
        "GlobalIPv6Address",
        "GlobalIPv6PrefixLen",
        "DNSNames",
    }
    return (
        isinstance(network_id, str)
        and bool(network_id)
        and set(endpoint) <= known_fields
        and state.get("Running") is False
        and _container_id(metadata) not in members
        and endpoint.get("NetworkID") == ""
        and endpoint.get("EndpointID") == ""
        and endpoint.get("Gateway") == ""
        and endpoint.get("IPAddress") == ""
        and endpoint.get("MacAddress") == ""
        and endpoint.get("IPPrefixLen") == 0
        and endpoint.get("IPv6Gateway") == ""
        and endpoint.get("GlobalIPv6Address") == ""
        and endpoint.get("GlobalIPv6PrefixLen") == 0
        and endpoint.get("IPAMConfig") in empty_or_missing
        and endpoint.get("Links") is None
        and endpoint.get("DriverOpts") in empty_or_missing
        and endpoint.get("GwPriority", 0) == 0
    )


def workload_endpoint_valid(network: Mapping, metadata: Mapping, cid: str, kind: str) -> bool:
    """Bind one workload's container-inspect endpoint to the exact plane and role aliases.

    Engine may omit created/stopped workloads from ``network inspect Containers``. Their retained
    container-inspect endpoint is still mandatory static proof and must not inherit the live-member
    inventory's omission allowance.
    """
    role = _workload_role(metadata, cid)
    expected_kinds = workload_network_kinds(metadata, cid)
    network_id = network.get("Id")
    endpoint = _network_endpoint(metadata, network_name(cid, kind))
    return (
        role is not None
        and expected_kinds is not None
        and kind in expected_kinds
        and network_identity_valid(network, cid, kind)
        and isinstance(network_id, str)
        and bool(network_id)
        and _endpoint_network_binding_valid(network, metadata, endpoint)
        and _aliases_valid(metadata, role, endpoint)
    )


def network_members_valid(
    network: Mapping,
    containers: Mapping[str, Mapping],
    cid: str,
    kind: str,
    *,
    require_brain: bool,
    require_dependencies: bool,
) -> bool:
    """Reject foreign, wrong-plane, duplicate, or alias-less network members."""
    if not network_identity_valid(network, cid, kind):
        return False
    members = _mapping(network.get("Containers"))
    network_id = network.get("Id")
    if not isinstance(network_id, str) or not network_id:
        return False
    seen: dict[str, int] = {}
    for member_id in members:
        metadata = containers.get(member_id)
        if not isinstance(metadata, Mapping) or _container_id(metadata) != member_id:
            return False
        role = _member_role(metadata, cid, kind)
        if role is None:
            return False
        role_name = role[0]
        seen[role_name] = seen.get(role_name, 0) + 1
        if role_name != "app" and seen[role_name] != 1:
            return False
        endpoint = _network_endpoint(metadata, network_name(cid, kind))
        if endpoint.get("NetworkID") != network_id:
            return False
        if not _aliases_valid(metadata, role, endpoint):
            return False
    if require_brain and seen.get("brain") != 1:
        return False
    return not require_dependencies or seen.get("postgres") == 1


def _security_options_valid(host_config: Mapping) -> bool:
    options = host_config.get("SecurityOpt")
    if not isinstance(options, list):
        return False
    normalized = {str(option) for option in options}
    nnp = {"no-new-privileges", "no-new-privileges:true"}
    # The manifests do not select a seccomp override, so the daemon's required built-in profile applies.
    # Accept Docker exposing that choice explicitly, but reject every custom/unconfined profile and an
    # apparent ``no-new-privileges:false`` drift instead of relying on substring checks.
    allowed = nnp | {"apparmor=docker-default", "seccomp=builtin"}
    return bool(normalized & nnp) and "apparmor=docker-default" in normalized and normalized <= allowed


def _tmpfs_valid(host_config: Mapping, *, size: int) -> bool:
    tmpfs = host_config.get("Tmpfs")
    if not isinstance(tmpfs, Mapping) or set(tmpfs) != {TMPFS_MOUNT_PATH}:
        return False
    raw_options = tmpfs.get(TMPFS_MOUNT_PATH)
    if not isinstance(raw_options, str):
        return False
    parsed: dict[str, str] = {}
    for option in raw_options.split(","):
        key, separator, value = option.strip().partition("=")
        if not separator or key in parsed or key not in {"size", "mode"} or not value:
            return False
        parsed[key] = value
    try:
        actual_size = _memory_bytes(parsed.get("size", ""), "tmpfs size")
    except ValueError:
        return False
    # Docker's default tmpfs mode is 01777; Engine inspect may omit that default or preserve an
    # explicitly equivalent value. Reject every other option or permission mode.
    return actual_size == size and parsed.get("mode", "1777") == "1777"


def _ulimits_valid(host_config: Mapping, *, nofile: int) -> bool:
    raw = host_config.get("Ulimits")
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], Mapping):
        return False
    limit = raw[0]
    return (
        set(limit) == {"Name", "Soft", "Hard"}
        and limit.get("Name") == "nofile"
        and limit.get("Soft") == nofile
        and limit.get("Hard") == nofile
    )


def _restart_policy_valid(host_config: Mapping) -> bool:
    policy = host_config.get("RestartPolicy")
    maximum_retries = policy.get("MaximumRetryCount") if isinstance(policy, Mapping) else None
    return (
        isinstance(policy, Mapping)
        and policy.get("Name") == "no"
        and (
            maximum_retries is None
            or (isinstance(maximum_retries, int) and not isinstance(maximum_retries, bool) and maximum_retries == 0)
        )
        and set(policy) <= {"Name", "MaximumRetryCount"}
    )


def _log_config_valid(host_config: Mapping) -> bool:
    log_config = host_config.get("LogConfig")
    if not isinstance(log_config, Mapping) or log_config.get("Type") != "json-file":
        return False
    config = log_config.get("Config")
    return isinstance(config, Mapping) and dict(config) == {
        "labels": "capsule.id",
        "max-size": CAP_LOG_MAX_SIZE,
        "max-file": CAP_LOG_MAX_FILE,
    }


def _resource_and_namespace_posture_valid(host_config: Mapping, role: str) -> bool:
    if role == "brain":
        expected = (
            BRAIN_MEMORY_BYTES,
            BRAIN_MEMORY_RESERVATION_BYTES,
            BRAIN_NANO_CPUS,
            BRAIN_PIDS_LIMIT,
        )
        tmpfs_size = 16 * 1024**2
        nofile = 256
    else:
        expected = (APP_MEMORY_BYTES, 0, APP_NANO_CPUS, APP_PIDS_LIMIT)
        tmpfs_size = 256 * 1024**2
        nofile = 4096
    memory_reservation = host_config.get("MemoryReservation")
    if memory_reservation is None:
        memory_reservation = 0
    return (
        (
            host_config.get("Memory"),
            memory_reservation,
            host_config.get("NanoCpus"),
            host_config.get("PidsLimit"),
        )
        == expected
        and host_config.get("MemorySwap") == expected[0]
        and _tmpfs_valid(host_config, size=tmpfs_size)
        and _ulimits_valid(host_config, nofile=nofile)
        and _restart_policy_valid(host_config)
        and _log_config_valid(host_config)
        and not bool(host_config.get("PortBindings"))
        and not bool(host_config.get("PublishAllPorts"))
        and not bool(host_config.get("Devices"))
        and not bool(host_config.get("DeviceRequests"))
        and not bool(host_config.get("Binds"))
        # Current Engine inspect normalizes the explicitly requested IPC/cgroup namespace modes to
        # concrete strings. Missing/null values are therefore lack of proof, not safe defaults. UTS
        # keeps Engine's empty spelling for its private default (and some API versions expose the
        # equivalent explicit spelling).
        and host_config.get("PidMode") == ""
        and host_config.get("IpcMode") == "private"
        and host_config.get("UTSMode") in {"", "private"}
        and host_config.get("CgroupnsMode") == "private"
        # Empty means "use the daemon's configured mapping". `host` would explicitly disable an
        # externally configured userns-remap policy, so it is never accepted. The mandatory runsc
        # Sentry user namespace is runtime-internal and must be proved from the host as documented in
        # GVISOR.md; Docker's HostConfig cannot attest to that namespace.
        and host_config.get("UsernsMode") == ""
    )


def workload_security_valid(
    metadata: Mapping,
    cid: str,
    required_runtime: str,
    *,
    expected_image_ref: str,
    expected_image_id: str,
) -> bool:
    """Validate immutable hostile-workload posture plus its exact core attachment."""
    role = _workload_role(metadata, cid)
    if role is None:
        return False
    host_config = _mapping(metadata.get("HostConfig"))
    config = _mapping(metadata.get("Config"))
    if (
        not expected_image_ref
        or not expected_image_id
        or config.get("Image") != expected_image_ref
        or metadata.get("Image") != expected_image_id
        or str(host_config.get("Runtime") or "runc") != required_runtime
        or bool(host_config.get("Privileged"))
        or host_config.get("NetworkMode") != network_name(cid, CORE_KIND)
        or not _security_options_valid(host_config)
        or not _resource_and_namespace_posture_valid(host_config, role[0])
        or metadata.get("AppArmorProfile") != "docker-default"
    ):
        return False
    networks = _mapping(_mapping(metadata.get("NetworkSettings")).get("Networks"))
    expected_kinds = workload_network_kinds(metadata, cid)
    if expected_kinds is None:
        return False
    expected_networks = {network_name(cid, kind) for kind in expected_kinds}
    if set(networks) != expected_networks:
        return False
    # Docker Engine 29 prefixes inspect results with ``CAP_`` while older Engines return the
    # unprefixed names accepted by create_container. Both spellings describe the same kernel set.
    cap_drop = _normalized_capabilities(host_config.get("CapDrop"))
    cap_add = _normalized_capabilities(host_config.get("CapAdd"))
    mounts = metadata.get("Mounts")
    if not isinstance(mounts, list):
        return False
    if role[0] == "brain":
        return host_config.get("ReadonlyRootfs") is True and cap_drop == {"ALL"} and not cap_add and not mounts
    return (
        config.get("User") == "10001:10001"
        and host_config.get("ReadonlyRootfs") is True
        and cap_drop == {"ALL"}
        and not cap_add
        and not mounts
    )


def daemon_security_options_valid(info: Mapping) -> bool:
    """Require the daemon's observable built-in seccomp and AppArmor defaults."""
    options = info.get("SecurityOptions")
    if not isinstance(options, list):
        return False
    normalized = {str(option) for option in options}
    return "name=apparmor" in normalized and any(
        option.startswith("name=seccomp") and "profile=builtin" in option for option in normalized
    )


def daemon_runtime_registration_valid(info: Mapping, required_runtime: str, required_path: str) -> bool:
    """Require one named Docker runtime to resolve to the reviewed absolute handler path."""
    runtimes = info.get("Runtimes")
    if not isinstance(runtimes, Mapping):
        return False
    runtime = runtimes.get(required_runtime)
    return (
        isinstance(runtime, Mapping)
        and runtime.get("path") == required_path
        and runtime.get("runtimeArgs") in (None, [])
    )


def daemon_isolation_valid(info: Mapping, required_runtime: str, required_path: str) -> bool:
    """Require the exact runtime binding plus observable built-in daemon security profiles."""
    return daemon_runtime_registration_valid(
        info,
        required_runtime,
        required_path,
    ) and daemon_security_options_valid(info)
