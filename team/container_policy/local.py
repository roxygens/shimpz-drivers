"""Pure isolation-profile admission for local Assistant containers."""

from __future__ import annotations

import egress_policy
from local_registry import is_digest_ref

ASSISTANT_UID = "10001:10001"
ASSISTANT_MEMORY = 128 * 1024 * 1024
ASSISTANT_NANO_CPUS = 250_000_000
ASSISTANT_PIDS = 64


def inspect_profile(
    attrs: dict,
    container_name: str,
    expected_labels: dict[str, str],
    expected_name: str,
    reviewed_image: str,
    network_name: str,
    cpuset_cpus: str,
) -> tuple[dict, dict[str, str]] | None:
    """Return admitted config/environment, or ``None`` for any profile drift."""
    config = attrs.get("Config") or {}
    host = attrs.get("HostConfig") or {}
    labels = config.get("Labels") or {}
    installed_image = labels.get("com.shimpz.local.image")
    security_options = host.get("SecurityOpt") or []
    networks = (attrs.get("NetworkSettings") or {}).get("Networks") or {}
    environment = egress_policy.environment_map(config.get("Env"))
    if (
        environment is None
        or not isinstance(labels, dict)
        or not all(labels.get(key) == value for key, value in expected_labels.items())
        or container_name != expected_name
        or not is_digest_ref(installed_image)
        or config.get("Image") != installed_image
        or installed_image.rpartition("@sha256:")[0] != reviewed_image.rpartition("@sha256:")[0]
        or config.get("User") != ASSISTANT_UID
        or host.get("ReadonlyRootfs") is not True
        or "ALL" not in (host.get("CapDrop") or [])
        or not any(str(option).startswith("no-new-privileges") for option in security_options)
        or any("seccomp=unconfined" in str(option) for option in security_options)
        or host.get("Privileged") is not False
        or host.get("NetworkMode") != network_name
        or host.get("Memory") != ASSISTANT_MEMORY
        or host.get("MemorySwap") != ASSISTANT_MEMORY
        or host.get("NanoCpus") != ASSISTANT_NANO_CPUS
        or host.get("CpusetCpus") != cpuset_cpus
        or host.get("PidsLimit") != ASSISTANT_PIDS
        or host.get("IpcMode") != "private"
        or host.get("CgroupnsMode") != "private"
        or host.get("Tmpfs") not in (None, {})
        or host.get("AutoRemove") is not False
        or (host.get("RestartPolicy") or {}).get("Name") not in {"", "no"}
        or (host.get("LogConfig") or {}).get("Type") != "json-file"
        or (host.get("LogConfig") or {}).get("Config") != {"max-file": "2", "max-size": "1m"}
        or host.get("PortBindings") not in (None, {})
        or host.get("Binds") not in (None, [])
        or host.get("Devices") not in (None, [])
        or host.get("DeviceRequests") not in (None, [])
        or attrs.get("Mounts") not in (None, [])
        or set(networks) != {network_name}
    ):
        return None
    return config, environment
