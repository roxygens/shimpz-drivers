#!/usr/local/bin/python3
"""Docker HEALTHCHECK probe: runtime/images/topology readiness and auth-gate enforcement.

The system interpreter intentionally has no docker SDK, so daemon checks read Docker's local Unix
socket with stdlib HTTP. The configured hostile-tenant runtime must remain bound to the reviewed absolute
handler while Docker advertises its built-in seccomp and AppArmor defaults; every advertised Brain image
must be present, and every existing Capsule Brain/App must actually use that runtime. The probe never
accepts Docker's default runc, a legacy workload, or a missing provider as a fallback. Then an
unauthenticated driver GET must be refused with 403 — a 2xx means the auth gate is not enforced.
"""

import http.client
import json
import os
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request

import marketplace
import network_policy

DOCKER_SOCKET = os.environ.get("DOCKER_HOST_SOCKET", "/var/run/docker.sock")
# Must stay identical to manifests.RUNTIME without importing the SDK-backed module. This is not an
# operator override: hostile-tenant readiness is always tied to the shipping gVisor runtime.
REQUIRED_RUNTIME = "runsc"
REQUIRED_RUNTIME_PATH = "/usr/local/bin/runsc"
REQUIRED_BRAIN_IMAGES = {
    "runtime": os.environ.get(
        "SHIMPZ_CAPSULE_IMAGE",
        "registry.k8s.io/pause:3.10.1@sha256:278fb9dbcca9518083ad1e11276933a2e96f23de604a3a08cc3c80002767d24c",
    ),
}
REQUIRED_IMAGES = tuple(REQUIRED_BRAIN_IMAGES.values())
LISTEN_PORT = int(os.environ.get("SHIMPZ_CAPSULEDRIVER_PORT", "7077"))


def _docker_json(path: str) -> tuple[int, object | None]:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(3)
    response = None
    try:
        client.connect(DOCKER_SOCKET)
        request = f"GET {path} HTTP/1.1\r\nHost: docker\r\nConnection: close\r\n\r\n"
        client.sendall(request.encode("ascii"))
        response = http.client.HTTPResponse(client)
        response.begin()
        payload = json.loads(response.read())
    except OSError, http.client.HTTPException, json.JSONDecodeError, UnicodeError, ValueError:
        return 0, None
    else:
        return response.status, payload
    finally:
        if response is not None:
            response.close()
        client.close()


def daemon_isolation_ready() -> bool:
    """Evaluate the runtime handler and daemon profiles from one Engine-info snapshot."""
    status, info = _docker_json("/info")
    return (
        status == 200
        and isinstance(info, dict)
        and network_policy.daemon_isolation_valid(info, REQUIRED_RUNTIME, REQUIRED_RUNTIME_PATH)
    )


def _image_id(image_ref: str) -> str | None:
    encoded = urllib.parse.quote(image_ref, safe="")
    status, metadata = _docker_json(f"/images/{encoded}/json")
    if status != 200 or not isinstance(metadata, dict):
        return None
    image_id = metadata.get("Id")
    return image_id if isinstance(image_id, str) and image_id else None


def images_ready() -> bool:
    """Require the exact local image references advertised by the provider registry."""
    return all(_image_id(image_ref) is not None for image_ref in REQUIRED_IMAGES)


def _expected_workload_image(metadata: dict, image_ids: dict[str, str]) -> tuple[str, str] | None:
    config = metadata.get("Config")
    labels = config.get("Labels") if isinstance(config, dict) else None
    if not isinstance(labels, dict):
        return None
    if labels.get("capsule.driver") == "1":
        provider = labels.get("capsule.brain")
        image_ref = REQUIRED_BRAIN_IMAGES.get(provider) if isinstance(provider, str) else None
    elif labels.get("capsule.app.driver") == "1":
        app_id = labels.get("capsule.app")
        app_spec = marketplace.APPS.get(app_id) if isinstance(app_id, str) else None
        image_ref = app_spec.image if app_spec is not None else None
    else:
        return None
    if not isinstance(image_ref, str) or not image_ref:
        return None
    if image_ref not in image_ids:
        resolved = _image_id(image_ref)
        if resolved is None:
            return None
        image_ids[image_ref] = resolved
    return image_ref, image_ids[image_ref]


def workloads_isolated() -> bool:
    """Reject deployment while any existing Capsule Brain or App uses a different runtime."""
    status, containers = _docker_json("/containers/json?all=1")
    if status != 200 or not isinstance(containers, list):
        return False
    for summary in containers:
        if not isinstance(summary, dict):
            return False
        labels = summary.get("Labels")
        if not isinstance(labels, dict) or not ({"capsule.driver", "capsule.app.driver"} & set(labels)):
            continue
        container_id = summary.get("Id")
        if not isinstance(container_id, str) or not container_id:
            return False
        inspect_status, metadata = _docker_json(f"/containers/{container_id}/json")
        if inspect_status != 200 or not isinstance(metadata, dict):
            return False
        host_config = metadata.get("HostConfig")
        if not isinstance(host_config, dict) or str(host_config.get("Runtime") or "runc") != REQUIRED_RUNTIME:
            return False
    return True


def _inspect_workloads(
    summaries: list,
) -> (
    tuple[
        dict[str, dict],
        set[str],
        dict[str, int],
        set[str],
        dict[str, tuple[str, frozenset[str], bool]],
    ]
    | None
):
    inspections: dict[str, dict] = {}
    cids: set[str] = set()
    brains_by_cid: dict[str, int] = {}
    running_brains: set[str] = set()
    workloads: dict[str, tuple[str, frozenset[str], bool]] = {}
    image_ids: dict[str, str] = {}
    for summary in summaries:
        if not isinstance(summary, dict):
            return None
        labels = summary.get("Labels")
        if not isinstance(labels, dict) or not ({"capsule.driver", "capsule.app.driver"} & set(labels)):
            continue
        container_id = summary.get("Id")
        cid = labels.get("capsule.id")
        if not isinstance(container_id, str) or not container_id or not isinstance(cid, str) or not cid:
            return None
        inspect_status, metadata = _docker_json(f"/containers/{container_id}/json")
        if inspect_status != 200 or not isinstance(metadata, dict):
            return None
        state = metadata.get("State")
        running = state.get("Running") if isinstance(state, dict) else None
        if not isinstance(running, bool):
            return None
        inspections[container_id] = metadata
        cids.add(cid)
        if labels.get("capsule.driver") == "1":
            brains_by_cid[cid] = brains_by_cid.get(cid, 0) + 1
            if running:
                running_brains.add(cid)
        expected_image = _expected_workload_image(metadata, image_ids)
        if expected_image is None or not network_policy.workload_security_valid(
            metadata,
            cid,
            REQUIRED_RUNTIME,
            expected_image_ref=expected_image[0],
            expected_image_id=expected_image[1],
        ):
            return None
        expected_kinds = network_policy.workload_network_kinds(metadata, cid)
        if expected_kinds is None:
            return None
        workloads[container_id] = (cid, expected_kinds, running)
    return inspections, cids, brains_by_cid, running_brains, workloads


def _load_network_members(network: dict, inspections: dict[str, dict]) -> bool:
    members = network.get("Containers")
    if not isinstance(members, dict):
        return False
    for member_id in members:
        if member_id in inspections:
            continue
        inspect_status, metadata = _docker_json(f"/containers/{member_id}/json")
        if inspect_status != 200 or not isinstance(metadata, dict):
            return False
        inspections[member_id] = metadata
    return True


def network_topology_ready() -> bool:
    """Require exact workload posture and two-plane membership for every Capsule."""
    status, summaries = _docker_json("/containers/json?all=1")
    if status != 200 or not isinstance(summaries, list):
        return False
    inspected = _inspect_workloads(summaries)
    if inspected is None:
        return False
    inspections, cids, brains_by_cid, running_brains, workloads = inspected
    if any(brains_by_cid.get(cid) != 1 for cid in cids):
        return False

    for cid in cids:
        for kind in (network_policy.CORE_KIND, network_policy.BRAIN_EGRESS_KIND):
            name = network_policy.network_name(cid, kind)
            encoded = urllib.parse.quote(name, safe="")
            network_status, network = _docker_json(f"/networks/{encoded}")
            if network_status != 200 or not isinstance(network, dict):
                return False
            if not _load_network_members(network, inspections) or not network_policy.network_members_valid(
                network,
                inspections,
                cid,
                kind,
                # Engine omits an intentionally stopped Brain from network inventory. Its immutable
                # image/resource/endpoints were proved above; only a running Brain must be a live member.
                require_brain=cid in running_brains,
                require_dependencies=True,
            ):
                return False
            for workload_id, (workload_cid, expected_kinds, running) in workloads.items():
                if (
                    workload_cid == cid
                    and kind in expected_kinds
                    and (
                        not network_policy.workload_endpoint_valid(
                            network,
                            inspections[workload_id],
                            cid,
                            kind,
                        )
                        or (
                            running
                            and not network_policy.workload_live_membership_valid(
                                network,
                                inspections[workload_id],
                                cid,
                                kind,
                            )
                        )
                    )
                ):
                    return False
    return True


def auth_gate_ready() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{LISTEN_PORT}/v1/capsules", timeout=3):
            return False
    except urllib.error.HTTPError as exc:
        return exc.code == 403
    except OSError:
        return False


def main() -> int:
    ready = (
        daemon_isolation_ready()
        and images_ready()
        and workloads_isolated()
        and network_topology_ready()
        and auth_gate_ready()
    )
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
