#!/usr/bin/env python3
"""Pure contracts for the Team core network and drift policy.

No Docker daemon and no mocks: these tests feed Engine-API-shaped immutable dictionaries into the
same stdlib policy used by team-driver admission and its shipping healthcheck.
"""

from __future__ import annotations

import copy
import unittest

import healthcheck as team_healthcheck
from container_policy import network as policy


def check(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


TEAM_ID = "tenant_workspace"
CORE = policy.network_name(TEAM_ID, policy.CORE_KIND)
BRAIN_IMAGE_REF = "trusted-brain:v1"
BRAIN_IMAGE_ID = "sha256:trusted-brain-id"
APP_IMAGE_REF = "trusted-app:v1"
APP_IMAGE_ID = "sha256:trusted-app-id"


def _endpoint(network_id: str, *aliases: str) -> dict:
    return {"NetworkID": network_id, "Aliases": list(aliases) if aliases else None}


def _pending_endpoint(*dns_names: str) -> dict:
    """Engine 29 container-inspect shape for an exact requested endpoint before start."""
    return {
        "IPAMConfig": {},
        "Links": None,
        "Aliases": None,
        "DriverOpts": {},
        "GwPriority": 0,
        "NetworkID": "",
        "EndpointID": "",
        "Gateway": "",
        "IPAddress": "",
        "MacAddress": "",
        "IPPrefixLen": 0,
        "IPv6Gateway": "",
        "GlobalIPv6Address": "",
        "GlobalIPv6PrefixLen": 0,
        "DNSNames": list(dns_names) if dns_names else None,
    }


def _container(
    container_id: str,
    name: str,
    **attributes: object,
) -> dict:
    allowed = {
        "labels",
        "networks",
        "host_config",
        "user",
        "apparmor",
        "mounts",
        "image_ref",
        "image_id",
        "hostname",
        "running",
    }
    if set(attributes) - allowed:
        raise ValueError("unknown container fixture attribute")
    return {
        "Id": container_id,
        "Name": f"/{name}",
        "Config": {
            "Labels": attributes.get("labels") or {},
            "User": attributes.get("user", ""),
            "Image": attributes.get("image_ref", ""),
            "Hostname": attributes.get("hostname", ""),
        },
        "Image": attributes.get("image_id", ""),
        "HostConfig": attributes.get("host_config") or {},
        "NetworkSettings": {"Networks": attributes.get("networks") or {}},
        "AppArmorProfile": attributes.get("apparmor", "docker-default"),
        "Mounts": attributes.get("mounts") or [],
        "State": {"Running": attributes.get("running", True)},
    }


def _network(kind: str, network_id: str, *member_ids: str) -> dict:
    return {
        "Id": network_id,
        "Name": policy.network_name(TEAM_ID, kind),
        "Driver": "bridge",
        "Scope": "local",
        "Internal": True,
        "Attachable": False,
        "Ingress": False,
        "ConfigOnly": False,
        "Labels": policy.network_labels(TEAM_ID, kind),
        "Containers": {container_id: {} for container_id in member_ids},
    }


def _valid_topology() -> tuple[dict, dict[str, dict]]:
    common_security = {
        "Runtime": "runsc",
        "Privileged": False,
        "NetworkMode": CORE,
        "SecurityOpt": ["no-new-privileges", "apparmor=docker-default"],
        "PortBindings": {},
        "PublishAllPorts": False,
        "Devices": [],
        "DeviceRequests": [],
        "Binds": [],
        "PidMode": "",
        "IpcMode": "private",
        "UTSMode": "",
        "CgroupnsMode": "private",
        "UsernsMode": "",
    }
    brain = _container(
        "brain-id",
        policy.team_container_name(TEAM_ID),
        labels={"team.driver": "1", "team.id": TEAM_ID, "team.brain": "runtime"},
        networks={CORE: _endpoint("core-id", policy.team_container_name(TEAM_ID))},
        host_config={
            **common_security,
            "CapDrop": ["ALL"],
            "CapAdd": sorted(policy.EXPECTED_BRAIN_CAP_ADD),
            "ReadonlyRootfs": True,
            "Memory": policy.BRAIN_MEMORY_BYTES,
            "MemorySwap": policy.BRAIN_MEMORY_BYTES,
            "MemoryReservation": policy.BRAIN_MEMORY_RESERVATION_BYTES,
            "NanoCpus": policy.BRAIN_NANO_CPUS,
            "PidsLimit": policy.BRAIN_PIDS_LIMIT,
            "Tmpfs": {"/tmp": "mode=1777,size=16m"},
            "Ulimits": [{"Name": "nofile", "Soft": 256, "Hard": 256}],
            "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
            "LogConfig": {
                "Type": "json-file",
                "Config": {
                    "labels": "team.id",
                    "max-size": policy.TEAM_LOG_MAX_SIZE,
                    "max-file": policy.TEAM_LOG_MAX_FILE,
                },
            },
        },
        mounts=[],
        image_ref=BRAIN_IMAGE_REF,
        image_id=BRAIN_IMAGE_ID,
        hostname=TEAM_ID,
    )
    app_id = "notification-center"
    app = _container(
        "app-id",
        policy.team_app_container_name(TEAM_ID, app_id),
        labels={"team.app.driver": "1", "team.id": TEAM_ID, "team.app": app_id},
        networks={CORE: _endpoint("core-id", app_id, f"{app_id}.team")},
        host_config={
            **common_security,
            "CapDrop": ["ALL"],
            "CapAdd": [],
            "ReadonlyRootfs": True,
            "Memory": policy.APP_MEMORY_BYTES,
            "MemorySwap": policy.APP_MEMORY_BYTES,
            "NanoCpus": policy.APP_NANO_CPUS,
            "PidsLimit": policy.APP_PIDS_LIMIT,
            "Tmpfs": {"/tmp": "size=256m"},
            "Ulimits": [{"Name": "nofile", "Soft": 4096, "Hard": 4096}],
            "RestartPolicy": {"Name": "no"},
            "LogConfig": {
                "Type": "json-file",
                "Config": {
                    "labels": "team.id",
                    "max-size": policy.TEAM_LOG_MAX_SIZE,
                    "max-file": policy.TEAM_LOG_MAX_FILE,
                },
            },
        },
        user="10001:10001",
        image_ref=APP_IMAGE_REF,
        image_id=APP_IMAGE_ID,
    )
    postgres = _container(
        "postgres-id",
        policy.POSTGRES_CONTAINER,
        labels=policy.shared_service_labels(policy.POSTGRES_ROLE),
        networks={CORE: _endpoint("core-id", "postgres")},
    )
    app_proxy = _container(
        "app-proxy-id",
        policy.APP_EGRESS_CONTAINER,
        labels=policy.shared_service_labels(policy.APP_EGRESS_ROLE),
        networks={CORE: _endpoint("core-id", "app-egress-proxy")},
    )
    containers = {item["Id"]: item for item in (brain, app, postgres, app_proxy)}
    core = _network(policy.CORE_KIND, "core-id", "brain-id", "app-id", "postgres-id", "app-proxy-id")
    return core, containers


def _members_valid(network: dict, containers: dict[str, dict], kind: str) -> bool:
    return policy.network_members_valid(
        network,
        containers,
        TEAM_ID,
        kind,
        require_brain=True,
        require_dependencies=True,
    )


def _workload_valid(metadata: dict) -> bool:
    labels = metadata["Config"]["Labels"]
    if labels.get("team.driver") == "1":
        expected_ref, expected_id = BRAIN_IMAGE_REF, BRAIN_IMAGE_ID
    else:
        expected_ref, expected_id = APP_IMAGE_REF, APP_IMAGE_ID
    return policy.workload_security_valid(
        metadata,
        TEAM_ID,
        "runsc",
        expected_image_ref=expected_ref,
        expected_image_id=expected_id,
    )


def test_network_names_are_injective_and_bounded() -> None:
    longest = policy.network_name("x" * 40, policy.CORE_KIND)
    check(len(longest.encode()) <= policy.DOCKER_NETWORK_NAME_MAX, "maximum Team ID stays inside Docker's limit")
    try:
        policy.network_name("x" * policy.DOCKER_NETWORK_NAME_MAX, policy.CORE_KIND)
        check(False, "an oversized derived Docker network name must be refused")
    except ValueError:
        check(True, "an oversized derived Docker network name is refused")
    try:
        policy.network_name("x", "brain-egress")
        check(False, "a retired network kind must be refused")
    except ValueError:
        check(True, "the retired Brain-egress network kind is refused")

    app_name = policy.team_app_container_name("x", "notification-center")
    adversarial_brain = policy.team_container_name("x_app_notification_center")
    check(app_name != adversarial_brain, "a valid Brain Team ID cannot collide with an App workload name")
    check(
        app_name.endswith("x.app.notification-center"),
        "App workload naming uses an out-of-Team-ID delimiter without lossy rewriting",
    )
    check(
        len(policy.team_app_container_name("x" * 40, "x" * 40).encode()) <= policy.DOCKER_RESOURCE_NAME_MAX,
        "maximum valid Team ID/App IDs stay inside Docker's resource-name limit",
    )

    foreign_brain = _container(
        "foreign-brain",
        policy.team_container_name("x"),
        labels={"team.driver": "1", "team.id": "somebody_else"},
    )
    check(not policy.brain_identity_valid(foreign_brain, "x"), "a matching name cannot forge Brain identity")


def test_valid_core_topology_and_security_posture() -> None:
    core, containers = _valid_topology()
    check(_members_valid(core, containers, policy.CORE_KIND), "core accepts only Brain, Apps, DB and app proxy")
    check(_workload_valid(containers["brain-id"]), "Brain posture is exact")
    check(_workload_valid(containers["app-id"]), "App posture is exact")
    check(
        policy.daemon_security_options_valid(
            {"SecurityOptions": ["name=apparmor", "name=seccomp,profile=builtin", "name=cgroupns"]}
        ),
        "daemon AppArmor and built-in seccomp posture validates",
    )


def test_daemon_admission_requires_exact_runsc_path_and_builtin_seccomp() -> None:
    valid = {
        "Runtimes": {"runsc": {"path": policy.TEAM_RUNTIME_PATH, "runtimeArgs": None}},
        "SecurityOptions": ["name=apparmor", "name=seccomp,profile=builtin", "name=cgroupns"],
    }
    check(
        policy.daemon_isolation_valid(valid, "runsc", policy.TEAM_RUNTIME_PATH),
        "exact runsc handler plus built-in daemon profiles passes admission",
    )

    wrong_handler = copy.deepcopy(valid)
    wrong_handler["Runtimes"]["runsc"]["path"] = "/usr/bin/runc"
    check(
        not policy.daemon_isolation_valid(wrong_handler, "runsc", policy.TEAM_RUNTIME_PATH),
        "a runsc registry alias to another handler fails admission",
    )

    injected_arguments = copy.deepcopy(valid)
    injected_arguments["Runtimes"]["runsc"]["runtimeArgs"] = ["--network=host"]
    check(
        not policy.daemon_isolation_valid(injected_arguments, "runsc", policy.TEAM_RUNTIME_PATH),
        "injected runsc runtime arguments fail admission despite the exact handler path",
    )

    missing_seccomp = copy.deepcopy(valid)
    missing_seccomp["SecurityOptions"] = ["name=apparmor", "name=cgroupns"]
    check(
        not policy.daemon_isolation_valid(missing_seccomp, "runsc", policy.TEAM_RUNTIME_PATH),
        "missing built-in seccomp fails admission despite the exact runtime path",
    )
    check(
        team_healthcheck.REQUIRED_RUNTIME_PATH == policy.TEAM_RUNTIME_PATH,
        "shipping readiness pins the same absolute runsc handler as lifecycle admission",
    )


def test_engine_29_capability_prefix_is_normalized() -> None:
    _core, containers = _valid_topology()
    brain = containers["brain-id"]
    brain["HostConfig"]["CapDrop"] = ["CAP_ALL"]
    brain["HostConfig"]["CapAdd"] = [f"CAP_{capability}" for capability in sorted(policy.EXPECTED_BRAIN_CAP_ADD)]
    check(
        _workload_valid(brain),
        "Engine 29 CAP_-prefixed inspect values preserve the exact capability contract",
    )


def test_health_resolves_each_workload_role_to_its_trusted_image_id() -> None:
    requested_refs: list[str] = []
    original_image_id = team_healthcheck._image_id
    team_healthcheck._image_id = lambda image_ref: requested_refs.append(image_ref) or f"sha256:{len(requested_refs)}"
    try:
        cache: dict[str, str] = {}
        brain_ref = team_healthcheck.REQUIRED_BRAIN_IMAGES["runtime"]
        brain = {"Config": {"Labels": {"team.driver": "1", "team.brain": "runtime"}}}
        check(
            team_healthcheck._expected_workload_image(brain, cache) == (brain_ref, "sha256:1"),
            "health maps the registered Brain provider to its resolved immutable image ID",
        )
        check(
            team_healthcheck._expected_workload_image(brain, cache) == (brain_ref, "sha256:1")
            and requested_refs == [brain_ref],
            "health caches one immutable resolution consistently across its inspection pass",
        )

        app_id, app_spec = next(iter(team_healthcheck.marketplace.APPS.items()))
        app = {"Config": {"Labels": {"team.app.driver": "1", "team.app": app_id}}}
        check(
            team_healthcheck._expected_workload_image(app, cache) == (app_spec.image, "sha256:2"),
            "health maps a registered App to its separately resolved immutable image ID",
        )
        unknown = {"Config": {"Labels": {"team.driver": "1", "team.brain": "unknown-provider"}}}
        check(
            team_healthcheck._expected_workload_image(unknown, cache) is None,
            "health fails closed for an unregistered Brain provider",
        )
    finally:
        team_healthcheck._image_id = original_image_id


def test_health_tracks_running_brains_without_weakening_stopped_posture() -> None:
    _core, containers = _valid_topology()
    brain = containers["brain-id"]
    app = containers["app-id"]
    brain_ref = team_healthcheck.REQUIRED_BRAIN_IMAGES["runtime"]
    app_ref = team_healthcheck.marketplace.APPS["notification-center"].image
    brain["Config"]["Image"], brain["Image"] = brain_ref, "sha256:health-brain"
    app["Config"]["Image"], app["Image"] = app_ref, "sha256:health-app"
    brain["State"]["Running"] = False
    metadata_by_id = {"brain-id": brain, "app-id": app}
    summaries = [
        {"Id": container_id, "Labels": metadata["Config"]["Labels"]}
        for container_id, metadata in metadata_by_id.items()
    ]
    original_docker_json = team_healthcheck._docker_json
    original_image_id = team_healthcheck._image_id
    team_healthcheck._docker_json = lambda path: (200, metadata_by_id[path.split("/")[2]])
    team_healthcheck._image_id = lambda image_ref: {
        brain_ref: "sha256:health-brain",
        app_ref: "sha256:health-app",
    }.get(image_ref)
    try:
        inspected = team_healthcheck._inspect_workloads(summaries)
        check(
            inspected is not None
            and inspected[3] == set()
            and inspected[4]
            == {
                "brain-id": (TEAM_ID, frozenset({policy.CORE_KIND}), False),
                "app-id": (TEAM_ID, frozenset({policy.CORE_KIND}), True),
            },
            "health tracks stopped and running workloads separately for static/live endpoint proof",
        )
        brain["State"]["Running"] = True
        inspected = team_healthcheck._inspect_workloads(summaries)
        check(
            inspected is not None and inspected[3] == {TEAM_ID}, "health requires live membership for a running Brain"
        )
        brain["State"]["Running"] = False
        brain["HostConfig"]["IpcMode"] = "host"
        check(
            team_healthcheck._inspect_workloads(summaries) is None,
            "stopped health normalization still rejects static namespace drift",
        )
    finally:
        team_healthcheck._docker_json = original_docker_json
        team_healthcheck._image_id = original_image_id


def test_foreign_services_and_extra_app_networks_fail_closed() -> None:
    core, containers = _valid_topology()
    broad_on_core = copy.deepcopy(core)
    broad_proxy = _container(
        "brain-proxy-id",
        "egress-proxy",
        labels={policy.SHARED_MANAGED_LABEL: "1", policy.SHARED_ROLE_LABEL: "brain-egress"},
        networks={CORE: _endpoint("core-id", "egress-proxy")},
    )
    containers["brain-proxy-id"] = broad_proxy
    broad_on_core["Containers"]["brain-proxy-id"] = {}
    check(not _members_valid(broad_on_core, containers, policy.CORE_KIND), "retired broad proxy on core fails closed")

    _core, containers = _valid_topology()
    containers["app-id"]["NetworkSettings"]["Networks"]["foreign"] = _endpoint("foreign-id", "notification-center")
    check(
        not _workload_valid(containers["app-id"]),
        "App with any extra network fails its workload posture",
    )


def test_stopped_brain_omission_keeps_static_proof_and_rejects_posture_drift() -> None:
    core, containers = _valid_topology()
    brain = containers["brain-id"]
    brain["State"]["Running"] = False
    del core["Containers"]["brain-id"]
    check(_workload_valid(brain), "stopped Brain keeps exact image/resource/endpoint posture in container inspect")
    check(
        policy.workload_endpoint_valid(core, brain, TEAM_ID, policy.CORE_KIND),
        "stopped Brain retains an exact container-inspect endpoint on core",
    )
    check(
        policy.network_members_valid(
            core,
            containers,
            TEAM_ID,
            policy.CORE_KIND,
            require_brain=False,
            require_dependencies=True,
        ),
        "stopped Brain omission is accepted on the exact core plane",
    )
    check(
        not policy.network_members_valid(
            core,
            containers,
            TEAM_ID,
            policy.CORE_KIND,
            require_brain=True,
            require_dependencies=True,
        ),
        "the same omission fails whenever core requires live Brain membership",
    )
    check(
        policy.workload_live_membership_valid(core, containers["app-id"], TEAM_ID, policy.CORE_KIND),
        "a running App remains valid on core while its exact Brain is intentionally stopped",
    )
    app_omitted = copy.deepcopy(core)
    del app_omitted["Containers"]["app-id"]
    check(
        policy.network_members_valid(
            app_omitted,
            containers,
            TEAM_ID,
            policy.CORE_KIND,
            require_brain=False,
            require_dependencies=True,
        ),
        "aggregate plane policy alone does not invent an omitted optional App",
    )
    check(
        not policy.workload_live_membership_valid(
            app_omitted,
            containers["app-id"],
            TEAM_ID,
            policy.CORE_KIND,
        ),
        "exact running-App membership proof rejects Engine inventory omission",
    )
    drifted = copy.deepcopy(brain)
    drifted["Config"]["Image"] = "attacker:stopped"
    check(not _workload_valid(drifted), "stopped-member normalization cannot bypass trusted image posture")
    wrong_network = copy.deepcopy(brain)
    wrong_network["NetworkSettings"]["Networks"][CORE]["NetworkID"] = "foreign-network-id"
    check(
        not policy.workload_endpoint_valid(core, wrong_network, TEAM_ID, policy.CORE_KIND),
        "stopped-member omission cannot bypass exact endpoint NetworkID binding",
    )
    pending = copy.deepcopy(brain)
    automatic_names = (policy.team_container_name(TEAM_ID), "brain-id", TEAM_ID)
    pending["NetworkSettings"]["Networks"] = {
        CORE: _pending_endpoint(*automatic_names),
    }
    check(
        policy.workload_endpoint_valid(core, pending, TEAM_ID, policy.CORE_KIND),
        "strict Engine 29 stopped-endpoint placeholder validates on core",
    )
    running_pending = copy.deepcopy(pending)
    running_pending["State"]["Running"] = True
    check(
        not policy.workload_endpoint_valid(core, running_pending, TEAM_ID, policy.CORE_KIND),
        "an empty endpoint binding can never admit a running workload",
    )
    inventoried_pending = copy.deepcopy(core)
    inventoried_pending["Containers"]["brain-id"] = {}
    check(
        not policy.workload_endpoint_valid(inventoried_pending, pending, TEAM_ID, policy.CORE_KIND),
        "an empty endpoint binding cannot contradict the network's live-member inventory",
    )
    addressed_pending = copy.deepcopy(pending)
    addressed_pending["NetworkSettings"]["Networks"][CORE]["IPAddress"] = "172.30.0.9"
    check(
        not policy.workload_endpoint_valid(core, addressed_pending, TEAM_ID, policy.CORE_KIND),
        "a partially populated stopped endpoint is not mistaken for Engine's empty placeholder",
    )
    extended_pending = copy.deepcopy(pending)
    extended_pending["NetworkSettings"]["Networks"][CORE]["FutureAttachmentField"] = ""
    check(
        not policy.workload_endpoint_valid(core, extended_pending, TEAM_ID, policy.CORE_KIND),
        "unknown pending-endpoint fields fail closed until their Engine semantics are reviewed",
    )
    reserved_alias = copy.deepcopy(brain)
    reserved_alias["NetworkSettings"]["Networks"][CORE]["Aliases"].append("postgres")
    check(
        not policy.workload_endpoint_valid(core, reserved_alias, TEAM_ID, policy.CORE_KIND),
        "stopped-member omission cannot smuggle a reserved endpoint alias",
    )


def test_network_reuse_rejects_wrong_identity_and_contamination() -> None:
    core, containers = _valid_topology()
    for field, bad in (
        ("Internal", False),
        ("Driver", "overlay"),
        ("Scope", "swarm"),
        ("Attachable", True),
        ("Ingress", True),
        ("ConfigOnly", True),
    ):
        drifted = copy.deepcopy(core)
        drifted[field] = bad
        check(not policy.network_identity_valid(drifted, TEAM_ID, policy.CORE_KIND), f"{field} drift is rejected")
    wrong_labels = copy.deepcopy(core)
    wrong_labels["Labels"][policy.NETWORK_TEAM_ID_LABEL] = "another_team"
    check(not policy.network_identity_valid(wrong_labels, TEAM_ID, policy.CORE_KIND), "wrong Team ID label is rejected")

    config_volume = {
        "Name": policy.volume_name(TEAM_ID, policy.CONFIG_VOLUME_KIND),
        "Driver": "local",
        "Scope": "local",
        "Options": {},
        "Labels": policy.volume_labels(TEAM_ID, policy.CONFIG_VOLUME_KIND),
    }
    check(
        policy.volume_identity_valid(config_volume, TEAM_ID, policy.CONFIG_VOLUME_KIND),
        "exact labeled Team volume identity validates",
    )
    unlabeled_volume = copy.deepcopy(config_volume)
    unlabeled_volume["Labels"] = {}
    check(
        not policy.volume_identity_valid(unlabeled_volume, TEAM_ID, policy.CONFIG_VOLUME_KIND),
        "same-name unlabeled volume reuse is rejected",
    )
    host_bind_volume = copy.deepcopy(config_volume)
    host_bind_volume["Options"] = {"type": "none", "o": "bind", "device": "/etc"}
    check(
        not policy.volume_identity_valid(host_bind_volume, TEAM_ID, policy.CONFIG_VOLUME_KIND),
        "same-name labeled local volume backed by a host bind is rejected",
    )

    foreign = _container(
        "foreign-id",
        "foreign-container",
        networks={CORE: _endpoint("core-id", "foreign-container")},
    )
    containers["foreign-id"] = foreign
    contaminated = copy.deepcopy(core)
    contaminated["Containers"]["foreign-id"] = {}
    check(not _members_valid(contaminated, containers, policy.CORE_KIND), "foreign member is rejected")
    check(
        not policy.network_member_managed(foreign, TEAM_ID, policy.CORE_KIND),
        "teardown never claims a foreign member",
    )
    check(
        policy.network_member_managed(containers["postgres-id"], TEAM_ID, policy.CORE_KIND),
        "teardown recognizes the exact configured core dependency",
    )
    app_proxy = _container(
        "app-egress",
        policy.APP_EGRESS_CONTAINER,
        labels=policy.shared_service_labels(policy.APP_EGRESS_ROLE),
    )
    check(
        policy.network_member_managed(app_proxy, TEAM_ID, policy.CORE_KIND),
        "cleanup recognizes the exact token proxy on the core plane",
    )
    check(
        not policy.network_member_managed(app_proxy, TEAM_ID, "brain-egress"),
        "cleanup never accepts the retired Brain-egress plane",
    )
    name_only_postgres = _container("name-only", policy.POSTGRES_CONTAINER)
    check(
        not policy.network_member_managed(name_only_postgres, TEAM_ID, policy.CORE_KIND),
        "an exact shared-service name without its role labels remains foreign",
    )


def test_alias_and_endpoint_identity_drift_fail_closed() -> None:
    core, containers = _valid_topology()
    containers["postgres-id"]["NetworkSettings"]["Networks"][CORE]["Aliases"] = []
    check(not _members_valid(core, containers, policy.CORE_KIND), "missing postgres alias is rejected")

    core, containers = _valid_topology()
    containers["postgres-id"]["NetworkSettings"]["Networks"][CORE]["NetworkID"] = "another-network"
    check(not _members_valid(core, containers, policy.CORE_KIND), "endpoint/network ID mismatch is rejected")

    core, containers = _valid_topology()
    containers["app-id"]["NetworkSettings"]["Networks"][CORE]["Aliases"].append("postgres")
    check(not _members_valid(core, containers, policy.CORE_KIND), "App cannot claim a reserved service alias")

    core, containers = _valid_topology()
    containers["postgres-id"]["NetworkSettings"]["Networks"][CORE]["Aliases"].extend(
        [policy.POSTGRES_CONTAINER, "postgres-id"]
    )
    check(_members_valid(core, containers, policy.CORE_KIND), "Docker name/id automatic aliases normalize safely")

    core, containers = _valid_topology()
    containers["postgres-id"]["NetworkSettings"]["Networks"][CORE]["DNSNames"] = [
        "postgres",
        policy.POSTGRES_CONTAINER,
        "postgres-id",
    ]
    check(_members_valid(core, containers, policy.CORE_KIND), "Engine 29 DNSNames normalize safely")

    core, containers = _valid_topology()
    containers["brain-id"]["Config"]["Hostname"] = "postgres"
    containers["brain-id"]["NetworkSettings"]["Networks"][CORE]["DNSNames"] = [
        policy.team_container_name(TEAM_ID),
        "brain-id",
        "postgres",
    ]
    check(not _members_valid(core, containers, policy.CORE_KIND), "automatic Brain hostname cannot claim postgres")

    core, containers = _valid_topology()
    containers["postgres-id"]["Config"]["Labels"][policy.SHARED_ROLE_LABEL] = policy.APP_EGRESS_ROLE
    check(not _members_valid(core, containers, policy.CORE_KIND), "shared service role-label drift is rejected")


def test_workload_security_drift_fail_closed() -> None:
    _core, containers = _valid_topology()
    mutations = (
        ("wrong runtime", lambda item: item["HostConfig"].update(Runtime="runc")),
        ("privileged", lambda item: item["HostConfig"].update(Privileged=True)),
        ("unconfined seccomp", lambda item: item["HostConfig"]["SecurityOpt"].append("seccomp=unconfined")),
        ("custom seccomp", lambda item: item["HostConfig"]["SecurityOpt"].append("seccomp=/tmp/custom.json")),
        ("wrong AppArmor", lambda item: item.update(AppArmorProfile="unconfined")),
        ("wrong UID", lambda item: item["Config"].update(User="0:0")),
        ("writable root", lambda item: item["HostConfig"].update(ReadonlyRootfs=False)),
        ("capability added", lambda item: item["HostConfig"].update(CapAdd=["NET_RAW"])),
        ("published port", lambda item: item["HostConfig"].update(PublishAllPorts=True)),
        ("host PID namespace", lambda item: item["HostConfig"].update(PidMode="host")),
        ("host IPC namespace", lambda item: item["HostConfig"].update(IpcMode="host")),
        ("shared IPC namespace", lambda item: item["HostConfig"].update(IpcMode="container:other")),
        ("host UTS namespace", lambda item: item["HostConfig"].update(UTSMode="host")),
        ("host cgroup namespace", lambda item: item["HostConfig"].update(CgroupnsMode="host")),
        ("disabled user namespace remap", lambda item: item["HostConfig"].update(UsernsMode="host")),
        ("missing IPC namespace proof", lambda item: item["HostConfig"].pop("IpcMode")),
        ("missing cgroup namespace proof", lambda item: item["HostConfig"].pop("CgroupnsMode")),
        ("null IPC namespace", lambda item: item["HostConfig"].update(IpcMode=None)),
        ("null cgroup namespace", lambda item: item["HostConfig"].update(CgroupnsMode=None)),
        ("malformed IPC namespace", lambda item: item["HostConfig"].update(IpcMode=False)),
        ("malformed user namespace", lambda item: item["HostConfig"].update(UsernsMode=0)),
        ("memory drift", lambda item: item["HostConfig"].update(Memory=policy.APP_MEMORY_BYTES + 1)),
        ("swap expansion", lambda item: item["HostConfig"].update(MemorySwap=policy.APP_MEMORY_BYTES * 2)),
        ("tmpfs expansion", lambda item: item["HostConfig"].update(Tmpfs={"/tmp": "size=1g"})),
        (
            "nofile expansion",
            lambda item: item["HostConfig"].update(Ulimits=[{"Name": "nofile", "Soft": 65536, "Hard": 65536}]),
        ),
        (
            "automatic restart enabled",
            lambda item: item["HostConfig"].update(RestartPolicy={"Name": "unless-stopped"}),
        ),
        (
            "unbounded logs",
            lambda item: item["HostConfig"].update(LogConfig={"Type": "json-file", "Config": {"labels": "team.id"}}),
        ),
        ("wrong configured image", lambda item: item["Config"].update(Image="attacker:v1")),
        ("wrong immutable image ID", lambda item: item.update(Image="sha256:attacker")),
    )
    for label, mutate in mutations:
        drifted = copy.deepcopy(containers["app-id"])
        mutate(drifted)
        check(not _workload_valid(drifted), f"App {label} is rejected")

    false_nnp = copy.deepcopy(containers["app-id"])
    false_nnp["HostConfig"]["SecurityOpt"] = ["no-new-privileges:false", "apparmor=docker-default"]
    check(not _workload_valid(false_nnp), "disabled no-new-privileges is rejected")

    brain = copy.deepcopy(containers["brain-id"])
    brain["HostConfig"]["CapAdd"].append("NET_RAW")
    check(not _workload_valid(brain), "Brain cap expansion is rejected")
    brain = copy.deepcopy(containers["brain-id"])
    brain["Mounts"].append({"Destination": "/var/run/docker.sock", "Type": "bind"})
    check(not _workload_valid(brain), "Brain foreign mount is rejected")
    brain = copy.deepcopy(containers["brain-id"])
    brain["Mounts"].append({"Destination": "/config", "Type": "volume", "Name": "foreign", "RW": True})
    check(not _workload_valid(brain), "Brain foreign volume is rejected")
    brain = copy.deepcopy(containers["brain-id"])
    brain["HostConfig"]["MemoryReservation"] = policy.BRAIN_MEMORY_RESERVATION_BYTES + 1
    check(not _workload_valid(brain), "Brain memory reservation drift is rejected")

    normalized = copy.deepcopy(containers["app-id"])
    normalized["HostConfig"]["UTSMode"] = "private"
    check(
        _workload_valid(normalized),
        "Engine's explicit private UTS spelling normalizes to the same isolated posture",
    )


def load_tests(
    _loader: unittest.TestLoader,
    _tests: unittest.TestSuite,
    _pattern: str | None,
) -> unittest.TestSuite:
    suite = unittest.TestSuite()
    functions = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for function in functions:
        suite.addTest(unittest.FunctionTestCase(function))
    return suite


if __name__ == "__main__":
    unittest.main()
