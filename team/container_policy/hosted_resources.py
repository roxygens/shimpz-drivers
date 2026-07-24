"""Hosted Team isolation, capacity, networks, and authorization."""

from __future__ import annotations

import contextlib
from collections import defaultdict
from dataclasses import dataclass
from http import HTTPStatus

import cleanup_state
import docker
import docker.errors
import inference_config
import manifests
import marketplace
import marketplace_image
import runtime_state

from container_policy import network as network_policy


def _get_container(name: str):
    try:
        return runtime_state._docker.containers.get(name)
    except docker.errors.NotFound:
        return None


def _require_team_runtime() -> None:
    """Fail closed unless Docker preserves the complete hostile-tenant daemon posture."""
    try:
        info = runtime_state._docker.info()
    except docker.errors.DockerException as exc:
        raise runtime_state.ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "cannot verify Docker isolation posture") from exc
    if not isinstance(info, dict):
        raise runtime_state.ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "cannot verify Docker isolation posture")
    if not network_policy.daemon_runtime_registration_valid(info, manifests.RUNTIME, manifests.RUNTIME_PATH):
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            f"required Team runtime {manifests.RUNTIME!r} is not loaded from {manifests.RUNTIME_PATH!r}",
        )
    if not network_policy.daemon_security_options_valid(info):
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "required Docker built-in seccomp and AppArmor defaults are unavailable",
        )


def _team_runtime(container) -> str:
    """Return Docker's immutable runtime selection for one existing Team."""
    try:
        container.reload()
    except docker.errors.DockerException as exc:
        raise runtime_state.ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "cannot verify Team runtime isolation") from exc
    runtime = container.attrs.get("HostConfig", {}).get("Runtime")
    return str(runtime or "runc")


def _trusted_image_id(image_ref: str) -> str:
    """Resolve one release-owned local reference to the immutable ID Engine will execute."""
    if not isinstance(image_ref, str) or not image_ref:
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE, "Team isolation is blocked: untrusted workload image role"
        )
    try:
        image = runtime_state._docker.images.get(image_ref)
        image_id = image.id
    except (AttributeError, docker.errors.DockerException) as exc:
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Team isolation is blocked: trusted workload image is unavailable",
        ) from exc
    if not isinstance(image_id, str) or not image_id:
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE, "Team isolation is blocked: invalid workload image identity"
        )
    return image_id


def _prepare_marketplace_image(spec: marketplace.AppSpec) -> None:
    """Materialize and prove only registry-owned digest artifacts before a new App can run."""
    if not marketplace.is_digest_image(spec.image):
        return
    try:
        marketplace_image.ensure_digest_artifact(runtime_state._docker.images, spec)
    except marketplace_image.ImageTrustError as exc:
        raise runtime_state.ApiError(HTTPStatus.SERVICE_UNAVAILABLE, str(exc)) from exc


def _trusted_workload_image(container, team_id: str) -> tuple[str, str]:
    """Resolve this exact workload role's configured ref to the currently trusted immutable ID."""
    labels = container.attrs.get("Config", {}).get("Labels", {})
    if network_policy.brain_identity_valid(container.attrs, team_id):
        provider = labels.get("team.brain")
        provider_spec = manifests.BRAINS.get(provider) if isinstance(provider, str) else None
        image_ref = provider_spec.get("image") if provider_spec is not None else None
    else:
        app_id = labels.get("team.app")
        app_spec = marketplace.APPS.get(app_id) if isinstance(app_id, str) else None
        image_ref = app_spec.image if app_spec is not None else None
    if not isinstance(image_ref, str) or not image_ref:
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE, "Team isolation is blocked: untrusted workload image role"
        )
    return image_ref, _trusted_image_id(image_ref)


def _require_team_isolation_mode(
    container,
    *,
    require_running: bool,
    inspect_memo: dict[str, dict[str, dict]] | None = None,
) -> None:
    """Validate exact static posture, plus live network membership whenever the workload is running."""
    runtime = _team_runtime(container)
    if runtime != manifests.RUNTIME:
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            f"Team isolation is blocked: required runtime {manifests.RUNTIME!r}, found {runtime!r}; "
            "destroy and recreate the Team",
        )
    state = container.attrs.get("State")
    running = state.get("Running") if isinstance(state, dict) else None
    if not isinstance(running, bool) or (require_running and not running):
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Team isolation is blocked: workload running state cannot be proved",
        )
    labels = container.attrs.get("Config", {}).get("Labels", {})
    team_id = labels.get("team.id") if isinstance(labels, dict) else None
    if not isinstance(team_id, str) or not team_id:
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE, "Team isolation is blocked: invalid workload identity"
        )
    image_ref, image_id = _trusted_workload_image(container, team_id)
    if not network_policy.workload_security_valid(
        container.attrs,
        team_id,
        manifests.RUNTIME,
        expected_image_ref=image_ref,
        expected_image_id=image_id,
    ):
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Team isolation is blocked: workload security or network attachment drifted; destroy and recreate the Team",
        )
    kind = network_policy.CORE_KIND
    try:
        network = runtime_state._docker.networks.get(network_policy.network_name(team_id, kind))
    except docker.errors.DockerException as exc:
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Team isolation is blocked: required network is missing",
        ) from exc
    _require_network_policy(
        network,
        team_id,
        kind,
        # Engine 29 omits created/stopped endpoints from network inspect while preserving their
        # exact attachments in container inspect. Static workload posture above proves those
        # endpoints; a running anchor must additionally be visible as the live Brain role.
        require_brain=running and network_policy.brain_identity_valid(container.attrs, team_id),
        require_dependencies=True,
        inspect_memo=inspect_memo,
    )
    if not network_policy.workload_endpoint_valid(
        network.attrs,
        container.attrs,
        team_id,
        kind,
    ):
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Team isolation is blocked: workload endpoint identity or aliases drifted",
        )
    if running and not network_policy.workload_live_membership_valid(
        network.attrs,
        container.attrs,
        team_id,
        kind,
    ):
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Team isolation is blocked: running workload is missing from its network inventory",
        )


def _require_team_isolation(container) -> None:
    """State-aware admission: stopped is exact/static; running additionally proves live membership."""
    _require_team_isolation_mode(container, require_running=False)


def _require_running_team_isolation(
    container,
    inspect_memo: dict[str, dict[str, dict]] | None = None,
) -> None:
    """Require a running workload and its complete live core-network membership."""
    _require_team_isolation_mode(container, require_running=True, inspect_memo=inspect_memo)


def _team_not_running(container) -> bool:
    """Return true only for an exact stopped state or an absent container identity."""
    try:
        container.reload()
    except docker.errors.NotFound:
        return True
    except docker.errors.DockerException:
        return False
    state = container.attrs.get("State")
    return isinstance(state, dict) and state.get("Running") is False


def _fail_stop_team(container, *, timeout: int = 10) -> None:
    """Actively stop/kill and prove a workload is no longer running."""
    try:
        container.stop(timeout=timeout)
    except docker.errors.NotFound:
        return
    except docker.errors.DockerException:
        pass
    if _team_not_running(container):
        return
    try:
        container.kill()
    except docker.errors.NotFound:
        return
    except docker.errors.DockerException:
        pass
    if not _team_not_running(container):
        raise runtime_state.ApiError(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Team isolation failed and the workload could not be proved stopped",
        )


def _remove_team_container(container, *, timeout: int = 10) -> bool:
    """Remove one workload by immutable ID, fail-stopping any survivor before returning false."""
    container_id = container.id
    try:
        _fail_stop_team(container, timeout=timeout)
    except runtime_state.ApiError:
        return False
    with contextlib.suppress(docker.errors.DockerException):
        container.remove(force=True)
    survivor = _remaining_container(container_id)
    if survivor is None:
        return True
    if survivor is _CONTAINER_LOOKUP_FAILED:
        return False
    try:
        _fail_stop_team(survivor, timeout=timeout)
    except runtime_state.ApiError:
        return False
    with contextlib.suppress(docker.errors.DockerException):
        survivor.remove(force=True)
    return _remaining_container(container_id) is None


_CONTAINER_LOOKUP_FAILED = object()


def _remaining_container(container_id: str):
    try:
        return runtime_state._docker.containers.get(container_id)
    except docker.errors.NotFound:
        return None
    except docker.errors.DockerException:
        return _CONTAINER_LOOKUP_FAILED


def _start_team_with_isolation(container) -> None:
    """Prove a stopped workload, start it, then fail-stop unless live membership also proves."""
    _require_team_isolation(container)
    # Re-read Docker's daemon posture at the final mutation boundary. A registration or default-profile
    # drift after create/preflight must leave the hostile workload stopped.
    _require_team_runtime()
    try:
        container.start()
    except docker.errors.DockerException as exc:
        # Engine may have committed a start before the client observed its response. Treat every
        # start error as ambiguous and actively fail-stop instead of assuming the workload stayed down.
        _fail_stop_team(container)
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Team start could not be proved; workload was stopped",
        ) from exc
    try:
        _require_running_team_isolation(container)
    except runtime_state.ApiError:
        _fail_stop_team(container)
        raise


@dataclass(frozen=True)
class _MemoryUsage:
    total: int
    by_owner: dict[str, int]


@dataclass(frozen=True)
class _CapacityReservation:
    key: str
    owner: str
    memory_bytes: int
    team_slot: bool


@dataclass(frozen=True)
class _CleanupResult:
    """Proof that both runtime artifacts and scoped database state were removed."""

    artifacts_removed: bool
    db_dropped: bool

    @property
    def complete(self) -> bool:
        return self.artifacts_removed and self.db_dropped


def _capacity_key(container) -> str:
    team_id = str(container.labels.get("team.id", ""))
    if container.labels.get("team.app.driver"):
        return f"app:{team_id}:{container.labels.get('team.app', '')}"
    return f"team:{team_id}"


def _admitted_resource_containers() -> list:
    """Return every hard-limited Team resource once, or fail closed on Docker inventory errors."""
    try:
        resources = {
            container.id: container
            for label in ("team.driver", "team.app.driver")
            for container in runtime_state._docker.containers.list(all=True, filters={"label": label})
        }
    except docker.errors.DockerException as exc:
        raise runtime_state.ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "cannot verify Team memory inventory") from exc
    return list(resources.values())


def _memory_usage(*, exclude_keys: frozenset[str] = frozenset()) -> _MemoryUsage:
    """Count inspected Docker hard limits; zero/missing limits are unsafe and reject admission."""
    total = 0
    by_owner: dict[str, int] = defaultdict(int)
    for container in _admitted_resource_containers():
        if _capacity_key(container) in exclude_keys:
            # The corresponding in-flight reservation already accounts for this resource. Docker
            # exposes a newly-created container before provisioning/health commits, so counting both
            # here would spuriously halve capacity during slow creates.
            continue
        try:
            container.reload()
            raw_limit = container.attrs.get("HostConfig", {}).get("Memory")
        except (AttributeError, docker.errors.DockerException) as exc:
            raise runtime_state.ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE, "cannot verify Team memory hard limits"
            ) from exc
        if isinstance(raw_limit, bool) or not isinstance(raw_limit, (int, float)) or raw_limit <= 0:
            raise runtime_state.ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                f"Team resource {container.name!r} has no verifiable memory hard limit",
            )
        limit = int(raw_limit)
        owner = str(container.labels.get("team.owner", ""))
        total += limit
        by_owner[owner] += limit
    return _MemoryUsage(total=total, by_owner=dict(by_owner))


def _physical_teams(*, exclude_keys: frozenset[str]) -> list:
    try:
        teams = runtime_state._docker.containers.list(all=True, filters={"label": "team.driver"})
    except docker.errors.DockerException as exc:
        raise runtime_state.ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "cannot verify Team count inventory") from exc
    return [container for container in teams if _capacity_key(container) not in exclude_keys]


def _validate_capacity(
    reservation: _CapacityReservation,
    physical: list,
    usage: _MemoryUsage,
    existing: tuple[_CapacityReservation, ...],
) -> None:
    if reservation.team_slot:
        team_reservations = [item for item in existing if item.team_slot]
        current = len(physical) + len(team_reservations)
        if current >= runtime_state.MAX_TEAMS:
            raise runtime_state.ApiError(
                HTTPStatus.TOO_MANY_REQUESTS, f"team limit reached ({current}/{runtime_state.MAX_TEAMS})"
            )
        owner_count = sum(container.labels.get("team.owner", "") == reservation.owner for container in physical) + sum(
            item.owner == reservation.owner for item in team_reservations
        )
        if owner_count >= runtime_state.MAX_TEAMS_PER_OWNER:
            raise runtime_state.ApiError(
                HTTPStatus.TOO_MANY_REQUESTS,
                f"team limit reached for this owner ({owner_count}/{runtime_state.MAX_TEAMS_PER_OWNER})",
            )
    reserved_total = sum(item.memory_bytes for item in existing)
    committed_total = usage.total + reserved_total
    if committed_total + reservation.memory_bytes > runtime_state.GLOBAL_MEMORY_BUDGET_BYTES:
        detail = (
            "global Team memory budget reached "
            f"({committed_total}/{runtime_state.GLOBAL_MEMORY_BUDGET_BYTES} bytes committed)"
        )
        raise runtime_state.ApiError(HTTPStatus.TOO_MANY_REQUESTS, detail)
    owner_used = usage.by_owner.get(reservation.owner, 0) + sum(
        item.memory_bytes for item in existing if item.owner == reservation.owner
    )
    if owner_used + reservation.memory_bytes > runtime_state.OWNER_MEMORY_BUDGET_BYTES:
        detail = (
            "Team memory budget reached for this owner "
            f"({owner_used}/{runtime_state.OWNER_MEMORY_BUDGET_BYTES} bytes committed)"
        )
        raise runtime_state.ApiError(HTTPStatus.TOO_MANY_REQUESTS, detail)


@contextlib.contextmanager
def _reserve_capacity(
    key: str,
    owner: str,
    requested: int,
    *,
    team_slot: bool,
):
    """Atomically reserve quota, then release the global lock before any slow provisioning.

    A resource can become visible in Docker while its PG/image/health transaction is still running.
    Inventory therefore excludes keys with live reservations and adds those reservations exactly once.
    Rollback completes before the `finally` drops the reservation, so another caller never observes a
    phantom free slot between the failed transaction and cleanup.
    """
    reservation = _CapacityReservation(
        key=key,
        owner=owner,
        memory_bytes=requested,
        team_slot=team_slot,
    )
    while True:
        with runtime_state._capacity_lock:
            if key in runtime_state._capacity_reservations:
                raise runtime_state.ApiError(HTTPStatus.CONFLICT, "Team resource admission is already in progress")
            generation = runtime_state._capacity_generation
            existing = tuple(runtime_state._capacity_reservations.values())
            reserved_keys = frozenset(runtime_state._capacity_reservations)
        physical = _physical_teams(exclude_keys=reserved_keys) if team_slot else []
        usage = _memory_usage(exclude_keys=reserved_keys)
        with runtime_state._capacity_lock:
            if key in runtime_state._capacity_reservations:
                raise runtime_state.ApiError(HTTPStatus.CONFLICT, "Team resource admission is already in progress")
            if generation != runtime_state._capacity_generation:
                continue
            _validate_capacity(reservation, physical, usage, existing)
            runtime_state._capacity_reservations[key] = reservation
            runtime_state._capacity_generation += 1
            break
    try:
        yield
    finally:
        with runtime_state._capacity_lock:
            if runtime_state._capacity_reservations.get(key) == reservation:
                runtime_state._capacity_reservations.pop(key, None)
                runtime_state._capacity_generation += 1


def _network_container_metadata(
    network,
    inspect_memo: dict[str, dict[str, dict]] | None = None,
) -> dict[str, dict]:
    network_id = getattr(network, "id", None)
    if inspect_memo is not None and isinstance(network_id, str) and network_id in inspect_memo:
        return inspect_memo[network_id]
    try:
        network.reload()
        member_ids = network.attrs.get("Containers", {})
        if not isinstance(member_ids, dict):
            raise TypeError("invalid network member inventory")
        containers: dict[str, dict] = {}
        for container_id in member_ids:
            container = runtime_state._docker.containers.get(container_id)
            metadata = dict(container.attrs)
            metadata.setdefault("Id", container.id)
            metadata.setdefault("Name", f"/{container.name}")
            containers[container_id] = metadata
    except (AttributeError, TypeError, docker.errors.DockerException) as exc:
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "cannot verify Team network isolation",
        ) from exc
    else:
        if inspect_memo is not None:
            if not isinstance(network_id, str) or not network_id:
                raise runtime_state.ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "cannot verify Team network isolation",
                )
            inspect_memo[network_id] = containers
        return containers


def _require_network_policy(
    network,
    team_id: str,
    kind: str,
    *,
    require_brain: bool,
    require_dependencies: bool,
    inspect_memo: dict[str, dict[str, dict]] | None = None,
) -> None:
    containers = _network_container_metadata(network, inspect_memo)
    if not network_policy.network_members_valid(
        network.attrs,
        containers,
        team_id,
        kind,
        require_brain=require_brain,
        require_dependencies=require_dependencies,
    ):
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            f"Team isolation is blocked: invalid or contaminated {kind} network",
        )


def _ensure_team_network_kind(team_id: str, kind: str):
    net_name = network_policy.network_name(team_id, kind)
    try:
        network = runtime_state._docker.networks.get(net_name)
    except docker.errors.NotFound:
        try:
            network = runtime_state._docker.networks.create(
                net_name,
                driver="bridge",
                internal=True,
                attachable=False,
                labels=network_policy.network_labels(team_id, kind),
            )
        except docker.errors.DockerException as exc:
            raise runtime_state.ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                f"could not create the Team {kind} network",
            ) from exc
    except docker.errors.DockerException as exc:
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            f"could not inspect the Team {kind} network",
        ) from exc
    _require_network_policy(
        network,
        team_id,
        kind,
        require_brain=False,
        require_dependencies=False,
    )
    return network


def _ensure_team_network(team_id: str):
    return _ensure_team_network_kind(team_id, network_policy.CORE_KIND)


def _already_connected(exc: docker.errors.APIError) -> bool:
    """True only for the ONE idempotent case: this container is already on this network (403)."""
    resp = exc.response
    return (
        resp is not None
        and resp.status_code == HTTPStatus.FORBIDDEN
        and "already exists in network" in (exc.explanation or "")
    )


def _safe_connect(network, container_name: str, *, aliases: list[str] | None = None, required: bool) -> None:
    try:
        container = runtime_state._docker.containers.get(container_name)
    except docker.errors.NotFound as exc:
        if required:
            raise runtime_state.ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR, f"required shared-plane container {container_name!r} not found"
            ) from exc
        return
    expected_shared_role = network_policy.shared_service_role_for_name(container_name)
    if expected_shared_role is not None:
        try:
            container.reload()
        except docker.errors.DockerException as exc:
            raise runtime_state.ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"failed to inspect required shared service {container_name!r}",
            ) from exc
        if not network_policy.shared_service_identity_valid(container.attrs, expected_shared_role):
            raise runtime_state.ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"required shared-plane container {container_name!r} has invalid role metadata",
            )
    try:
        network.connect(container, aliases=aliases)
    except docker.errors.APIError as exc:
        if _already_connected(exc):
            return
        if required:
            raise runtime_state.ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"failed to connect required service {container_name!r} to its Team network",
            ) from exc


def _wire_network_deps(network, dependencies: list[tuple[str, list[str]]]) -> None:
    for container_name, aliases in dependencies:
        _safe_connect(network, container_name, aliases=aliases, required=True)


def _teardown_team_network_kind(team_id: str, kind: str) -> bool:
    """Remove one owned plane, or report residue without mutating any foreign identity."""
    network = _teardown_network(team_id, kind)
    if network is None:
        return True
    if network is _NETWORK_LOOKUP_FAILED:
        return False
    try:
        network.reload()
    except docker.errors.DockerException:
        return False
        # A same-name foreign network is not ours to mutate. Reuse rejects it, and teardown leaves it for
        # an operator instead of disconnecting unrelated containers based on a name alone.
    if not network_policy.network_identity_valid(network.attrs, team_id, kind):
        return False
    cleanup_complete = True
    for container_id in dict(network.attrs.get("Containers", {})):
        try:
            container = runtime_state._docker.containers.get(container_id)
            container.reload()
        except docker.errors.DockerException:
            cleanup_complete = False
            continue
        if not network_policy.network_member_managed(container.attrs, team_id, kind):
            cleanup_complete = False
            continue
        try:
            network.disconnect(container_id, force=True)
        except docker.errors.APIError:
            cleanup_complete = False
    return cleanup_complete and _remove_empty_network(network)


_NETWORK_LOOKUP_FAILED = object()


def _teardown_network(team_id: str, kind: str):
    try:
        return runtime_state._docker.networks.get(network_policy.network_name(team_id, kind))
    except docker.errors.NotFound:
        return None
    except docker.errors.DockerException:
        return _NETWORK_LOOKUP_FAILED


def _remove_empty_network(network) -> bool:
    try:
        network.reload()
        if network.attrs.get("Containers"):
            return False
        network.remove()
    except docker.errors.DockerException:
        return False
    return True


def _teardown_team_networks(team_id: str) -> bool:
    return _teardown_team_network_kind(team_id, network_policy.CORE_KIND)


def _describe(container) -> dict:
    team_id = str(container.labels.get("team.id", ""))
    try:
        inference = runtime_state._inference_store.load(team_id)
    except inference_config.InferenceConfigError:
        inference = None
    return {
        "team_id": team_id,
        "team_name": container.labels.get("team.name"),
        "owner": container.labels.get("team.owner", ""),
        "provider": inference.provider if inference is not None else None,
        "model": inference.model if inference is not None else None,
        "status": container.status,
        "container": container.name,
    }


@dataclass(frozen=True)
class _AuthorizationLease:
    team_id: str
    container_id: str
    owner: str
    principal: tuple[str, str | None]
    cleanup_nonce: str = ""


def _cleanup_record(team_id: str) -> cleanup_state.Record | None:
    try:
        return cleanup_state.load(team_id)
    except cleanup_state.CleanupStateError as exc:
        raise runtime_state.ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Team cleanup state is unavailable") from exc


def _authorize_container(team_id: str, principal: tuple[str, str | None], container) -> _AuthorizationLease:
    if not network_policy.brain_identity_valid(container.attrs, team_id):
        raise runtime_state.ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
    owner = str(container.labels.get("team.owner", ""))
    kind, account_id = principal
    if kind != "operator" and owner != account_id:
        raise runtime_state.ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
    return _AuthorizationLease(
        team_id=team_id,
        container_id=container.id,
        owner=owner,
        principal=principal,
    )


def _authorize(team_id: str, principal: tuple[str, str | None]) -> _AuthorizationLease:
    """Operator may touch any team; an account may only touch a team it owns.

    This first pass returns an identity lease. Every sensitive operation must revalidate it only after
    acquiring its lifecycle/chat lock: authorization that waited behind destroy/recreate is never
    transferable to the new container that happens to reuse the same TEAM_ID.
    """
    container = _get_container(manifests.team_container_name(team_id))
    if container is None:
        raise runtime_state.ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
    return _authorize_container(team_id, principal, container)


def _authorize_destroy(team_id: str, principal: tuple[str, str | None]) -> _AuthorizationLease:
    """Authorize against the Brain, or its durable non-runnable cleanup successor."""
    container = _get_container(manifests.team_container_name(team_id))
    if container is not None:
        return _authorize_container(team_id, principal, container)
    record = _cleanup_record(team_id)
    if record is None or not cleanup_state.principal_authorized(record, principal):
        raise runtime_state.ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
    return _AuthorizationLease(
        team_id=team_id,
        container_id=record.brain_id,
        owner=record.owner,
        principal=principal,
        cleanup_nonce=record.nonce,
    )


def _require_cleanup_authorization(team_id: str, lease: _AuthorizationLease) -> cleanup_state.Record:
    """Revalidate the exact durable ownership record after acquiring the lifecycle lock."""
    record = _cleanup_record(team_id)
    if (
        not lease.cleanup_nonce
        or lease.team_id != team_id
        or record is None
        or record.nonce != lease.cleanup_nonce
        or record.owner != lease.owner
        or record.brain_id != lease.container_id
        or not cleanup_state.principal_authorized(record, lease.principal)
        or _get_container(manifests.team_container_name(team_id)) is not None
    ):
        raise runtime_state.ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
    return record


def _require_current_authorization(
    team_id: str,
    lease: _AuthorizationLease,
    *,
    require_isolation: bool = True,
    allow_pending_cleanup: bool = False,
):
    """Revalidate owner + immutable Docker identity; caller already holds the operation lock."""
    container = _get_container(manifests.team_container_name(team_id))
    if (
        lease.cleanup_nonce
        or lease.team_id != team_id
        or container is None
        or not network_policy.brain_identity_valid(container.attrs, team_id)
        or container.id != lease.container_id
        or str(container.labels.get("team.owner", "")) != lease.owner
    ):
        # Accounts must not learn that a different tenant recreated this name. Operators receive the
        # same retry-safe 404 contract instead of accidentally mutating an object they never selected.
        raise runtime_state.ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
    kind, account_id = lease.principal
    if kind != "operator" and lease.owner != account_id:
        raise runtime_state.ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
    if not allow_pending_cleanup and _cleanup_record(team_id) is not None:
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, f"team {team_id!r} has an incomplete teardown; retry destroy")
    if require_isolation:
        _require_team_isolation(container)
    return container

    # ── installed apps (the P4 deploy arm) ───────────────────────────────────────
