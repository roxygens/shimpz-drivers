"""Hosted Team app admission, egress, installation, and inventory."""

from __future__ import annotations

import contextlib
import http.client
import time
from http import HTTPStatus
from typing import NoReturn

import assistant_secret_store
import docker
import docker.errors
import egress_policy
import manifests
import marketplace
import oauth_account_store
import pgdriver_client
import runtime_state
from assistant_human import hosted_assistants

from container_policy import hosted_resources
from container_policy import network as network_policy


def _egress_store() -> egress_policy.EgressPolicyStore:
    return egress_policy.EgressPolicyStore(
        runtime_state.APP_EGRESS_POLICY_DIR,
        runtime_state.APP_EGRESS_POLICY_GID,
        "localhost,127.0.0.1,::1,postgres,.team",
    )


def _raise_egress_error(exc: egress_policy.EgressPolicyError) -> NoReturn:
    status = (
        HTTPStatus.CONFLICT if isinstance(exc, egress_policy.EgressPolicyDriftError) else HTTPStatus.SERVICE_UNAVAILABLE
    )
    raise runtime_state.ApiError(status, "installed Assistant egress policy failed its contract") from exc


def _team_app_containers(team_id: str) -> list:
    """Every installed-app container of team `team_id` (its OWN label set — never `team.driver`)."""
    return runtime_state._docker.containers.list(all=True, filters={"label": ["team.app.driver", f"team.id={team_id}"]})


def _app_egress_token(
    team_id: str,
    app_id: str,
    *,
    create: bool = True,
    store: egress_policy.EgressPolicyStore | None = None,
) -> str | None:
    """The app instance's stable egress token (its Proxy-Authorization to app-egress-proxy).

    Kept in the policy volume (drivers + proxy only) and reused across reinstalls, exactly like
    shimpz-driver's per-app tokens — the proxy maps token → this instance's own allowlist.
    """
    try:
        current_store = store if store is not None else _egress_store()
        return current_store.token(
            manifests.team_app_container_name(team_id, app_id),
            create=create,
        )
    except egress_policy.EgressPolicyError as exc:
        _raise_egress_error(exc)


def _write_egress_policy(
    token: str,
    allowed_hosts: tuple[str, ...],
    store: egress_policy.EgressPolicyStore | None = None,
) -> None:
    try:
        current_store = store if store is not None else _egress_store()
        current_store.write(token, allowed_hosts)
    except egress_policy.EgressPolicyError as exc:
        _raise_egress_error(exc)


def _validate_egress_policy(
    team_id: str,
    app_id: str,
    allowed_hosts: tuple[str, ...],
    store: egress_policy.EgressPolicyStore | None = None,
) -> str:
    try:
        current_store = store if store is not None else _egress_store()
        return current_store.validate(
            manifests.team_app_container_name(team_id, app_id),
            allowed_hosts,
        )
    except egress_policy.EgressPolicyError as exc:
        _raise_egress_error(exc)


def _validate_admitted_egress(
    team_id: str,
    app_id: str,
    allowed_hosts: tuple[str, ...],
    store: egress_policy.EgressPolicyStore | None = None,
) -> str | None:
    if allowed_hosts:
        return _validate_egress_policy(team_id, app_id, allowed_hosts, store)
    return None


def _egress_proxy_environment(
    token: str,
    store: egress_policy.EgressPolicyStore | None = None,
) -> dict[str, str]:
    try:
        current_store = store if store is not None else _egress_store()
        return current_store.proxy_environment(token)
    except egress_policy.EgressPolicyError as exc:
        _raise_egress_error(exc)


def _validate_assistant_proxy_environment(
    container,
    token: str | None,
    allowed_hosts: tuple[str, ...],
    store: egress_policy.EgressPolicyStore | None = None,
) -> None:
    config = container.attrs.get("Config")
    raw_environment = config.get("Env") if isinstance(config, dict) else None
    environment = egress_policy.environment_map(raw_environment)
    if environment is None:
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "installed Assistant proxy environment is invalid")
    proxy_environment = {key: value for key, value in environment.items() if key.upper().endswith("_PROXY")}
    if allowed_hosts and token is None:
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "installed Assistant proxy environment failed its contract")
    expected = _egress_proxy_environment(token, store) if token is not None else {}
    if proxy_environment != expected:
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "installed Assistant proxy environment failed its contract")


def _reserve_egress_environment(
    team_id: str,
    app_id: str,
    allowed_hosts: tuple[str, ...],
    store: egress_policy.EgressPolicyStore | None = None,
) -> tuple[str | None, dict[str, str]]:
    if not allowed_hosts:
        return None, {}
    current_store = store if store is not None else _egress_store()
    token = _app_egress_token(team_id, app_id, store=current_store)
    if token is None:
        raise runtime_state.ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Assistant egress token is unavailable")
    return token, _egress_proxy_environment(token, current_store)


def _activate_admitted_egress(
    network,
    token: str | None,
    allowed_hosts: tuple[str, ...],
    store: egress_policy.EgressPolicyStore | None = None,
) -> None:
    if not allowed_hosts:
        return
    if token is None:
        raise runtime_state.ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, "Assistant egress admission failed")
    _write_egress_policy(token, allowed_hosts, store)
    # Only the authenticated app proxy may join the core network. The broad Brain proxy
    # is confined to the separate Brain-egress network and is unreachable from this App.
    hosted_resources._safe_connect(
        network,
        manifests.APP_EGRESS_CONTAINER,
        aliases=["app-egress-proxy"],
        required=True,
    )


def _remove_egress_policy(team_id: str, app_id: str) -> bool:
    """Remove an App's policy and token without losing the token needed for a retry."""
    try:
        _egress_store().remove(manifests.team_app_container_name(team_id, app_id))
    except egress_policy.EgressPolicyError:
        return False
    return True


def _probe_app_health(container, port: int, health_path: str) -> bool:
    """Probe the registry-declared endpoint; only an exact HTTP 200 proves the App ready."""
    url = f"http://127.0.0.1:{port}{health_path}"
    script = (
        "import http.client,sys\n"
        "connection=http.client.HTTPConnection('127.0.0.1', int(sys.argv[1]), timeout=3)\n"
        "try:\n"
        "    connection.request('GET', sys.argv[2])\n"
        "    print(connection.getresponse().status)\n"
        "finally:\n"
        "    connection.close()\n"
    )
    probes = (
        [
            "curl",
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "--max-time",
            "3",
            url,
        ],
        ["python3", "-c", script, str(port), health_path],
    )
    for probe in probes:
        try:
            rc, out = container.exec_run(probe)
        except docker.errors.APIError:  # the binary isn't in this image — try the other one
            continue
        answer = out.decode(errors="replace").strip() if rc == 0 else ""
        if answer.isdigit():
            return marketplace.health_response_ok(int(answer))
    return False


def _wait_app_healthy(container, port: int, health_path: str) -> tuple[bool, str]:
    for attempt in range(runtime_state.HEALTH_RETRIES):
        container.reload()
        if container.status in ("exited", "dead"):
            return False, f"container not running (status={container.status})"
        if container.status == "running" and _probe_app_health(container, port, health_path):
            return True, "ok"
        if attempt < runtime_state.HEALTH_RETRIES - 1:
            time.sleep(runtime_state.HEALTH_DELAY_SECONDS)
    return False, "health probe never answered"


def _app_ready_now(container, port: int, health_path: str) -> tuple[bool, str]:
    """Re-prove running + exact endpoint health at the install response commit seam."""
    try:
        container.reload()
        if container.status != "running":
            return False, f"container not running (status={container.status})"
        if not _probe_app_health(container, port, health_path):
            return False, "declared health endpoint did not answer 200"
            # The endpoint may have answered while the process was exiting. Reload once more so a
            # container that died during or immediately after the probe cannot be reported as running.
        container.reload()
    except docker.errors.DockerException:
        return False, "container readiness could not be verified"
    if container.status != "running":
        return False, f"container exited during its health probe (status={container.status})"
    return True, container.status


def _teardown_app(
    team_id: str,
    app_id: str,
    *,
    container=None,
    drop_db: bool = True,
) -> hosted_resources._CleanupResult:
    """Remove one exact managed App, retaining retry state whenever cleanup is incomplete."""
    admitted = _admit_teardown_app(team_id, app_id, container, drop_db)
    if isinstance(admitted, hosted_resources._CleanupResult):
        return admitted
    container, drop_db = admitted
    policy_removed = _remove_egress_policy(team_id, app_id)
    container_removed = container is None
    if container is not None and policy_removed:
        container_id = getattr(container, "id", None)
        container_removed = hosted_resources._remove_team_container(container)
        if container_removed:
            runtime_state._assistant_genesis_cache.discard(container_id)
            runtime_state._assistant_allowed_hosts_cache.discard(container_id)
            runtime_state._assistant_machine_contract_cache.discard(container_id)
    elif container is not None:
        # Preserve the labeled retry anchor, but do not leave tenant code running after a failed removal.
        with contextlib.suppress(runtime_state.ApiError):
            hosted_resources._fail_stop_team(container)

    artifacts_removed = policy_removed and container_removed
    if not drop_db:
        return hosted_resources._CleanupResult(artifacts_removed, True)
    if not artifacts_removed:
        # Keep the DB registration intact until the retryable container/policy phase has completed.
        return hosted_resources._CleanupResult(False, False)
    return _drop_app_database(team_id, app_id)


def _admit_teardown_app(team_id: str, app_id: str, container, drop_db: bool):
    if container is None:
        try:
            container = hosted_resources._get_container(manifests.team_app_container_name(team_id, app_id))
        except docker.errors.DockerException:
            return hosted_resources._CleanupResult(False, not drop_db)

    if container is not None:
        try:
            container.reload()
        except docker.errors.DockerException:
            return hosted_resources._CleanupResult(False, not drop_db)
        if not network_policy.app_identity_valid(container.attrs, team_id, app_id):
            # A deterministic-name collision or drifted ownership label is not ours to delete.
            return hosted_resources._CleanupResult(False, not drop_db)
        db_label = container.labels.get("team.app.db")
        if db_label not in (None, "0", "1"):
            return hosted_resources._CleanupResult(False, not drop_db)
            # Missing means a legacy App from before the label existed; conservatively assume it has a DB.
        drop_db = drop_db and db_label != "0"
    return container, drop_db


def _drop_app_database(team_id: str, app_id: str) -> hosted_resources._CleanupResult:
    try:
        pgdriver_client.drop_app_db(team_id, app_id)
    except pgdriver_client.PgDriverError, http.client.HTTPException, OSError, ValueError:
        return hosted_resources._CleanupResult(True, False)
    return hosted_resources._CleanupResult(True, True)


def _retain_admitted_assistant_secrets(team_id: str, app_id: str, spec: marketplace.AppSpec) -> None:
    """Prune credentials removed from the exact Assistant contract that just passed admission."""
    if spec.assistant is None:
        return
    try:
        pruned = runtime_state._assistant_secrets.retain_declared(
            team_id,
            app_id,
            tuple(sorted(spec.assistant.secrets)),
        )
    except assistant_secret_store.AssistantSecretError as exc:
        hosted_assistants._raise_assistant_secret_error(exc)
    if pruned:
        # A paused turn may still reference a secret removed by this admitted release.
        runtime_state._assistant_secret_challenges.cancel_team(team_id)
        runtime_state._assistant_input_challenges.cancel_team(team_id)
        runtime_state._assistant_approval_challenges.cancel_team(team_id)


def _retain_admitted_assistant_accounts(team_id: str, app_id: str, spec: marketplace.AppSpec) -> None:
    """Prune OAuth grants removed from the exact Assistant contract admitted at install."""
    if spec.assistant is None:
        return
    try:
        pruned = runtime_state._assistant_accounts.retain_declared(
            team_id,
            app_id,
            tuple(sorted(spec.assistant.accounts)),
        )
    except oauth_account_store.OAuthAccountStoreError as exc:
        raise runtime_state.ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Assistant account state is unavailable") from exc
    if pruned:
        runtime_state._assistant_account_challenges.cancel_team(team_id)


def _retain_admitted_assistant_private_state(team_id: str, app_id: str, spec: marketplace.AppSpec) -> None:
    _retain_admitted_assistant_secrets(team_id, app_id, spec)
    _retain_admitted_assistant_accounts(team_id, app_id, spec)


@runtime_state._serialize_against_team_chat
def _install_app(
    team_id: str,
    app_id: str,
    spec: marketplace.AppSpec,
    owner: str,
    lease: hosted_resources._AuthorizationLease,
) -> dict:
    with runtime_state._lock_for(team_id):
        team = hosted_resources._require_current_authorization(team_id, lease)
        if owner != lease.owner:
            raise runtime_state.ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
        hosted_resources._prepare_marketplace_image(spec)
        team_name = team.labels.get("team.name", "")
        existing = hosted_resources._get_container(manifests.team_app_container_name(team_id, app_id))
        if existing is not None:  # idempotent only for this exact, still-isolated installed App
            return _admit_existing_app(team_id, app_id, spec, owner, existing)
        return _provision_app(team_id, app_id, spec, owner, team_name)


def _admit_existing_app(
    team_id: str,
    app_id: str,
    spec: marketplace.AppSpec,
    owner: str,
    existing,
) -> dict[str, object]:
    egress_store = _egress_store()
    try:
        existing.reload()
    except docker.errors.DockerException as exc:
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            f"cannot verify installed app {app_id!r}",
        ) from exc
    expected_labels = {
        "team.app.driver": "1",
        "team.id": team_id,
        "team.app": app_id,
        "team.owner": owner,
    }
    if any(str(existing.labels.get(key, "")) != value for key, value in expected_labels.items()):
        raise runtime_state.ApiError(
            HTTPStatus.CONFLICT,
            f"existing container for app {app_id!r} has invalid ownership metadata; uninstall it first",
        )
    hosted_resources._require_team_isolation(existing)
    configured_image = str(existing.attrs.get("Config", {}).get("Image", ""))
    if configured_image != spec.image:
        raise runtime_state.ApiError(
            HTTPStatus.CONFLICT,
            f"installed app {app_id!r} uses a different image; uninstall it before reinstalling",
        )
    admitted_hosts = hosted_assistants._admit_app_contract(spec, existing)
    token = _validate_admitted_egress(team_id, app_id, admitted_hosts, egress_store)
    _validate_assistant_proxy_environment(existing, token, admitted_hosts, egress_store)
    ready, status = _app_ready_now(existing, spec.port, spec.health_path)
    if not ready:
        raise runtime_state.ApiError(
            HTTPStatus.CONFLICT,
            f"installed app {app_id!r} is not ready ({status}); uninstall it before reinstalling",
        )
    _retain_admitted_assistant_private_state(team_id, app_id, spec)
    return {"team_id": team_id, "app": app_id, "status": status, "installed": False}


def _provision_app(
    team_id: str,
    app_id: str,
    spec: marketplace.AppSpec,
    owner: str,
    team_name: str,
) -> dict[str, object]:
    if spec.assistant is not None:
        hosted_assistants._revoke_assistant_approval_grants(team_id, app_id)
    if len(_team_app_containers(team_id)) >= runtime_state.MAX_APPS_PER_TEAM:
        raise runtime_state.ApiError(
            HTTPStatus.TOO_MANY_REQUESTS, f"app limit reached for {team_id!r} ({runtime_state.MAX_APPS_PER_TEAM})"
        )
    key = f"app:{team_id}:{app_id}"
    with hosted_resources._reserve_capacity(key, owner, manifests.APP_MEM_LIMIT_BYTES, team_slot=False):
        committed_status = _provision_app_transaction(team_id, app_id, spec, owner, team_name)
    return {
        "team_id": team_id,
        "app": app_id,
        "status": committed_status,
        "installed": True,
        **({"database": manifests.team_app_db_project(team_id, app_id)} if spec.db else {}),
    }


def _provision_app_transaction(
    team_id: str,
    app_id: str,
    spec: marketplace.AppSpec,
    owner: str,
    team_name: str,
) -> str:
    # Return the same explicit admission error as Team create instead of relying on a
    # lower-level Docker create failure when the hostile-tenant runtime is unavailable.
    hosted_resources._require_team_runtime()
    egress_store = _egress_store()
    try:
        database_url = pgdriver_client.create_app_db(team_id, app_id)["database_url"] if spec.db else ""
        network = hosted_resources._ensure_team_network(team_id)
        token, proxy_env = _reserve_egress_environment(team_id, app_id, spec.allowed_hosts, egress_store)
        kwargs = manifests.build_team_app_kwargs(
            team_id,
            app_id,
            spec,
            database_url=database_url,
            proxy_env=proxy_env,
            owner=owner,
            team_name=team_name,
        )
        hosted_resources._require_team_runtime()
        container = runtime_state._docker.containers.create(**kwargs)
        network.disconnect(container)
        network.connect(container, aliases=[app_id, f"{app_id}.team"])
        admitted_hosts = hosted_assistants._admit_app_contract(spec, container)
        _validate_assistant_proxy_environment(container, token, admitted_hosts, egress_store)
        _activate_admitted_egress(network, token, admitted_hosts, egress_store)
        hosted_resources._start_team_with_isolation(container)
        healthy, reason = _wait_app_healthy(container, spec.port, spec.health_path)
        if not healthy:
            raise runtime_state.ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"app {app_id!r} failed its health probe ({reason}; rolled back)",
            )
        hosted_resources._require_team_isolation(container)
        ready, committed_status = _app_ready_now(container, spec.port, spec.health_path)
        if not ready:
            raise runtime_state.ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"app {app_id!r} lost readiness before install commit ({committed_status}; rolled back)",
            )
        _retain_admitted_assistant_private_state(team_id, app_id, spec)
    except Exception as exc:
        cleanup = _teardown_app(team_id, app_id, drop_db=spec.db)
        if not cleanup.complete:
            raise runtime_state.ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "app install failed and rollback is incomplete; retry uninstall or contact the operator",
            ) from exc
        if isinstance(exc, runtime_state.ApiError):
            raise
        raise runtime_state.ApiError(
            HTTPStatus.INTERNAL_SERVER_ERROR, "app install failed and was rolled back"
        ) from exc
    else:
        return committed_status


@runtime_state._serialize_against_team_chat
def _uninstall_app(team_id: str, app_id: str, lease: hosted_resources._AuthorizationLease) -> dict:
    with runtime_state._lock_for(team_id):
        # Removal is a remediation operation and must remain available for a legacy blocked Team.
        hosted_resources._require_current_authorization(team_id, lease, require_isolation=False)
        runtime_state._assistant_secret_challenges.cancel_team(team_id)
        runtime_state._assistant_account_challenges.cancel_team(team_id)
        runtime_state._assistant_input_challenges.cancel_team(team_id)
        runtime_state._assistant_approval_challenges.cancel_team(team_id)
        cleanup = _teardown_app(team_id, app_id)
        if not cleanup.complete:
            raise runtime_state.ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "app teardown is incomplete; retry uninstall or contact the operator",
            )
        try:
            runtime_state._assistant_secrets.delete_assistant(team_id, app_id)
        except assistant_secret_store.AssistantSecretError as exc:
            hosted_assistants._raise_assistant_secret_error(exc)
        try:
            runtime_state._assistant_accounts.delete_assistant(team_id, app_id)
        except oauth_account_store.OAuthAccountStoreError as exc:
            raise runtime_state.ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE, "Assistant account state is unavailable"
            ) from exc
        hosted_assistants._revoke_assistant_approval_grants(team_id, app_id)
        return {"team_id": team_id, "app": app_id, "uninstalled": True, "db_dropped": cleanup.db_dropped}


def _list_apps(team_id: str, lease: hosted_resources._AuthorizationLease) -> dict:
    with runtime_state._lock_for(team_id):
        # Read-only inventory lets the owner see and remove residual Apps without executing tenant code.
        hosted_resources._require_current_authorization(team_id, lease, require_isolation=False)
        apps = [
            {
                "app": app_id,
                "status": c.status,
                "container": c.name,
                "powers": sorted(spec.assistant.powers) if spec is not None and spec.assistant is not None else [],
            }
            for c in _team_app_containers(team_id)
            for app_id in [c.labels.get("team.app")]
            for spec in [marketplace.APPS.get(app_id) if isinstance(app_id, str) else None]
        ]
        return {"team_id": team_id, "apps": apps}
