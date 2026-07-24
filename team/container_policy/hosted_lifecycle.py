"""Hosted Team files, provisioning, teardown, status, and lifecycle operations."""

from __future__ import annotations

import http.client
from http import HTTPStatus

import assistant_secret_store
import audit
import brain_runtime_client
import cleanup_state
import docker.errors
import inference_config
import manifests
import marketplace
import oauth_account_store
import pgdriver_client
import power_journal
import runtime_state
import team_storage
from assistant_human import hosted_assistants

from container_policy import hosted_apps, hosted_resources
from container_policy import network as network_policy


def _put_inbox_file(
    team_id: str,
    filename: object,
    data: object,
    media_type: object,
    lease: hosted_resources._AuthorizationLease,
) -> dict:
    """Store an opaque object outside every Brain and Assistant filesystem."""
    if not isinstance(data, bytes) or not data or len(data) > hosted_assistants.MAX_INBOX_FILE_BYTES:
        raise runtime_state.ApiError(
            HTTPStatus.BAD_REQUEST, f"file must be 1..{hosted_assistants.MAX_INBOX_FILE_BYTES} bytes"
        )
    with runtime_state._lock_for(team_id):
        hosted_resources._require_current_authorization(team_id, lease, require_isolation=False)
        try:
            stored = runtime_state._storage().put(team_id, filename, data, media_type)
        except team_storage.StorageQuotaError as exc:
            raise runtime_state.ApiError(HTTPStatus.INSUFFICIENT_STORAGE, str(exc)) from exc
        except team_storage.StorageInputError as exc:
            raise runtime_state.ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        except team_storage.StorageError as exc:
            raise runtime_state.ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE, "Team storage failed its safety checks"
            ) from exc
        return {"team_id": team_id, "file": stored}


def _list_team_files(team_id: str, lease: hosted_resources._AuthorizationLease) -> dict:
    with runtime_state._lock_for(team_id):
        hosted_resources._require_current_authorization(team_id, lease, require_isolation=False)
        try:
            listing = runtime_state._storage().list(team_id)
        except team_storage.StorageError as exc:
            raise runtime_state.ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE, "Team storage failed its safety checks"
            ) from exc
        return {"team_id": team_id, **listing}


def _delete_team_file(team_id: str, file_id: object, lease: hosted_resources._AuthorizationLease) -> dict:
    with runtime_state._lock_for(team_id):
        hosted_resources._require_current_authorization(team_id, lease, require_isolation=False)
        try:
            result = runtime_state._storage().delete(team_id, file_id)
        except team_storage.StorageNotFoundError as exc:
            raise runtime_state.ApiError(HTTPStatus.NOT_FOUND, "file not found") from exc
        except team_storage.StorageError as exc:
            raise runtime_state.ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE, "Team storage failed its safety checks"
            ) from exc
        return {"team_id": team_id, **result}


# ── operations ───────────────────────────────────────────────────────────────
def _remove_volume(team_id: str, kind: str) -> bool:
    name = network_policy.volume_name(team_id, kind)
    try:
        volume = runtime_state._docker.volumes.get(name)
        volume.reload()
        if not network_policy.volume_identity_valid(volume.attrs, team_id, kind):
            return False
        volume.remove(force=True)
    except docker.errors.NotFound:
        return True
    except docker.errors.DockerException:
        return False
    return True


def _owned_teardown_brain(team_id: str, owner: str, brain_id: str):
    try:
        brain = hosted_resources._get_container(manifests.team_container_name(team_id))
    except docker.errors.DockerException:
        return False, None
    if brain is None:
        return True, None
    try:
        brain.reload()
    except docker.errors.DockerException:
        return False, None
    valid = (
        network_policy.brain_identity_valid(brain.attrs, team_id)
        and brain.id == brain_id
        and str(brain.labels.get("team.owner", "")) == owner
    )
    return valid, brain


def _stop_teardown_brain(brain) -> bool:
    if brain is None:
        return True
    try:
        hosted_resources._fail_stop_team(brain, timeout=30)
    except runtime_state.ApiError:
        return False
    return True


def _teardown_apps(team_id: str) -> bool:
    try:
        app_containers = hosted_apps._team_app_containers(team_id)
    except docker.errors.DockerException:
        return False
    cleanup_complete = True
    for app_container in app_containers:
        app_id = app_container.labels.get("team.app", "")
        if not isinstance(app_id, str) or marketplace.APP_ID_RE.fullmatch(app_id) is None:
            cleanup_complete = False
            continue
        # The Team-level database drop removes every registered App database in one scoped call.
        result = hosted_apps._teardown_app(team_id, app_id, container=app_container, drop_db=False)
        cleanup_complete = result.artifacts_removed and cleanup_complete
    return cleanup_complete


def _teardown_network_planes(team_id: str) -> bool:
    return hosted_resources._teardown_team_networks(team_id)


def _remove_teardown_brain(brain) -> bool:
    if brain is None:
        return True
    return hosted_resources._remove_team_container(brain, timeout=30)


def _teardown_volumes(team_id: str) -> bool:
    results = [
        _remove_volume(team_id, kind)
        for kind in (network_policy.CONFIG_VOLUME_KIND, network_policy.WORKSPACE_VOLUME_KIND)
    ]
    return all(results)


def _teardown_storage(team_id: str) -> bool:
    if runtime_state._storage_instance is None and not runtime_state.TEAM_STORAGE_ROOT.exists():
        return True
    try:
        runtime_state._storage().destroy(team_id)
    except team_storage.StorageError:
        return False
    return True


def _teardown_inference(team_id: str) -> bool:
    try:
        runtime_state._inference_store.delete(team_id)
    except inference_config.InferenceConfigError:
        return False
    return True


def _teardown_assistant_secrets(team_id: str) -> bool:
    runtime_state._assistant_secret_challenges.cancel_team(team_id)
    runtime_state._assistant_input_challenges.cancel_team(team_id)
    runtime_state._assistant_approval_challenges.cancel_team(team_id)
    try:
        runtime_state._assistant_secrets.delete_team(team_id)
    except assistant_secret_store.AssistantSecretError:
        return False
    return True


def _teardown_assistant_accounts(team_id: str) -> bool:
    runtime_state._assistant_account_challenges.cancel_team(team_id)
    try:
        runtime_state._assistant_accounts.delete_team(team_id)
    except oauth_account_store.OAuthAccountStoreError:
        return False
    return runtime_state._teardown_team_approval_grants(team_id)


def _drop_teardown_database(team_id: str, record: cleanup_state.Record) -> cleanup_state.Record | None:
    if record.db_dropped:
        return record
    try:
        pgdriver_client.drop_team(team_id)
        return cleanup_state.mark_db_dropped(record)
    except (
        pgdriver_client.PgDriverError,
        cleanup_state.CleanupStateError,
        http.client.HTTPException,
        OSError,
        ValueError,
    ):
        return None


def _finalize_teardown(team_id: str, record: cleanup_state.Record) -> bool:
    try:
        pgdriver_client.finalize_team_drop(team_id)
        cleanup_state.finish(record)
    except (
        pgdriver_client.PgDriverError,
        cleanup_state.CleanupStateError,
        http.client.HTTPException,
        OSError,
        ValueError,
    ):
        return False
    return True


def _teardown(team_id: str, *, owner: str, brain_id: str) -> hosted_resources._CleanupResult:
    """Remove every Team artifact, preserving a durable owner-bound retry anchor throughout."""
    brain_valid, brain = _owned_teardown_brain(team_id, owner, brain_id)
    if not brain_valid:
        return hosted_resources._CleanupResult(False, False)

    # Persist the immutable tenant/Brain identity before the first mutation. Once Docker releases the
    # Brain's volume references this record—not a runnable workload—authorizes only a retrying DELETE.
    try:
        record = cleanup_state.begin(team_id, owner, brain_id)
    except cleanup_state.CleanupStateError:
        return hosted_resources._CleanupResult(False, False)
    if (
        not _stop_teardown_brain(brain)
        or not _teardown_apps(team_id)
        or not _teardown_storage(team_id)
        or not _teardown_inference(team_id)
        or not _teardown_assistant_secrets(team_id)
        or not _teardown_assistant_accounts(team_id)
        or not _teardown_network_planes(team_id)
        or not _remove_teardown_brain(brain)
        or not _teardown_volumes(team_id)
    ):
        return hosted_resources._CleanupResult(False, record.db_dropped)
    record = _drop_teardown_database(team_id, record)
    if record is None:
        return hosted_resources._CleanupResult(False, False)
    # pg-driver keeps a retired, idempotent principal until this provisioner-authorized finalizer;
    # only then is the controller's cleartext principal removed. Both operations are retry-safe.
    if not _finalize_teardown(team_id, record):
        return hosted_resources._CleanupResult(False, True)
    return hosted_resources._CleanupResult(True, True)


def _create(team_id: str, body: dict, owner: str = "") -> dict:
    try:
        team_name = hosted_resources._validated_team_name(body.get("team_name", team_id))
    except ValueError as exc:
        raise runtime_state.ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    try:
        inference = inference_config.normalize(body.get("provider"), body.get("model"))
    except inference_config.InferenceConfigError as exc:
        raise runtime_state.ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    # The current hosted Team identity remains a sandboxed lifecycle anchor. Model inference is
    # now a separate service, so changing provider/model never replaces this container.
    anchor_brain = manifests.DEFAULT_BRAIN
    anchor_model = manifests.model_for_brain(anchor_brain)
    with runtime_state._lock_for(team_id):
        pending_cleanup = hosted_resources._cleanup_record(team_id)
        if pending_cleanup is not None:
            if owner and pending_cleanup.owner != owner:
                raise runtime_state.ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
            raise runtime_state.ApiError(
                HTTPStatus.CONFLICT,
                f"team {team_id!r} has an incomplete teardown; retry destroy before creating it",
            )
        existing = hosted_resources._get_container(manifests.team_container_name(team_id))
        if existing is not None:
            # An account may only "re-create" (get) its OWN team; a name collision with a different
            # owner is invisible (404), never a hijack of someone else's team.
            existing_owner = existing.labels.get("team.owner", "")
            if owner and existing_owner != owner:
                raise runtime_state.ApiError(HTTPStatus.NOT_FOUND, f"team {team_id!r} not found")
            # Upgrade fail-close: idempotent create must not bless a legacy runc container. Test
            # data can be destroyed/recreated; production migration must be an explicit release step.
            hosted_resources._require_team_runtime()
            hosted_resources._require_team_isolation(existing)
            existing_name = hosted_resources._team_name_from_anchor(existing)
            if "team_name" in body and team_name != existing_name:
                raise runtime_state.ApiError(HTTPStatus.CONFLICT, "Team name differs from the persisted identity")
            runtime_state._inference_store.save(team_id, inference)
            return {
                "team_id": team_id,
                "team_name": existing_name,
                "provider": inference.provider,
                "model": inference.model,
                "status": existing.status,
                "created": False,
            }
        if not _teardown_storage(team_id):
            raise runtime_state.ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "stale Team storage could not be cleared before creation",
            )
        # Reserve count + memory atomically, then let unrelated Teams enter admission while the
        # runtime check, credential service, Postgres, Docker start and health work proceed.
        with hosted_resources._reserve_capacity(f"team:{team_id}", owner, manifests.MEM_LIMIT_BYTES, team_slot=True):
            # Quotas are an admission decision of their own: an owner already at the limit must receive
            # 429 even while the hostile-tenant runtime is unavailable. A different owner reaches this
            # independent fail-closed host gate and still cannot provision without the required runtime.
            hosted_resources._require_team_runtime()
            # Transactional: on ANY failure, roll back everything partially created before surfacing —
            # never leak an orphan DB/role, network, or volume for an operator to hunt down later.
            container = None
            try:
                db = pgdriver_client.provision_team(team_id)
                network = hosted_resources._ensure_team_network(team_id)
                hosted_resources._wire_network_deps(network, manifests.core_deps())
                hosted_resources._require_network_policy(
                    network,
                    team_id,
                    network_policy.CORE_KIND,
                    require_brain=False,
                    require_dependencies=True,
                )
                kwargs = manifests.build_team_kwargs(
                    team_id,
                    team_name,
                    database_url=db["database_url"],
                    owner=owner,
                    brain=anchor_brain,
                    model=anchor_model,
                )
                hosted_resources._require_team_runtime()
                container = runtime_state._docker.containers.create(**kwargs)
                hosted_resources._start_team_with_isolation(container)
                runtime_state._inference_store.save(team_id, inference)
            except Exception as exc:
                cleanup = _teardown(
                    team_id,
                    owner=owner,
                    brain_id=container.id if container is not None else "",
                )
                if not cleanup.complete:
                    raise runtime_state.ApiError(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "Team create failed and rollback is incomplete; contact the operator",
                    ) from exc
                if isinstance(exc, runtime_state.ApiError):
                    raise
                raise runtime_state.ApiError(
                    HTTPStatus.INTERNAL_SERVER_ERROR, "Team create failed and was rolled back"
                ) from exc
        return {
            "team_id": team_id,
            "team_name": team_name,
            "provider": inference.provider,
            "model": inference.model,
            "status": "running",
            "created": True,
            "database": manifests.team_db_project(team_id),
        }


def _destroy(team_id: str, lease: hosted_resources._AuthorizationLease) -> dict:
    with runtime_state._lock_for(team_id):
        # Destruction is the supported remediation for a legacy or drifted runtime.
        if lease.cleanup_nonce:
            hosted_resources._require_cleanup_authorization(team_id, lease)
            container = None
        else:
            container = hosted_resources._require_current_authorization(
                team_id,
                lease,
                require_isolation=False,
                allow_pending_cleanup=True,
            )
            # A running chat is terminated by stopping the Brain before its lock can drain. Commit
            # the retry authorization first so even a timeout or ambiguous Docker stop leaves the
            # owner with a durable path back into DELETE.
            try:
                cleanup_state.begin(team_id, lease.owner, lease.container_id)
            except cleanup_state.CleanupStateError as exc:
                raise runtime_state.ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Team cleanup state is unavailable",
                ) from exc
        chat_lock = runtime_state._chat_lock_for(team_id)
        if container is not None:
            container.reload()
            if container.status == "running":
                hosted_resources._fail_stop_team(container, timeout=30)
        if not chat_lock.acquire(timeout=30):
            raise runtime_state.ApiError(HTTPStatus.CONFLICT, "the active chat turn did not stop in time")
        try:
            try:
                runtime_state._brain_runtime.delete_thread(
                    hosted_resources._brain_thread_id(team_id, lease.container_id)
                )
            except brain_runtime_client.BrainRuntimeError as exc:
                raise runtime_state.ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Team conversation state could not be deleted",
                ) from exc
            try:
                runtime_state._power_execution_journal().purge(lease.container_id)
            except power_journal.PowerJournalError as exc:
                raise runtime_state.ApiError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Team Power execution state could not be deleted",
                ) from exc
            cleanup = _teardown(team_id, owner=lease.owner, brain_id=lease.container_id)
            runtime_state._clear_team_id_runtime_state(team_id)
            if not cleanup.complete:
                raise runtime_state.ApiError(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    "Team teardown is incomplete; retry destroy or contact the operator",
                )
            return {"team_id": team_id, "destroyed": True, "db_dropped": cleanup.db_dropped}
        finally:
            chat_lock.release()


def _list(owner: str | None = None) -> dict:
    """All teams for the operator; only the account's own when `owner` is set."""
    teams = runtime_state._docker.containers.list(all=True, filters={"label": "team.driver"})
    if owner is not None:
        teams = [container for container in teams if container.labels.get("team.owner", "") == owner]
    return {"teams": [hosted_resources._describe(container) for container in teams]}


def _status(team_id: str, lease: hosted_resources._AuthorizationLease) -> dict:
    with runtime_state._lock_for(team_id):
        # Status remains readable so the UI can offer Stop/Destroy remediation.
        return hosted_resources._describe(
            hosted_resources._require_current_authorization(team_id, lease, require_isolation=False)
        )


def _inference_status(team_id: str, lease: hosted_resources._AuthorizationLease) -> dict:
    with runtime_state._lock_for(team_id):
        hosted_resources._require_current_authorization(team_id, lease)
        try:
            config = runtime_state._inference_store.load(team_id)
        except inference_config.InferenceConfigError as exc:
            raise runtime_state.ApiError(HTTPStatus.CONFLICT, "Team model provider is not configured") from exc
    return {"team_id": team_id, "provider": config.provider, "model": config.model}


def _configure_inference(team_id: str, body: object, lease: hosted_resources._AuthorizationLease) -> dict:
    if not isinstance(body, dict) or set(body) != {"provider", "model"}:
        raise runtime_state.ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "inference requires provider and model")
    try:
        config = inference_config.normalize(body["provider"], body["model"])
    except inference_config.InferenceConfigError as exc:
        raise runtime_state.ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    with runtime_state._lock_for(team_id):
        hosted_resources._require_current_authorization(team_id, lease)
        try:
            runtime_state._inference_store.save(team_id, config)
        except inference_config.InferenceConfigError as exc:
            raise runtime_state.ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE, "Team model provider could not be saved"
            ) from exc
    audit.log("inference_configure", team_id, result="ok", provider=config.provider, model=config.model)
    return {"team_id": team_id, "provider": config.provider, "model": config.model}


def _logs(team_id: str, lines: int, lease: hosted_resources._AuthorizationLease) -> dict:
    with runtime_state._lock_for(team_id):
        container = hosted_resources._require_current_authorization(team_id, lease, require_isolation=False)
        return {"team_id": team_id, "logs": container.logs(tail=lines).decode("utf-8", "replace")}


@runtime_state._serialize_against_team_chat
def _lifecycle(team_id: str, op: str, lease: hosted_resources._AuthorizationLease) -> dict:
    with runtime_state._lock_for(team_id):
        # Stop is always available as remediation. Start/restart require both an exact per-container
        # runtime and a currently registered daemon runtime; Docker may never fall back to runc.
        container = hosted_resources._require_current_authorization(team_id, lease, require_isolation=op != "stop")
        if op in {"start", "restart"}:
            hosted_resources._require_team_runtime()
        container.reload()
        # The outer Team chat slot proves no turn can observe a partially changed runtime.
        if op in {"stop", "restart"} and container.status == "running":
            hosted_resources._fail_stop_team(container, timeout=30)
        container.reload()
        if op in {"start", "restart"} and container.status != "running":
            hosted_resources._start_team_with_isolation(container)
    return {"team_id": team_id, "op": op, "status": "ok"}
