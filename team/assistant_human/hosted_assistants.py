"""Hosted Assistant contracts, RPC, private state, and Power execution."""

from __future__ import annotations

import contextlib
import socket
from dataclasses import dataclass
from http import HTTPStatus
from typing import NoReturn

import assistant_account_flow
import assistant_chat
import assistant_genesis
import assistant_help
import assistant_manifest
import assistant_secret_flow
import assistant_secret_store
import audit
import brain_credentials_client
import brain_runtime_client
import chat_orchestrator
import chat_turn_engine
import docker
import docker.errors
import manifests
import marketplace
import oauth_account_store
import oauth_http_client
import power_execution
import power_journal
import team_storage
from container_policy import network as network_policy
from http_boundary import controller_binding

from assistant_human import approval_grants as assistant_approval_grants

_controller = controller_binding.current()


# ── Controller-owned Assistant chat ─────────────────────────────────────────────────────────────
CHAT_OUTPUT_CAP = 60000
MAX_INBOX_FILE_BYTES = 25 * 1024 * 1024
MAX_FILE_BODY_BYTES = MAX_INBOX_FILE_BYTES
MAX_ASSISTANT_RPC_OUTPUT_BYTES = assistant_help.MAX_HELP_BYTES * 6 + 1024
ASSISTANT_RPC_TIMEOUT_SECONDS = 8
MAX_CHAT_FILES = 8
MAX_CHAT_ASSISTANTS = 16
CHAT_PAUSED_STATUSES = chat_turn_engine.CHAT_PAUSED_STATUSES


@dataclass(frozen=True, slots=True)
class _ActiveAssistant:
    assistant_id: str
    contract: marketplace.AssistantContract
    container: object


@dataclass(frozen=True, slots=True)
class _HostedAssistantSecretSpec:
    """Small adapter shared by the closed secret and account contracts."""

    assistant_id: str
    name: str
    powers: dict[str, object]
    secrets: dict[str, marketplace.SecretSpec]
    accounts: dict[str, marketplace.AccountSpec]


@dataclass(frozen=True, slots=True)
class _HostedPowerSecretSpec:
    secrets: tuple[str, ...]
    accounts: tuple[str, ...]
    summary: str


@dataclass(frozen=True, slots=True)
class _HostedAssistantSecretBinding:
    spec: _HostedAssistantSecretSpec


@dataclass(frozen=True, slots=True)
class _PendingHostedChat:
    """Secret-free, process-local state for one paused hosted Team turn."""

    continuation: chat_orchestrator.ChatContinuation
    assistant_ids: tuple[str, ...]
    file_ids: tuple[str, ...]
    owner: str
    identity: tuple[object, ...]
    answer_logs: tuple[tuple[str, tuple[object, ...]], ...] = ()


def _hosted_secret_spec(active: _ActiveAssistant) -> _HostedAssistantSecretSpec:
    name = active.assistant_id.replace("-", " ").title()
    return _controller._HostedAssistantSecretSpec(
        assistant_id=active.assistant_id,
        name=name,
        powers={
            power_id: _controller._HostedPowerSecretSpec(
                tuple(getattr(power, "secrets", ())),
                tuple(getattr(power, "accounts", ())),
                str(getattr(power, "summary", "")),
            )
            for power_id, power in active.contract.powers.items()
        },
        secrets=getattr(active.contract, "secrets", {}),
        accounts=getattr(active.contract, "accounts", {}),
    )


def _secret_bindings(
    bindings: dict[str, _ActiveAssistant],
) -> dict[str, _HostedAssistantSecretBinding]:
    return {
        assistant_id: _controller._HostedAssistantSecretBinding(_controller._hosted_secret_spec(active))
        for assistant_id, active in bindings.items()
    }


def _require_assistant_genesis(container) -> str:
    """Admit only one immutable, bounded Genesis file and hide package details on failure."""
    try:
        return _controller._assistant_genesis_cache.get(container)
    except assistant_genesis.GenesisError as exc:
        raise _controller.ApiError(HTTPStatus.CONFLICT, "installed Assistant Genesis failed its contract") from exc


def _require_assistant_allowed_hosts(spec: marketplace.AppSpec, container) -> tuple[str, ...]:
    """Admit the complete security manifest and return its reviewed egress set."""
    contract = spec.assistant
    if contract is None:
        raise _controller.ApiError(HTTPStatus.CONFLICT, "installed Assistant has no reviewed manifest contract")
    try:
        reviewed = assistant_manifest.reviewed_manifest_contract(
            allowed_hosts=spec.allowed_hosts,
            accounts=contract.accounts,
        )
        declared = _controller._assistant_allowed_hosts_cache.get(container, reviewed)
        _controller._assistant_machine_contract_cache.get(container, declared.accounts, contract.machine_contract)
    except assistant_manifest.ManifestError as exc:
        raise _controller.ApiError(
            HTTPStatus.CONFLICT, "installed Assistant manifest failed its reviewed contract"
        ) from exc
    else:
        return declared.allowed_hosts


def _admit_app_contract(spec: marketplace.AppSpec, container) -> tuple[str, ...]:
    if spec.assistant is not None:
        allowed_hosts = _controller._require_assistant_allowed_hosts(spec, container)
        _controller._require_assistant_genesis(container)
        return allowed_hosts
    return spec.allowed_hosts


def _power_operation(
    request: brain_runtime_client.PowerRequest,
    assistant_container_id: object,
    secret_generations: tuple[tuple[str, int], ...] = (),
    account_generations: tuple[tuple[str, int], ...] = (),
) -> power_journal.Operation:
    spec = marketplace.APPS.get(request.assistant_id)
    image = spec.image if spec is not None else ""
    return power_execution.power_operation(
        request,
        assistant_container_id,
        image,
        secret_generations,
        account_generations,
    )


def _hosted_power_identity(active: _ActiveAssistant) -> tuple[object, object]:
    config = getattr(active.container, "attrs", {}).get("Config", {})
    image = config.get("Image") if isinstance(config, dict) else None
    if not isinstance(image, str) or not image:
        spec = marketplace.APPS.get(active.assistant_id)
        image = spec.image if spec is not None else ""
    return active.container.id, image


def _close_exec_stream(stream) -> None:
    power_execution.close_exec_stream(stream)


def _installed_assistant(
    team_id: str,
    assistant_id: object,
    inspect_memo: dict[str, dict[str, dict]] | None = None,
    candidate=None,
):
    assistant_id, spec = marketplace.resolve(assistant_id)
    contract = spec.assistant
    if contract is None:
        raise _controller.ApiError(HTTPStatus.NOT_FOUND, f"{assistant_id!r} is not an Assistant")
    container = candidate
    if container is None:
        container = _controller._get_container(manifests.team_app_container_name(team_id, assistant_id))
    if container is None:
        raise _controller.ApiError(HTTPStatus.CONFLICT, f"Assistant {assistant_id!r} is not installed in this Team")
    with _controller._active_chat_guard:
        if (team_id, container.id) in _controller._blocked_power_workloads:
            raise _controller.ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Assistant Power execution is blocked until this Assistant is reinstalled",
            )
    if candidate is None:
        try:
            container.reload()
        except docker.errors.DockerException as exc:
            raise _controller.ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE, "installed Assistant could not be verified"
            ) from exc
    if (
        not network_policy.app_identity_valid(container.attrs, team_id, assistant_id)
        or str(container.attrs.get("Config", {}).get("Image", "")) != spec.image
    ):
        raise _controller.ApiError(HTTPStatus.CONFLICT, "installed Assistant failed its identity contract")
    _controller._require_running_team_isolation(container, inspect_memo)
    allowed_hosts = _controller._require_assistant_allowed_hosts(spec, container)
    egress_store = _controller._egress_store()
    token = _controller._validate_admitted_egress(team_id, assistant_id, allowed_hosts, egress_store)
    _controller._validate_assistant_proxy_environment(container, token, allowed_hosts, egress_store)
    return assistant_id, contract, container


def _active_team_assistants(team_id: str) -> tuple[_ActiveAssistant, ...]:
    active: list[_controller._ActiveAssistant] = []
    seen: set[str] = set()
    inspect_memo: dict[str, dict[str, dict]] = {}
    try:
        installed = _controller._team_app_containers(team_id)
    except docker.errors.DockerException as exc:
        raise _controller.ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "installed Assistants could not be listed") from exc
    for candidate in installed:
        assistant_id = (candidate.labels or {}).get("team.app")
        spec = marketplace.APPS.get(assistant_id) if isinstance(assistant_id, str) else None
        if spec is None or spec.assistant is None:
            continue
        try:
            candidate.reload()
        except docker.errors.DockerException as exc:
            raise _controller.ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE, "installed Assistant could not be inspected"
            ) from exc
        if candidate.status != "running":
            continue
        if assistant_id in seen:
            raise _controller.ApiError(HTTPStatus.CONFLICT, "duplicate installed Assistant identity")
        current_id, contract, container = _controller._installed_assistant(
            team_id,
            assistant_id,
            inspect_memo,
            candidate,
        )
        seen.add(current_id)
        active.append(_controller._ActiveAssistant(current_id, contract, container))
    active.sort(key=lambda item: item.assistant_id)
    return tuple(active)


def _chat_assistant_ids(value: object) -> tuple[str, ...]:
    """Return one explicit, bounded Assistant scope; empty means Brain-only."""
    if not isinstance(value, list) or len(value) > _controller.MAX_CHAT_ASSISTANTS:
        raise _controller.ApiError(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            f"assistant_ids must contain at most {_controller.MAX_CHAT_ASSISTANTS} ids",
        )
    try:
        assistant_ids = tuple(marketplace.validate_app_id(item) for item in value)
    except marketplace.MarketplaceError:
        raise _controller.ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "assistant_ids contains an invalid id") from None
    if len(set(assistant_ids)) != len(assistant_ids):
        raise _controller.ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "assistant_ids must not contain duplicate ids")
    return tuple(sorted(assistant_ids))


def _select_team_assistants(
    active: tuple[_ActiveAssistant, ...],
    assistant_ids: tuple[str, ...],
) -> tuple[_ActiveAssistant, ...]:
    active_by_id = {assistant.assistant_id: assistant for assistant in active}
    try:
        return tuple(active_by_id[assistant_id] for assistant_id in assistant_ids)
    except KeyError:
        raise _controller.ApiError(HTTPStatus.CONFLICT, "a selected Assistant is unavailable") from None


def _read_rpc_frames(raw_socket: socket.socket, deadline: float) -> tuple[bytes, bytes]:
    return power_execution.read_rpc_frames(raw_socket, deadline, _controller.MAX_ASSISTANT_RPC_OUTPUT_BYTES)


def _register_active_power(team_id: str, token: str, container) -> None:
    with _controller._active_chat_guard:
        if _controller._active_chat_tokens.get(team_id) != token or token in _controller._cancelled_chat_tokens:
            raise _controller.ApiError(HTTPStatus.CONFLICT, "brain turn stopped")
        if team_id in _controller._active_power_container_ids:
            raise _controller.ApiError(HTTPStatus.CONFLICT, "Team already has an active Assistant Power")
        _controller._active_power_container_ids[team_id] = (token, container.id)


def _release_active_power(team_id: str, token: str, container_id: str) -> None:
    with _controller._active_chat_guard:
        if _controller._active_power_container_ids.get(team_id) == (token, container_id):
            _controller._active_power_container_ids.pop(team_id, None)


def _register_optional_power(team_id: str, token: str | None, container) -> None:
    if token is not None:
        _controller._register_active_power(team_id, token, container)


def _release_optional_power(team_id: str, token: str | None, container_id: str) -> None:
    if token is not None:
        _controller._release_active_power(team_id, token, container_id)


def _raise_if_rpc_cancelled(token: str | None, exc: BaseException | None = None) -> None:
    if token is not None and _controller._token_cancelled(token):
        raise _controller.ApiError(HTTPStatus.CONFLICT, "brain turn stopped") from exc


def _fail_stop_power(team_id: str, container) -> None:
    """Prove an ambiguous Assistant RPC can no longer execute before returning an error."""
    try:
        _controller._fail_stop_team(container, timeout=3)
    except _controller.ApiError as exc:
        with _controller._active_chat_guard:
            _controller._blocked_power_workloads.add((team_id, container.id))
        raise _controller.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            "Assistant Power termination could not be proved; reinstall the Assistant",
        ) from exc


@dataclass(frozen=True, slots=True)
class AssistantRpcRequest:
    team_id: str
    container: object
    command: str
    method: str
    path: str
    payload: dict
    token: str | None
    operation: str
    detect_unsupported_path: bool = False


def _assistant_rpc_exchange(request: AssistantRpcRequest) -> object:
    team_id = request.team_id
    container = request.container
    token = request.token
    try:
        encoded = assistant_secret_flow.encode_private_rpc_envelope(request.payload)
    except assistant_secret_flow.SecretFlowError as exc:
        raise _controller.ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Power input is too large") from exc
    _controller._register_optional_power(team_id, token, container)

    def close_stream(stream: object) -> None:
        with contextlib.suppress(Exception):
            _controller._close_exec_stream(stream)

    try:
        try:
            return power_execution.rpc_exchange(
                container.id,
                [request.command, request.method, request.path],
                encoded,
                power_execution.RpcExchangeStrategy(
                    api=_controller._docker.api,
                    user="10001:10001",
                    workdir=manifests.CONTAINER_TMP,
                    timeout=_controller.ASSISTANT_RPC_TIMEOUT_SECONDS,
                    maximum=_controller.MAX_ASSISTANT_RPC_OUTPUT_BYTES,
                    transport_errors=(docker.errors.DockerException,),
                    fail_stop=lambda: _controller._fail_stop_power(team_id, container),
                    cancelled=lambda exc: _controller._raise_if_rpc_cancelled(token, exc),
                    close_stream=close_stream,
                ),
                detect_unsupported_path=request.detect_unsupported_path,
            )
        except power_execution.RpcExchangeError as exc:
            if exc.kind == "unsupported-path":
                raise _controller._UnsupportedAssistantRpcPathError(request.path) from None
            suffix = {
                "timeout": "timed out",
                "ambiguous": "status is ambiguous",
                "invalid-result": "returned an invalid result",
                "failed": "failed",
            }.get(exc.kind)
            status = power_execution.rpc_failure_status(exc.kind)
            raise _controller.ApiError(status, f"{request.operation} {suffix}") from exc
    finally:
        _controller._release_optional_power(team_id, token, container.id)


def _assistant_rpc(
    team_id: str,
    token: str,
    container,
    command: str,
    method: str,
    path: str,
    payload: dict,
) -> object:
    return _controller._assistant_rpc_exchange(
        _controller.AssistantRpcRequest(
            team_id=team_id,
            container=container,
            command=command,
            method=method,
            path=path,
            payload=payload,
            token=token,
            operation="Assistant Power",
        )
    )


def _assistant_help(
    team_id: str,
    assistant_id: str,
    lease: _controller._AuthorizationLease,
    locale: str = "en",
) -> dict[str, str]:
    """Read bounded Markdown through one fixed RPC from an installed running Assistant."""
    try:
        locale = assistant_help.validate_locale(locale)
    except ValueError as exc:
        raise _controller.ApiError(HTTPStatus.BAD_REQUEST, "Assistant Help locale is not supported") from exc
    with _controller._lock_for(team_id):
        _controller._require_current_authorization(team_id, lease)
        current_id, contract, container = _controller._installed_assistant(team_id, assistant_id)
        try:
            raw_result = _controller._assistant_rpc_exchange(
                _controller.AssistantRpcRequest(
                    team_id=team_id,
                    container=container,
                    command=contract.rpc_command,
                    method="GET",
                    path=f"/v1/help/{locale}",
                    payload={},
                    token=None,
                    operation="Assistant Help",
                    detect_unsupported_path=True,
                )
            )
        except _controller._UnsupportedAssistantRpcPathError:
            raw_result = _controller._assistant_rpc_exchange(
                _controller.AssistantRpcRequest(
                    team_id=team_id,
                    container=container,
                    command=contract.rpc_command,
                    method="GET",
                    path="/v1/help",
                    payload={},
                    token=None,
                    operation="Assistant Help",
                )
            )
    try:
        help_payload = assistant_help.validate_payload(raw_result)
    except ValueError as exc:
        raise _controller.ApiError(HTTPStatus.BAD_GATEWAY, f"Assistant Help from {current_id!r} is invalid") from exc
    return {"assistant": current_id, **help_payload}


def _raise_assistant_secret_error(exc: assistant_secret_store.AssistantSecretError) -> NoReturn:
    if isinstance(exc, assistant_secret_store.AssistantSecretMissingError):
        raise _controller.ApiError(HTTPStatus.PRECONDITION_REQUIRED, "Assistant secrets are required") from exc
    if isinstance(exc, assistant_secret_store.AssistantSecretValidationError):
        raise _controller.ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "Assistant secret values are invalid") from exc
    raise _controller.ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Assistant secret state is unavailable") from exc


def _revoke_assistant_approval_grants(team_id: str, assistant_id: str) -> None:
    try:
        _controller._assistant_approval_grants.revoke_assistant(team_id, assistant_id)
    except assistant_approval_grants.ApprovalGrantError as exc:
        raise _controller.ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Assistant approval state is unavailable") from exc


def _teardown_team_approval_grants(team_id: str) -> bool:
    try:
        _controller._assistant_approval_grants.revoke_team(team_id)
    except assistant_approval_grants.ApprovalGrantError:
        return False
    return True


def _power_secret_generations(
    team_id: str,
    active: _ActiveAssistant,
    power_id: str,
) -> tuple[tuple[str, int], ...]:
    try:
        return power_execution.secret_generations(
            active.contract.powers,
            power_id,
            lambda secret_ids: _controller._assistant_secrets.metadata(
                team_id,
                active.assistant_id,
                secret_ids,
            ),
        )
    except assistant_secret_store.AssistantSecretError as exc:
        raise power_journal.PowerJournalConflictError("Power secret state is unavailable") from exc


def _resolve_power_secrets(
    team_id: str,
    assistant_id: str,
    contract: marketplace.AssistantContract,
    power_id: str,
) -> dict[str, str]:
    power = contract.powers.get(power_id)
    if power is None:
        raise _controller.ApiError(power_execution.UNDECLARED_POWER_STATUS, "Assistant requested an undeclared Power")
    secret_ids = tuple(getattr(power, "secrets", ()))
    if not secret_ids:
        return {}
    try:
        return _controller._assistant_secrets.resolve_many(team_id, assistant_id, secret_ids)
    except assistant_secret_store.AssistantSecretError as exc:
        _controller._raise_assistant_secret_error(exc)


def _power_account_generations(
    team_id: str,
    active: _ActiveAssistant,
    power_id: str,
) -> tuple[tuple[str, int], ...]:
    try:
        return power_execution.account_generations(
            active.contract.powers,
            getattr(active.contract, "accounts", {}),
            power_id,
            lambda declarations: _controller._assistant_accounts.metadata(
                team_id,
                active.assistant_id,
                declarations,
            ),
        )
    except oauth_account_store.OAuthAccountStoreError as exc:
        raise power_journal.PowerJournalConflictError("Power account state is unavailable") from exc


def _refresh_oauth_account(
    provider: str,
    scopes: tuple[str, ...],
    refresh_token: str,
    _broker_lease: str | None,
) -> object:
    try:
        return _controller._oauth_http.refresh(
            provider_id=provider,
            client_id=_controller._cloudflare_oauth_client_id,
            client_secret=_controller._cloudflare_oauth_client_secret,
            refresh_token=refresh_token,
            scopes=scopes,
        )
    except oauth_http_client.OAuthHTTPError as exc:
        raise oauth_account_store.OAuthAccountReauthorizationError("OAuth account requires reauthorization") from exc


def _resolve_power_accounts(
    team_id: str,
    active: _ActiveAssistant,
    power_id: str,
) -> dict[str, dict[str, str]]:
    try:
        return assistant_account_flow.resolve_power_accounts(
            team_id,
            _controller._hosted_secret_spec(active),
            power_id,
            _controller._assistant_accounts,
            _controller._refresh_oauth_account,
        )
    except assistant_account_flow.AccountFlowError as exc:
        raise _controller.ApiError(
            power_execution.ACCOUNT_PRECONDITION_STATUS, "Assistant account is unavailable"
        ) from exc


def _require_hosted_power_rpc_envelope(
    team_id: str,
    bindings: dict[str, _ActiveAssistant],
    request: brain_runtime_client.PowerRequest,
    answers: tuple[object, ...] = (),
) -> None:
    active = bindings.get(request.assistant_id)
    if active is None:
        raise _controller.ApiError(HTTPStatus.CONFLICT, "Brain requested an unavailable Assistant")
    try:
        power_execution.require_rpc_envelope(
            active,
            request,
            lambda binding, power_id: _controller._resolve_power_secrets(
                team_id,
                binding.assistant_id,
                binding.contract,
                power_id,
            ),
            lambda binding, power_id: _controller._resolve_power_accounts(team_id, binding, power_id),
            answers,
        )
    except assistant_secret_flow.SecretFlowError as exc:
        raise _controller.ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Assistant Power input is too large") from exc


def _contains_secret(value: object, secrets_by_id: dict[str, str]) -> bool:
    return power_execution.contains_secret(value, secrets_by_id)


def _assistant_secret_inventory(
    team_id: str,
    lease: _controller._AuthorizationLease,
) -> dict[str, object]:
    with _controller._lock_for(team_id):
        _controller._require_current_authorization(team_id, lease, require_isolation=False)
        try:
            return assistant_secret_flow.inventory_payload(
                team_id,
                _controller._installed_assistant_secret_specs(team_id),
                _controller._assistant_secrets,
            )
        except assistant_secret_store.AssistantSecretError as exc:
            _controller._raise_assistant_secret_error(exc)


def _assistant_account_inventory(
    team_id: str,
    lease: _controller._AuthorizationLease,
) -> dict[str, object]:
    with _controller._lock_for(team_id):
        _controller._require_current_authorization(team_id, lease, require_isolation=False)
        try:
            payload = assistant_account_flow.inventory_payload(
                team_id,
                _controller._installed_assistant_secret_specs(team_id),
                _controller._assistant_accounts,
            )
        except oauth_account_store.OAuthAccountStoreError as exc:
            raise _controller.ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE, "Assistant account state is unavailable"
            ) from exc
        except assistant_account_flow.AccountFlowError as exc:
            raise _controller.ApiError(HTTPStatus.CONFLICT, "Assistant account contract is unavailable") from exc
    return {"team_id": team_id, **payload}


def _installed_assistant_secret_specs(team_id: str) -> tuple[_HostedAssistantSecretSpec, ...]:
    specs: list[_controller._HostedAssistantSecretSpec] = []
    seen: set[str] = set()
    try:
        containers = _controller._team_app_containers(team_id)
    except docker.errors.DockerException as exc:
        raise _controller.ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "installed Assistants could not be listed") from exc
    for container in containers:
        assistant_id = (container.labels or {}).get("team.app")
        app_spec = marketplace.APPS.get(assistant_id) if isinstance(assistant_id, str) else None
        if app_spec is None or app_spec.assistant is None:
            continue
        if assistant_id in seen:
            raise _controller.ApiError(HTTPStatus.CONFLICT, "duplicate installed Assistant identity")
        seen.add(assistant_id)
        specs.append(
            _controller._hosted_secret_spec(_controller._ActiveAssistant(assistant_id, app_spec.assistant, container))
        )
    return tuple(specs)


@_controller._serialize_against_team_chat
def _replace_assistant_secrets(
    team_id: str,
    body: object,
    lease: _controller._AuthorizationLease,
) -> dict[str, object]:
    """Atomically rotate declared credentials after revalidating the exact installed Assistant."""
    with _controller._lock_for(team_id):
        _controller._require_current_authorization(team_id, lease, require_isolation=False)
        if not isinstance(body, dict):
            raise _controller.ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "Assistant secret replacement is invalid")
        try:
            assistant_id, contract, container = _controller._installed_assistant(team_id, body.get("assistant_id"))
            spec = _controller._hosted_secret_spec(_controller._ActiveAssistant(assistant_id, contract, container))
            replacements = assistant_secret_flow.replacement_values(spec, body)
        except (marketplace.MarketplaceError, assistant_secret_flow.SecretFlowError) as exc:
            raise _controller.ApiError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "Assistant secret replacement is invalid",
            ) from exc
        inventory_specs = _controller._installed_assistant_secret_specs(team_id)
        # A paused continuation is generation-bound; it must never overwrite this rotation later.
        _controller._assistant_secret_challenges.cancel_team(team_id)
        _controller._assistant_input_challenges.cancel_team(team_id)
        _controller._assistant_approval_challenges.cancel_team(team_id)
        try:
            _controller._assistant_secrets.put_many(team_id, assistant_id, replacements)
            return assistant_secret_flow.inventory_payload(team_id, inventory_specs, _controller._assistant_secrets)
        except assistant_secret_store.AssistantSecretError as exc:
            _controller._raise_assistant_secret_error(exc)


def _pending_chat_secrets(
    team_id: str,
    lease: _controller._AuthorizationLease,
) -> dict[str, object]:
    with _controller._lock_for(team_id):
        _controller._require_current_authorization(team_id, lease, require_isolation=False)
        challenge = _controller._assistant_secret_challenges.current(team_id)
    return (
        assistant_secret_flow.challenge_payload(challenge)
        if challenge is not None
        else {"team_id": team_id, "status": "none"}
    )


@dataclass(frozen=True, slots=True)
class PowerInvocationRequest:
    team_id: str
    token: str
    assistant_id: str
    contract: marketplace.AssistantContract
    container: object
    power: object
    payload: object
    answers: tuple[object, ...] = ()
    inspect_memo: dict[str, dict[str, dict]] | None = None


def _invoke_assistant_power(request: PowerInvocationRequest) -> dict[str, object]:
    team_id = request.team_id
    assistant_id = request.assistant_id
    contract = request.contract
    container = request.container
    power = request.power
    answers = request.answers
    if (
        not isinstance(power, str)
        or assistant_chat.POWER_ID_RE.fullmatch(power) is None
        or power not in contract.powers
    ):
        raise _controller.ApiError(power_execution.UNDECLARED_POWER_STATUS, "Assistant requested an undeclared Power")
    try:
        safe_input = marketplace.validate_power_input(assistant_id, power, request.payload)
    except ValueError as exc:
        raise _controller.ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc)) from exc
    _current_id, _current_contract, current_container = _controller._installed_assistant(
        team_id,
        assistant_id,
        request.inspect_memo,
    )
    if current_container.id != container.id:
        raise _controller.ApiError(HTTPStatus.CONFLICT, "installed Assistant changed during the chat turn")
    power_spec = contract.powers[power]
    secret_values = _controller._resolve_power_secrets(team_id, assistant_id, contract, power)
    active = _controller._ActiveAssistant(assistant_id, contract, container)
    account_values = _controller._resolve_power_accounts(team_id, active, power)
    audit.log(
        "assistant_power",
        team_id,
        result="ok",
        phase="started",
        assistant=assistant_id,
        power=power,
    )
    try:
        raw_result = _controller._assistant_rpc(
            team_id,
            request.token,
            container,
            contract.rpc_command,
            power_spec.method,
            power_spec.path,
            {
                "input": safe_input,
                "secrets": secret_values,
                "accounts": account_values,
                "answers": list(answers),
            },
        )
    except _controller.ApiError as exc:
        audit.log(
            "assistant_power",
            team_id,
            result="error",
            assistant=assistant_id,
            power=power,
            status=int(exc.status),
        )
        raise
    try:
        projected = power_execution.project_rpc_result(
            raw_result,
            secret_values,
            account_values,
            answers,
            lambda value: marketplace.validate_power_output(assistant_id, power, value),
        )
    except power_execution.RpcSecretExposureError:
        audit.log(
            "assistant_power",
            team_id,
            result="error",
            assistant=assistant_id,
            power=power,
            reason="secret-exposure",
        )
        raise _controller.ApiError(HTTPStatus.BAD_GATEWAY, "Assistant Power exposed protected data") from None
    except power_execution.RpcInvalidResultError as exc:
        audit.log(
            "assistant_power",
            team_id,
            result="error",
            assistant=assistant_id,
            power=power,
            reason="invalid-output",
        )
        raise _controller.ApiError(HTTPStatus.BAD_GATEWAY, "Assistant Power returned an invalid result") from exc
    if projected.suspended:
        audit.log(
            "assistant_power",
            team_id,
            result="ok",
            phase="suspended",
            assistant=assistant_id,
            power=power,
        )
        return {"assistant": assistant_id, "power": power, "suspend": projected.value}
    audit.log(
        "assistant_power",
        team_id,
        result="ok",
        phase="completed",
        assistant=assistant_id,
        power=power,
    )
    return {"assistant": assistant_id, "power": power, "result": projected.value}


def _validate_assistant_power_input(bindings, assistant_id: str, power: str, power_input) -> object:
    """Normalize one hosted Power input without touching Docker or another external system."""
    if assistant_id not in bindings:
        raise _controller.ApiError(HTTPStatus.CONFLICT, "Brain requested an unavailable Assistant")
    try:
        return marketplace.validate_power_input(assistant_id, power, power_input)
    except ValueError as exc:
        raise _controller.ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc)) from exc


def _chat_file_metadata(team_id: str, file_ids: object) -> list[dict[str, object]]:
    if file_ids is None:
        return []
    if not isinstance(file_ids, list) or len(file_ids) > _controller.MAX_CHAT_FILES:
        raise _controller.ApiError(
            HTTPStatus.BAD_REQUEST, f"files must contain at most {_controller.MAX_CHAT_FILES} opaque ids"
        )
    try:
        return _controller._storage().metadata(team_id, file_ids)
    except team_storage.StorageNotFoundError as exc:
        raise _controller.ApiError(HTTPStatus.NOT_FOUND, "selected file not found in this Team") from exc
    except team_storage.StorageInputError as exc:
        raise _controller.ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    except team_storage.StorageError as exc:
        raise _controller.ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Team storage failed its safety checks") from exc


def _model_credential(owner: str, provider: str) -> tuple[str, int]:
    if not owner:
        raise _controller.ApiError(HTTPStatus.CONFLICT, "this Team has no account owner for model credentials")
    try:
        credential = brain_credentials_client.resolve(owner, provider)
    except brain_credentials_client.BrainCredentialError as exc:
        raise _controller.ApiError(HTTPStatus.BAD_GATEWAY, "model credential service is unavailable") from exc
    if credential is None:
        raise _controller.ApiError(HTTPStatus.CONFLICT, f"configure the {provider!r} API key before chatting")
    auth_type, api_key, generation = credential
    if auth_type != "api_key":
        raise _controller.ApiError(HTTPStatus.CONFLICT, "the selected model provider requires an API key")
    return api_key, generation


def _require_model_credential_current(owner: str, provider: str, generation: int) -> None:
    try:
        current = brain_credentials_client.generation_is_current(owner, provider, generation)
    except brain_credentials_client.BrainCredentialError as exc:
        raise _controller.ApiError(HTTPStatus.BAD_GATEWAY, "model credential could not be verified") from exc
    if not current:
        raise _controller.ApiError(HTTPStatus.CONFLICT, "model credential changed or was revoked; retry")
