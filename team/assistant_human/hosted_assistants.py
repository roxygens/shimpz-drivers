"""Hosted Assistant contracts, RPC, private state, and Power execution."""

from __future__ import annotations

import contextlib
import socket
from dataclasses import dataclass
from http import HTTPStatus

import assistant_account_flow
import assistant_chat
import assistant_help
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
import runtime_state
import team_storage
from container_policy import hosted_apps, hosted_resources
from container_policy import network as network_policy

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
    return _HostedAssistantSecretSpec(
        assistant_id=active.assistant_id,
        name=name,
        powers={
            power_id: _HostedPowerSecretSpec(
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
        assistant_id: _HostedAssistantSecretBinding(_hosted_secret_spec(active))
        for assistant_id, active in bindings.items()
    }


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
        raise runtime_state.ApiError(HTTPStatus.NOT_FOUND, f"{assistant_id!r} is not an Assistant")
    container = candidate
    if container is None:
        container = hosted_resources._get_container(manifests.team_app_container_name(team_id, assistant_id))
    if container is None:
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, f"Assistant {assistant_id!r} is not installed in this Team")
    with runtime_state._active_chat_guard:
        if (team_id, container.id) in runtime_state._blocked_power_workloads:
            raise runtime_state.ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Assistant Power execution is blocked until this Assistant is reinstalled",
            )
    if candidate is None:
        try:
            container.reload()
        except docker.errors.DockerException as exc:
            raise runtime_state.ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE, "installed Assistant could not be verified"
            ) from exc
    if (
        not network_policy.app_identity_valid(container.attrs, team_id, assistant_id)
        or str(container.attrs.get("Config", {}).get("Image", "")) != spec.image
    ):
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "installed Assistant failed its identity contract")
    hosted_resources._require_running_team_isolation(container, inspect_memo)
    allowed_hosts = hosted_apps._require_assistant_allowed_hosts(spec, container)
    egress_store = hosted_apps._egress_store()
    token = hosted_apps._validate_admitted_egress(team_id, assistant_id, allowed_hosts, egress_store)
    hosted_apps._validate_assistant_proxy_environment(container, token, allowed_hosts, egress_store)
    return assistant_id, contract, container


def _active_team_assistants(team_id: str) -> tuple[_ActiveAssistant, ...]:
    active: list[_ActiveAssistant] = []
    seen: set[str] = set()
    inspect_memo: dict[str, dict[str, dict]] = {}
    try:
        installed = hosted_apps._team_app_containers(team_id)
    except docker.errors.DockerException as exc:
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE, "installed Assistants could not be listed"
        ) from exc
    for candidate in installed:
        assistant_id = (candidate.labels or {}).get("team.app")
        spec = marketplace.APPS.get(assistant_id) if isinstance(assistant_id, str) else None
        if spec is None or spec.assistant is None:
            continue
        try:
            candidate.reload()
        except docker.errors.DockerException as exc:
            raise runtime_state.ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE, "installed Assistant could not be inspected"
            ) from exc
        if candidate.status != "running":
            continue
        if assistant_id in seen:
            raise runtime_state.ApiError(HTTPStatus.CONFLICT, "duplicate installed Assistant identity")
        current_id, contract, container = _installed_assistant(
            team_id,
            assistant_id,
            inspect_memo,
            candidate,
        )
        seen.add(current_id)
        active.append(_ActiveAssistant(current_id, contract, container))
    active.sort(key=lambda item: item.assistant_id)
    return tuple(active)


def _chat_assistant_ids(value: object) -> tuple[str, ...]:
    """Return one explicit, bounded Assistant scope; empty means Brain-only."""
    if not isinstance(value, list) or len(value) > MAX_CHAT_ASSISTANTS:
        raise runtime_state.ApiError(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            f"assistant_ids must contain at most {MAX_CHAT_ASSISTANTS} ids",
        )
    try:
        assistant_ids = tuple(marketplace.validate_app_id(item) for item in value)
    except marketplace.MarketplaceError:
        raise runtime_state.ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "assistant_ids contains an invalid id") from None
    if len(set(assistant_ids)) != len(assistant_ids):
        raise runtime_state.ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "assistant_ids must not contain duplicate ids")
    return tuple(sorted(assistant_ids))


def _select_team_assistants(
    active: tuple[_ActiveAssistant, ...],
    assistant_ids: tuple[str, ...],
) -> tuple[_ActiveAssistant, ...]:
    active_by_id = {assistant.assistant_id: assistant for assistant in active}
    try:
        return tuple(active_by_id[assistant_id] for assistant_id in assistant_ids)
    except KeyError:
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "a selected Assistant is unavailable") from None


def _read_rpc_frames(raw_socket: socket.socket, deadline: float) -> tuple[bytes, bytes]:
    return power_execution.read_rpc_frames(raw_socket, deadline, MAX_ASSISTANT_RPC_OUTPUT_BYTES)


def _register_active_power(team_id: str, token: str, container) -> None:
    with runtime_state._active_chat_guard:
        if runtime_state._active_chat_tokens.get(team_id) != token or token in runtime_state._cancelled_chat_tokens:
            raise runtime_state.ApiError(HTTPStatus.CONFLICT, "brain turn stopped")
        if team_id in runtime_state._active_power_container_ids:
            raise runtime_state.ApiError(HTTPStatus.CONFLICT, "Team already has an active Assistant Power")
        runtime_state._active_power_container_ids[team_id] = (token, container.id)


def _release_active_power(team_id: str, token: str, container_id: str) -> None:
    with runtime_state._active_chat_guard:
        if runtime_state._active_power_container_ids.get(team_id) == (token, container_id):
            runtime_state._active_power_container_ids.pop(team_id, None)


def _register_optional_power(team_id: str, token: str | None, container) -> None:
    if token is not None:
        _register_active_power(team_id, token, container)


def _release_optional_power(team_id: str, token: str | None, container_id: str) -> None:
    if token is not None:
        _release_active_power(team_id, token, container_id)


def _raise_if_rpc_cancelled(token: str | None, exc: BaseException | None = None) -> None:
    if token is not None and runtime_state._token_cancelled(token):
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "brain turn stopped") from exc


def _fail_stop_power(team_id: str, container) -> None:
    """Prove an ambiguous Assistant RPC can no longer execute before returning an error."""
    try:
        hosted_resources._fail_stop_team(container, timeout=3)
    except runtime_state.ApiError as exc:
        with runtime_state._active_chat_guard:
            runtime_state._blocked_power_workloads.add((team_id, container.id))
        raise runtime_state.ApiError(
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
        raise runtime_state.ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Power input is too large") from exc
    _register_optional_power(team_id, token, container)

    def close_stream(stream: object) -> None:
        with contextlib.suppress(Exception):
            _close_exec_stream(stream)

    try:
        try:
            return power_execution.rpc_exchange(
                container.id,
                [request.command, request.method, request.path],
                encoded,
                power_execution.RpcExchangeStrategy(
                    api=runtime_state._docker.api,
                    user="10001:10001",
                    workdir=manifests.CONTAINER_TMP,
                    timeout=ASSISTANT_RPC_TIMEOUT_SECONDS,
                    maximum=MAX_ASSISTANT_RPC_OUTPUT_BYTES,
                    transport_errors=(docker.errors.DockerException,),
                    fail_stop=lambda: _fail_stop_power(team_id, container),
                    cancelled=lambda exc: _raise_if_rpc_cancelled(token, exc),
                    close_stream=close_stream,
                ),
                detect_unsupported_path=request.detect_unsupported_path,
            )
        except power_execution.RpcExchangeError as exc:
            if exc.kind == "unsupported-path":
                raise runtime_state._UnsupportedAssistantRpcPathError(request.path) from None
            suffix = {
                "timeout": "timed out",
                "ambiguous": "status is ambiguous",
                "invalid-result": "returned an invalid result",
                "failed": "failed",
            }.get(exc.kind)
            status = power_execution.rpc_failure_status(exc.kind)
            raise runtime_state.ApiError(status, f"{request.operation} {suffix}") from exc
    finally:
        _release_optional_power(team_id, token, container.id)


def _assistant_rpc(
    team_id: str,
    token: str,
    container,
    command: str,
    method: str,
    path: str,
    payload: dict,
) -> object:
    return _assistant_rpc_exchange(
        AssistantRpcRequest(
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
    lease: hosted_resources._AuthorizationLease,
    locale: str = "en",
) -> dict[str, str]:
    """Read bounded Markdown through one fixed RPC from an installed running Assistant."""
    try:
        locale = assistant_help.validate_locale(locale)
    except ValueError as exc:
        raise runtime_state.ApiError(HTTPStatus.BAD_REQUEST, "Assistant Help locale is not supported") from exc
    with runtime_state._lock_for(team_id):
        hosted_resources._require_current_authorization(team_id, lease)
        current_id, contract, container = _installed_assistant(team_id, assistant_id)
        try:
            raw_result = _assistant_rpc_exchange(
                AssistantRpcRequest(
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
        except runtime_state._UnsupportedAssistantRpcPathError:
            raw_result = _assistant_rpc_exchange(
                AssistantRpcRequest(
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
        raise runtime_state.ApiError(HTTPStatus.BAD_GATEWAY, f"Assistant Help from {current_id!r} is invalid") from exc
    return {"assistant": current_id, **help_payload}


def _power_secret_generations(
    team_id: str,
    active: _ActiveAssistant,
    power_id: str,
) -> tuple[tuple[str, int], ...]:
    try:
        return power_execution.secret_generations(
            active.contract.powers,
            power_id,
            lambda secret_ids: runtime_state._assistant_secrets.metadata(
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
        raise runtime_state.ApiError(power_execution.UNDECLARED_POWER_STATUS, "Assistant requested an undeclared Power")
    secret_ids = tuple(getattr(power, "secrets", ()))
    if not secret_ids:
        return {}
    try:
        return runtime_state._assistant_secrets.resolve_many(team_id, assistant_id, secret_ids)
    except assistant_secret_store.AssistantSecretError as exc:
        runtime_state._raise_assistant_secret_error(exc)


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
            lambda declarations: runtime_state._assistant_accounts.metadata(
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
        return runtime_state._oauth_http.refresh(
            provider_id=provider,
            client_id=runtime_state._cloudflare_oauth_client_id,
            client_secret=runtime_state._cloudflare_oauth_client_secret,
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
            _hosted_secret_spec(active),
            power_id,
            runtime_state._assistant_accounts,
            _refresh_oauth_account,
        )
    except assistant_account_flow.AccountFlowError as exc:
        raise runtime_state.ApiError(
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
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "Brain requested an unavailable Assistant")
    try:
        power_execution.require_rpc_envelope(
            active,
            request,
            lambda binding, power_id: _resolve_power_secrets(
                team_id,
                binding.assistant_id,
                binding.contract,
                power_id,
            ),
            lambda binding, power_id: _resolve_power_accounts(team_id, binding, power_id),
            answers,
        )
    except assistant_secret_flow.SecretFlowError as exc:
        raise runtime_state.ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Assistant Power input is too large") from exc


def _contains_secret(value: object, secrets_by_id: dict[str, str]) -> bool:
    return power_execution.contains_secret(value, secrets_by_id)


def _assistant_secret_inventory(
    team_id: str,
    lease: hosted_resources._AuthorizationLease,
) -> dict[str, object]:
    with runtime_state._lock_for(team_id):
        hosted_resources._require_current_authorization(team_id, lease, require_isolation=False)
        try:
            return assistant_secret_flow.inventory_payload(
                team_id,
                _installed_assistant_secret_specs(team_id),
                runtime_state._assistant_secrets,
            )
        except assistant_secret_store.AssistantSecretError as exc:
            runtime_state._raise_assistant_secret_error(exc)


def _assistant_account_inventory(
    team_id: str,
    lease: hosted_resources._AuthorizationLease,
) -> dict[str, object]:
    with runtime_state._lock_for(team_id):
        hosted_resources._require_current_authorization(team_id, lease, require_isolation=False)
        try:
            payload = assistant_account_flow.inventory_payload(
                team_id,
                _installed_assistant_secret_specs(team_id),
                runtime_state._assistant_accounts,
            )
        except oauth_account_store.OAuthAccountStoreError as exc:
            raise runtime_state.ApiError(
                HTTPStatus.SERVICE_UNAVAILABLE, "Assistant account state is unavailable"
            ) from exc
        except assistant_account_flow.AccountFlowError as exc:
            raise runtime_state.ApiError(HTTPStatus.CONFLICT, "Assistant account contract is unavailable") from exc
    return {"team_id": team_id, **payload}


def _installed_assistant_secret_specs(team_id: str) -> tuple[_HostedAssistantSecretSpec, ...]:
    specs: list[_HostedAssistantSecretSpec] = []
    seen: set[str] = set()
    try:
        containers = hosted_apps._team_app_containers(team_id)
    except docker.errors.DockerException as exc:
        raise runtime_state.ApiError(
            HTTPStatus.SERVICE_UNAVAILABLE, "installed Assistants could not be listed"
        ) from exc
    for container in containers:
        assistant_id = (container.labels or {}).get("team.app")
        app_spec = marketplace.APPS.get(assistant_id) if isinstance(assistant_id, str) else None
        if app_spec is None or app_spec.assistant is None:
            continue
        if assistant_id in seen:
            raise runtime_state.ApiError(HTTPStatus.CONFLICT, "duplicate installed Assistant identity")
        seen.add(assistant_id)
        specs.append(_hosted_secret_spec(_ActiveAssistant(assistant_id, app_spec.assistant, container)))
    return tuple(specs)


@runtime_state._serialize_against_team_chat
def _replace_assistant_secrets(
    team_id: str,
    body: object,
    lease: hosted_resources._AuthorizationLease,
) -> dict[str, object]:
    """Atomically rotate declared credentials after revalidating the exact installed Assistant."""
    with runtime_state._lock_for(team_id):
        hosted_resources._require_current_authorization(team_id, lease, require_isolation=False)
        if not isinstance(body, dict):
            raise runtime_state.ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "Assistant secret replacement is invalid")
        try:
            assistant_id, contract, container = _installed_assistant(team_id, body.get("assistant_id"))
            spec = _hosted_secret_spec(_ActiveAssistant(assistant_id, contract, container))
            replacements = assistant_secret_flow.replacement_values(spec, body)
        except (marketplace.MarketplaceError, assistant_secret_flow.SecretFlowError) as exc:
            raise runtime_state.ApiError(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "Assistant secret replacement is invalid",
            ) from exc
        inventory_specs = _installed_assistant_secret_specs(team_id)
        # A paused continuation is generation-bound; it must never overwrite this rotation later.
        runtime_state._assistant_secret_challenges.cancel_team(team_id)
        runtime_state._assistant_input_challenges.cancel_team(team_id)
        runtime_state._assistant_approval_challenges.cancel_team(team_id)
        try:
            runtime_state._assistant_secrets.put_many(team_id, assistant_id, replacements)
            return assistant_secret_flow.inventory_payload(team_id, inventory_specs, runtime_state._assistant_secrets)
        except assistant_secret_store.AssistantSecretError as exc:
            runtime_state._raise_assistant_secret_error(exc)


def _pending_chat_secrets(
    team_id: str,
    lease: hosted_resources._AuthorizationLease,
) -> dict[str, object]:
    with runtime_state._lock_for(team_id):
        hosted_resources._require_current_authorization(team_id, lease, require_isolation=False)
        challenge = runtime_state._assistant_secret_challenges.current(team_id)
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
        raise runtime_state.ApiError(power_execution.UNDECLARED_POWER_STATUS, "Assistant requested an undeclared Power")
    try:
        safe_input = marketplace.validate_power_input(assistant_id, power, request.payload)
    except ValueError as exc:
        raise runtime_state.ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc)) from exc
    _current_id, _current_contract, current_container = _installed_assistant(
        team_id,
        assistant_id,
        request.inspect_memo,
    )
    if current_container.id != container.id:
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "installed Assistant changed during the chat turn")
    power_spec = contract.powers[power]
    secret_values = _resolve_power_secrets(team_id, assistant_id, contract, power)
    active = _ActiveAssistant(assistant_id, contract, container)
    account_values = _resolve_power_accounts(team_id, active, power)
    audit.log(
        "assistant_power",
        team_id,
        result="ok",
        phase="started",
        assistant=assistant_id,
        power=power,
    )
    try:
        raw_result = _assistant_rpc(
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
    except runtime_state.ApiError as exc:
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
        raise runtime_state.ApiError(HTTPStatus.BAD_GATEWAY, "Assistant Power exposed protected data") from None
    except power_execution.RpcInvalidResultError as exc:
        audit.log(
            "assistant_power",
            team_id,
            result="error",
            assistant=assistant_id,
            power=power,
            reason="invalid-output",
        )
        raise runtime_state.ApiError(HTTPStatus.BAD_GATEWAY, "Assistant Power returned an invalid result") from exc
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
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "Brain requested an unavailable Assistant")
    try:
        return marketplace.validate_power_input(assistant_id, power, power_input)
    except ValueError as exc:
        raise runtime_state.ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, str(exc)) from exc


def _chat_file_metadata(team_id: str, file_ids: object) -> list[dict[str, object]]:
    if file_ids is None:
        return []
    if not isinstance(file_ids, list) or len(file_ids) > MAX_CHAT_FILES:
        raise runtime_state.ApiError(HTTPStatus.BAD_REQUEST, f"files must contain at most {MAX_CHAT_FILES} opaque ids")
    try:
        return runtime_state._storage().metadata(team_id, file_ids)
    except team_storage.StorageNotFoundError as exc:
        raise runtime_state.ApiError(HTTPStatus.NOT_FOUND, "selected file not found in this Team") from exc
    except team_storage.StorageInputError as exc:
        raise runtime_state.ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
    except team_storage.StorageError as exc:
        raise runtime_state.ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Team storage failed its safety checks") from exc


def _model_credential(owner: str, provider: str) -> tuple[str, int]:
    if not owner:
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "this Team has no account owner for model credentials")
    try:
        credential = brain_credentials_client.resolve(owner, provider)
    except brain_credentials_client.BrainCredentialError as exc:
        raise runtime_state.ApiError(HTTPStatus.BAD_GATEWAY, "model credential service is unavailable") from exc
    if credential is None:
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, f"configure the {provider!r} API key before chatting")
    auth_type, api_key, generation = credential
    if auth_type != "api_key":
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "the selected model provider requires an API key")
    return api_key, generation


def _require_model_credential_current(owner: str, provider: str, generation: int) -> None:
    try:
        current = brain_credentials_client.generation_is_current(owner, provider, generation)
    except brain_credentials_client.BrainCredentialError as exc:
        raise runtime_state.ApiError(HTTPStatus.BAD_GATEWAY, "model credential could not be verified") from exc
    if not current:
        raise runtime_state.ApiError(HTTPStatus.CONFLICT, "model credential changed or was revoked; retry")
