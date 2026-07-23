"""Local Assistant container install, update, and uninstall lifecycle."""

from collections.abc import Callable
from contextlib import suppress
from http import HTTPStatus

from container_policy import local as local_container_policy
from docker.errors import DockerException, NotFound
from docker.types import LogConfig, Ulimit
from local_registry import AssistantSpec

from local_support.assistant_rpc import ASSISTANT_UID
from local_support.chat_types import ActiveAssistant as _ActiveAssistant
from local_support.errors import ApiProblemError as ApiProblem
from local_support.validation import validate_team_id

ASSISTANT_MEMORY = local_container_policy.ASSISTANT_MEMORY
ASSISTANT_NANO_CPUS = local_container_policy.ASSISTANT_NANO_CPUS
ASSISTANT_PIDS = local_container_policy.ASSISTANT_PIDS
READINESS_RECOVERY_ASSISTANTS = frozenset({"shimpz-cloudflare"})


def _is_replaceable_readiness_failure(assistant_id: str, problem: ApiProblem) -> bool:
    return assistant_id in READINESS_RECOVERY_ASSISTANTS and problem.code == "assistant-not-ready"


def _serialize_against_local_team_chat(
    operation: Callable[..., dict[str, object]],
) -> Callable[..., dict[str, object]]:
    """Reject Assistant mutation before its first side effect while a Team turn owns the slot."""

    def guarded(controller, team_id: str, *args, **kwargs) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        lock = controller._chat_lock(team_id)
        if not lock.acquire(blocking=False):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant lifecycle cannot change during an active Team chat turn",
                code="chat-active",
            )
        try:
            return operation(controller, team_id, *args, **kwargs)
        finally:
            lock.release()

    return guarded


class LocalAssistantLifecycleMixin:
    def _rollback_assistant_install(
        self,
        team_id: str,
        spec: AssistantSpec,
        network,
        container,
        *,
        egress_prepared: bool,
    ) -> ApiProblem | None:
        incomplete = False
        if container is not None:
            self._assistant_genesis_cache.discard(container.id)
            self._assistant_allowed_hosts_cache.discard(container.id)
            self._assistant_machine_contract_cache.discard(container.id)
            try:
                container.remove(force=True)
            except NotFound:
                pass
            except DockerException:
                incomplete = True
                with suppress(ApiProblem):
                    self._fail_stop_power(container)
        if egress_prepared:
            try:
                self._release_assistant_egress(team_id, spec.assistant_id, network)
            except ApiProblem:
                incomplete = True
        if incomplete:
            return ApiProblem(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Assistant install rollback is incomplete",
                code="assistant-install-rollback-incomplete",
            )
        return None

    def _create_assistant_container(self, team_id: str, spec: AssistantSpec, network, image) -> None:
        container = None
        egress_prepared = False
        try:
            proxy_environment: dict[str, str] = {}
            if spec.allowed_hosts:
                token = self._egress_token(team_id, spec.assistant_id, create=True)
                if token is None:
                    raise ApiProblem(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "Assistant egress token could not be saved",
                        code="egress-policy-unavailable",
                    )
                proxy_environment = self._proxy_environment(token)
                egress_prepared = True
            container = self.client.containers.create(
                image=spec.image,
                name=self._container_name(team_id, spec.assistant_id),
                command=None,
                detach=True,
                user=ASSISTANT_UID,
                network=network.name,
                labels=self._assistant_labels(team_id, spec),
                environment={
                    "SHIMPZ_ASSISTANT_ID": spec.assistant_id,
                    "SHIMPZ_TEAM_ID": team_id,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    **proxy_environment,
                },
                read_only=True,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                privileged=False,
                ipc_mode="private",
                cgroupns="private",
                mem_limit=ASSISTANT_MEMORY,
                memswap_limit=ASSISTANT_MEMORY,
                nano_cpus=ASSISTANT_NANO_CPUS,
                cpuset_cpus=self.cpuset_cpus,
                pids_limit=ASSISTANT_PIDS,
                ulimits=[Ulimit(name="nofile", soft=1024, hard=1024)],
                restart_policy={"Name": "no"},
                log_config=LogConfig(type=LogConfig.types.JSON, config={"max-size": "1m", "max-file": "2"}),
            )
            container.reload()
            if container.attrs.get("Image") != image.id:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "Docker resolved an unexpected Assistant image",
                    code="image-resolution-mismatch",
                )
            allowed_hosts = self._admit_assistant_allowed_hosts(container, spec)
            if allowed_hosts:
                self._activate_assistant_egress(team_id, spec, network, allowed_hosts)
            container.start()
            self._validate_container(container, team_id, spec, network.name)
            self._wait_ready(container, spec)
            self._active_assistant_genesis(_ActiveAssistant(spec, container.id, container))
        except ApiProblem as exc:
            cleanup_error = self._rollback_assistant_install(
                team_id,
                spec,
                network,
                container,
                egress_prepared=egress_prepared,
            )
            if cleanup_error is not None:
                raise cleanup_error from exc
            raise
        except DockerException as exc:
            cleanup_error = self._rollback_assistant_install(
                team_id,
                spec,
                network,
                container,
                egress_prepared=egress_prepared,
            )
            if cleanup_error is not None:
                raise cleanup_error from exc
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Docker could not install the Assistant",
                code="docker-install-failed",
            ) from exc

    def _replace_unready_assistant(self, team_id: str, spec: AssistantSpec, network, existing) -> None:
        # The reference Assistant is the only explicitly stateless recovery target. Resolve its trusted image before
        # removing anything, then revalidate ownership to close the pull/remove race.
        image = self._trusted_image(spec)
        self._validate_container(existing, team_id, spec, network.name)
        try:
            self._assistant_genesis_cache.discard(existing.id)
            self._assistant_allowed_hosts_cache.discard(existing.id)
            self._assistant_machine_contract_cache.discard(existing.id)
            existing.remove(force=True)
        except DockerException as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Docker could not replace the Assistant",
                code="docker-remove-failed",
            ) from exc
        self._create_assistant_container(team_id, spec, network, image)

    def _replace_outdated_assistant(self, team_id: str, spec: AssistantSpec, network, existing) -> None:
        image = self._trusted_image(spec)
        config = self._validate_container_security(existing, team_id, spec, network.name)
        if self._has_current_assistant_artifact(config, spec):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "the installed Assistant changed during update",
                code="assistant-update-conflict",
            )
        self._retain_declared_assistant_account_state(team_id, spec)
        remaining_egress = (
            self._team_has_egress_assistant(team_id, excluding=spec.assistant_id) if spec.allowed_hosts else None
        )
        try:
            self._assistant_genesis_cache.discard(existing.id)
            self._assistant_allowed_hosts_cache.discard(existing.id)
            self._assistant_machine_contract_cache.discard(existing.id)
            existing.remove(force=True)
        except DockerException as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Docker could not replace the Assistant",
                code="docker-remove-failed",
            ) from exc
        self._revoke_assistant_approval_grants(team_id, spec.assistant_id)
        if spec.allowed_hosts:
            self._release_assistant_egress(
                team_id,
                spec.assistant_id,
                network,
                remaining_egress=remaining_egress,
            )
        self._create_assistant_container(team_id, spec, network, image)

    @_serialize_against_local_team_chat
    def install_assistant(self, team_id: str, assistant_id: str) -> dict[str, object]:
        spec = self._resolve(assistant_id)
        with self._lock(team_id):
            network = self._network(team_id)
            existing = self._assistant_container(team_id, assistant_id, required=False)
            if existing is not None:
                config = self._validate_container_security(existing, team_id, spec, network.name)
                if not self._has_current_assistant_artifact(config, spec):
                    self._replace_outdated_assistant(team_id, spec, network, existing)
                    return {"assistant": assistant_id, "installed": False}
                self._validate_container_security(existing, team_id, spec, network.name)
                existing.reload()
                if existing.status != "running":
                    try:
                        existing.start()
                    except DockerException as exc:
                        raise ApiProblem(
                            HTTPStatus.SERVICE_UNAVAILABLE,
                            "Docker could not start the Assistant",
                            code="docker-start-failed",
                        ) from exc
                try:
                    self._wait_ready(existing, spec)
                except ApiProblem as exc:
                    if not _is_replaceable_readiness_failure(assistant_id, exc):
                        raise
                    self._replace_unready_assistant(team_id, spec, network, existing)
                else:
                    self._active_assistant_genesis(_ActiveAssistant(spec, existing.id, existing))
                return {"assistant": assistant_id, "installed": False}

            image = self._trusted_image(spec)
            self._revoke_assistant_approval_grants(team_id, spec.assistant_id)
            self._create_assistant_container(team_id, spec, network, image)
            return {"assistant": assistant_id, "installed": True}

    @_serialize_against_local_team_chat
    def uninstall_assistant(self, team_id: str, assistant_id: str) -> dict[str, object]:
        spec = self._resolve(assistant_id)
        self.secret_challenges.cancel_team(team_id)
        self.approval_challenges.cancel_team(team_id)
        self.input_challenges.cancel_team(team_id)
        self._delete_chat_continuation(team_id)
        with self._lock(team_id):
            network = self._network(team_id)
            self._revoke_assistant_approval_grants(team_id, assistant_id)
            container = self._assistant_container(team_id, assistant_id, required=False)
            if container is None:
                if self._egress_token(team_id, assistant_id, create=False) is not None:
                    remaining_egress = self._team_has_egress_assistant(team_id, excluding=assistant_id)
                    self._release_assistant_egress(
                        team_id,
                        assistant_id,
                        network,
                        remaining_egress=remaining_egress,
                    )
                self._delete_assistant_secret_state(team_id, assistant_id)
                self._delete_assistant_account_state(team_id, assistant_id)
                return {"assistant": assistant_id, "uninstalled": False}
            self._validate_container_security(container, team_id, spec, network.name)
            remaining_egress = (
                self._team_has_egress_assistant(team_id, excluding=assistant_id) if spec.allowed_hosts else None
            )
            try:
                container.remove(force=True)
            except DockerException as exc:
                raise ApiProblem(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Docker could not uninstall the Assistant",
                    code="docker-remove-failed",
                ) from exc
            self._blocked_power_workloads.discard(container.id)
            self._assistant_genesis_cache.discard(container.id)
            self._assistant_allowed_hosts_cache.discard(container.id)
            self._assistant_machine_contract_cache.discard(container.id)
            if spec.allowed_hosts:
                self._release_assistant_egress(
                    team_id,
                    assistant_id,
                    network,
                    remaining_egress=remaining_egress,
                )
            self._delete_assistant_secret_state(team_id, assistant_id)
            self._delete_assistant_account_state(team_id, assistant_id)
            return {"assistant": assistant_id, "uninstalled": True}
