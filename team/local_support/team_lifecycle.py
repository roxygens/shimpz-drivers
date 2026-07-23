"""Local Team destruction and whole-Space reset lifecycle."""

from __future__ import annotations

from contextlib import ExitStack
from http import HTTPStatus

import brain_runtime_client
import inference_config
import power_journal
import team_storage
from docker.errors import DockerException

from local_support.egress import PROFILE
from local_support.errors import ApiProblemError as ApiProblem
from local_support.labels import (
    ASSISTANT_LABEL,
    IMAGE_LABEL,
    KIND_LABEL,
    MANAGED_LABEL,
    PROFILE_LABEL,
    SPACE_LABEL,
    TEAM_LABEL,
)
from local_support.validation import ASSISTANT_ID_RE as _ASSISTANT_ID
from local_support.validation import MAX_ASSISTANT_ID_LENGTH, MAX_TEAM_ID_LENGTH, validate_team_id
from local_support.validation import TEAM_ID_RE as _TEAM_ID
from local_support.validation import brain_thread_id as _brain_thread_id


class LocalTeamLifecycleMixin:
    def _purge_power_generation(self, generation: str) -> None:
        try:
            self.power_state.purge(generation)
        except power_journal.PowerJournalError as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Team Power execution state could not be deleted",
                code="power-state-unavailable",
            ) from exc

    def _team_assistant_containers(self, team_id: str) -> list:
        try:
            return self.client.containers.list(**self._assistant_filters(team_id))
        except DockerException as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Docker is unavailable",
                code="docker-unavailable",
            ) from exc

    def _validate_destroy_containers(self, containers: list, team_id: str, network) -> None:
        for container in containers:
            assistant_id = container.labels.get(ASSISTANT_LABEL)
            spec = self.registry.get(assistant_id)
            if spec is None or network is None:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "Team resources failed their ownership contract",
                    code="ownership-conflict",
                )
            self._validate_container_security(container, team_id, spec, network.name)

    def _delete_team_conversation(self, team_id: str, network) -> None:
        if network is None:
            return
        thread_id = _brain_thread_id(self.space_id, team_id, network.id)
        try:
            self.brain_runtime.delete_thread(thread_id)
        except brain_runtime_client.BrainRuntimeError as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Team conversation state could not be deleted",
                code="brain-runtime-failed",
            ) from exc
        self._purge_power_generation(network.id)

    def _remove_team_assistants(self, team_id: str, containers: list) -> int:
        for container in containers:
            assistant_id = container.labels[ASSISTANT_LABEL]
            spec = self.registry[assistant_id]
            try:
                container.remove(force=True)
            except DockerException as exc:
                raise ApiProblem(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Docker could not destroy the Team",
                    code="docker-remove-failed",
                ) from exc
            self._blocked_power_workloads.discard(container.id)
            self._remove_assistant_policy_if_needed(team_id, assistant_id, spec)
        return len(containers)

    def _delete_team_persistence(self, team_id: str) -> bool:
        try:
            storage_removed = self.storage.destroy(team_id)
        except team_storage.StorageError as exc:
            self._raise_storage_problem(exc)
        try:
            self.inference_store.delete(team_id)
        except inference_config.InferenceConfigError as exc:
            self._raise_inference_problem(exc)
        return storage_removed

    def _delete_team_private_state(self, team_id: str) -> None:
        self._delete_team_secret_state(team_id)
        self._delete_team_account_state(team_id)

    def _remove_team_network(self, network) -> bool:
        if network is None:
            return False
        self._disconnect_egress_proxy_if_attached(network)
        try:
            network.remove()
        except DockerException as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Docker could not destroy the Team",
                code="docker-remove-failed",
            ) from exc
        return True

    def destroy_team(self, team_id: str) -> dict[str, object]:
        team_id = validate_team_id(team_id)
        self.secret_challenges.cancel_team(team_id)
        self.approval_challenges.cancel_team(team_id)
        self.input_challenges.cancel_team(team_id)
        self._delete_chat_continuation(team_id)
        self._cancel_chat_for_destroy(team_id)

        chat_lock = self._chat_lock(team_id)
        if not chat_lock.acquire(timeout=30):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "active Team chat did not stop in time",
                code="chat-active",
            )
        try:
            with self._lock(team_id):
                self._revoke_team_approval_grants(team_id)
                network = self._network(team_id, required=False)
                containers = self._team_assistant_containers(team_id)
                self._validate_destroy_containers(containers, team_id, network)
                self._delete_team_conversation(team_id, network)
                removed = self._remove_team_assistants(team_id, containers)
                storage_removed = self._delete_team_persistence(team_id)
                destroyed = self._remove_team_network(network)
                self._delete_team_private_state(team_id)
                return {
                    "team_id": team_id,
                    "destroyed": destroyed,
                    "assistants_removed": removed,
                    "storage_removed": storage_removed,
                }
        finally:
            chat_lock.release()

    def _validate_reset_container(self, container) -> None:
        container.reload()
        labels = container.attrs.get("Config", {}).get("Labels") or {}
        team_id = labels.get(TEAM_LABEL)
        assistant_id = labels.get(ASSISTANT_LABEL)
        if (
            not isinstance(team_id, str)
            or len(team_id) > MAX_TEAM_ID_LENGTH
            or _TEAM_ID.fullmatch(team_id) is None
            or not isinstance(assistant_id, str)
            or len(assistant_id) > MAX_ASSISTANT_ID_LENGTH
            or _ASSISTANT_ID.fullmatch(assistant_id) is None
            or not isinstance(labels.get(IMAGE_LABEL), str)
            or not self._labels_include(labels, self._base_labels(team_id, "assistant"))
            or container.name != self._container_name(team_id, assistant_id)
        ):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "a labeled Space resource failed its ownership contract",
                code="ownership-conflict",
            )

    def _reset_inventory(self) -> tuple[list, list]:
        base_labels = [
            f"{MANAGED_LABEL}=1",
            f"{PROFILE_LABEL}={PROFILE}",
            f"{SPACE_LABEL}={self.space_id}",
        ]
        containers = self.client.containers.list(
            all=True,
            filters={"label": [*base_labels, f"{KIND_LABEL}=assistant"]},
        )
        networks = self.client.networks.list(filters={"label": [*base_labels, f"{KIND_LABEL}=team"]})
        for container in containers:
            self._validate_reset_container(container)
        return containers, networks

    def _reset_assistant_identities(self, containers: list, networks: list) -> set[tuple[str, str]]:
        owned_assistants = {
            (
                container.attrs["Config"]["Labels"][TEAM_LABEL],
                container.attrs["Config"]["Labels"][ASSISTANT_LABEL],
            )
            for container in containers
        }
        owned_team_ids: set[str] = set()
        for network in networks:
            labels = network.attrs.get("Labels") or {}
            team_id = labels.get(TEAM_LABEL)
            if not isinstance(team_id, str):
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "a labeled Space resource failed its ownership contract",
                    code="ownership-conflict",
                )
            validate_team_id(team_id)
            self._validate_network(network, team_id)
            owned_team_ids.add(team_id)
        owned_assistants.update((team_id, assistant_id) for team_id in owned_team_ids for assistant_id in self.registry)
        return owned_assistants

    def _remove_space_resources(
        self,
        containers: list,
        networks: list,
        owned_assistants: set[tuple[str, str]],
    ) -> bool:
        self._delete_all_secret_state()
        self._delete_all_account_state()
        self._revoke_all_approval_grants()
        for container in containers:
            container.remove(force=True)
            self._blocked_power_workloads.discard(container.id)
        for team_id, assistant_id in sorted(owned_assistants):
            self._remove_egress_policy(team_id, assistant_id)
        for network in networks:
            self._disconnect_egress_proxy_if_attached(network)
        storage_removed = self.storage.destroy_all()
        for network in networks:
            team_id = network.attrs["Labels"][TEAM_LABEL]
            self.inference_store.delete(team_id)
        for network in networks:
            network.remove()
        return storage_removed

    def reset_space(self) -> dict[str, object]:
        """Remove every exactly owned workload/network without accepting resource ids."""
        self.secret_challenges.cancel_all()
        self.approval_challenges.cancel_all()
        self.input_challenges.cancel_all()
        self._clear_chat_continuations()
        with ExitStack() as locks:
            for lock in self._locks:
                locks.enter_context(lock)
            try:
                containers, networks = self._reset_inventory()
                owned_assistants = self._reset_assistant_identities(containers, networks)
                storage_removed = self._remove_space_resources(containers, networks, owned_assistants)
            except ApiProblem:
                raise
            except team_storage.StorageError as exc:
                self._raise_storage_problem(exc)
            except inference_config.InferenceConfigError as exc:
                self._raise_inference_problem(exc)
            except DockerException as exc:
                raise ApiProblem(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Docker could not reset the Space",
                    code="docker-reset-failed",
                ) from exc
            return {
                "reset": True,
                "assistants_removed": len(containers),
                "teams_removed": len(networks),
                "storage_removed": storage_removed,
            }
