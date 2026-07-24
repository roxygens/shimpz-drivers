"""Local Assistant resolution, image trust, and isolation validation."""

from http import HTTPStatus

import assistant_manifest
from container_policy import local as local_container_policy
from docker.errors import DockerException, ImageNotFound, NotFound
from local_registry import AssistantSpec

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


class LocalAssistantResourcesMixin:
    def _assistant_filters(self, team_id: str) -> dict[str, list[str] | bool]:
        return {
            "all": True,
            "filters": {
                "label": [
                    f"{MANAGED_LABEL}=1",
                    f"{PROFILE_LABEL}={PROFILE}",
                    f"{SPACE_LABEL}={self.space_id}",
                    f"{KIND_LABEL}=assistant",
                    f"{TEAM_LABEL}={team_id}",
                ]
            },
        }

    def _assistant_container(self, team_id: str, assistant_id: str, *, required: bool = True):
        name = self._container_name(team_id, assistant_id)
        try:
            container = self.client.containers.get(name)
        except NotFound:
            if required:
                raise ApiProblem(
                    HTTPStatus.NOT_FOUND,
                    "Assistant is not installed",
                    code="assistant-not-found",
                ) from None
            return None
        return container

    def _resolve(self, assistant_id: str) -> AssistantSpec:
        spec = self.registry.get(assistant_id)
        if spec is None:
            # Resolution is intentionally completed before any image lookup/pull.
            raise ApiProblem(HTTPStatus.NOT_FOUND, "Assistant is not allowlisted", code="assistant-not-allowlisted")
        return spec

    @staticmethod
    def _image_labels_valid(image, spec: AssistantSpec) -> bool:
        labels = (image.attrs.get("Config") or {}).get("Labels") or {}
        return (
            labels.get("org.shimpz.assistant.id") == spec.assistant_id and labels.get("org.shimpz.assistant.api") == "1"
        )

    def _trusted_image(self, spec: AssistantSpec):
        try:
            image = self.client.images.get(spec.image)
        except ImageNotFound:
            try:
                image = self.client.images.pull(spec.image)
            except DockerException as exc:
                raise ApiProblem(
                    HTTPStatus.BAD_GATEWAY,
                    "the trusted Assistant image could not be pulled",
                    code="image-pull-failed",
                ) from exc
        except DockerException as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Docker is unavailable",
                code="docker-unavailable",
            ) from exc
        image.reload()
        repo_digests = image.attrs.get("RepoDigests") or []
        if spec.image not in repo_digests or not self._image_labels_valid(image, spec):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "the Assistant image does not match its trusted contract",
                code="image-contract-mismatch",
            )
        return image

    def _assistant_labels(self, team_id: str, spec: AssistantSpec) -> dict[str, str]:
        labels = self._base_labels(team_id, "assistant")
        labels.update({ASSISTANT_LABEL: spec.assistant_id, IMAGE_LABEL: spec.image})
        return labels

    def _validate_container_profile(
        self,
        container,
        team_id: str,
        spec: AssistantSpec,
        network_name: str,
    ) -> tuple[dict, dict[str, str]]:
        container.reload()
        expected_labels = self._assistant_labels(team_id, spec)
        expected_labels.pop(IMAGE_LABEL)
        admitted = local_container_policy.inspect_profile(
            container.attrs,
            container.name,
            expected_labels,
            self._container_name(team_id, spec.assistant_id),
            spec.image,
            network_name,
            self.cpuset_cpus,
        )
        if admitted is None:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "the installed Assistant failed its isolation profile",
                code="assistant-isolation-drift",
            )
        return admitted

    def _validate_container_egress(
        self,
        team_id: str,
        spec: AssistantSpec,
        network_name: str,
        environment: dict[str, str],
        egress_proxy=None,
    ) -> tuple[str, ...]:
        try:
            reviewed_hosts = assistant_manifest.canonical_allowed_hosts(spec.allowed_hosts)
        except assistant_manifest.ManifestError as exc:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "the reviewed Assistant allowed_hosts contract is invalid",
                code="assistant-registry-drift",
            ) from exc
        expected_proxy_environment = None
        if reviewed_hosts:
            expected_proxy_environment = self._validate_egress_policy(team_id, spec, reviewed_hosts)
            proxy = egress_proxy() if egress_proxy is not None else None
            self._validate_egress_proxy_attachment(network_name, proxy)
        if not local_container_policy.egress_environment_valid(environment, expected_proxy_environment):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "the installed Assistant failed its isolation profile",
                code="assistant-isolation-drift",
            )
        return reviewed_hosts

    def _validate_container_isolation(
        self,
        container,
        team_id: str,
        spec: AssistantSpec,
        network_name: str,
        egress_proxy=None,
    ) -> dict:
        config, environment = self._validate_container_profile(container, team_id, spec, network_name)
        self._validate_container_egress(
            team_id,
            spec,
            network_name,
            environment,
            egress_proxy,
        )
        return config

    def _validate_container_security(
        self,
        container,
        team_id: str,
        spec: AssistantSpec,
        network_name: str,
        egress_proxy=None,
    ) -> dict:
        config = self._validate_container_isolation(container, team_id, spec, network_name, egress_proxy)
        self._admit_assistant_allowed_hosts(container, spec)
        return config

    @staticmethod
    def _has_current_assistant_artifact(config: dict, spec: AssistantSpec) -> bool:
        labels = config.get("Labels") or {}
        return config.get("Image") == spec.image and labels.get(IMAGE_LABEL) == spec.image

    def _validate_current_assistant_artifact(self, config: dict, spec: AssistantSpec) -> None:
        if not self._has_current_assistant_artifact(config, spec):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "the installed Assistant must be updated",
                code="assistant-update-required",
            )

    def _validate_container(self, container, team_id: str, spec: AssistantSpec, network_name: str) -> None:
        config = self._validate_container_security(container, team_id, spec, network_name)
        self._validate_current_assistant_artifact(config, spec)
