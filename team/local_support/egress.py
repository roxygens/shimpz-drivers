"""Local Team resource naming, network validation, and Assistant egress."""

import os
import re
from http import HTTPStatus
from pathlib import Path
from typing import NoReturn

import egress_policy
from docker.errors import DockerException, NotFound
from local_registry import AssistantSpec

from local_support.errors import ApiProblemError as ApiProblem
from local_support.labels import (
    ASSISTANT_LABEL,
    KIND_LABEL,
    MANAGED_LABEL,
    PROFILE_LABEL,
    SPACE_LABEL,
    TEAM_LABEL,
    TEAM_NAME_LABEL,
)
from local_support.validation import space_prefix as _space_prefix
from local_support.validation import validate_team_name

PROFILE = "single-owner-local-v1"
MAX_EGRESS_POLICY_BYTES = egress_policy.MAX_POLICY_BYTES
APP_EGRESS_PROXY_ALIAS = "app-egress-proxy"
APP_EGRESS_PROXY_PORT = 8889
APP_EGRESS_PROXY_KIND = "app-egress-proxy"
APP_EGRESS_POLICY_GID = 10017
APP_EGRESS_PROXY_CONTAINER = os.environ.get("SHIMPZ_APP_EGRESS_PROXY_CONTAINER", "").strip()
APP_EGRESS_POLICY_DIR = Path(
    os.environ.get(
        "SHIMPZ_APP_EGRESS_POLICY_DIR",
        "/var/lib/shimpz-local/app-egress",
    )
)
_CONTAINER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def _egress_store() -> egress_policy.EgressPolicyStore:
    return egress_policy.EgressPolicyStore(
        APP_EGRESS_POLICY_DIR,
        APP_EGRESS_POLICY_GID,
        "127.0.0.1,localhost",
        APP_EGRESS_PROXY_ALIAS,
        APP_EGRESS_PROXY_PORT,
    )


def _raise_egress_problem(exc: egress_policy.EgressPolicyError) -> NoReturn:
    if isinstance(exc, egress_policy.EgressPolicyDriftError):
        raise ApiProblem(
            HTTPStatus.CONFLICT,
            "Assistant egress policy failed its ownership contract",
            code="egress-policy-drift",
        ) from exc
    raise ApiProblem(
        HTTPStatus.SERVICE_UNAVAILABLE,
        "Assistant egress policy storage is unavailable",
        code="egress-policy-unavailable",
    ) from exc


class LocalEgressMixin:
    def _base_labels(self, team_id: str, kind: str) -> dict[str, str]:
        return {
            MANAGED_LABEL: "1",
            PROFILE_LABEL: PROFILE,
            SPACE_LABEL: self.space_id,
            KIND_LABEL: kind,
            TEAM_LABEL: team_id,
        }

    def _network_name(self, team_id: str) -> str:
        return f"shimpz-local-{_space_prefix(self.space_id)}-team-{team_id}"

    def _container_name(self, team_id: str, assistant_id: str) -> str:
        return f"shimpz-local-{_space_prefix(self.space_id)}-{team_id}-assistant-{assistant_id}"

    def _egress_policy_identity(self, team_id: str, assistant_id: str) -> str:
        return f"{self.space_id}\0{team_id}\0{assistant_id}"

    def _egress_token(
        self,
        team_id: str,
        assistant_id: str,
        *,
        create: bool,
        store: egress_policy.EgressPolicyStore | None = None,
    ) -> str | None:
        try:
            current_store = store if store is not None else _egress_store()
            return current_store.token(
                self._egress_policy_identity(team_id, assistant_id),
                create=create,
            )
        except egress_policy.EgressPolicyError as exc:
            _raise_egress_problem(exc)

    @staticmethod
    def _proxy_environment(
        token: str,
        store: egress_policy.EgressPolicyStore | None = None,
    ) -> dict[str, str]:
        try:
            current_store = store if store is not None else _egress_store()
            return current_store.proxy_environment(token)
        except egress_policy.EgressPolicyError as exc:
            _raise_egress_problem(exc)

    def _reserve_assistant_egress_environment(
        self,
        team_id: str,
        assistant_id: str,
    ) -> tuple[str | None, dict[str, str], egress_policy.EgressPolicyStore]:
        store = _egress_store()
        token = self._egress_token(team_id, assistant_id, create=True, store=store)
        environment = self._proxy_environment(token, store) if token is not None else {}
        return token, environment, store

    def _write_egress_policy(
        self,
        team_id: str,
        spec: AssistantSpec,
        allowed_hosts: tuple[str, ...],
        store: egress_policy.EgressPolicyStore | None = None,
    ) -> dict[str, str]:
        try:
            current_store = store if store is not None else _egress_store()
            token = current_store.token(
                self._egress_policy_identity(team_id, spec.assistant_id),
                create=True,
            )
            if token is None:
                raise egress_policy.EgressPolicyUnavailableError("egress token was not created")
            current_store.write(token, allowed_hosts)
            return current_store.proxy_environment(token)
        except egress_policy.EgressPolicyError as exc:
            _raise_egress_problem(exc)

    def _validate_egress_policy(
        self,
        team_id: str,
        spec: AssistantSpec,
        allowed_hosts: tuple[str, ...],
        store: egress_policy.EgressPolicyStore | None = None,
    ) -> dict[str, str]:
        try:
            current_store = store if store is not None else _egress_store()
            admitted = self._read_admitted_egress_policy(team_id, spec.assistant_id, current_store)
            token = current_store.validate_admitted(
                admitted,
                allowed_hosts,
            )
            return current_store.proxy_environment(token)
        except egress_policy.EgressPolicyError as exc:
            _raise_egress_problem(exc)

    def _read_admitted_egress_policy(
        self,
        team_id: str,
        assistant_id: str,
        store: egress_policy.EgressPolicyStore | None = None,
    ) -> tuple[str, tuple[str, ...]] | None:
        """Read only a canonical policy previously admitted and owned by this controller."""
        try:
            current_store = store if store is not None else _egress_store()
            return current_store.admitted(
                self._egress_policy_identity(team_id, assistant_id),
            )
        except egress_policy.EgressPolicyError as exc:
            _raise_egress_problem(exc)

    def _remove_egress_policy(
        self,
        team_id: str,
        assistant_id: str,
        store: egress_policy.EgressPolicyStore | None = None,
    ) -> None:
        try:
            current_store = store if store is not None else _egress_store()
            current_store.remove(
                self._egress_policy_identity(team_id, assistant_id),
            )
        except egress_policy.EgressPolicyError as exc:
            _raise_egress_problem(exc)

    def _egress_proxy(self):
        if not APP_EGRESS_PROXY_CONTAINER or _CONTAINER_NAME.fullmatch(APP_EGRESS_PROXY_CONTAINER) is None:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Assistant egress proxy is unavailable",
                code="egress-proxy-unavailable",
            )
        try:
            proxy = self.client.containers.get(APP_EGRESS_PROXY_CONTAINER)
        except (NotFound, DockerException) as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Assistant egress proxy is unavailable",
                code="egress-proxy-unavailable",
            ) from exc
        attrs = proxy.attrs
        config = attrs.get("Config") or {}
        host = attrs.get("HostConfig") or {}
        labels = config.get("Labels") or {}
        expected_labels = {
            MANAGED_LABEL: "1",
            PROFILE_LABEL: PROFILE,
            SPACE_LABEL: self.space_id,
            KIND_LABEL: APP_EGRESS_PROXY_KIND,
        }
        security_options = host.get("SecurityOpt") or []
        mounts = attrs.get("Mounts") or []
        policy_mounts = [mount for mount in mounts if mount.get("Destination") == "/policy"]
        if (
            proxy.name != APP_EGRESS_PROXY_CONTAINER
            or proxy.status != "running"
            or not self._labels_include(labels, expected_labels)
            or config.get("User") not in {"10005", "10005:10005"}
            or host.get("ReadonlyRootfs") is not True
            or "ALL" not in (host.get("CapDrop") or [])
            or not any(str(option).startswith("no-new-privileges") for option in security_options)
            or any("seccomp=unconfined" in str(option) for option in security_options)
            or host.get("Privileged") is not False
            or host.get("PortBindings") not in (None, {})
            or len(policy_mounts) != 1
            or policy_mounts[0].get("RW") is not False
        ):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant egress proxy failed its isolation profile",
                code="egress-proxy-drift",
            )
        return proxy

    def _connect_egress_proxy(self, network) -> None:
        proxy = self._egress_proxy()
        attached = ((proxy.attrs.get("NetworkSettings") or {}).get("Networks") or {}).get(network.name)
        if attached is None:
            try:
                network.connect(proxy, aliases=[APP_EGRESS_PROXY_ALIAS])
                proxy.reload()
            except DockerException as exc:
                raise ApiProblem(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "Assistant egress proxy could not join the Team",
                    code="egress-proxy-unavailable",
                ) from exc
            attached = ((proxy.attrs.get("NetworkSettings") or {}).get("Networks") or {}).get(network.name)
        if not isinstance(attached, dict) or APP_EGRESS_PROXY_ALIAS not in (attached.get("Aliases") or []):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant egress proxy failed its Team attachment contract",
                code="egress-proxy-drift",
            )

    def _validate_egress_proxy_attachment(self, network_name: str, proxy=None) -> None:
        if proxy is None:
            proxy = self._egress_proxy()
        attached = ((proxy.attrs.get("NetworkSettings") or {}).get("Networks") or {}).get(network_name)
        if not isinstance(attached, dict) or APP_EGRESS_PROXY_ALIAS not in (attached.get("Aliases") or []):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant egress proxy failed its Team attachment contract",
                code="egress-proxy-drift",
            )

    def _disconnect_egress_proxy(self, network) -> None:
        proxy = self._egress_proxy()
        attached = ((proxy.attrs.get("NetworkSettings") or {}).get("Networks") or {}).get(network.name)
        if attached is None:
            return
        try:
            network.disconnect(proxy)
            proxy.reload()
        except DockerException as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Assistant egress proxy could not leave the Team",
                code="egress-proxy-unavailable",
            ) from exc
        if network.name in ((proxy.attrs.get("NetworkSettings") or {}).get("Networks") or {}):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Assistant egress proxy failed its Team attachment contract",
                code="egress-proxy-drift",
            )

    def _disconnect_egress_proxy_if_attached(self, network) -> None:
        try:
            network.reload()
        except DockerException as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Team network could not be inspected",
                code="docker-unavailable",
            ) from exc
        endpoints = network.attrs.get("Containers") or {}
        if not isinstance(endpoints, dict):
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Team resource ownership conflict",
                code="ownership-conflict",
            )
        if any(endpoint.get("Name") == APP_EGRESS_PROXY_CONTAINER for endpoint in endpoints.values()):
            self._disconnect_egress_proxy(network)

    def _team_has_egress_assistant(self, team_id: str, *, excluding: str | None = None) -> bool:
        try:
            containers = self.client.containers.list(**self._assistant_filters(team_id))
        except DockerException as exc:
            raise ApiProblem(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Docker is unavailable",
                code="docker-unavailable",
            ) from exc
        for container in containers:
            assistant_id = (container.labels or {}).get(ASSISTANT_LABEL)
            if assistant_id == excluding:
                continue
            spec = self.registry.get(assistant_id)
            if spec is None:
                raise ApiProblem(
                    HTTPStatus.CONFLICT,
                    "an installed Assistant is no longer allowlisted",
                    code="assistant-registry-drift",
                )
            self._validate_container_security(
                container,
                team_id,
                spec,
                self._network_name(team_id),
            )
            if spec.allowed_hosts:
                return True
        return False

    def _release_assistant_egress(
        self,
        team_id: str,
        assistant_id: str,
        network,
        *,
        remaining_egress: bool | None = None,
    ) -> None:
        self._remove_egress_policy(team_id, assistant_id)
        if remaining_egress is None:
            remaining_egress = self._team_has_egress_assistant(team_id)
        if not remaining_egress:
            self._disconnect_egress_proxy(network)

    def _remove_assistant_policy_if_needed(
        self,
        team_id: str,
        assistant_id: str,
        spec: AssistantSpec,
    ) -> None:
        if spec.allowed_hosts:
            self._remove_egress_policy(team_id, assistant_id)

    def _activate_assistant_egress(
        self,
        team_id: str,
        spec: AssistantSpec,
        network,
        allowed_hosts: tuple[str, ...],
        store: egress_policy.EgressPolicyStore | None = None,
    ) -> dict[str, str]:
        if not allowed_hosts:
            return {}
        current_store = store if store is not None else _egress_store()
        environment = self._write_egress_policy(team_id, spec, allowed_hosts, current_store)
        try:
            self._connect_egress_proxy(network)
        except ApiProblem:
            self._remove_egress_policy(team_id, spec.assistant_id, current_store)
            raise
        return environment

    @staticmethod
    def _labels_include(actual: object, expected: dict[str, str]) -> bool:
        return isinstance(actual, dict) and all(actual.get(key) == value for key, value in expected.items())

    def _validate_network(self, network, team_id: str, *, refresh: bool = True) -> str:
        if refresh:
            network.reload()
        attrs = network.attrs
        expected = self._base_labels(team_id, "team")
        labels = attrs.get("Labels") or {}
        if (
            not self._labels_include(labels, expected)
            or attrs.get("Name") != self._network_name(team_id)
            or attrs.get("Driver") != "bridge"
            or attrs.get("Internal") is not True
            or attrs.get("Attachable") is not False
        ):
            raise ApiProblem(HTTPStatus.CONFLICT, "Team resource ownership conflict", code="ownership-conflict")
        try:
            return validate_team_name(labels.get(TEAM_NAME_LABEL))
        except ApiProblem as exc:
            raise ApiProblem(
                HTTPStatus.CONFLICT,
                "Team resource ownership conflict",
                code="ownership-conflict",
            ) from exc

    def _network(self, team_id: str, *, required: bool = True):
        try:
            network = self.client.networks.get(self._network_name(team_id))
        except NotFound:
            if required:
                raise ApiProblem(HTTPStatus.NOT_FOUND, "Team not found", code="team-not-found") from None
            return None
        self._validate_network(network, team_id)
        return network
