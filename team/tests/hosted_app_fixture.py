"""Import-isolated hosted Team driver fixture shared by its contract suites."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM))
_MODULES_BEFORE_APP_LOAD = dict(sys.modules)


class _DockerError(Exception):
    pass


class _NotFoundError(_DockerError):
    pass


class _APIError(_DockerError):
    pass


class _Passthru:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


class _LogConfig(_Passthru):
    types = types.SimpleNamespace(JSON="json-file")


class _EmptyCollection:
    @staticmethod
    def get(_identity):
        raise _NotFoundError

    @staticmethod
    def list(**_kwargs):
        return []


_engine = types.SimpleNamespace(
    containers=_EmptyCollection(),
    networks=_EmptyCollection(),
    volumes=_EmptyCollection(),
    images=_EmptyCollection(),
)
_docker_types = types.ModuleType("docker.types")
_docker_types.Mount = _Passthru
_docker_types.Ulimit = _Passthru
_docker_types.Healthcheck = _Passthru
_docker_types.LogConfig = _LogConfig
_docker_errors = types.ModuleType("docker.errors")
_docker_errors.DockerException = _DockerError
_docker_errors.NotFound = _NotFoundError
_docker_errors.APIError = _APIError
_docker_errors.ImageNotFound = _NotFoundError
_docker_socket = types.ModuleType("docker.utils.socket")
_docker_utils = types.ModuleType("docker.utils")
_docker_utils.socket = _docker_socket
_docker = types.ModuleType("docker")
_docker.from_env = lambda: _engine
_docker.types = _docker_types
_docker.errors = _docker_errors
_docker.utils = _docker_utils
sys.modules.update(
    {
        "docker": _docker,
        "docker.types": _docker_types,
        "docker.errors": _docker_errors,
        "docker.utils": _docker_utils,
        "docker.utils.socket": _docker_socket,
    }
)


def _stub(name: str, **members):
    module = types.ModuleType(name)
    for key, value in members.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _BrainCredentialError(Exception):
    pass


class _PgDriverError(Exception):
    pass


_stub("accounts_client", verify=lambda _token: None)
_stub("audit", log=lambda *_args, **_kwargs: "trace")
_stub(
    "brain_credentials_client",
    BrainCredentialError=_BrainCredentialError,
    resolve=lambda *_args: None,
    generation_is_current=lambda *_args: True,
)
_stub(
    "pgdriver_client",
    PgDriverError=_PgDriverError,
    provision_team=lambda _team_id: {"database_url": "postgres://scoped"},
    create_app_db=lambda *_args: {},
    drop_app_db=lambda *_args: {},
    drop_team=lambda *_args: {},
    finalize_team_drop=lambda *_args: {},
)
_stub("token_store", ensure_token=lambda: "operator-token")

spec = importlib.util.spec_from_file_location("team_app_hosted_test", TEAM / "app.py")
app = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = app
_hosted_test_state = tempfile.TemporaryDirectory()
with mock.patch.dict(
    os.environ,
    {
        "SHIMPZ_TEAM_ASSISTANT_APPROVAL_GRANTS_PATH": str(
            Path(_hosted_test_state.name) / "assistant-approvals" / "grants.sqlite3"
        )
    },
):
    spec.loader.exec_module(app)

runtime_state = sys.modules["runtime_state"]
hosted_resources = sys.modules["container_policy.hosted_resources"]
hosted_apps = sys.modules["container_policy.hosted_apps"]
hosted_lifecycle = sys.modules["container_policy.hosted_lifecycle"]
hosted_assistants = sys.modules["assistant_human.hosted_assistants"]
hosted_chat_api = sys.modules["assistant_human.hosted_chat_api"]
hosted_chat_segment = sys.modules["assistant_human.hosted_chat_segment"]
hosted_controller = sys.modules["http_boundary.hosted_controller"]

# The loaded app keeps direct references to its fakes. Restore the process import table so discovery
# order can never make unrelated tests import a partial Docker/client module.
for module_name, module in tuple(sys.modules.items()):
    source = getattr(module, "__file__", None)
    if source is None:
        continue
    try:
        belongs_to_team = Path(source).resolve().is_relative_to(TEAM)
    except OSError, RuntimeError, ValueError:
        belongs_to_team = False
    if belongs_to_team and module_name not in {__name__, spec.name}:
        previous = _MODULES_BEFORE_APP_LOAD.get(module_name)
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
for module_name in (
    "docker",
    "docker.types",
    "docker.errors",
    "docker.utils",
    "docker.utils.socket",
    "accounts_client",
    "audit",
    "brain_credentials_client",
    "pgdriver_client",
    "token_store",
):
    previous = _MODULES_BEFORE_APP_LOAD.get(module_name)
    if previous is None:
        sys.modules.pop(module_name, None)
    else:
        sys.modules[module_name] = previous

ANCHOR_ID = "a" * 64
