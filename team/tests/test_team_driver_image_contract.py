from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UV_IMAGE = "ghcr.io/astral-sh/uv:0.11.25@sha256:1e3808aa9023d0980e7c15b1fa7c1ac16ff35925780cf5c459858b2d693f01a9"


def _runtime_import_closure(*entrypoints: str) -> set[str]:
    pending = list(entrypoints)
    modules = set()
    while pending:
        module = pending.pop()
        if module in modules:
            continue
        path = ROOT / f"{module}.py"
        if not path.is_file():
            continue
        modules.add(module)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            imported = []
            if isinstance(node, ast.Import):
                imported = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = [node.module.split(".", 1)[0]]
            pending.extend(name for name in imported if (ROOT / f"{name}.py").is_file())
    return {f"{module}.py" for module in modules}


class StaticTeamDriverImageContractTests(unittest.TestCase):
    def test_static_build_context_excludes_dependencies_caches_and_secrets(self) -> None:
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

        self.assertLessEqual(
            {
                ".env",
                ".env.*",
                "**/.env",
                "**/.env.*",
                ".venv",
                "**/__pycache__",
                "**/*.pyc",
            },
            set(dockerignore),
        )

    def test_static_image_packages_the_exact_runtime_import_closure(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        logical_lines = re.sub(r"\\\n\s*", " ", dockerfile).splitlines()
        runtime_copy = next((line for line in logical_lines if line.startswith("COPY ") and "app.py" in line), "")
        packaged = set(re.findall(r"\b[a-z][a-z0-9_]*[.]py\b", runtime_copy))

        self.assertEqual(packaged, _runtime_import_closure("app", "healthcheck"))

    def test_static_image_keeps_brain_access_and_private_state_narrow(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("ARG SHIMPZ_BRAIN_RUNTIME_TOKEN_GID=10016", dockerfile)
        self.assertIn(
            'groupadd -g "${SHIMPZ_BRAIN_RUNTIME_TOKEN_GID}" shimpzbrain-runtime-token',
            dockerfile,
        )
        self.assertNotIn("r2", dockerfile.lower())
        self.assertIn(
            "chown teamdriver:shimpzbrain-runtime-token /run/shimpz-brain-runtime",
            dockerfile,
        )
        self.assertIn("chmod 0750 /run/shimpz-brain-runtime", dockerfile)
        self.assertIn("/var/lib/team-driver/inference", dockerfile)
        self.assertIn("/var/lib/team-driver/power-journal", dockerfile)
        self.assertIn("/var/lib/team-driver/assistant-secrets/state", dockerfile)
        self.assertIn("/var/lib/team-driver/assistant-secrets/key", dockerfile)
        self.assertIn("/var/lib/team-driver/assistant-accounts/state", dockerfile)
        self.assertIn("/var/lib/team-driver/assistant-accounts/key", dockerfile)
        self.assertIn(
            "/var/lib/team-driver/cleanup \\\n"
            "        /var/lib/team-driver/inference \\\n"
            "        /var/lib/team-driver/power-journal \\",
            dockerfile,
        )

    def test_static_local_image_copies_only_builder_resolved_dependencies(self) -> None:
        dockerfile = (ROOT / "Dockerfile.local").read_text(encoding="utf-8")
        runtime = dockerfile.split(" AS runtime\n", 1)[1]

        self.assertIn(f"FROM {UV_IMAGE} AS uv", dockerfile)
        self.assertIn("COPY --from=uv /uv /usr/local/bin/uv", dockerfile)
        self.assertIn("COPY --from=dependencies /opt/venv /opt/venv", runtime)
        self.assertNotIn("uv-install.sh", dockerfile)
        self.assertNotIn("apt-get", runtime)
        self.assertNotIn("curl", runtime)
        self.assertNotIn("/usr/local/bin/uv", runtime)


if __name__ == "__main__":
    unittest.main()
