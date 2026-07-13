# syntax=docker/dockerfile:1
# check=skip=SecretsUsedInArgOrEnv ; false positive: the *_TOKEN_GID/DOCKER_GID ARGs are numeric group IDs, never secrets
#
# capsule-driver — the SECOND container (besides shimpz-driver) holding /var/run/docker.sock, dedicated
# to Capsule lifecycle. `shimpz-brain` never mounts the socket (SECURITY_ENGINEERING_PLAN.md 1.1); the
# authenticated admin panel calls this driver's restricted, allowlisted, audited API instead. This
# process's own UID is unprivileged (10001) — defense-in-depth against trivial filesystem tampering,
# not a claim that socket access itself is contained (holding the socket is host-root-equivalent).
# Pinned by digest (same as drivers/apps/Dockerfile — bump both together).
FROM python:3.14-slim@sha256:b877e50bd90de10af8d82c57a022fc2e0dc731c5320d762a27986facfc3355c1

ARG UV_VERSION=0.11.25
# Same version AND hash as drivers/apps/Dockerfile (astral.sh serves an immutable versioned artifact).
ARG UV_INSTALL_SHA256=ca2de1bca2913ba30ce88658b6d90a663c627ecac378803aa58084a9adb35a46
# Must match the GID that owns /var/run/docker.sock on the host (stat -c '%g' /var/run/docker.sock).
ARG DOCKER_GID=989
# Fixed GID for THIS driver's token group — the caller (admin, uid 1000) joins it to read the token.
# Distinct from every other sidecar's (10002–10009) so no token is readable via another's group.
ARG SHIMPZ_CAPSULEDRIVER_TOKEN_GID=10010
# The pg-driver token group — this driver joins it (read-only) to request a scoped DB from pg-driver.
# MUST match drivers/apps + brain's SHIMPZ_PGDRIVER_TOKEN_GID so the shared 0440 token is readable.
ARG SHIMPZ_PGDRIVER_TOKEN_GID=10004

# Downloaded + hash-checked BEFORE execution — never `curl | sh` (SECURITY_ENGINEERING_PLAN.md item 6).
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" -o /tmp/uv-install.sh && \
    echo "${UV_INSTALL_SHA256}  /tmp/uv-install.sh" | sha256sum -c - && \
    env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh /tmp/uv-install.sh && \
    rm -f /tmp/uv-install.sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# `capsuledriver` needs THREE groups: `dockersock` (the socket), `shimpzcapsuledriver-token` (so the
# token it writes is group-readable by the admin's uid 1000), and `shimpzpgdriver-token` (so it can
# READ pg-driver's 0440 token to request a scoped DB). None is its PRIMARY group.
RUN groupadd -g "${DOCKER_GID}" dockersock \
    && groupadd -g "${SHIMPZ_CAPSULEDRIVER_TOKEN_GID}" shimpzcapsuledriver-token \
    && groupadd -g "${SHIMPZ_PGDRIVER_TOKEN_GID}" shimpzpgdriver-token \
    && useradd -u 10001 -g dockersock -G shimpzcapsuledriver-token,shimpzpgdriver-token -M -s /usr/sbin/nologin capsuledriver

WORKDIR /app
COPY pyproject.toml .
# Pinned EXACT (not the open range pyproject documents for humans) — SECURITY_ENGINEERING_PLAN.md item 6.
RUN uv venv /opt/venv && uv pip install --python /opt/venv/bin/python --no-cache-dir docker==7.1.0

COPY app.py validate.py manifests.py audit.py token_store.py pgdriver_client.py accounts_client.py healthcheck.py ./

# Pre-create + own every named-volume mountpoint so the fresh (root:root) volume is writable by the
# non-root user. /run/shimpz-capsuledriver gets GROUP `shimpzcapsuledriver-token` so the fresh volume's
# perms already match what the token file inside needs — readable by the admin panel via that group.
RUN mkdir -p /run/shimpz-capsuledriver /var/log/capsule-driver \
    && chown capsuledriver:shimpzcapsuledriver-token /run/shimpz-capsuledriver && chmod 750 /run/shimpz-capsuledriver \
    && chown -R capsuledriver:dockersock /var/log/capsule-driver

USER capsuledriver
EXPOSE 7077
ENTRYPOINT ["/opt/venv/bin/python", "/app/app.py"]
