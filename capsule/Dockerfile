# syntax=docker/dockerfile:1@sha256:87999aa3d42bdc6bea60565083ee17e86d1f3339802f543c0d03998580f9cb89
# check=skip=SecretsUsedInArgOrEnv ; false positive: the *_TOKEN_GID/DOCKER_GID ARGs are numeric group IDs, never secrets
#
# capsule-driver — the SECOND container (besides shimpz-driver) holding /var/run/docker.sock, dedicated
# to Capsule lifecycle. `shimpz-brain` never mounts the socket (SECURITY_ENGINEERING_PLAN.md 1.1); the
# authenticated admin panel calls this driver's restricted, allowlisted, audited API instead. This
# process's own UID is unprivileged (10001) — defense-in-depth against trivial filesystem tampering,
# not a claim that socket access itself is contained (holding the socket is host-root-equivalent).
# Pinned by digest (same as drivers/apps/Dockerfile — bump both together).
FROM python:3.14-slim@sha256:b877e50bd90de10af8d82c57a022fc2e0dc731c5320d762a27986facfc3355c1
ARG SOURCE_DATE_EPOCH=0

ARG DEBIAN_SNAPSHOT=20260623T000000Z
RUN set -eux; \
    . /etc/os-release; \
    archive_keyring="$(find /usr/share/keyrings -maxdepth 1 -type f -name 'debian-archive-keyring.*' -print -quit)"; \
    test -n "$archive_keyring"; \
    rm -f /etc/apt/sources.list; \
    find /etc/apt/sources.list.d -maxdepth 1 -type f -delete; \
    printf '%s\n' \
        "deb [signed-by=${archive_keyring}] https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT} ${VERSION_CODENAME} main" \
        "deb [signed-by=${archive_keyring}] https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT} ${VERSION_CODENAME}-updates main" \
        "deb [signed-by=${archive_keyring}] https://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT} ${VERSION_CODENAME}-security main" \
        > /etc/apt/sources.list.d/debian-snapshot.list; \
    printf 'Acquire::Check-Valid-Until "false";\n' > /etc/apt/apt.conf.d/99shimpz-snapshot; \
    test "$(grep -Fc "https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}" /etc/apt/sources.list.d/debian-snapshot.list)" -eq 2; \
    test "$(grep -Fc "https://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}" /etc/apt/sources.list.d/debian-snapshot.list)" -eq 1

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
ARG SHIMPZ_BRAINCRED_UNSEAL_TOKEN_GID=10012
ARG SHIMPZ_ACCOUNTS_BRAIN_RESOLVE_TOKEN_GID=10013
# R2 lifecycle capability only. This is deliberately distinct from the legacy R2 operational token.
ARG SHIMPZ_R2PROVISIONER_TOKEN_GID=10015

# Downloaded + hash-checked BEFORE execution — never `curl | sh` (SECURITY_ENGINEERING_PLAN.md item 6).
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" -o /tmp/uv-install.sh && \
    echo "${UV_INSTALL_SHA256}  /tmp/uv-install.sh" | sha256sum -c - && \
    env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh /tmp/uv-install.sh && \
    rm -f /tmp/uv-install.sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/lib/apt/periodic/* /var/cache/apt/* /var/cache/fontconfig/* \
        /var/cache/ldconfig/aux-cache /var/cache/man/* /var/log/apt/* \
        /var/log/alternatives.log /var/log/dpkg.log /root/.cache/uv

# Every shared capability has a distinct supplementary group. In particular, R2 grants only its
# provisioner token here; capsule-driver never receives the legacy R2 operational bearer.
RUN groupadd -g "${DOCKER_GID}" dockersock \
    && groupadd -g "${SHIMPZ_CAPSULEDRIVER_TOKEN_GID}" shimpzcapsuledriver-token \
    && groupadd -g "${SHIMPZ_PGDRIVER_TOKEN_GID}" shimpzpgdriver-token \
    && groupadd -g "${SHIMPZ_BRAINCRED_UNSEAL_TOKEN_GID}" shimpzbraincred-unseal-token \
    && groupadd -g "${SHIMPZ_ACCOUNTS_BRAIN_RESOLVE_TOKEN_GID}" shimpzbrain-resolve \
    && groupadd -g "${SHIMPZ_R2PROVISIONER_TOKEN_GID}" shimpzr2provisioner-token \
    && useradd -u 10001 -g dockersock \
        -G shimpzcapsuledriver-token,shimpzpgdriver-token,shimpzbraincred-unseal-token,shimpzbrain-resolve,shimpzr2provisioner-token \
        -M -s /usr/sbin/nologin capsuledriver

WORKDIR /app
COPY pyproject.toml uv.lock ./
# The frozen lock binds Docker SDK's complete transitive graph and artifact hashes.
RUN UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --no-install-project --no-dev --python 3.14 && \
    rm -rf /root/.cache/uv

COPY app.py validate.py manifests.py network_policy.py marketplace.py marketplace_image.py cleanup_state.py audit.py token_store.py \
     pgdriver_client.py r2driver_client.py accounts_client.py brain_credentials_client.py healthcheck.py ./

# Pre-create + own every named-volume mountpoint so the fresh (root:root) volume is writable by the
# non-root user. /run/shimpz-capsuledriver gets GROUP `shimpzcapsuledriver-token` so the fresh volume's
# perms already match what the token file inside needs — readable by the admin panel via that group.
RUN mkdir -p /run/shimpz-capsuledriver /var/log/capsule-driver /var/lib/capsule-driver/pg-principals \
        /var/lib/capsule-driver/r2-principals \
        /var/lib/capsule-driver/cleanup \
        /app-egress-policy /run/shimpz-braincred-unseal /run/shimpz-accounts-brain-resolve \
    && chown capsuledriver:shimpzcapsuledriver-token /run/shimpz-capsuledriver && chmod 750 /run/shimpz-capsuledriver \
    && chown capsuledriver:shimpzbraincred-unseal-token /run/shimpz-braincred-unseal \
    && chown capsuledriver:shimpzbrain-resolve /run/shimpz-accounts-brain-resolve \
    && chmod 0750 /run/shimpz-braincred-unseal /run/shimpz-accounts-brain-resolve \
    && chown -R capsuledriver:dockersock /var/log/capsule-driver /var/lib/capsule-driver \
    && chmod 0700 /var/lib/capsule-driver/pg-principals /var/lib/capsule-driver/r2-principals \
        /var/lib/capsule-driver/cleanup \
    # egress-policy mountpoint owned by the driver uid, so a FRESH policy volume copies up writable
    # (an existing root-owned volume needs the one-time `chown 10001` — see docs/TESTS.md)
    && chown capsuledriver:dockersock /app-egress-policy

USER capsuledriver
EXPOSE 7077
ENTRYPOINT ["/opt/venv/bin/python", "/app/app.py"]
