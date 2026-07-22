# syntax=docker/dockerfile:1@sha256:87999aa3d42bdc6bea60565083ee17e86d1f3339802f543c0d03998580f9cb89
# check=skip=SecretsUsedInArgOrEnv ; false positive: the *_TOKEN_GID/*_TOKEN_* ARGs here hold numeric group IDs and file paths, never secret values
#
# shimpz-driver — the ONLY container in this stack that holds /var/run/docker.sock.
# `shimpz-brain` never mounts it. This process's own UID is
# unprivileged (10001), but note holding the socket is host-root-equivalent regardless of
# UID — the non-root user is defense-in-depth against trivial container-filesystem tampering,
# not a claim that socket access itself is contained.
# Pinned by digest (not just the `3.14-slim` tag, which can move), with extra weight because this
# is the ONE container holding docker.sock. `docker pull
# python:3.14-slim` resolved to this exact digest; bump both together.
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
# Same version AND hash as the main Dockerfile's UV_INSTALL_SHA256 (astral.sh serves an immutable
# versioned artifact — confirmed by downloading this exact URL and computing its sha256 directly,
# not copied on faith). Bump both together if UV_VERSION ever changes here.
ARG UV_INSTALL_SHA256=ca2de1bca2913ba30ce88658b6d90a663c627ecac378803aa58084a9adb35a46
# Must match the GID that owns /var/run/docker.sock on the host running this compose stack
# (checked with `stat -c '%g' /var/run/docker.sock`); override via compose build-arg if it differs.
ARG DOCKER_GID=989
# Fixed GID for the token-sharing group — MUST match the same groupadd in the main Dockerfile
# (shimpz-brain's `abc` user joins it there) so both sides can agree on a numeric GID without a runtime
# lookup. Picked well outside the base image's/DOCKER_GID's normal range to avoid collision.
ARG SHIMPZ_DRIVER_TOKEN_GID=10002

# Downloaded to a file and hash-checked BEFORE execution — never `curl | sh` (an unverified
# pipe-to-shell is a full supply-chain compromise if astral.sh, or anything on
# the path to it) is ever compromised or MITM'd, pinned version or not).
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" -o /tmp/uv-install.sh && \
    echo "${UV_INSTALL_SHA256}  /tmp/uv-install.sh" | sha256sum -c - && \
    env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh /tmp/uv-install.sh && \
    rm -f /tmp/uv-install.sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/lib/apt/periodic/* /var/cache/apt/* /var/cache/fontconfig/* \
        /var/cache/ldconfig/aux-cache /var/cache/man/* /var/log/apt/* \
        /var/log/alternatives.log /var/log/dpkg.log /root/.cache/uv

# `driver` needs TWO supplementary groups: `dockersock` (to read/write the Docker socket) and
# `shimpzdriver-token` (so the token file it writes can be group-readable by `shimpz-brain`'s `abc` user,
# which is NOT UID 10001 and was previously locked out entirely). Neither is driver's PRIMARY
# group so a plain `chown driver:X` still says what X means.
RUN groupadd -g "${DOCKER_GID}" dockersock \
    && groupadd -g "${SHIMPZ_DRIVER_TOKEN_GID}" shimpzdriver-token \
    && useradd -u 10001 -g dockersock -G shimpzdriver-token -M -s /usr/sbin/nologin driver

WORKDIR /app
COPY pyproject.toml uv.lock ./
# The frozen lock binds Docker SDK's complete transitive graph and artifact hashes.
RUN UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --no-install-project --no-dev --python 3.14 && \
    rm -rf /root/.cache/uv

COPY app.py validate.py manifests.py audit.py token_store.py caddy_routes.py egress_lock.py ./

# Pre-create + own every path a NAMED VOLUME will be mounted onto — Docker copies an image
# directory's ownership/content into a fresh empty volume on first mount, which is otherwise
# root:root/0755 and unwritable by the non-root `driver` user. /run/shimpz-driver specifically gets
# GROUP `shimpzdriver-token` (not `dockersock`) so the fresh volume's directory perms already match
# what the token file inside it will need — readable by shimpz-brain's `abc` user via that shared group.
RUN mkdir -p /run/shimpz-driver /var/log/shimpz-driver /caddy-sites \
    && chown driver:shimpzdriver-token /run/shimpz-driver && chmod 750 /run/shimpz-driver \
    && chown -R driver:dockersock /var/log/shimpz-driver /caddy-sites

USER driver
EXPOSE 7070
ENTRYPOINT ["/opt/venv/bin/python", "/app/app.py"]
