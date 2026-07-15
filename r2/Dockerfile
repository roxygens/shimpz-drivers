# syntax=docker/dockerfile:1@sha256:87999aa3d42bdc6bea60565083ee17e86d1f3339802f543c0d03998580f9cb89
# check=skip=SecretsUsedInArgOrEnv ; false positive: the *_TOKEN_GID/*_TOKEN_* ARGs here hold numeric group IDs and file paths, never secret values
#
# r2-driver — the ONLY container that decrypts managed or per-Capsule Cloudflare R2 credentials.
# (SECURITY_ENGINEERING_PLAN.md item 7). `shimpz-brain` never sees the R2 secret; it calls this restricted,
# allowlisted, audited HTTP API instead. It deliberately has no docker.sock or unrelated keyring access.
# Its blast radius is the explicitly configured managed and BYOK R2 accounts/buckets. Python uses the
# pinned cryptography package for AES-256-GCM; rclone is the only binary it shells (fixed argv/env).
# Pinned by digest (same base the other stdlib sidecars pin) — SECURITY_ENGINEERING_PLAN.md item 6.
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

# rclone = the S3 client for Cloudflare R2. SAME pinned version + sha256 as the main Dockerfile once
# installed in the brain — bump both together. Official binary (Debian's is frozen years behind).
ARG RCLONE_VERSION=1.74.3
ARG RCLONE_SHA256=dbee7ccd7a5d617e4ed4cd4555c16669b511abfe8d31164f61be35ac9e999bd2
ARG UV_VERSION=0.11.25
ARG UV_INSTALL_SHA256=ca2de1bca2913ba30ce88658b6d90a663c627ecac378803aa58084a9adb35a46
RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl unzip; \
    curl -fsSL "https://downloads.rclone.org/v${RCLONE_VERSION}/rclone-v${RCLONE_VERSION}-linux-amd64.zip" -o /tmp/rclone.zip; \
    echo "${RCLONE_SHA256}  /tmp/rclone.zip" | sha256sum -c -; \
    unzip -j /tmp/rclone.zip '*/rclone' -d /usr/local/bin/; \
    chmod +x /usr/local/bin/rclone; \
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" -o /tmp/uv-install.sh; \
    echo "${UV_INSTALL_SHA256}  /tmp/uv-install.sh" | sha256sum -c -; \
    env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh /tmp/uv-install.sh; \
    rm -f /tmp/rclone.zip /tmp/uv-install.sh; \
    apt-get purge -y curl unzip; \
    apt-get autoremove -y; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /var/lib/apt/periodic/* /var/cache/apt/* /var/cache/fontconfig/* \
        /var/cache/ldconfig/aux-cache /var/cache/man/* /var/log/apt/* \
        /var/log/alternatives.log /var/log/dpkg.log

# Fixed GID for the token-sharing group — MUST match the same groupadd in the main Dockerfile (shimpz-brain's
# `abc` user joins it there). DISTINCT from every other sidecar's token GID (10002-10006) so no
# sidecar's token is ever readable via another's group.
ARG SHIMPZ_R2DRIVER_TOKEN_GID=10007
ARG SHIMPZ_R2PROVISIONER_TOKEN_GID=10015
RUN groupadd -g "${SHIMPZ_R2DRIVER_TOKEN_GID}" shimpzr2driver-token \
    && groupadd -g "${SHIMPZ_R2PROVISIONER_TOKEN_GID}" shimpzr2provisioner-token \
    && useradd -u 10007 -G shimpzr2driver-token,shimpzr2provisioner-token -M -s /usr/sbin/nologin r2driver

WORKDIR /app
COPY pyproject.toml uv.lock ./
# Frozen artifacts include cryptography's AES-256-GCM implementation and complete hashes.
RUN UV_PROJECT_ENVIRONMENT=/opt/venv uv sync --frozen --no-install-project --no-dev --python 3.14 \
    && rm -rf /root/.cache/uv

COPY app.py validate.py r2_client.py audit.py backup_gate.py token_store.py healthcheck.py \
    credential_bundle.py credential_store.py principal_store.py driver_manifest.py shimpz.driver.toml \
    shimpz.credentials.json ./

# Pre-create + own every path a NAMED VOLUME mounts onto — Docker seeds a fresh volume from the image
# dir's ownership, otherwise root:root/0755 and unwritable by the non-root `r2driver` user.
RUN mkdir -p /run/shimpz-r2driver /run/shimpz-r2backup /run/shimpz-r2provisioner \
        /var/log/r2-driver /var/lib/shimpz-r2backup \
        /var/lib/shimpz-r2credentials /var/lib/shimpz-r2keyring /var/lib/shimpz-r2principals \
    && chown r2driver:shimpzr2driver-token /run/shimpz-r2driver && chmod 750 /run/shimpz-r2driver \
    && chown r2driver:r2driver /run/shimpz-r2backup && chmod 700 /run/shimpz-r2backup \
    && chown r2driver:shimpzr2provisioner-token /run/shimpz-r2provisioner \
    && chmod 750 /run/shimpz-r2provisioner \
    && chown -R r2driver:shimpzr2driver-token /var/log/r2-driver \
    && chown r2driver:r2driver /var/lib/shimpz-r2backup /var/lib/shimpz-r2credentials \
        /var/lib/shimpz-r2keyring /var/lib/shimpz-r2principals \
    && chmod 700 /var/lib/shimpz-r2backup /var/lib/shimpz-r2credentials \
        /var/lib/shimpz-r2keyring /var/lib/shimpz-r2principals

USER r2driver
EXPOSE 7075
ENTRYPOINT ["/opt/venv/bin/python", "/app/app.py"]
