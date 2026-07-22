# syntax=docker/dockerfile:1@sha256:87999aa3d42bdc6bea60565083ee17e86d1f3339802f543c0d03998580f9cb89
# check=skip=SecretsUsedInArgOrEnv ; false positive: the *_TOKEN_GID/*_TOKEN_* ARGs here hold numeric group IDs and file paths, never secret values
#
# pg-driver — the ONLY container that holds SHIMPZ_PG_DSN (the Postgres superuser DSN).
# Neither the platform Brain nor a Team sees it. Team-driver provisions a random hashed principal,
# and each Team principal can mutate only its exact registered database set. This sidecar deliberately
# holds no Cloudflare, Docker, or account credential.
# The only third-party piece is postgresql-client (psql/createdb/dropdb); no Python DB driver is needed.
# Pinned by digest (not just the `3.14-slim` tag, which can move). `docker pull
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

# Fixed GID for the provisioner-token group. Only team-driver joins it; neither the platform
# brain nor any Team can read this control-plane credential.
ARG SHIMPZ_PGDRIVER_TOKEN_GID=10004

RUN apt-get update && apt-get install -y --no-install-recommends postgresql-client && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/lib/apt/periodic/* /var/cache/apt/* /var/cache/fontconfig/* \
        /var/cache/ldconfig/aux-cache /var/cache/man/* /var/log/apt/* \
        /var/log/alternatives.log /var/log/dpkg.log

RUN groupadd -g "${SHIMPZ_PGDRIVER_TOKEN_GID}" shimpzpgdriver-token \
    && useradd -u 10001 -G shimpzpgdriver-token -M -s /usr/sbin/nologin pgdriver

WORKDIR /app
COPY app.py validate.py pg_client.py principal_store.py audit.py token_store.py healthcheck.py \
    driver_manifest.py shimpz.driver.toml ./

# Pre-create + own every path a NAMED VOLUME will be mounted onto — Docker copies an image
# directory's ownership/content into a fresh empty volume on first mount, which is otherwise
# root:root/0755 and unwritable by the non-root `pgdriver` user.
RUN mkdir -p /run/shimpz-pgdriver /var/log/pg-driver /var/lib/pg-driver \
    && chown pgdriver:shimpzpgdriver-token /run/shimpz-pgdriver && chmod 750 /run/shimpz-pgdriver \
    && chown -R pgdriver:shimpzpgdriver-token /var/log/pg-driver /var/lib/pg-driver

USER pgdriver
EXPOSE 7072
ENTRYPOINT ["python3", "/app/app.py"]
