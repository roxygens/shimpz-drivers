# syntax=docker/dockerfile:1@sha256:87999aa3d42bdc6bea60565083ee17e86d1f3339802f543c0d03998580f9cb89
# check=skip=SecretsUsedInArgOrEnv ; false positive: the *_TOKEN_GID/*_TOKEN_* ARGs here hold numeric group IDs and file paths, never secret values
#
# bus-driver — the ONLY container that holds SHIMPZ_BUS_ADMIN_* (the Redpanda cluster superuser).
# `shimpz-brain` never sees it; it calls this restricted, allowlisted, audited HTTP API instead. Deliberately
# holds NO other credential — a compromise here can only ever touch this Redpanda cluster's topics/
# ACLs, never Cloudflare, Postgres, or Docker.
# Only third-party dependency: confluent-kafka==2.14.2 — the SAME exact pinned version already
# proven working (against this same cluster, same SASL/SCRAM auth) in the main `shimpz-brain` image; a
# manylinux wheel ships prebuilt for this Python version, so no compiler/build-essential needed.
# Pinned by digest (not just the `3.14-slim` tag, which can move) — SECURITY_ENGINEERING_PLAN.md
# item 6. `docker pull python:3.14-slim` resolved to this exact digest; bump both together.
FROM python:3.14-slim@sha256:b877e50bd90de10af8d82c57a022fc2e0dc731c5320d762a27986facfc3355c1
ARG SOURCE_DATE_EPOCH=0

# Fixed GID for the token-sharing group — MUST match the same groupadd in the main Dockerfile
# (shimpz-brain's `abc` user joins it there). Distinct from every other sidecar's own GID so no
# driver's token is ever readable via another's group.
ARG SHIMPZ_BUSDRIVER_TOKEN_GID=10005

RUN groupadd -g "${SHIMPZ_BUSDRIVER_TOKEN_GID}" shimpzbusdriver-token \
    && useradd -u 10001 -G shimpzbusdriver-token -M -s /usr/sbin/nologin busdriver

COPY requirements.lock /tmp/requirements.lock
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --require-hashes --requirement /tmp/requirements.lock \
    && rm -f /tmp/requirements.lock \
    && rm -rf /root/.cache/pip \
    && /opt/venv/bin/python -c 'import confluent_kafka'

WORKDIR /app
COPY app.py validate.py bus_client.py audit.py token_store.py healthcheck.py ./

# Pre-create + own every path a NAMED VOLUME will be mounted onto — Docker copies an image
# directory's ownership/content into a fresh empty volume on first mount, which is otherwise
# root:root/0755 and unwritable by the non-root `busdriver` user.
RUN mkdir -p /run/shimpz-busdriver /var/log/bus-driver \
    && chown busdriver:shimpzbusdriver-token /run/shimpz-busdriver && chmod 750 /run/shimpz-busdriver \
    && chown -R busdriver:shimpzbusdriver-token /var/log/bus-driver

USER busdriver
EXPOSE 7073
ENTRYPOINT ["/opt/venv/bin/python", "/app/app.py"]
