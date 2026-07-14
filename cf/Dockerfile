# syntax=docker/dockerfile:1@sha256:87999aa3d42bdc6bea60565083ee17e86d1f3339802f543c0d03998580f9cb89
# check=skip=SecretsUsedInArgOrEnv ; false positive: the *_TOKEN_GID/*_TOKEN_* ARGs here hold numeric group IDs and file paths, never secret values
#
# cf-driver — the ONLY container that holds SHIMPZ_CF_TOKEN/SHIMPZ_CF_ACCOUNT
# (SECURITY_ENGINEERING_PLAN.md item 3). `shimpz-brain` never sees the Cloudflare token; it calls this
# restricted, allowlisted, audited HTTP API instead. Deliberately holds NO other credential and
# NO docker.sock — a compromise here can only ever touch Cloudflare DNS/Tunnel/Access for the
# domains Shimpz already manages, never containers, never the rest of the keyring.
# stdlib-only (urllib) — zero third-party dependencies, so no uv/venv needed at all.
# Pinned by digest (not just the `3.14-slim` tag, which can move) — SECURITY_ENGINEERING_PLAN.md
# item 6. `docker pull python:3.14-slim` resolved to this exact digest; bump both together.
FROM python:3.14-slim@sha256:b877e50bd90de10af8d82c57a022fc2e0dc731c5320d762a27986facfc3355c1
ARG SOURCE_DATE_EPOCH=0

# Fixed GID for the token-sharing group — MUST match the same groupadd in the main Dockerfile
# (shimpz-brain's `abc` user joins it there) so both sides can agree on a numeric GID without a runtime
# lookup. Distinct from shimpz-driver's own shimpzdriver-token GID (10002) so the two sidecars'
# tokens are never readable by each other's group.
ARG SHIMPZ_CFDRIVER_TOKEN_GID=10003

RUN groupadd -g "${SHIMPZ_CFDRIVER_TOKEN_GID}" shimpzcfdriver-token \
    && useradd -u 10001 -G shimpzcfdriver-token -M -s /usr/sbin/nologin cfdriver

WORKDIR /app
COPY app.py validate.py cf_client.py audit.py token_store.py healthcheck.py ./

# Pre-create + own every path a NAMED VOLUME will be mounted onto — Docker copies an image
# directory's ownership/content into a fresh empty volume on first mount, which is otherwise
# root:root/0755 and unwritable by the non-root `cfdriver` user.
RUN mkdir -p /run/shimpz-cfdriver /var/log/cf-driver \
    && chown cfdriver:shimpzcfdriver-token /run/shimpz-cfdriver && chmod 750 /run/shimpz-cfdriver \
    && chown -R cfdriver:shimpzcfdriver-token /var/log/cf-driver

USER cfdriver
EXPOSE 7071
ENTRYPOINT ["python3", "/app/app.py"]
