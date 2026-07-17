# syntax=docker/dockerfile:1@sha256:87999aa3d42bdc6bea60565083ee17e86d1f3339802f543c0d03998580f9cb89
# check=skip=SecretsUsedInArgOrEnv ; false positive: the *_TOKEN_GID/*_TOKEN_* ARGs here hold numeric group IDs and file paths, never secret values
#
# openai-driver — the private media credential boundary. It implements audited image generation,
# transcription, and speech operations with closed request shapes and allowlisted models. They are
# not yet exposed as Assistant Powers. Holds NO other credential and NO docker.sock, and provides no
# generic OpenAI passthrough. Uses the `openai` SDK, so STT multipart + TTS streaming are the SDK's
# battle-tested code, not hand-rolled.
# Pinned by digest (same base the other stdlib sidecars pin) — SECURITY_ENGINEERING_PLAN.md item 6.
FROM python:3.14-slim@sha256:b877e50bd90de10af8d82c57a022fc2e0dc731c5320d762a27986facfc3355c1
ARG SOURCE_DATE_EPOCH=0

# The ONLY OpenAI SDK install in the fleet. The generated lock binds the complete transitive graph and
# every accepted artifact hash; --require-hashes fails closed on an undeclared package or artifact.
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes --requirement /tmp/requirements.lock \
    && rm -f /tmp/requirements.lock \
    && rm -rf /root/.cache/pip

# Fixed GID for the driver token group. It stays distinct from every other sidecar's token GID
# (10002-10007) so no sidecar can read this token through another service's group membership.
ARG SHIMPZ_OPENAIDRIVER_TOKEN_GID=10008
RUN groupadd -g "${SHIMPZ_OPENAIDRIVER_TOKEN_GID}" shimpzopenaidriver-token \
    && useradd -u 10008 -G shimpzopenaidriver-token -M -s /usr/sbin/nologin openaidriver

WORKDIR /app
COPY app.py validate.py oa_client.py audit.py token_store.py healthcheck.py ./

# Pre-create + own every path a NAMED VOLUME mounts onto — Docker seeds a fresh volume from the image
# dir's ownership, otherwise root:root/0755 and unwritable by the non-root user.
RUN mkdir -p /run/shimpz-openaidriver /var/log/openai-driver \
    && chown openaidriver:shimpzopenaidriver-token /run/shimpz-openaidriver && chmod 750 /run/shimpz-openaidriver \
    && chown -R openaidriver:shimpzopenaidriver-token /var/log/openai-driver

USER openaidriver
EXPOSE 7076
ENTRYPOINT ["python3", "/app/app.py"]
