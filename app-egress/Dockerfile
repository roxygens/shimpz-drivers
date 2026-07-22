# syntax=docker/dockerfile:1@sha256:87999aa3d42bdc6bea60565083ee17e86d1f3339802f543c0d03998580f9cb89
#
# app-egress-proxy — the ONLY internet route for DEPLOYED APPS (Shimpz L2, deny-by-default egress). A
# CONNECT-only forward proxy that serves MANY apps: each app carries its own token (Basic proxy auth) and
# reaches ONLY the hosts it declared in [needs].egress (its effective_egress), read per-app from the
# policy dir the driver writes. Holds NO credential and NO docker.sock; a compromise here can, at
# worst, reach hosts apps already declared. stdlib-only (socket/select/base64) — zero third-party deps.
# Pinned by digest (not the movable tag). Same digest as the other
# driver sidecars; bump all together.
FROM python:3.14-slim@sha256:b877e50bd90de10af8d82c57a022fc2e0dc731c5320d762a27986facfc3355c1
ARG SOURCE_DATE_EPOCH=0

# IPv6 egress is broken on this host (see the main Dockerfile) → prefer IPv4 so an outbound CONNECT to a
# dual-stack allowlisted host doesn't stall on an unreachable AAAA. Same fix as every other image here.
RUN echo 'precedence ::ffff:0:0/96 100' >> /etc/gai.conf

RUN groupadd --gid 10017 shimpzapp-egress-policy && \
    useradd -u 10005 -M -s /usr/sbin/nologin --groups shimpzapp-egress-policy appegressproxy

WORKDIR /app
COPY app.py audit.py healthcheck.py ./

# Pre-create + own the audit-log volume mountpoint (Docker copies image-dir ownership into a fresh empty
# named volume on first mount). The policy dir is mounted read-only by the driver; not owned here.
RUN mkdir -p /var/log/app-egress-proxy /policy && \
    chown -R appegressproxy:appegressproxy /var/log/app-egress-proxy && \
    chown appegressproxy:shimpzapp-egress-policy /policy && chmod 0770 /policy

USER appegressproxy
EXPOSE 8889
# The Docker HEALTHCHECK is defined in docker-compose.yml (matching the sibling drivers);
# healthcheck.py proves the CONNECT-only gate is live regardless of any app's policy.
ENTRYPOINT ["python3", "/app/app.py"]
