# syntax=docker/dockerfile:1@sha256:87999aa3d42bdc6bea60565083ee17e86d1f3339802f543c0d03998580f9cb89
FROM python:3.14-slim@sha256:b877e50bd90de10af8d82c57a022fc2e0dc731c5320d762a27986facfc3355c1

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY app.py /opt/shimpz-cloudflare/app.py
COPY GENESIS.md /opt/shimpz-cloudflare/GENESIS.md
COPY shimpz.assistant.toml /opt/shimpz-cloudflare/shimpz.assistant.toml
COPY shimpz-cloudflare-rpc /usr/local/bin/shimpz-cloudflare-rpc
RUN groupadd --gid 10001 assistant && \
    useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin assistant && \
    chmod 0444 /opt/shimpz-cloudflare/app.py /opt/shimpz-cloudflare/GENESIS.md \
        /opt/shimpz-cloudflare/shimpz.assistant.toml && \
    chmod 0555 /usr/local/bin/shimpz-cloudflare-rpc

LABEL org.shimpz.assistant.id="shimpz-cloudflare" org.shimpz.assistant.api="1"
USER 10001:10001
HEALTHCHECK --interval=5s --timeout=3s --start-period=2s --retries=3 \
    CMD ["/usr/local/bin/shimpz-cloudflare-rpc", "GET", "/healthz"]
ENTRYPOINT ["/usr/local/bin/python3", "/opt/shimpz-cloudflare/app.py"]
