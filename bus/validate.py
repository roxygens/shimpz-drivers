"""Allowlist validation for bus-driver — runs BEFORE any Kafka admin/produce/consume call.

Nothing here talks to Redpanda; it only decides yes/no and returns validated values the caller
(app.py) turns into bus_client.py calls. Same shape as the other drivers' own validate.py
modules — the actual security boundary, not the client that acts on its output.
"""

from __future__ import annotations

import re

# Kafka/Redpanda topic-name charset (letters, digits, '.', '_', '-'); 249 bytes is the broker's own
# limit (leaves headroom for the internal "<topic>.dlq"/changelog suffixes Redpanda itself appends).
TOPIC_RE = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
PARTITIONS_MIN, PARTITIONS_MAX = 1, 100
TAIL_N_MIN, TAIL_N_MAX = 1, 1000


class ValidationError(Exception):
    """A bus-driver request failed the allowlist — nothing was touched."""


def sanitize_proj(name: str) -> str:
    """Port of shimpzdetect.sh's _sanitize_proj.

    MUST match every other driver's own sanitize_proj exactly (shimpz-bus/shimpz-app and the app/PG
    drivers all independently agree).
    """
    lowered = re.sub(r"[^a-z0-9_]+", "_", str(name).lower())
    return lowered.strip("_")


def validate_project(name: object) -> str:
    if not isinstance(name, str) or not name:
        raise ValidationError(f"project name must be a non-empty string: {name!r}")
    sanitized = sanitize_proj(name)
    if not sanitized:
        raise ValidationError(f"project name sanitizes to empty: {name!r}")
    return sanitized


def validate_topic(topic: object) -> str:
    if not isinstance(topic, str) or not TOPIC_RE.match(topic):
        raise ValidationError(f"topic must match {TOPIC_RE.pattern!r}: {topic!r}")
    return topic


def validate_grant(consumer: object, topic: object) -> tuple[str, str]:
    """A cross-project consume grant: a validated (consumer_project, foreign_topic) pair (R131).

    consumer is sanitized to its proj_<name> identity (same rule as provision); topic is a real
    topic name. Nothing here decides POLICY (who may read what) — that's the trusted brain's call,
    same as every other bus op; this only rejects malformed input before it reaches an ACL write.
    """
    return validate_project(consumer), validate_topic(topic)


def validate_partitions(partitions: object) -> int:
    if not isinstance(partitions, int) or isinstance(partitions, bool):
        raise ValidationError(f"partitions must be an integer: {partitions!r}")
    if not (PARTITIONS_MIN <= partitions <= PARTITIONS_MAX):
        raise ValidationError(f"partitions {partitions} outside {PARTITIONS_MIN}-{PARTITIONS_MAX}")
    return partitions


def validate_tail_n(n: object) -> int:
    if not isinstance(n, int) or isinstance(n, bool):
        raise ValidationError(f"n must be an integer: {n!r}")
    if not (TAIL_N_MIN <= n <= TAIL_N_MAX):
        raise ValidationError(f"n {n} outside {TAIL_N_MIN}-{TAIL_N_MAX}")
    return n
