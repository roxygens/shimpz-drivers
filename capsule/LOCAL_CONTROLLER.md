# Local Capsule controller v1

`Dockerfile.local` is the single-owner local controller used by an installed Shimpz Space. It is a
separate runtime from the hosted `Dockerfile`/`app.py` controller: it has no Brain, PostgreSQL, R2,
egress-policy, or `runsc` dependency.

An empty Capsule is one internal Docker bridge network. Every network and Assistant container carries
these exact ownership values:

- `com.shimpz.local.managed=1`
- `com.shimpz.local.profile=single-owner-local-v1`
- `com.shimpz.local.space-id=$SPACE_ID`
- a fixed kind plus the Capsule/Assistant identity needed to derive its deterministic Docker name
- Capsule networks also carry the validated 1–80 character display name used by the Admin

The Admin does not receive the Docker socket. It reads the controller's volume-backed token and calls
the internal port with `Authorization: Bearer …`. Request bodies are bounded JSON objects and every
accepted or rejected call writes metadata-only audit JSONL; tokens, request bodies, and Assistant output
are never audited.

## Runtime contract

- Required environment: `SHIMPZ_SPACE_ID`, a stable lowercase/dash-separated ID (maximum 48 bytes).
- Internal HTTP port: `7077`; publish it only on the private Compose network, never a host interface.
- Process identity: UID/GID `10001:10001`, with fixed supplementary token GID `10010`.
- Writable controller volumes: `/run/shimpz-local` for the token, `/var/log/shimpz-local` for the
  bounded audit journal, and `/var/lib/shimpz-local/storage` for opaque per-Capsule blobs. Storage is
  never mounted into the Admin or an Assistant. The rest of the controller root filesystem may be
  read-only, with `/tmp` as a small `noexec,nosuid,nodev` tmpfs.
- Each Capsule starts with an exact 100 MiB payload quota, at most 256 files, and at most 25 MiB per
  upload. The trusted controller enforces quota transactionally and applies a SQLite page ceiling;
  plan-specific limits can later replace the trusted quota resolver without changing the API. This
  is intentionally not described as a portable kernel project quota: Docker Desktop and ordinary
  Linux Docker volumes do not expose one consistently, while no workload receives any storage mount.
- Docker access: bind `/var/run/docker.sock` read/write only into this controller and add the socket's
  numeric host GID with Compose `group_add`. The Admin must never mount the socket.
- On the first start, the controller atomically creates 32 random bytes as 64 lowercase hex characters
  at `/run/shimpz-local/token`, owned by `10001:10010`, mode `0440`, one hard link. The Admin mounts the
  same setgid token volume read-only and joins GID `10010`; the bearer is never supplied through environment.
- Image healthcheck: the bundled probe reads that token and performs authenticated `GET /healthz` on
  loopback. The endpoint returns `{status:"ok",trace_id:"…"}` and is not public/unauthenticated.

## API

All routes require the bearer token, including health and read routes.

| Method | Path | Body | Result |
| --- | --- | --- | --- |
| `GET` | `/v1/assistants` | none | `{assistants:[{id,title,summary,powers}]}` |
| `GET` | `/v1/capsules` | none | `{capsules:[{id,name,status:"running"}]}` |
| `POST` | `/v1/capsules/{capsule}/create` | `{name:"My Capsule"}` | idempotently creates an empty Capsule |
| `DELETE` | `/v1/capsules/{capsule}` | none | idempotently removes its Assistants, then its network |
| `GET` | `/v1/capsules/{capsule}/assistants` | none | `{assistants:[{assistant,status}]}` |
| `POST` | `/v1/capsules/{capsule}/assistants` | `{assistant:"hello-pulse"}` | idempotently installs the allowlisted digest |
| `DELETE` | `/v1/capsules/{capsule}/assistants/{assistant}` | none | idempotently uninstalls it |
| `POST` | `/v1/capsules/{capsule}/assistants/hello-pulse/powers/hello` | `{name:"Captain"}` | invokes only the declared Power |
| `GET` | `/v1/capsules/{capsule}/files` | none | lists opaque metadata and the 100 MiB logical quota |
| `POST` | `/v1/capsules/{capsule}/files` | `{filename,media_type?,content_b64}` | stores one opaque object, up to 25 MiB |
| `DELETE` | `/v1/capsules/{capsule}/files/{opaque_id}` | none | deletes one object from that Capsule only |
| `DELETE` | `/v1/space` | none | idempotent installer reset for this `SPACE_ID` |

`DELETE /v1/space` accepts no resource IDs. It acquires every Capsule mutation lock, selects only
resources with the full managed/profile/`SPACE_ID`/kind label set, verifies their deterministic names,
removes Assistant containers first, every safely shaped directory in the dedicated storage volume second,
and Capsule networks last. This also removes storage orphaned by a prior daemon failure. It does not remove shared
images, the controller itself, or any Docker resource without the exact ownership values. The installer
must call it before removing the controller and its token volume; retries are safe after a partial
daemon failure.

## Assistant release binding

The source image contains a deliberate all-zero placeholder and fails closed at startup. Release
automation binds the published first-party digest without changing runtime configuration:

```sh
docker build \
  --file capsule/Dockerfile.local \
  --build-arg HELLO_PULSE_IMAGE='ghcr.io/roxygens/shimpz-space@sha256:<digest>' \
  capsule
```

Unknown Assistant IDs are resolved before an image lookup. A missing trusted digest is pulled once;
the resulting repository digest and the Assistant v1 image labels must match before creation.
Assistant containers run as `10001:10001`, with a read-only root, all capabilities dropped,
`no-new-privileges`, Docker's default seccomp profile, no mounts or published ports, one internal
Capsule network, 0.25 CPU, 128 MiB memory/swap, 64 PIDs, and a 1,024-file-descriptor ceiling.
