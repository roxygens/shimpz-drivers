from __future__ import annotations

import json
import os
import stat
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import oauth_account_store
from oauth_http_client import OAuthTokenSet

ACCESS = "access-token-private-material-123456789"
REFRESH = "refresh-token-private-material-987654321"
SCOPES = ("offline.access", "tweet.read", "tweet.write", "users.read")
DECLARATIONS = {"x": {"provider": "x", "scopes": SCOPES}}
ACCOUNT = {"id": "2244994945", "username": "XDevelopers", "name": "X Developers"}


def tokens(
    *,
    access: str = ACCESS,
    refresh: str | None = REFRESH,
    scopes: tuple[str, ...] = SCOPES,
    expires_in: int = 3600,
    broker_lease: str | None = None,
) -> OAuthTokenSet:
    return OAuthTokenSet(access, refresh, scopes, expires_in, broker_lease)


class OAuthAccountStoreTests(unittest.TestCase):
    def _store(
        self,
        root: Path,
        *,
        clock=lambda: 1_000_000_000,
    ) -> oauth_account_store.OAuthAccountStore:
        return oauth_account_store.OAuthAccountStore(
            root / "state" / "accounts.json",
            root / "key" / "aes256.key",
            clock=clock,
        )

    def test_inventory_includes_missing_and_encrypted_account_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            missing = store.metadata("team_1", "shimpz-assistant", DECLARATIONS)
            self.assertEqual(
                missing,
                (oauth_account_store.OAuthAccountMetadata("x", "x", SCOPES, "missing", None, None, 0),),
            )

            stored = store.put("team_1", "shimpz-assistant", "x", "x", SCOPES, tokens(), ACCOUNT)
            self.assertEqual(stored.generation, 1)
            self.assertEqual(stored.status, "connected")
            self.assertEqual(stored.account, oauth_account_store.OAuthAccountIdentity(**ACCOUNT))
            self.assertEqual(store.metadata("team_1", "shimpz-assistant", DECLARATIONS), (stored,))
            self.assertEqual(
                store.resolve(
                    "team_1",
                    "shimpz-assistant",
                    "x",
                    "x",
                    SCOPES,
                    lambda _token, _lease: self.fail("unexpired token must not refresh"),
                ),
                ACCESS,
            )

            state = (root / "state" / "accounts.json").read_text(encoding="utf-8")
            key = (root / "key" / "aes256.key").read_bytes()
            for private in (ACCESS, REFRESH, "2244994945", "XDevelopers", "X Developers"):
                self.assertNotIn(private, state)
                self.assertNotIn(private.encode(), key)
            self.assertNotIn("access_token", state)
            self.assertNotIn("refresh_token", state)
            self.assertNotIn(ACCESS, repr(stored))
            self.assertEqual(stat.S_IMODE(store.state_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(store.key_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(store.state_path.parent.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(store.key_path.parent.stat().st_mode), 0o700)

    def test_rotation_is_atomic_and_increments_authenticated_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(Path(directory))
            writes = 0
            original = store._write_state

            def counted(state) -> None:
                nonlocal writes
                writes += 1
                original(state)

            store._write_state = counted
            first = store.put("team_1", "shimpz-assistant", "x", "x", SCOPES, tokens(), ACCOUNT)
            second = store.put(
                "team_1",
                "shimpz-assistant",
                "x",
                "x",
                SCOPES,
                tokens(access="new-access-token-123456789"),
                ACCOUNT,
            )
            self.assertEqual((first.generation, second.generation, writes), (1, 2, 2))
            self.assertEqual(
                store.resolve(
                    "team_1",
                    "shimpz-assistant",
                    "x",
                    "x",
                    SCOPES,
                    lambda _token, _lease: None,
                ),
                "new-access-token-123456789",
            )

    def test_expired_account_refresh_is_single_flight_and_preserves_account(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            now = [1_000]
            store = self._store(Path(directory), clock=lambda: now[0])
            store.put(
                "team_1",
                "shimpz-assistant",
                "x",
                "x",
                SCOPES,
                tokens(expires_in=30, broker_lease="broker-lease-private-material-123456789"),
                ACCOUNT,
            )
            self.assertEqual(
                store.metadata("team_1", "shimpz-assistant", DECLARATIONS)[0].status,
                "connected",
            )
            now[0] = 1_031
            self.assertEqual(
                store.metadata("team_1", "shimpz-assistant", DECLARATIONS)[0].status,
                "refresh-required",
            )

            entered = threading.Event()
            release = threading.Event()
            calls: list[str] = []

            def refresh(value: str, lease: str | None) -> OAuthTokenSet:
                calls.append(value)
                self.assertEqual(lease, "broker-lease-private-material-123456789")
                entered.set()
                self.assertTrue(release.wait(2))
                return tokens(access="refreshed-access-token-123456789", expires_in=3600)

            def resolve() -> str:
                return store.resolve("team_1", "shimpz-assistant", "x", "x", SCOPES, refresh)

            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(resolve)
                self.assertTrue(entered.wait(2))
                second = pool.submit(resolve)
                release.set()
                self.assertEqual(first.result(2), "refreshed-access-token-123456789")
                self.assertEqual(second.result(2), "refreshed-access-token-123456789")
            self.assertEqual(calls, [REFRESH])
            metadata = store.metadata("team_1", "shimpz-assistant", DECLARATIONS)[0]
            self.assertEqual(metadata.generation, 2)
            self.assertEqual(metadata.account, oauth_account_store.OAuthAccountIdentity(**ACCOUNT))

    def test_missing_refresh_and_declaration_drift_require_reauthorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            now = [1_000]
            store = self._store(Path(directory), clock=lambda: now[0])
            reduced_scopes = ("tweet.read", "users.read")
            store.put(
                "team_1",
                "shimpz-assistant",
                "x",
                "x",
                reduced_scopes,
                tokens(refresh=None, scopes=reduced_scopes, expires_in=30),
                None,
            )
            drifted = store.metadata("team_1", "shimpz-assistant", DECLARATIONS)[0]
            self.assertEqual(drifted.status, "reauthorization-required")
            self.assertEqual(drifted.scopes, SCOPES)
            self.assertIsNone(drifted.account)
            with self.assertRaises(oauth_account_store.OAuthAccountReauthorizationError):
                store.resolve(
                    "team_1",
                    "shimpz-assistant",
                    "x",
                    "x",
                    SCOPES,
                    lambda _token, _lease: None,
                )

            reduced = {"x": {"provider": "x", "scopes": reduced_scopes}}
            now[0] = 1_031
            self.assertEqual(
                store.metadata("team_1", "shimpz-assistant", reduced)[0].status,
                "reauthorization-required",
            )
            with self.assertRaises(oauth_account_store.OAuthAccountReauthorizationError):
                store.resolve(
                    "team_1",
                    "shimpz-assistant",
                    "x",
                    "x",
                    reduced_scopes,
                    lambda _token, _lease: None,
                )

    def test_aad_rejects_cross_identity_copy_and_metadata_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            store.put("team_1", "shimpz-assistant", "x", "x", SCOPES, tokens(), ACCOUNT)
            store.put(
                "team_2",
                "shimpz-assistant",
                "x",
                "x",
                SCOPES,
                tokens(access="other-access-token-123456789"),
                ACCOUNT,
            )
            state_path = root / "state" / "accounts.json"
            original = json.loads(state_path.read_text(encoding="utf-8"))
            copied = json.loads(json.dumps(original))
            copied["teams"]["team_2"]["shimpz-assistant"]["x"] = copied["teams"]["team_1"]["shimpz-assistant"]["x"]
            state_path.write_text(json.dumps(copied, separators=(",", ":")), encoding="utf-8")
            state_path.chmod(0o600)
            with self.assertRaises(oauth_account_store.OAuthAccountStoreError):
                store.metadata("team_2", "shimpz-assistant", DECLARATIONS)

            for field, value in (
                ("expires_at", 1_000_003_601),
                ("status", "reauthorization-required"),
                ("generation", 2),
                ("scopes", ["tweet.read", "users.read"]),
            ):
                tampered = json.loads(json.dumps(original))
                tampered["teams"]["team_1"]["shimpz-assistant"]["x"][field] = value
                state_path.write_text(json.dumps(tampered, separators=(",", ":")), encoding="utf-8")
                state_path.chmod(0o600)
                with self.subTest(field=field), self.assertRaises(oauth_account_store.OAuthAccountStoreError):
                    store.metadata("team_1", "shimpz-assistant", DECLARATIONS)

    def test_missing_or_substituted_key_fails_closed_without_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            store.put("team_1", "shimpz-assistant", "x", "x", SCOPES, tokens(), ACCOUNT)
            original_state = store.state_path.read_bytes()
            store.key_path.unlink()
            with self.assertRaises(oauth_account_store.OAuthAccountStoreError):
                store.put(
                    "team_1",
                    "shimpz-assistant",
                    "x",
                    "x",
                    SCOPES,
                    tokens(access="replacement-token-123456789"),
                    ACCOUNT,
                )
            self.assertFalse(store.key_path.exists())
            self.assertEqual(store.state_path.read_bytes(), original_state)

            store.key_path.write_bytes(os.urandom(32))
            store.key_path.chmod(0o600)
            with self.assertRaises(oauth_account_store.OAuthAccountStoreError):
                store.metadata("team_1", "shimpz-assistant", DECLARATIONS)

    def test_invalid_tokens_permissions_symlinks_and_duplicate_json_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = self._store(root)
            store.put("team_1", "shimpz-assistant", "x", "x", SCOPES, tokens(), ACCOUNT)
            original = store.state_path.read_bytes()
            invalid = (
                tokens(access="line\nbreak"),
                tokens(scopes=("dm.read",)),
                tokens(expires_in=29),
            )
            for value in invalid:
                with (
                    self.subTest(value=value),
                    self.assertRaises(oauth_account_store.OAuthAccountValidationError),
                ):
                    store.put("team_1", "shimpz-assistant", "x", "x", SCOPES, value, ACCOUNT)
                self.assertEqual(store.state_path.read_bytes(), original)

            store.state_path.chmod(0o644)
            with self.assertRaises(oauth_account_store.OAuthAccountStoreError):
                store.metadata("team_1", "shimpz-assistant", DECLARATIONS)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "state").mkdir(mode=0o700)
            target = root / "target.json"
            target.write_text('{"schema":1,"teams":{},"teams":{}}', encoding="utf-8")
            target.chmod(0o600)
            symlink = root / "state" / "accounts.json"
            symlink.symlink_to(target)
            with self.assertRaises(oauth_account_store.OAuthAccountStoreError):
                self._store(root).metadata("team_1", "shimpz-assistant", DECLARATIONS)
            symlink.unlink()
            target.replace(symlink)
            with self.assertRaisesRegex(oauth_account_store.OAuthAccountStoreError, "duplicate"):
                self._store(root).metadata("team_1", "shimpz-assistant", DECLARATIONS)

    def test_retention_and_deletion_are_exactly_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(Path(directory))
            for team, assistant in (
                ("team_1", "first-assistant"),
                ("team_1", "second-assistant"),
                ("team_2", "second-assistant"),
            ):
                store.put(team, assistant, "x", "x", SCOPES, tokens(), ACCOUNT)

            self.assertFalse(store.retain_declared("team_1", "first-assistant", {"x": object()}))
            self.assertTrue(store.retain_declared("team_1", "first-assistant", {}))
            self.assertFalse(store.delete_account("team_1", "first-assistant", "x"))
            self.assertEqual(
                store.metadata("team_1", "second-assistant", DECLARATIONS)[0].status,
                "connected",
            )
            self.assertTrue(store.delete_team("team_1"))
            self.assertEqual(
                store.metadata("team_1", "second-assistant", DECLARATIONS)[0].status,
                "missing",
            )
            self.assertTrue(store.delete_assistant("team_2", "second-assistant"))
            self.assertFalse(store.delete_assistant("team_2", "second-assistant"))
            self.assertFalse(store.delete_all())

    def test_revocation_transaction_keeps_authenticated_custody_until_callback_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self._store(Path(directory))
            store.put("team_1", "shimpz-assistant", "x", "x", SCOPES, tokens(), ACCOUNT)
            observed: list[tuple[str, str, str | None, str | None]] = []

            def fail(provider: str, access: str, refresh: str | None, lease: str | None) -> None:
                observed.append((provider, access, refresh, lease))
                raise RuntimeError("synthetic upstream failure")

            with self.assertRaisesRegex(RuntimeError, "upstream failure"):
                store.revoke_then_delete("team_1", "shimpz-assistant", "x", fail)
            self.assertEqual(observed, [("x", ACCESS, REFRESH, None)])
            self.assertEqual(
                store.metadata("team_1", "shimpz-assistant", DECLARATIONS)[0].status,
                "connected",
            )

            self.assertTrue(
                store.revoke_then_delete(
                    "team_1",
                    "shimpz-assistant",
                    "x",
                    lambda provider, access, refresh, lease: observed.append((provider, access, refresh, lease)),
                )
            )
            self.assertEqual(
                observed,
                [("x", ACCESS, REFRESH, None), ("x", ACCESS, REFRESH, None)],
            )
            self.assertFalse(
                store.revoke_then_delete(
                    "team_1",
                    "shimpz-assistant",
                    "x",
                    lambda *_tokens: self.fail("missing account must not invoke revocation"),
                )
            )


if __name__ == "__main__":
    unittest.main()
