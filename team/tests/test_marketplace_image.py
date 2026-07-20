from __future__ import annotations

import unittest

import docker
import marketplace
import marketplace_image


class _Image:
    def __init__(
        self,
        *,
        repo_digests: list[str] | None = None,
        labels: dict[str, str] | None = None,
        image_id: str = "sha256:" + "b" * 64,
    ) -> None:
        self.id = image_id
        self.attrs = {
            "RepoDigests": repo_digests,
            "Config": {"Labels": labels},
        }


class _Images:
    def __init__(self, image: _Image | None, *, missing_once: bool = False) -> None:
        self.image = image
        self.missing_once = missing_once
        self.gets: list[str] = []
        self.pulls: list[str] = []

    def get(self, image_ref: str) -> _Image:
        self.gets.append(image_ref)
        if self.missing_once:
            self.missing_once = False
            raise docker.errors.ImageNotFound("missing")
        if self.image is None:
            raise docker.errors.ImageNotFound("missing")
        return self.image

    def pull(self, image_ref: str) -> _Image | None:
        self.pulls.append(image_ref)
        return self.image


def _assistant_image(
    *,
    digest: str = marketplace.SHIMPZ_ASSISTANT_IMAGE,
    assistant_id: str = "shimpz-assistant",
    assistant_api: str = "1",
) -> _Image:
    return _Image(
        repo_digests=[digest],
        labels={
            "org.shimpz.assistant.id": assistant_id,
            "org.shimpz.assistant.api": assistant_api,
        },
    )


class MarketplaceImageTests(unittest.TestCase):
    def test_shimpz_assistant_registry_contract_is_digest_backed_and_resource_free(self) -> None:
        spec = marketplace.APPS["shimpz-assistant"]
        self.assertEqual(
            spec.image,
            "ghcr.io/roxygens/shimpz-space@sha256:1fb163897668d639a685cfc0fef8e91009a9c6ac1387d6dbf3b8cac8c024f1ff",
        )
        self.assertTrue(marketplace.is_digest_image(spec.image))
        self.assertEqual((spec.port, spec.health_path), (8080, "/health"))
        self.assertFalse(spec.db)
        self.assertEqual(spec.allowed_hosts, ("api.mux.com", "api.x.com"))
        self.assertTrue(spec.first_party)
        self.assertEqual(
            dict(spec.required_image_labels),
            {"org.shimpz.assistant.id": "shimpz-assistant", "org.shimpz.assistant.api": "1"},
        )
        self.assertIsNotNone(spec.assistant)
        self.assertEqual(
            set(spec.assistant.powers),
            {
                "public-user-lookup",
                "identity-me",
                "create-post",
                "delete-post",
                "list-direct-uploads",
                "create-test-direct-upload",
                "cancel-direct-upload",
                "verify-mux-webhook",
            },
        )
        self.assertEqual(spec.assistant.powers["identity-me"].path, "/v1/powers/identity-me")
        self.assertEqual(
            set(spec.assistant.secrets),
            {"mux-token-id", "mux-token-secret", "mux-webhook-signing-secret"},
        )
        self.assertEqual(spec.assistant.accounts["x"].provider, "x")
        self.assertEqual(
            spec.assistant.accounts["x"].scopes,
            ("offline.access", "tweet.read", "tweet.write", "users.read"),
        )
        x_powers = {"public-user-lookup", "identity-me", "create-post", "delete-post"}
        mux_api_powers = {"list-direct-uploads", "create-test-direct-upload", "cancel-direct-upload"}
        for power_id, power in spec.assistant.powers.items():
            self.assertEqual(power.accounts, ("x",) if power_id in x_powers else ())
            self.assertEqual(
                power.secrets,
                ("mux-token-id", "mux-token-secret")
                if power_id in mux_api_powers
                else ("mux-webhook-signing-secret",)
                if power_id == "verify-mux-webhook"
                else (),
            )

    def test_x_powers_expose_closed_runtime_schemas_and_explicit_approval(self) -> None:
        powers = marketplace.APPS["shimpz-assistant"].assistant.powers

        approved = {"create-post", "delete-post", "create-test-direct-upload", "cancel-direct-upload"}
        for power_id, power in powers.items():
            self.assertEqual(power.approval, "each-run" if power_id in approved else "none")
            self.assertEqual(power.input_schema["type"], "object")
            self.assertFalse(power.input_schema["additionalProperties"])
            self.assertFalse(power.output_schema["additionalProperties"])

    def test_missing_digest_is_pulled_by_the_exact_registry_reference_then_rechecked(self) -> None:
        spec = marketplace.APPS["shimpz-assistant"]
        images = _Images(_assistant_image(), missing_once=True)

        image_id = marketplace_image.ensure_digest_artifact(images, spec)

        self.assertEqual(image_id, "sha256:" + "b" * 64)
        self.assertEqual(images.gets, [spec.image, spec.image])
        self.assertEqual(images.pulls, [spec.image])

    def test_digest_or_assistant_label_mismatch_is_refused_without_a_pull(self) -> None:
        spec = marketplace.APPS["shimpz-assistant"]
        mismatches = (
            _assistant_image(digest="ghcr.io/roxygens/shimpz-space@sha256:" + "c" * 64),
            _assistant_image(assistant_id="other-assistant"),
            _assistant_image(assistant_api="2"),
        )
        for image in mismatches:
            with self.subTest(attrs=image.attrs):
                images = _Images(image)
                with self.assertRaises(marketplace_image.ImageTrustError):
                    marketplace_image.ensure_digest_artifact(images, spec)
                self.assertEqual(images.pulls, [])

    def test_tag_backed_notification_center_is_not_eligible_for_registry_pull(self) -> None:
        spec = marketplace.APPS["notification-center"]
        images = _Images(_assistant_image())
        self.assertFalse(marketplace.is_digest_image(spec.image))
        with self.assertRaises(marketplace_image.ImageTrustError):
            marketplace_image.ensure_digest_artifact(images, spec)
        self.assertEqual(images.gets, [])
        self.assertEqual(images.pulls, [])

    def test_shimpz_assistant_power_input_and_output_contracts_are_closed(self) -> None:
        self.assertEqual(
            marketplace.validate_power_input("shimpz-assistant", "public-user-lookup", {"username": "XDevelopers"}),
            {"username": "XDevelopers"},
        )
        self.assertEqual(
            marketplace.validate_power_input(
                "shimpz-assistant",
                "create-post",
                {"text": "Hello from Shimpz"},
            ),
            {"text": "Hello from Shimpz"},
        )
        current = {"id": "2244994945", "name": "X Developers", "username": "XDevelopers"}
        self.assertEqual(
            marketplace.validate_power_output("shimpz-assistant", "identity-me", current),
            current,
        )
        for payload in ({"username": 12}, {"username": "bad-name"}, {"username": "XDevelopers", "shell": "id"}, []):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                marketplace.validate_power_input("shimpz-assistant", "public-user-lookup", payload)
        with self.assertRaises(ValueError):
            marketplace.validate_power_output("shimpz-assistant", "identity-me", current | {"path": "/host/x"})


if __name__ == "__main__":
    unittest.main()
