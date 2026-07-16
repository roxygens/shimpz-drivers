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


def _hello_image(
    *,
    digest: str = marketplace.HELLO_PULSE_IMAGE,
    assistant_id: str = "hello-pulse",
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
    def test_hello_pulse_registry_contract_is_digest_backed_and_resource_free(self) -> None:
        spec = marketplace.APPS["hello-pulse"]
        self.assertEqual(
            spec.image,
            "ghcr.io/roxygens/shimpz-space@sha256:2b051d2db0b83ec4689901033f8ca38459265a38c7a3e84f53271a3f786471b5",
        )
        self.assertTrue(marketplace.is_digest_image(spec.image))
        self.assertEqual((spec.port, spec.health_path), (8080, "/health"))
        self.assertFalse(spec.db)
        self.assertEqual(spec.egress, ())
        self.assertTrue(spec.first_party)
        self.assertEqual(
            dict(spec.required_image_labels),
            {"org.shimpz.assistant.id": "hello-pulse", "org.shimpz.assistant.api": "1"},
        )

    def test_missing_digest_is_pulled_by_the_exact_registry_reference_then_rechecked(self) -> None:
        spec = marketplace.APPS["hello-pulse"]
        images = _Images(_hello_image(), missing_once=True)

        image_id = marketplace_image.ensure_digest_artifact(images, spec)

        self.assertEqual(image_id, "sha256:" + "b" * 64)
        self.assertEqual(images.gets, [spec.image, spec.image])
        self.assertEqual(images.pulls, [spec.image])

    def test_digest_or_assistant_label_mismatch_is_refused_without_a_pull(self) -> None:
        spec = marketplace.APPS["hello-pulse"]
        mismatches = (
            _hello_image(digest="ghcr.io/roxygens/shimpz-space@sha256:" + "c" * 64),
            _hello_image(assistant_id="other-assistant"),
            _hello_image(assistant_api="2"),
        )
        for image in mismatches:
            with self.subTest(attrs=image.attrs):
                images = _Images(image)
                with self.assertRaises(marketplace_image.ImageTrustError):
                    marketplace_image.ensure_digest_artifact(images, spec)
                self.assertEqual(images.pulls, [])

    def test_tag_backed_notification_center_is_not_eligible_for_registry_pull(self) -> None:
        spec = marketplace.APPS["notification-center"]
        images = _Images(_hello_image())
        self.assertFalse(marketplace.is_digest_image(spec.image))
        with self.assertRaises(marketplace_image.ImageTrustError):
            marketplace_image.ensure_digest_artifact(images, spec)
        self.assertEqual(images.gets, [])
        self.assertEqual(images.pulls, [])


if __name__ == "__main__":
    unittest.main()
