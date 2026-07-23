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
        self.attrs = {"RepoDigests": repo_digests, "Config": {"Labels": labels}}


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
    digest: str = marketplace.SHIMPZ_CLOUDFLARE_ASSISTANT_IMAGE,
    assistant_id: str = "shimpz-cloudflare",
    assistant_api: str = "1",
) -> _Image:
    return _Image(
        repo_digests=[digest],
        labels={"org.shimpz.assistant.id": assistant_id, "org.shimpz.assistant.api": assistant_api},
    )


class MarketplaceImageTests(unittest.TestCase):
    def test_cloudflare_is_the_only_first_party_assistant(self) -> None:
        self.assertEqual(set(marketplace.APPS), {"notification-center", "shimpz-cloudflare"})
        spec = marketplace.APPS["shimpz-cloudflare"]
        self.assertEqual(spec.image, marketplace.SHIMPZ_CLOUDFLARE_ASSISTANT_IMAGE)
        self.assertTrue(marketplace.is_digest_image(spec.image))
        self.assertEqual((spec.port, spec.health_path), (8080, "/healthz"))
        self.assertFalse(spec.db)
        self.assertEqual(spec.allowed_hosts, ("api.cloudflare.com",))
        self.assertTrue(spec.first_party)
        self.assertEqual(
            dict(spec.required_image_labels),
            {"org.shimpz.assistant.id": "shimpz-cloudflare", "org.shimpz.assistant.api": "1"},
        )
        self.assertIsNotNone(spec.assistant)
        assert spec.assistant is not None
        self.assertEqual(set(spec.assistant.powers), {"list-zones", "list-dns-records"})
        self.assertEqual(spec.assistant.secrets, {})
        self.assertEqual(spec.assistant.accounts["cloudflare"].provider, "cloudflare")
        self.assertEqual(
            spec.assistant.accounts["cloudflare"].scopes,
            ("dns.read", "offline_access", "zone.read"),
        )
        self.assertTrue(all(power.accounts == ("cloudflare",) for power in spec.assistant.powers.values()))
        self.assertTrue(all(not hasattr(power, "approval") for power in spec.assistant.powers.values()))

    def test_missing_digest_is_pulled_by_the_exact_registry_reference_then_rechecked(self) -> None:
        spec = marketplace.APPS["shimpz-cloudflare"]
        images = _Images(_assistant_image(), missing_once=True)
        self.assertEqual(marketplace_image.ensure_digest_artifact(images, spec), "sha256:" + "b" * 64)
        self.assertEqual(images.gets, [spec.image, spec.image])
        self.assertEqual(images.pulls, [spec.image])

    def test_digest_or_assistant_label_mismatch_is_refused_without_a_pull(self) -> None:
        spec = marketplace.APPS["shimpz-cloudflare"]
        mismatches = (
            _assistant_image(digest="ghcr.io/theshimpz/shimpz-space@sha256:" + "c" * 64),
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

    def test_cloudflare_power_input_and_output_contracts_are_closed(self) -> None:
        request = {"page": 1, "per_page": 25}
        self.assertEqual(marketplace.validate_power_input("shimpz-cloudflare", "list-zones", request), request)
        zones = {
            "zones": [],
            "pagination": {"page": 1, "per_page": 25, "count": 0, "total_count": 0, "total_pages": 0},
        }
        self.assertEqual(marketplace.validate_power_output("shimpz-cloudflare", "list-zones", zones), zones)
        for payload in ({"page": 1, "per_page": 25, "shell": "id"}, {"page": 0, "per_page": 25}, []):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                marketplace.validate_power_input("shimpz-cloudflare", "list-zones", payload)


if __name__ == "__main__":
    unittest.main()
