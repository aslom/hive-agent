"""Tests for scripts/db.py — URL canonicalization and url_id."""
import hashlib
import sys
import unittest
from pathlib import Path

# Make scripts/ importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from db import canonicalize_url, url_id


class TestCanonicalizeUrl(unittest.TestCase):

    # ── scheme + host ──────────────────────────────────────────────────────
    def test_lowercases_scheme(self):
        self.assertEqual(
            canonicalize_url("HTTP://Example.COM/path"),
            "http://example.com/path",
        )

    def test_lowercases_host(self):
        self.assertEqual(
            canonicalize_url("https://Example.COM/"),
            "https://example.com/",
        )

    # ── trailing slash ─────────────────────────────────────────────────────
    def test_strips_trailing_slash_on_path(self):
        result = canonicalize_url("https://example.com/foo/bar/")
        self.assertFalse(result.endswith("/bar/"))
        self.assertTrue(result.endswith("/foo/bar"))

    def test_preserves_root_slash(self):
        # Root "/" must stay — stripping it would produce an empty path.
        result = canonicalize_url("https://example.com/")
        self.assertTrue(result.endswith("/"))

    # ── tracking-param stripping ───────────────────────────────────────────
    def test_strips_utm_params(self):
        url = "https://example.com/post?utm_source=twitter&utm_medium=social"
        result = canonicalize_url(url)
        self.assertNotIn("utm_source", result)
        self.assertNotIn("utm_medium", result)

    def test_strips_fbclid(self):
        url = "https://example.com/post?fbclid=abc123"
        self.assertNotIn("fbclid", canonicalize_url(url))

    def test_strips_gclid(self):
        url = "https://example.com/post?gclid=abc123"
        self.assertNotIn("gclid", canonicalize_url(url))

    def test_strips_ref_param(self):
        url = "https://example.com/post?ref=hn"
        self.assertNotIn("ref=", canonicalize_url(url))

    def test_preserves_non_tracking_params(self):
        url = "https://example.com/search?q=agents&page=2"
        result = canonicalize_url(url)
        self.assertIn("q=agents", result)
        self.assertIn("page=2", result)

    def test_mixed_tracking_and_real_params(self):
        url = "https://example.com/page?q=llm&utm_source=feed&page=1"
        result = canonicalize_url(url)
        self.assertIn("q=llm", result)
        self.assertIn("page=1", result)
        self.assertNotIn("utm_source", result)

    # ── arXiv normalisation ────────────────────────────────────────────────
    def test_arxiv_pdf_to_abs(self):
        self.assertEqual(
            canonicalize_url("https://arxiv.org/pdf/2401.12345.pdf"),
            "https://arxiv.org/abs/2401.12345",
        )

    def test_arxiv_pdf_versioned(self):
        self.assertEqual(
            canonicalize_url("https://arxiv.org/pdf/2401.12345v2"),
            "https://arxiv.org/abs/2401.12345",
        )

    def test_arxiv_abs_versioned(self):
        self.assertEqual(
            canonicalize_url("https://arxiv.org/abs/2401.12345v3"),
            "https://arxiv.org/abs/2401.12345",
        )

    def test_arxiv_html_variant(self):
        self.assertEqual(
            canonicalize_url("https://arxiv.org/html/2401.12345"),
            "https://arxiv.org/abs/2401.12345",
        )

    def test_arxiv_pdf_and_abs_same_canonical(self):
        pdf = canonicalize_url("https://arxiv.org/pdf/2401.12345.pdf")
        abs_ = canonicalize_url("https://arxiv.org/abs/2401.12345")
        self.assertEqual(pdf, abs_)

    def test_arxiv_different_ids_differ(self):
        a = canonicalize_url("https://arxiv.org/abs/2401.00001")
        b = canonicalize_url("https://arxiv.org/abs/2401.00002")
        self.assertNotEqual(a, b)

    # ── whitespace / leading-trailing strip ───────────────────────────────
    def test_strips_surrounding_whitespace(self):
        self.assertEqual(
            canonicalize_url("  https://example.com/path  "),
            canonicalize_url("https://example.com/path"),
        )


class TestUrlId(unittest.TestCase):

    def test_returns_hex_string(self):
        result = url_id("https://example.com/")
        self.assertRegex(result, r"^[0-9a-f]{64}$")

    def test_deterministic(self):
        url = "https://example.com/foo"
        self.assertEqual(url_id(url), url_id(url))

    def test_canonical_equivalence(self):
        # PDF and abs variants of the same arXiv paper must produce the same id.
        self.assertEqual(
            url_id("https://arxiv.org/pdf/2401.12345v2"),
            url_id("https://arxiv.org/abs/2401.12345"),
        )

    def test_tracking_param_stripped_same_id(self):
        clean = url_id("https://example.com/post")
        dirty = url_id("https://example.com/post?utm_source=newsletter")
        self.assertEqual(clean, dirty)

    def test_different_urls_differ(self):
        self.assertNotEqual(
            url_id("https://example.com/a"),
            url_id("https://example.com/b"),
        )

    def test_matches_manual_sha256(self):
        url = "https://example.com/test"
        canonical = canonicalize_url(url)
        expected = hashlib.sha256(canonical.encode()).hexdigest()
        self.assertEqual(url_id(url), expected)


if __name__ == "__main__":
    unittest.main()
