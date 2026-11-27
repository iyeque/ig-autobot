"""Tests for the Cloudflare-backed main-brand quote image renderer.

Covers: unconfigured -> None (fallback), API failure -> None (fallback),
successful render dimensions/format, correct Cloudflare request shape, and
exact parity of the quote-text cleaning with the PIL fallback renderer.
"""
import io
import base64
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_quote_image_cloudflare as cfq  # noqa: E402


REAL_QUOTE = (
    "If you can't evade it, embrace it. — from Chapter 2: If you can't evade "
    "it, embrace it in The Nine Stitches by M.W.E. Wigman."
)


def _fake_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1024, 1024), (30, 40, 60)).save(buf, format="PNG")
    return buf.getvalue()


class FakeResponse:
    def __init__(self, payload: bytes, content_type: str = "image/png"):
        self.content = payload
        self.headers = {"content-type": content_type}
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return {"result": {"image": base64.b64encode(self._payload).decode()}}


class QuoteCloudflareFallbackTest(unittest.TestCase):
    def test_unconfigured_returns_none(self):
        with patch.dict("os.environ", {"CLOUDFLARE_ACCOUNT_ID": "", "CLOUDFLARE_API_TOKEN": ""}, clear=False):
            with patch.object(cfq.requests, "post") as post:
                self.assertIsNone(cfq.generate_quote_image_cloudflare(REAL_QUOTE, "Chapter 2"))
                post.assert_not_called()

    def test_partial_config_returns_none(self):
        with patch.dict("os.environ", {"CLOUDFLARE_ACCOUNT_ID": "acct", "CLOUDFLARE_API_TOKEN": ""}, clear=False):
            with patch.object(cfq.requests, "post") as post:
                self.assertIsNone(cfq.generate_quote_image_cloudflare(REAL_QUOTE, "Chapter 2"))
                post.assert_not_called()

    def test_api_failure_returns_none(self):
        with patch.dict("os.environ", {"CLOUDFLARE_ACCOUNT_ID": "acct", "CLOUDFLARE_API_TOKEN": "tok"}, clear=False):
            with patch.object(cfq.requests, "post", side_effect=RuntimeError("offline")):
                self.assertIsNone(cfq.generate_quote_image_cloudflare(REAL_QUOTE, "Chapter 2"))


class QuoteCloudflareRenderTest(unittest.TestCase):
    def _configured(self):
        return patch.dict("os.environ", {"CLOUDFLARE_ACCOUNT_ID": "acct", "CLOUDFLARE_API_TOKEN": "tok"}, clear=False)

    def test_success_renders_1080x1350_jpeg(self):
        import tempfile
        with self._configured(), patch.object(cfq.requests, "post", return_value=FakeResponse(_fake_png())):
            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "quote_post.jpg"
                result = cfq.generate_quote_image_cloudflare(REAL_QUOTE, "Chapter 2", out_path=str(out))
                self.assertEqual(result, str(out))
                self.assertTrue(out.exists())
                with Image.open(out) as im:
                    self.assertEqual(im.format, "JPEG")
                    self.assertEqual(im.size, (1080, 1350))

    def test_request_uses_flux_model_and_auth(self):
        import tempfile
        with self._configured(), patch.object(cfq.requests, "post", return_value=FakeResponse(_fake_png())) as post:
            with tempfile.TemporaryDirectory() as td:
                cfq.generate_quote_image_cloudflare(REAL_QUOTE, "Chapter 2",
                                                    out_path=str(Path(td) / "quote.jpg"))
            args, kwargs = post.call_args
            self.assertIn(cfq.CLOUDFLARE_MODEL, args[0])
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer tok")
            self.assertIn("prompt", kwargs["json"])
            self.assertNotIn("embrace it", kwargs["json"]["prompt"])  # book text not sent to the AI

    def test_prompt_is_deterministic_per_quote(self):
        self.assertEqual(cfq._pick_prompt(REAL_QUOTE), cfq._pick_prompt(REAL_QUOTE))


class QuoteCleaningParityTest(unittest.TestCase):
    def test_strips_chapter_author_book_attribution(self):
        # Matches exactly what publish_instagram_quotes.generate_quote_image shows.
        self.assertEqual(
            cfq._clean_quote_text(REAL_QUOTE),
            '"If you can\'t evade it, embrace it"',
        )

    def test_introduction_attribution(self):
        raw = ("The body, wrapped in flesh, marks the boundary where the physical meets "
               "the metaphysical. — from Introduction in The Nine Stitches by M.W.E. Wigman.")
        self.assertEqual(
            cfq._clean_quote_text(raw),
            '"The body, wrapped in flesh, marks the boundary where the physical meets the metaphysical"',
        )


if __name__ == "__main__":
    unittest.main()
