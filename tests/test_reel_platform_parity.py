import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import shared_utils  # noqa: E402


class ReelParityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        (self.base / "reels").mkdir()
        self.hf = self.base / "reels" / "reel_42_hyperframes.mp4"
        self.pil = self.base / "reels" / "reel_2026_pil.mp4"
        self.pil.write_bytes(b"pil")
        self.active = {"post_id": 42, "reel": "reels/reel_2026_pil.mp4"}

    def test_hyperframes_preferred_when_present(self):
        self.hf.write_bytes(b"hyperframes")
        path = shared_utils.resolve_reel_path(self.active, state_dir=str(self.base))
        self.assertTrue(path.endswith("reel_42_hyperframes.mp4"), path)

    def test_pil_fallback_when_hyperframes_missing(self):
        path = shared_utils.resolve_reel_path(self.active, state_dir=str(self.base))
        self.assertTrue(path.endswith("reel_2026_pil.mp4"), path)

    def test_reel_mp4_last_fallback(self):
        (self.base / "reel.mp4").write_bytes(b"local")
        path = shared_utils.resolve_reel_path({"post_id": 99}, state_dir=str(self.base))
        self.assertTrue(path.endswith("reel.mp4"), path)
        self.assertNotIn("hyperframes", path)

    def test_public_url_points_at_hyperframes(self):
        self.hf.write_bytes(b"hyperframes")
        media = shared_utils.resolve_bundle_media(
            self.active, state_dir=str(self.base),
            base_url="https://example.test/site/",
        )
        self.assertEqual(
            media["reel"],
            "https://example.test/site/reels/reel_42_hyperframes.mp4",
        )
        self.assertTrue(media["reel_local"].endswith("reel_42_hyperframes.mp4"))

    def test_public_url_falls_back_to_pil(self):
        media = shared_utils.resolve_bundle_media(
            self.active, state_dir=str(self.base),
            base_url="https://example.test/site/",
        )
        self.assertEqual(
            media["reel"],
            "https://example.test/site/reels/reel_2026_pil.mp4",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
