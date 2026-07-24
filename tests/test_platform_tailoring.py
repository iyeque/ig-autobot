import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import scripts.prepare_assets as prepare_assets
import scripts.publish as publish


class PlatformTailoringTests(unittest.TestCase):
    def test_bluesky_tailoring_adds_linkedin_cta_and_limits_length(self):
        caption = "A short bluesky caption"
        tailored = prepare_assets._apply_platform_tailoring(caption, "bluesky")
        self.assertIn("Want to read more?... check out my LinkedIn", tailored)
        self.assertLessEqual(len(tailored), 280)

    def test_threads_tailoring_shortens_caption(self):
        caption = "A" * 500
        tailored = prepare_assets._apply_platform_tailoring(caption, "threads")
        self.assertLessEqual(len(tailored), 420)

    def test_youtube_tailoring_uses_short_caption(self):
        caption = "A" * 500
        tailored = prepare_assets._apply_platform_tailoring(caption, "youtube")
        self.assertLessEqual(len(tailored), 400)

    def test_instagram_cadence_selects_expected_format_by_day(self):
        expected = {
            0: "carousel",
            1: "reel",
            2: "carousel",
            3: "reel",
            4: "carousel",
            5: "reel",
            6: "static",
        }
        for weekday, expected_format in expected.items():
            with self.subTest(weekday=weekday):
                self.assertEqual(prepare_assets._instagram_format_for_weekday(weekday), expected_format)

    def test_publish_uses_instagram_format_marker_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "instagram_format.txt"), "w", encoding="utf-8") as handle:
                handle.write("reel")
            self.assertEqual(publish._get_instagram_preferred_format(tmpdir), "reel")


if __name__ == "__main__":
    unittest.main()
