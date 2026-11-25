import sys
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))  # same module object bot.py imports

import bot  # noqa: E402
import openrouter_captions as orc  # noqa: E402


NEW_RESPONSE = """The quiet hour before anyone wakes is the real boundary
Most families protect the device and leave the moment unguarded
A phone cannot answer what the silence is asking
Put one chair by a window and let it stay empty
What would you notice if the screen stayed off
---CAPTION---
The quiet hour before anyone wakes is where the day is actually decided. Most families protect the device and leave the moment itself unguarded.

A phone cannot answer what the silence is asking. It only arrives faster than the question. One empty chair by a window is enough to give the question somewhere to land.

What would you notice if the screen stayed off for the first twenty minutes?"""

LEGACY_RESPONSE = """slide1: The quiet hour before anyone wakes
slide2: Most families protect the device
slide3: A phone cannot answer the silence
slide4: Leave one chair by the window
slide5: What would you notice
post_caption: A short single-line carousel caption."""


class ParseCarouselResponseTest(unittest.TestCase):
    def test_new_delimiter_keeps_multiparagraph_caption(self):
        parsed = orc._parse_carousel_response(NEW_RESPONSE)
        self.assertEqual(len(parsed["slides"]), 5)
        self.assertIn("\n\n", parsed["post_caption"])
        self.assertTrue(parsed["post_caption"].endswith("?"))
        self.assertNotIn("---CAPTION---", parsed["post_caption"])

    def test_new_delimiter_strips_slide_labels(self):
        text = NEW_RESPONSE.replace("The quiet hour", "1) The quiet hour", 1)
        parsed = orc._parse_carousel_response(text)
        self.assertEqual(parsed["slides"][0], "The quiet hour before anyone wakes is the real boundary")

    def test_legacy_six_line_still_parses(self):
        parsed = orc._parse_carousel_response(LEGACY_RESPONSE)
        self.assertEqual(len(parsed["slides"]), 5)
        self.assertEqual(parsed["post_caption"], "A short single-line carousel caption.")

    def test_too_few_slides_rejected(self):
        self.assertIsNone(orc._parse_carousel_response("a\nb\nc\n---CAPTION---\ntext"))


class CarouselCaptionTest(unittest.TestCase):
    def _deterministic(self, style):
        # Force both hosted routes to fail so we exercise the guaranteed path.
        with patch.object(orc, "configured", return_value=False), \
             patch.object(bot, "_generate_text_ai_horde", side_effect=RuntimeError("offline")):
            return bot._build_carousel_narrative("Safety", "Why the first hour matters", style=style)

    def test_deterministic_captions_are_coherent_for_both_brands(self):
        tags = {
            "dark": "#TheNineStitches",
            "wilma": "#DigitalGuardian #DigitalParenting #DigitalSafety #ParentingTips",
        }
        for style, tag in tags.items():
            with self.subTest(style=style):
                n = self._deterministic(style)
                cap = n["post_caption"]
                self.assertNotIn("5 slides, no fluff", cap)
                self.assertNotIn("Swipe", cap)
                self.assertGreaterEqual(cap.count("\n\n"), 3)
                self.assertTrue(cap.rstrip().endswith(tag))
                self.assertEqual(cap.count(tag), 1)

    def test_signature_dedupes_existing_hashtags(self):
        cap = "A note.\n\n#OldTag #AnotherTag"
        out = bot._with_carousel_signature(cap, "dark")
        self.assertEqual(out, "A note.\n\n#TheNineStitches")
        self.assertEqual(out.count("#"), 1)

    def test_imagery_hint_passed_to_openrouter(self):
        captured = {}

        def fake(topic, voice="", model=None, timeout=60.0, retries=2, imagery=""):
            captured["imagery"] = imagery
            return {
                "slides": ["one two three", "four five six", "seven eight nine",
                           "ten eleven twelve", "thirteen fourteen fifteen"],
                "post_caption": "A coherent note.\n\nSecond paragraph with a real question?",
            }

        with patch.object(orc, "carousel_narrative", side_effect=fake), \
             patch.object(bot, "_carousel_narrative_ok", return_value=True):
            n = bot._build_carousel_narrative("Safety", "First hour", style="wilma")
        self.assertIn("sunlit", captured["imagery"])
        self.assertTrue(n["post_caption"].endswith(
            "#DigitalGuardian #DigitalParenting #DigitalSafety #ParentingTips"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
