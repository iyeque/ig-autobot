import sys
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import bot  # noqa: E402
from scripts import brand_guardian as bg  # noqa: E402
import openrouter_captions as orc  # noqa: E402


BROKEN_LINKEDIN = (
    "Writing comes in fits and starts. Perfection is a mirage that continues to "
    "tantalize us every single day.\n\n"
    "#TheNineStitches\n\n### CTA:\n"
    "So tell me, have you ever finished things badly?\n\n#LinkedIn\n"
    "Let's get this thread rolling! What are your thoughts on the allure of "
    "perfection vs. the power of imperfection in"
)

GOOD_LINKEDIN = (
    "The first hour of the day is rarely about productivity. It is about which "
    "part of you gets to speak first.\n\n"
    "I stopped trying to optimise it and started protecting it. The difference "
    "shows up everywhere else.\n\n"
    "Which part of your morning would you protect first?\n\n#TheNineStitches"
)


class LinkedInTruncationTest(unittest.TestCase):
    def test_linkedin_hard_limit_matches_brand_brief(self):
        self.assertEqual(bot._PLATFORM_CHAR_LIMITS["linkedin"], 2000)
        self.assertEqual(bg.PLATFORM_LIMITS["linkedin"], 2000)

    def test_bundle370_style_caption_is_flagged(self):
        self.assertTrue(bg.is_cropped_or_truncated(BROKEN_LINKEDIN, "linkedin"))

    def test_healthy_caption_not_flagged(self):
        self.assertFalse(bg.is_cropped_or_truncated(GOOD_LINKEDIN, "linkedin"))

    def test_hard_cap_ellipsis_near_limit_is_flagged(self):
        capped = ("A thought. " * 170).strip() + "..."
        self.assertTrue(bg.is_cropped_or_truncated(capped, "linkedin"))

    def test_scaffolding_stripped_by_cleaner(self):
        cleaned = bot.clean_caption_formatting(BROKEN_LINKEDIN)
        self.assertNotIn("#LinkedIn", cleaned)
        self.assertNotIn("### CTA", cleaned)


class PillarLensTest(unittest.TestCase):
    def test_all_main_pillars_have_lenses(self):
        for pillar in ("micro_philosophy", "nature_metaphor",
                       "systems_psychology", "author_voice", "quote"):
            self.assertIn(pillar, bot.PILLAR_CAPTION_LENSES)
            self.assertGreater(len(bot.PILLAR_CAPTION_LENSES[pillar]), 40)

    def test_lens_reaches_openrouter_system_prompt(self):
        seen = {}

        def fake_post(url, headers=None, json=None, timeout=None, **kwargs):
            seen["system"] = json["messages"][0]["content"]
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.json.return_value = {
                "choices": [{"message": {"content": "A tailored caption."},
                             "finish_reason": "stop"}]
            }
            return resp

        lens = bot.PILLAR_CAPTION_LENSES["systems_psychology"]
        with patch.object(orc.requests, "post", side_effect=fake_post), \
             patch.object(orc, "configured", return_value=True):
            result = orc.tailor("A master reflection about feedback loops.", "linkedin", lens=lens)
        self.assertEqual(result, "A tailored caption.")
        self.assertIn("Name the loop, incentive or feedback pattern", seen["system"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
