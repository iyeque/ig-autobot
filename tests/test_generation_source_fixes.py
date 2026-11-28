import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import bot  # noqa: E402
import agent_orchestrator as ao  # noqa: E402


class PromptSanitizingTest(unittest.TestCase):
    def test_strips_think_reasoning(self):
        raw = "<think>\nThinking Process:\n1. analyze request\n</think>A weathered stone path at dawn."
        self.assertEqual(bot.sanitize_image_prompt(raw), "A weathered stone path at dawn.")

    def test_caps_at_cloudflare_limit(self):
        self.assertEqual(len(bot.sanitize_image_prompt("x" * 5000)), 2048)

    def test_plain_prompt_unchanged(self):
        self.assertEqual(bot.sanitize_image_prompt("  a quiet room  "), "a quiet room")


class ReflectionGateTest(unittest.TestCase):
    def test_short_llm_answer_is_expanded_past_gate(self):
        with patch.object(ao, "llm_call", return_value="Too short."):
            reflection = ao.generate_master_reflection(
                {"title": "A Quiet Choice", "topic": "small decisions"},
                {"persona": "writer", "name": "The Nine Stitches", "book_title": "The Nine Stitches"},
            )
        self.assertGreaterEqual(len(reflection), 100)
        self.assertNotIn("<think>", reflection.lower())
        self.assertIn("A Quiet Choice", reflection)

    def test_empty_llm_uses_prose_fallback(self):
        with patch.object(ao, "llm_call", return_value=""):
            reflection = ao.generate_master_reflection(
                {"title": "Winter Light", "topic": "fallow seasons"},
                {"persona": "writer", "name": "The Nine Stitches", "book_title": "The Nine Stitches"},
            )
        self.assertGreaterEqual(len(reflection), 100)
        self.assertTrue(reflection.startswith("I keep returning to"))

    def test_clean_llm_answer_preserved(self):
        good = (
            "I used to think more was the answer — more effort, more polish, "
            "more proof that I had earned the right to be heard. But the quiet "
            "choice keeps winning, the one I make without an audience, and it "
            "outlasts every performance I could have staged."
        )
        with patch.object(ao, "llm_call", return_value=f"<think>reasoning</think>{good}"):
            reflection = ao.generate_master_reflection(
                {"title": "T", "topic": "A"},
                {"persona": "writer", "name": "N", "book_title": "B"},
            )
        self.assertEqual(reflection, good)


if __name__ == "__main__":
    unittest.main(verbosity=2)
