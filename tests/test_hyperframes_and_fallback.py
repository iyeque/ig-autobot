import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import bot  # noqa: E402
import generate_hyperframes_reel as hf  # noqa: E402


class FallbackVarietyTest(unittest.TestCase):
    def test_same_pillar_different_ids_get_different_prompts(self):
        post = {"id": 1, "pillar": "systems_psychology", "title": "The Feedback Loop We Live In"}
        prompts = [bot._refresh_fallback_post(post, i)["caption_prompt"] for i in range(1, 9)]
        self.assertEqual(len(set(prompts)), 5)  # five distinct angles

    def test_stale_phrases_removed(self):
        stale_prompt = "Reflect on feedback loops. What does it teach us about productive failure?"
        post = {"id": 1, "pillar": "systems_psychology",
                "title": "Feedback Loops", "caption_prompt": stale_prompt}
        refreshed = bot._refresh_fallback_post(post, 1)
        for field in ("caption_prompt", "image_prompt"):
            for phrase in ("productive failure", "feedback loop", "tree roots", "circuit diagram"):
                self.assertNotIn(phrase, refreshed[field].lower())
        self.assertEqual(post["caption_prompt"], stale_prompt)  # input not mutated

    def test_visuals_rotate_across_ids(self):
        post = {"id": 1, "pillar": "nature_metaphor", "title": "Roots Before Branches"}
        visuals = [bot._refresh_fallback_post(post, i)["image_prompt"] for i in range(1, 13)]
        self.assertGreaterEqual(len(set(visuals)), 3)

    def test_generated_posts_are_pillar_aware(self):
        posts = bot._generate_new_posts()
        self.assertEqual(len(posts), 20)
        self.assertTrue(all("End with one concrete question" in p["caption_prompt"] for p in posts))
        self.assertFalse(any("productive failure" in p["caption_prompt"] for p in posts))


class HyperFramesResolverTest(unittest.TestCase):
    def test_posted_bundle_content_is_found(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            image = td_path / "sample.jpg"
            image.write_bytes(b"\xff\xd8\xff\xd9")
            state = {
                "active_bundle": None,
                "content_queue": [],
                "posted_bundle_content": {
                    "371": {
                        "post_id": 371,
                        "pillar": "micro_philosophy",
                        "topic": "The Myth of the Perfect Moment",
                        "image": str(image),
                        "captions": {"instagram": "A caption long enough to render."},
                    }
                },
            }
            sp = td_path / "state.json"
            sp.write_text(json.dumps(state), encoding="utf-8")
            out = td_path / "composition.html"

            argv = ["prog", "--post_id", "371", "--state_path", str(sp), "--duration_s", "9"]
            with patch.object(sys, "argv", argv), \
                 patch.object(hf, "COMPOSITIONS_DIR", td_path), \
                 patch.object(hf, "generate_composition", return_value=out) as gen:
                hf.main()
            gen.assert_called_once()
            self.assertEqual(str(gen.call_args.kwargs["post_id"]), "371")


if __name__ == "__main__":
    unittest.main(verbosity=2)
