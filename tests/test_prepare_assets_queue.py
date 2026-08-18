import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import scripts.prepare_assets as prepare_assets


class PrepareAssetsQueueTests(unittest.TestCase):
    def test_selects_next_bundle_when_current_one_was_already_posted(self):
        state = {
            "active_bundle": {"post_id": "bundle-a", "captions": {"instagram": "first"}, "platforms_posted": []},
            "content_queue": [
                {"post_id": "bundle-a", "captions": {"instagram": "first"}, "platforms_posted": []},
                {"post_id": "bundle-b", "captions": {"instagram": "second"}, "platforms_posted": []},
            ],
            "platform_posted_bundles": {"instagram": ["bundle-a"]},
        }

        bundle = prepare_assets._select_next_bundle_for_platform(state, "instagram", "state.json")[0]

        self.assertIsNotNone(bundle)
        self.assertEqual(bundle["post_id"], "bundle-b")

    def test_queue_order_preserved_after_removing_first_candidate(self):
        state = {
            "active_bundle": None,
            "content_queue": [
                {"post_id": "q1"},
                {"post_id": "q2"},
                {"post_id": "q3"},
            ],
            "platform_posted_bundles": {},
        }

        _, queue = prepare_assets._select_next_bundle_for_platform(state, "instagram", "state.json")
        self.assertEqual([b["post_id"] for b in queue], ["q2", "q3"])


if __name__ == "__main__":
    unittest.main()
