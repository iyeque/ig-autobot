import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shared_utils
import scripts.prepare_assets as prepare_assets


class BundleReservationTests(unittest.TestCase):
    def test_detects_same_platform_consumption(self):
        state = {
            "platform_posted_bundles": {
                "instagram": ["bundle-a"],
            }
        }
        bundle = {"post_id": "bundle-a"}

        self.assertTrue(shared_utils.is_bundle_consumed_for_platform(bundle, "instagram", "state.json", state))

    def test_prepare_assets_skips_already_consumed_bundle_for_same_platform(self):
        state = {
            "active_bundle": None,
            "content_queue": [
                {"post_id": "bundle-a", "captions": {"instagram": "first"}, "platforms_posted": []},
                {"post_id": "bundle-b", "captions": {"instagram": "second"}, "platforms_posted": []},
            ],
            "platform_posted_bundles": {
                "instagram": ["bundle-a"],
            },
        }

        bundle, queue = prepare_assets._select_next_bundle_for_platform(state, "threads", "state.json")

        self.assertIsNotNone(bundle)
        self.assertEqual(bundle["post_id"], "bundle-a")
        self.assertEqual(len(queue), 1)


if __name__ == "__main__":
    unittest.main()
