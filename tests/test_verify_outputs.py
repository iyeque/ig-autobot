import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_outputs.py"


class VerifyOutputsTests(unittest.TestCase):
    def test_accepts_valid_assets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "output.jpg").write_bytes(b"fake-image")
            Path(tmpdir, "caption.txt").write_text("A valid caption", encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(SCRIPT_PATH)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Outputs verified", result.stdout)

    def test_rejects_missing_caption(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "output.jpg").write_bytes(b"fake-image")

            result = subprocess.run(
                [sys.executable, str(SCRIPT_PATH)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            combined_output = result.stdout + result.stderr
            self.assertIn("Caption not generated", combined_output)


if __name__ == "__main__":
    unittest.main()
