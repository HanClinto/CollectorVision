import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class QuickstartIntegrationTests(unittest.TestCase):
    def test_quickstart_script_prints_expected_card(self) -> None:
        result = subprocess.run(
            [sys.executable, "examples/quickstart.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        if result.returncode != 0:
            self.fail(
                "examples/quickstart.py failed\n"
                f"exit={result.returncode}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        stdout = result.stdout
        self.assertIn("Detected corner sharpness=", stdout)
        self.assertIn(
            "Top match 7286819f-6c57-4503-898c-528786ad86e9",
            stdout,
        )
        self.assertIn("Name      Scrying Glass", stdout)
        self.assertIn("Set       Urza's Destiny (UDS)", stdout)


if __name__ == "__main__":
    unittest.main()
