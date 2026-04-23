import contextlib
import io
import json
import runpy
import unittest
import urllib.request
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class QuickstartIntegrationTests(unittest.TestCase):
    def test_quickstart_script_prints_expected_card(self) -> None:
        real_urlopen = urllib.request.urlopen

        def fake_urlopen(request, *args, **kwargs):
            url = request.full_url if hasattr(request, "full_url") else str(request)
            # Keep the example itself untouched: run the real script, but stub the
            # final Scryfall lookup so this integration test stays deterministic.
            if url == "https://api.scryfall.com/cards/7286819f-6c57-4503-898c-528786ad86e9":
                return _FakeResponse(
                    {
                        "name": "Scrying Glass",
                        "set_name": "Urza's Destiny",
                        "set": "uds",
                        "prices": {"usd": "0.40"},
                    }
                )
            return real_urlopen(request, *args, **kwargs)

        output = io.StringIO()
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with contextlib.redirect_stdout(output):
                runpy.run_path(
                    str(ROOT / "examples" / "quickstart.py"),
                    run_name="__main__",
                )

        stdout = output.getvalue()
        self.assertIn("Detected corner sharpness=", stdout)
        self.assertIn(
            "Top match 7286819f-6c57-4503-898c-528786ad86e9",
            stdout,
        )
        self.assertIn("Name      Scrying Glass", stdout)
        self.assertIn("Set       Urza's Destiny (UDS)", stdout)


if __name__ == "__main__":
    unittest.main()
