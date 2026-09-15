import importlib.util
import os
import sys
import unittest
from unittest.mock import patch


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

spec = importlib.util.spec_from_file_location(
    "tts_routing_under_test", os.path.join(SRC_DIR, "classes", "Tts.py")
)
tts_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tts_module)


class TtsRoutingTests(unittest.TestCase):
    @patch.object(tts_module, "get_tts_provider", return_value="auto")
    @patch.object(tts_module.TTS, "_synthesize_gemini", return_value="voice.wav")
    def test_auto_routes_portuguese_to_gemini(
        self, gemini_mock, _provider_mock
    ) -> None:
        result = tts_module.TTS().synthesize(
            "Olá", "voice.wav", language="Português brasileiro"
        )

        self.assertEqual(result, "voice.wav")
        gemini_mock.assert_called_once_with("Olá", "voice.wav", "pt-BR")

    @patch.object(tts_module, "get_tts_provider", return_value="kitten")
    def test_kitten_rejects_portuguese(self, _provider_mock) -> None:
        with self.assertRaisesRegex(ValueError, "English-only"):
            tts_module.TTS().synthesize("Olá", "voice.wav", language="pt-BR")


if __name__ == "__main__":
    unittest.main()
