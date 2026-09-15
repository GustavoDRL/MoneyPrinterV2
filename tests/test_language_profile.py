import os
import sys
import unittest


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from language_profile import language_code, normalize_locale


class LanguageProfileTests(unittest.TestCase):
    def test_normalizes_portuguese_names(self) -> None:
        self.assertEqual(normalize_locale("Português brasileiro"), "pt-BR")
        self.assertEqual(normalize_locale("pt_BR"), "pt-BR")

    def test_normalizes_english_names(self) -> None:
        self.assertEqual(normalize_locale("English"), "en-US")

    def test_returns_whisper_language_code(self) -> None:
        self.assertEqual(language_code("pt-BR"), "pt")

    def test_rejects_ambiguous_free_form_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "Use BCP-47"):
            normalize_locale("my preferred language")


if __name__ == "__main__":
    unittest.main()
