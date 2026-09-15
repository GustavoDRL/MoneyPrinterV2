import os
import sys
import unittest
from types import SimpleNamespace


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from subtitle_service import (
    build_captions,
    build_captions_from_words,
    captions_to_srt,
    get_subtitle_profile,
)


def segment(texts: list[str]) -> SimpleNamespace:
    words = [
        SimpleNamespace(start=index * 0.5, end=(index + 1) * 0.5, word=text)
        for index, text in enumerate(texts)
    ]
    return SimpleNamespace(start=0.0, end=len(words) * 0.5, text=" ".join(texts), words=words)


class SubtitleServiceTests(unittest.TestCase):
    def test_portuguese_captions_keep_words_and_readable_groups(self) -> None:
        captions = build_captions(
            [segment(["Separe", "50", "%", "do", "salário", "para", "contas", "essenciais."])],
            "pt-BR",
        )

        self.assertEqual(" ".join(item.text for item in captions), "Separe 50% do salário para contas essenciais.")
        self.assertTrue(all(len(item.text) <= 32 for item in captions))
        self.assertTrue(all(len(item.text.split()) <= 5 for item in captions))
        self.assertTrue(all(len(item.text.split()) > 1 for item in captions))

    def test_english_uses_its_own_profile(self) -> None:
        profile = get_subtitle_profile("en-US")

        self.assertEqual(profile.locale, "en-US")
        self.assertEqual(profile.max_words, 6)

    def test_provider_independent_words_use_language_profile(self) -> None:
        words = [
            (index * 0.4, (index + 1) * 0.4, word)
            for index, word in enumerate(
                ["Keep", "30%", "for", "personal", "spending."]
            )
        ]
        captions = build_captions_from_words(words, "en-US")

        self.assertEqual(len(captions), 1)
        self.assertEqual(captions[0].text, "Keep 30% for personal spending.")

    def test_srt_serialization_preserves_utf8_and_timing(self) -> None:
        captions = build_captions([segment(["Reserva", "de", "emergência."])], "pt-BR")
        output = captions_to_srt(captions)

        self.assertIn("00:00:00,000 --> 00:00:01,500", output)
        self.assertIn("Reserva de emergência.", output)


if __name__ == "__main__":
    unittest.main()
