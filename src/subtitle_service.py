"""Language-aware subtitle segmentation and SRT serialization."""

from dataclasses import dataclass
from typing import Iterable

from language_profile import normalize_locale


@dataclass(frozen=True)
class SubtitleProfile:
    locale: str
    max_chars: int
    max_words: int
    font_size: int


@dataclass(frozen=True)
class Caption:
    start: float
    end: float
    text: str


SUBTITLE_PROFILES = {
    "pt": SubtitleProfile("pt-BR", max_chars=32, max_words=5, font_size=88),
    "en": SubtitleProfile("en-US", max_chars=36, max_words=6, font_size=92),
}
DEFAULT_SUBTITLE_PROFILE = SubtitleProfile(
    "multilingual", max_chars=32, max_words=5, font_size=88
)


def get_subtitle_profile(locale: str) -> SubtitleProfile:
    """Return readable short-form subtitle limits for a canonical locale."""
    language = normalize_locale(locale).split("-", 1)[0]
    return SUBTITLE_PROFILES.get(language, DEFAULT_SUBTITLE_PROFILE)


def _timed_words(segments: Iterable) -> list[tuple[float, float, str]]:
    words = []

    def append_word(start: float, end: float, text: str) -> None:
        if text in {"%", ",", ".", ";", ":", "!", "?"} and words:
            previous = words[-1]
            words[-1] = (previous[0], end, previous[2] + text)
        else:
            words.append((start, end, text))

    for segment in segments:
        segment_words = list(getattr(segment, "words", None) or [])
        if segment_words:
            for word in segment_words:
                text = str(getattr(word, "word", "")).strip()
                if text:
                    append_word(float(word.start), float(word.end), text)
            continue

        tokens = str(getattr(segment, "text", "")).strip().split()
        if not tokens:
            continue
        start = float(segment.start)
        duration = max(0.001, float(segment.end) - start)
        step = duration / len(tokens)
        for index, token in enumerate(tokens):
            append_word(start + index * step, start + (index + 1) * step, token)
    return words


def build_captions(segments: Iterable, locale: str) -> list[Caption]:
    """Group timestamped words into readable, language-specific captions."""
    return build_captions_from_words(_timed_words(segments), locale)


def build_captions_from_words(
    words: Iterable[tuple[float, float, str]], locale: str
) -> list[Caption]:
    """Group provider-independent ``(start, end, text)`` word tuples."""
    profile = get_subtitle_profile(locale)
    captions = []
    sentence = []
    normalized_words = []

    for start, end, text in words:
        text = str(text).strip()
        if not text:
            continue
        if text in {"%", ",", ".", ";", ":", "!", "?"} and normalized_words:
            previous = normalized_words[-1]
            normalized_words[-1] = (previous[0], float(end), previous[2] + text)
        else:
            normalized_words.append((float(start), float(end), text))

    def flush_sentence() -> None:
        if not sentence:
            return

        word_count = len(sentence)
        best: list[tuple[int, list] | None] = [None] * (word_count + 1)
        best[word_count] = (0, [])

        for start_index in range(word_count - 1, -1, -1):
            for end_index in range(start_index + 1, word_count + 1):
                chunk = sentence[start_index:end_index]
                text = " ".join(word[2] for word in chunk)
                if len(chunk) > profile.max_words:
                    break
                if len(text) > profile.max_chars and len(chunk) > 1:
                    break
                tail = best[end_index]
                if tail is None:
                    continue

                orphan_penalty = 5000 if len(chunk) == 1 and word_count > 1 else 0
                ragged_penalty = (profile.max_chars - min(len(text), profile.max_chars)) ** 2
                score = 1000 + orphan_penalty + ragged_penalty + tail[0]
                if best[start_index] is None or score < best[start_index][0]:
                    best[start_index] = (score, [chunk] + tail[1])

        chunks = best[0][1] if best[0] else [sentence[:]]
        for chunk in chunks:
            captions.append(
                Caption(
                    start=chunk[0][0],
                    end=chunk[-1][1],
                    text=" ".join(word[2] for word in chunk),
                )
            )
        sentence.clear()

    for word in normalized_words:
        if sentence and word[0] - sentence[-1][1] > 0.8:
            flush_sentence()
        sentence.append(word)
        if word[2].endswith((".", "!", "?")):
            flush_sentence()

    flush_sentence()
    return captions


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp."""
    total_millis = max(0, int(round(seconds * 1000)))
    hours = total_millis // 3600000
    minutes = (total_millis % 3600000) // 60000
    secs = (total_millis % 60000) // 1000
    millis = total_millis % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def captions_to_srt(captions: Iterable[Caption]) -> str:
    """Serialize captions to valid UTF-8 SRT content."""
    blocks = []
    for index, caption in enumerate(captions, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_timestamp(caption.start)} --> "
                    f"{format_srt_timestamp(caption.end)}",
                    caption.text,
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")
