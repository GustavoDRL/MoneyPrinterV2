"""Canonical language handling shared by text, speech, and transcription."""

import re
import unicodedata


LANGUAGE_ALIASES = {
    "english": "en-US",
    "english us": "en-US",
    "ingles": "en-US",
    "ingles americano": "en-US",
    "portuguese": "pt-BR",
    "portuguese brazil": "pt-BR",
    "portugues": "pt-BR",
    "portugues brasileiro": "pt-BR",
    "brazilian portuguese": "pt-BR",
}


def normalize_locale(language: str) -> str:
    """Convert common language names and BCP-47-like values to one locale."""
    value = str(language or "").strip()
    if not value:
        raise ValueError("Language is empty. Use a locale such as 'pt-BR' or 'en-US'.")

    alias = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    alias = re.sub(r"[_-]+", " ", alias.lower()).strip()
    if alias in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[alias]

    locale_match = re.fullmatch(r"([A-Za-z]{2,3})(?:[-_]([A-Za-z]{2}))?", value)
    if not locale_match:
        raise ValueError(
            f"Unsupported language format '{language}'. Use BCP-47, for example "
            "'pt-BR' or 'en-US'."
        )

    language_code, region = locale_match.groups()
    return language_code.lower() + (f"-{region.upper()}" if region else "")


def language_code(locale: str) -> str:
    """Return the base language code used by speech recognition."""
    return normalize_locale(locale).split("-", 1)[0]
