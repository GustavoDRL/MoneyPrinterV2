import base64
import os
import re
import wave

import requests
import soundfile as sf
from kittentts import KittenTTS as KittenModel

from config import (
    ROOT_DIR,
    get_gemini_tts_model,
    get_gemini_tts_voice,
    get_kitten_tts_voice,
    get_nanobanana2_api_base_url,
    get_nanobanana2_api_key,
    get_tts_provider,
)
from language_profile import normalize_locale

KITTEN_MODEL = "KittenML/kitten-tts-mini-0.8"
KITTEN_SAMPLE_RATE = 24000

class TTS:
    def __init__(self) -> None:
        self._kitten_model = None

    def _synthesize_kitten(self, text: str, output_file: str) -> str:
        if self._kitten_model is None:
            self._kitten_model = KittenModel(KITTEN_MODEL)
        audio = self._kitten_model.generate(text, voice=get_kitten_tts_voice())
        sf.write(output_file, audio, KITTEN_SAMPLE_RATE)
        return output_file

    def _synthesize_gemini(
        self, text: str, output_file: str, locale: str
    ) -> str:
        api_key = get_nanobanana2_api_key()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for multilingual TTS.")

        model = get_gemini_tts_model()
        endpoint = (
            f"{get_nanobanana2_api_base_url().rstrip('/')}"
            f"/models/{model}:generateContent"
        )
        response = requests.post(
            endpoint,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json={
                "contents": [
                    {
                        "parts": [
                            {
                                "text": (
                                    f"Speak in locale {locale} with natural pronunciation, "
                                    f"clear pacing, and an engaging narrator tone:\n{text}"
                                )
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": get_gemini_tts_voice()
                            }
                        }
                    },
                },
            },
            timeout=300,
        )
        response.raise_for_status()

        for candidate in response.json().get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                inline_data = part.get("inlineData") or part.get("inline_data") or {}
                encoded_audio = inline_data.get("data")
                mime_type = str(
                    inline_data.get("mimeType") or inline_data.get("mime_type", "")
                )
                if not encoded_audio or not mime_type.startswith("audio/"):
                    continue

                audio_bytes = base64.b64decode(encoded_audio)
                if mime_type.lower().startswith(("audio/wav", "audio/x-wav")):
                    with open(output_file, "wb") as audio_file:
                        audio_file.write(audio_bytes)
                    return output_file

                rate_match = re.search(r"rate=(\d+)", mime_type)
                sample_rate = int(rate_match.group(1)) if rate_match else 24000
                with wave.open(output_file, "wb") as audio_file:
                    audio_file.setnchannels(1)
                    audio_file.setsampwidth(2)
                    audio_file.setframerate(sample_rate)
                    audio_file.writeframes(audio_bytes)
                return output_file

        raise RuntimeError("Gemini TTS did not return an audio payload.")

    def synthesize(
        self,
        text: str,
        output_file: str = os.path.join(ROOT_DIR, ".mp", "audio.wav"),
        language: str = "en-US",
    ) -> str:
        locale = normalize_locale(language)
        provider = get_tts_provider()
        if provider == "auto":
            provider = "kitten" if locale.startswith("en") else "gemini"

        if provider == "kitten":
            if not locale.startswith("en"):
                raise ValueError(
                    f"KittenTTS is English-only and cannot synthesize {locale}. "
                    "Use tts_provider='auto' or 'gemini'."
                )
            return self._synthesize_kitten(text, output_file)
        if provider == "gemini":
            return self._synthesize_gemini(text, output_file, locale)

        raise ValueError(
            f"Unsupported tts_provider '{provider}'. Expected auto, kitten, or gemini."
        )
