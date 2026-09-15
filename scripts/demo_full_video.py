#!/usr/bin/env python3
"""Run the complete local content pipeline without opening a browser or uploading."""

import json
import os
import shutil
import sys
from datetime import datetime


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from classes.Tts import TTS
from classes.YouTube import YouTube
from config import (
    get_codex_model,
    get_gemini_tts_model,
    get_gemini_tts_voice,
    get_llm_provider,
    get_ollama_model,
    get_tts_provider,
)
from llm_provider import select_model
from subtitle_service import get_subtitle_profile


def main() -> int:
    provider = get_llm_provider()
    model = get_codex_model() if provider == "codex" else get_ollama_model()
    if model:
        select_model(model)

    output_dir = os.path.join(ROOT_DIR, ".mp", "demo_full")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[DEMO] Text provider: {provider} ({model or 'logged-in default'})")
    print("[DEMO] Browser and publishing are disabled.")

    youtube = YouTube(
        account_uuid="local-demo",
        account_nickname="Local Demo",
        fp_profile_path="",
        niche="educação financeira prática para iniciantes",
        language="pt-BR",
        browser_enabled=False,
    )
    video_path = youtube.generate_video(TTS())
    subtitle_profile = get_subtitle_profile(youtube.language)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    final_video_path = os.path.join(output_dir, f"{stamp}.mp4")
    result_path = os.path.join(output_dir, f"{stamp}.json")
    shutil.copy2(video_path, final_video_path)

    result = {
        "language": "pt-BR",
        "voice_profile": {
            "provider": get_tts_provider(),
            "resolved_provider": "gemini",
            "model": get_gemini_tts_model(),
            "voice": get_gemini_tts_voice(),
        },
        "subtitle_profile": {
            "locale": subtitle_profile.locale,
            "max_chars": subtitle_profile.max_chars,
            "max_words": subtitle_profile.max_words,
            "font_size": subtitle_profile.font_size,
        },
        "subject": youtube.subject,
        "script": youtube.script,
        "metadata": youtube.metadata,
        "image_prompts": youtube.image_prompts,
        "images": youtube.images,
        "narration": youtube.tts_path,
        "video": final_video_path,
        "published": False,
    }
    with open(result_path, "w", encoding="utf-8") as result_file:
        json.dump(result, result_file, ensure_ascii=False, indent=2)
        result_file.write("\n")

    print(f"\n[RESULT] Subject: {youtube.subject}")
    print(f"[RESULT] Title: {youtube.metadata['title']}")
    print(f"[RESULT] Video: {final_video_path}")
    print(f"[RESULT] Details: {result_path}")
    print("[RESULT] Published: no")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}")
        raise SystemExit(1)
