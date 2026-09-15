#!/usr/bin/env python3
"""Generate one short-form content concept and one image without publishing."""

import base64
import json
import os
import sys
from datetime import datetime

import requests


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import (
    get_codex_model,
    get_llm_provider,
    get_nanobanana2_api_base_url,
    get_nanobanana2_api_key,
    get_nanobanana2_aspect_ratio,
    get_nanobanana2_model,
    get_ollama_model,
)
from llm_provider import generate_text, select_model


def _parse_concept(raw_response: str) -> dict:
    cleaned = raw_response.strip().removeprefix("```json").removeprefix("```")
    cleaned = cleaned.removesuffix("```").strip()
    concept = json.loads(cleaned)
    required = ("title", "script", "image_prompt")
    if not all(isinstance(concept.get(key), str) and concept[key].strip() for key in required):
        raise ValueError(f"Codex response must contain: {', '.join(required)}")
    return {key: concept[key].strip() for key in required}


def _generate_image(prompt: str) -> tuple[bytes, str]:
    api_key = get_nanobanana2_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    endpoint = (
        f"{get_nanobanana2_api_base_url().rstrip('/')}"
        f"/models/{get_nanobanana2_model()}:generateContent"
    )
    response = requests.post(
        endpoint,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"aspectRatio": get_nanobanana2_aspect_ratio()},
            },
        },
        timeout=300,
    )
    response.raise_for_status()

    for candidate in response.json().get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            inline_data = part.get("inlineData") or part.get("inline_data") or {}
            mime_type = inline_data.get("mimeType") or inline_data.get("mime_type", "")
            if inline_data.get("data") and str(mime_type).startswith("image/"):
                extension = {
                    "image/jpeg": ".jpg",
                    "image/png": ".png",
                    "image/webp": ".webp",
                }.get(str(mime_type).lower(), ".img")
                return base64.b64decode(inline_data["data"]), extension

    raise RuntimeError("Gemini did not return an image payload.")


def main() -> int:
    provider = get_llm_provider()
    model = get_codex_model() if provider == "codex" else get_ollama_model()
    if model:
        select_model(model)

    print(f"[1/2] Generating concept with {provider}...")
    concept = _parse_concept(
        generate_text(
            "Crie um conceito simples para um YouTube Short de 15 segundos sobre "
            "como economizar R$ 5 por dia. Responda em português brasileiro somente "
            "como JSON válido, sem markdown, com exatamente três strings: title, "
            "script e image_prompt. O image_prompt deve descrever uma ilustração "
            "vertical atraente, sem palavras nem logotipos."
        )
    )

    print("[2/2] Generating one vertical image with Gemini...")
    image_bytes, image_extension = _generate_image(concept["image_prompt"])

    output_dir = os.path.join(ROOT_DIR, ".mp", "demo")
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    image_path = os.path.join(output_dir, f"{stamp}{image_extension}")
    result_path = os.path.join(output_dir, f"{stamp}.json")

    with open(image_path, "wb") as image_file:
        image_file.write(image_bytes)
    with open(result_path, "w", encoding="utf-8") as result_file:
        json.dump(concept, result_file, ensure_ascii=False, indent=2)
        result_file.write("\n")

    print(f"\nTitle: {concept['title']}")
    print(f"Script: {concept['script']}")
    print(f"Image: {image_path}")
    print(f"Concept: {result_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}")
        raise SystemExit(1)
