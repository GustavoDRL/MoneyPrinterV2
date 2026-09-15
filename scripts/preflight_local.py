#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from typing import Tuple

import requests
from dotenv import load_dotenv


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")
load_dotenv(os.path.join(ROOT_DIR, ".env"))


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def check_url(url: str, timeout: int = 3) -> Tuple[bool, str]:
    try:
        response = requests.get(url, timeout=timeout)
        return True, f"HTTP {response.status_code}"
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    if not os.path.exists(CONFIG_PATH):
        fail(f"Missing config file: {CONFIG_PATH}")
        return 1

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    failures = 0

    stt_provider = str(cfg.get("stt_provider", "local_whisper")).lower()

    ok(f"stt_provider={stt_provider}")

    tts_provider = str(cfg.get("tts_provider", "auto")).strip().lower()
    if tts_provider in {"auto", "kitten", "gemini"}:
        ok(f"tts_provider={tts_provider}")
    else:
        fail(
            f"Unsupported tts_provider '{tts_provider}'. "
            "Expected auto, kitten, or gemini."
        )
        failures += 1

    if tts_provider in {"auto", "kitten"}:
        try:
            import kittentts  # noqa: F401

            ok("KittenTTS is installed for English narration")
        except Exception as exc:
            fail(f"KittenTTS is not importable: {exc}")
            failures += 1

    if tts_provider in {"auto", "gemini"}:
        ok(
            "Gemini TTS handles non-English narration "
            f"with voice={cfg.get('gemini_tts_voice', 'Kore')}"
        )

    imagemagick_path = cfg.get("imagemagick_path", "")
    if imagemagick_path and os.path.exists(imagemagick_path):
        ok(f"imagemagick_path exists: {imagemagick_path}")
    else:
        warn(
            "imagemagick_path is not set to a valid executable path. "
            "MoviePy subtitle rendering may fail."
        )

    firefox_profile = cfg.get("firefox_profile", "")
    if firefox_profile:
        if os.path.isdir(firefox_profile):
            ok(f"firefox_profile exists: {firefox_profile}")
        else:
            warn(f"firefox_profile does not exist: {firefox_profile}")
    else:
        warn("firefox_profile is empty. Twitter/YouTube automation requires this.")

    # Text generation (logged-in Codex or local Ollama)
    llm_provider = str(cfg.get("llm_provider", "local_ollama")).strip().lower()
    ok(f"llm_provider={llm_provider}")

    if llm_provider == "codex":
        model = str(cfg.get("codex_model", "")).strip()
        if model:
            ok(f"codex_model={model}")
        else:
            ok("codex_model uses the logged-in default")

        try:
            import openai_codex  # noqa: F401

            ok("OpenAI Codex Python SDK is installed")
        except Exception as exc:
            fail(f"OpenAI Codex Python SDK is not importable: {exc}")
            failures += 1

        try:
            auth = subprocess.run(
                ["codex", "login", "status"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            auth_detail = "\n".join(
                part.strip() for part in (auth.stdout, auth.stderr) if part.strip()
            )
            if auth.returncode == 0 and "Logged in" in auth_detail:
                login_line = next(
                    line for line in auth_detail.splitlines() if "Logged in" in line
                )
                ok(login_line)
            else:
                fail("Codex is not logged in. Run 'codex login'.")
                failures += 1
        except Exception as exc:
            fail(f"Could not check Codex login: {exc}")
            failures += 1
    elif llm_provider == "local_ollama":
        base = str(cfg.get("ollama_base_url", "http://127.0.0.1:11434")).rstrip("/")
        reachable, detail = check_url(f"{base}/api/tags")
        if not reachable:
            fail(f"Ollama is not reachable at {base}: {detail}")
            failures += 1
        else:
            ok(f"Ollama reachable at {base}")
            try:
                tags = requests.get(f"{base}/api/tags", timeout=5).json()
                models = [m.get("name") for m in tags.get("models", [])]
                if models:
                    ok(f"Ollama models available: {', '.join(models[:10])}")
                else:
                    warn("No models found on Ollama. Pull a model first (e.g. 'ollama pull llama3.2:3b').")
            except Exception as exc:
                warn(f"Could not validate Ollama model list: {exc}")
    else:
        fail(
            f"Unsupported llm_provider '{llm_provider}'. "
            "Expected 'codex' or 'local_ollama'."
        )
        failures += 1

    # Nano Banana 2 (image generation)
    api_key = cfg.get("nanobanana2_api_key", "") or os.environ.get("GEMINI_API_KEY", "")
    nb2_base = str(
        cfg.get(
            "nanobanana2_api_base_url",
            "https://generativelanguage.googleapis.com/v1beta",
        )
    ).rstrip("/")
    if api_key:
        ok("nanobanana2_api_key is set")
    else:
        fail("nanobanana2_api_key is empty (and GEMINI_API_KEY is not set)")
        failures += 1

    reachable, detail = check_url(nb2_base, timeout=8)
    if not reachable:
        warn(f"Nano Banana 2 base URL could not be reached: {detail}")
    else:
        ok(f"Nano Banana 2 base URL reachable: {nb2_base}")

    if stt_provider == "local_whisper":
        try:
            import faster_whisper  # noqa: F401

            ok("faster-whisper is installed")
        except Exception as exc:
            fail(f"faster-whisper is not importable: {exc}")
            failures += 1

    if failures:
        print("")
        print(f"Preflight completed with {failures} blocking issue(s).")
        return 1

    print("")
    print("Preflight passed. Local setup looks ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
