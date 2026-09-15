#!/usr/bin/env python3
"""Run a safe text-only LLM demo without publishing or creating media."""

import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import get_codex_model, get_llm_provider, get_ollama_model
from llm_provider import generate_text, select_model


def main() -> int:
    provider = get_llm_provider()
    model = get_codex_model() if provider == "codex" else get_ollama_model()
    if not model and provider != "codex":
        print("[FAIL] No model is configured in config.json.")
        return 1

    if model:
        select_model(model)
    print(f"[INFO] provider={provider} model={model or 'logged-in default'}")
    response = generate_text(
        "Crie uma ideia curta para um YouTube Short sobre educação financeira. "
        "Responda em português brasileiro com exatamente uma frase."
    )
    print(f"[DEMO] {response}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}")
        raise SystemExit(1)
