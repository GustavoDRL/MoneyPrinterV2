import ollama
from openai_codex import Codex, CodexConfig, Sandbox

from config import (
    ROOT_DIR,
    get_codex_model,
    get_llm_provider,
    get_ollama_base_url,
)

_selected_model: str | None = None


def _client() -> ollama.Client:
    return ollama.Client(host=get_ollama_base_url())


def list_models() -> list[str]:
    """
    Lists all models available on the local Ollama server.

    Returns:
        models (list[str]): Sorted list of model names.
    """
    response = _client().list()
    return sorted(m.model for m in response.models)


def select_model(model: str) -> None:
    """
    Sets the model to use for all subsequent generate_text calls.

    Args:
        model (str): Model name for the configured provider.
    """
    global _selected_model
    _selected_model = model


def get_active_model() -> str | None:
    """
    Returns the currently selected model, or None if none has been selected.
    """
    return _selected_model


def generate_text(prompt: str, model_name: str = None) -> str:
    """
    Generates text using the logged-in Codex SDK or local Ollama provider.

    Args:
        prompt (str): User prompt
        model_name (str): Optional model name override

    Returns:
        response (str): Generated text
    """
    provider = get_llm_provider()
    model = model_name or _selected_model

    if provider == "codex" and not model:
        model = get_codex_model() or None

    if provider == "codex":
        with Codex(CodexConfig(cwd=ROOT_DIR)) as codex:
            thread = codex.thread_start(
                model=model,
                sandbox=Sandbox.read_only,
                ephemeral=True,
                developer_instructions=(
                    "You are a text-generation component. Do not inspect files, "
                    "run commands, or use tools. Return only the requested final text."
                ),
            )
            result = thread.run(prompt)

        output_text = str(result.final_response or "").strip()
        if not output_text:
            raise RuntimeError("Codex returned an empty text response.")
        return output_text

    if not model:
        raise RuntimeError(
            "No LLM model selected. Call select_model() first or configure a model."
        )

    if provider != "local_ollama":
        raise ValueError(
            f"Unsupported llm_provider '{provider}'. "
            "Expected 'codex' or 'local_ollama'."
        )

    response = _client().chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"].strip()
