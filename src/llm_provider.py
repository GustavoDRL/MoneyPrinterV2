import ollama
from openai import OpenAI

from config import (
    get_llm_provider,
    get_ollama_base_url,
    get_openai_api_key,
    get_openai_model,
)

_selected_model: str | None = None


def _client() -> ollama.Client:
    return ollama.Client(host=get_ollama_base_url())


def _openai_client() -> OpenAI:
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before using the OpenAI provider."
        )
    return OpenAI(api_key=api_key)


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
    Generates text using the configured OpenAI or local Ollama provider.

    Args:
        prompt (str): User prompt
        model_name (str): Optional model name override

    Returns:
        response (str): Generated text
    """
    provider = get_llm_provider()
    model = model_name or _selected_model

    if provider == "openai" and not model:
        model = get_openai_model()

    if not model:
        raise RuntimeError(
            "No LLM model selected. Call select_model() first or configure a model."
        )

    if provider == "openai":
        response = _openai_client().responses.create(
            model=model,
            input=prompt,
            store=False,
        )
        output_text = response.output_text.strip()
        if not output_text:
            raise RuntimeError("OpenAI returned an empty text response.")
        return output_text

    if provider != "local_ollama":
        raise ValueError(
            f"Unsupported llm_provider '{provider}'. "
            "Expected 'openai' or 'local_ollama'."
        )

    response = _client().chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"].strip()
