import importlib.util
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

spec = importlib.util.spec_from_file_location(
    "llm_provider_under_test", os.path.join(SRC_DIR, "llm_provider.py")
)
llm_provider = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_provider)


class LlmProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        llm_provider._selected_model = None

    @patch.object(llm_provider, "get_llm_provider", return_value="openai")
    @patch.object(llm_provider, "_openai_client")
    def test_openai_responses_returns_output_text(
        self, client_factory_mock: Mock, _provider_mock: Mock
    ) -> None:
        client = client_factory_mock.return_value
        client.responses.create.return_value = SimpleNamespace(output_text=" resposta ")
        llm_provider.select_model("gpt-test")

        result = llm_provider.generate_text("prompt")

        self.assertEqual(result, "resposta")
        client.responses.create.assert_called_once_with(
            model="gpt-test", input="prompt", store=False
        )

    @patch.object(llm_provider, "get_llm_provider", return_value="invalid")
    def test_unknown_provider_fails_clearly(self, _provider_mock: Mock) -> None:
        llm_provider.select_model("model")

        with self.assertRaisesRegex(ValueError, "Unsupported llm_provider"):
            llm_provider.generate_text("prompt")

    @patch.object(llm_provider, "get_openai_api_key", return_value="")
    def test_openai_client_requires_environment_key(self, _key_mock: Mock) -> None:
        with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY is not set"):
            llm_provider._openai_client()


if __name__ == "__main__":
    unittest.main()
