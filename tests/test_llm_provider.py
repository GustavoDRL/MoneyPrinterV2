import importlib.util
import os
import sys
import unittest
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

    @patch.object(llm_provider, "get_llm_provider", return_value="codex")
    @patch.object(llm_provider, "Codex")
    def test_codex_sdk_returns_final_response(
        self, codex_cls_mock: Mock, _provider_mock: Mock
    ) -> None:
        codex = codex_cls_mock.return_value.__enter__.return_value
        thread = codex.thread_start.return_value
        thread.run.return_value.final_response = " resposta "

        result = llm_provider.generate_text("prompt")

        self.assertEqual(result, "resposta")
        thread.run.assert_called_once_with("prompt")
        thread_start_kwargs = codex.thread_start.call_args.kwargs
        self.assertIsNone(thread_start_kwargs["model"])
        self.assertEqual(thread_start_kwargs["sandbox"], llm_provider.Sandbox.read_only)
        self.assertTrue(thread_start_kwargs["ephemeral"])

    @patch.object(llm_provider, "get_llm_provider", return_value="invalid")
    def test_unknown_provider_fails_clearly(self, _provider_mock: Mock) -> None:
        llm_provider.select_model("model")

        with self.assertRaisesRegex(ValueError, "Unsupported llm_provider"):
            llm_provider.generate_text("prompt")

    @patch.object(llm_provider, "get_llm_provider", return_value="codex")
    @patch.object(llm_provider, "Codex")
    def test_codex_empty_response_fails_clearly(
        self, codex_cls_mock: Mock, _provider_mock: Mock
    ) -> None:
        codex = codex_cls_mock.return_value.__enter__.return_value
        thread = codex.thread_start.return_value
        thread.run.return_value.final_response = ""

        with self.assertRaisesRegex(RuntimeError, "Codex returned an empty"):
            llm_provider.generate_text("prompt")


if __name__ == "__main__":
    unittest.main()
