import unittest
import time
from unittest.mock import patch, MagicMock
from core.tests.tools import TEST_INITIAL_STATES
from core.workflow import generate_code_node


class TestGenerateCodeNode(unittest.TestCase):
    def setUp(self):
        """Создаём базовое состояние перед каждым тестом."""
        self.sample_state = TEST_INITIAL_STATES[0]

    def _mock_llm_generate(self, text="```python\nprint('hello')\n```"):
        """Формирует поддельный ответ llm.generate()."""
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text=text)]
        mock_llm = MagicMock()
        mock_llm.generate.return_value = [mock_output]
        return mock_llm

    @patch("workflow.tokenizer")
    @patch("workflow.llm")
    def test_extracts_code_from_python_block(self, mock_llm, mock_tokenizer):
        """Проверяет извлечение кода из блока ```python ...```"""
        mock_llm.return_value = self._mock_llm_generate()
        mock_llm.generate = self._mock_llm_generate().generate
        mock_tokenizer.apply_chat_template.side_effect = lambda messages, **_: str(messages)

        result = generate_code_node(self.sample_state.copy())

        self.assertEqual(result["generated_code"], "print('hello')")
        self.assertIn("response_message", result)
        self.assertTrue(result["response_message"].startswith("Here's the generated code")) # type: ignore
        self.assertIn("generate_code_sec", result["timing_info"])
        self.assertGreaterEqual(result["timing_info"]["generate_code_sec"], 0)

    @patch("workflow.tokenizer")
    @patch("workflow.llm")
    def test_no_code_block_fallback_to_full_text(self, mock_llm, mock_tokenizer):
        """Если нет ```python, берётся весь текст."""
        mock_llm.return_value = self._mock_llm_generate("print('fallback')")
        mock_llm.generate = self._mock_llm_generate("print('fallback')").generate
        mock_tokenizer.apply_chat_template.side_effect = lambda messages, **_: str(messages)

        result = generate_code_node(self.sample_state.copy())

        self.assertEqual(result["generated_code"], "print('fallback')")

    @patch("workflow.tokenizer")
    @patch("workflow.llm")
    def test_preserves_existing_timing_info(self, mock_llm, mock_tokenizer):
        """Проверяем, что предыдущие значения в timing_info не затираются."""
        state = self.sample_state.copy()
        state["timing_info"] = {"prev_step_sec": 1.23}

        mock_llm.return_value = self._mock_llm_generate()
        mock_llm.generate = self._mock_llm_generate().generate
        mock_tokenizer.apply_chat_template.side_effect = lambda messages, **_: str(messages)

        result = generate_code_node(state)

        self.assertIn("prev_step_sec", result["timing_info"])
        self.assertIn("generate_code_sec", result["timing_info"])
        self.assertEqual(result["timing_info"]["prev_step_sec"], 1.23)

    @patch("workflow.tokenizer")
    @patch("workflow.llm")
    def test_llm_generate_called_once(self, mock_llm, mock_tokenizer):
        """Проверяет, что LLM вызывается ровно один раз с правильными аргументами."""
        mock_instance = self._mock_llm_generate()
        mock_llm.generate = mock_instance.generate
        mock_tokenizer.apply_chat_template.side_effect = lambda messages, **_: str(messages)

        generate_code_node(self.sample_state.copy())

        mock_llm.generate.assert_called_once()
        args, kwargs = mock_llm.generate.call_args
        self.assertIsInstance(args[0], list)
        # SamplingParams передаётся вторым аргументом
        self.assertTrue(len(args) >= 2)

    @patch("workflow.tokenizer")
    @patch("workflow.llm")
    def test_timing_value_is_reasonable(self, mock_llm, mock_tokenizer):
        """Проверяем, что generate_code_sec записан и реалистичен."""
        mock_llm.return_value = self._mock_llm_generate()
        mock_llm.generate = self._mock_llm_generate().generate
        mock_tokenizer.apply_chat_template.side_effect = lambda messages, **_: str(messages)

        start = time.perf_counter()
        result = generate_code_node(self.sample_state.copy())
        elapsed = result["timing_info"]["generate_code_sec"]
        total_elapsed = time.perf_counter() - start

        self.assertGreaterEqual(elapsed, 0)
        self.assertLess(elapsed, total_elapsed + 0.5)

