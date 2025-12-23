import copy
import logging
import time
import unittest

from core.tests.tools import TEST_INITIAL_STATES
from core.workflow import WorkflowEngine

logger = logging.getLogger(__name__)


class _StubLLM:
    def __init__(self, response_text: str = """```python
print('hello world')
```"""):
        self._response_text = response_text

    class _Output:
        def __init__(self, text: str):
            self.outputs = [self._Result(text)]

        class _Result:
            def __init__(self, text: str):
                self.text = text

    def generate(self, prompts, sampling_params):
        return [self._Output(self._response_text)]


class _StubTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join([message["content"] for message in messages])


class TestGenerateCode(unittest.TestCase):
    def setUp(self):
        self.sample_state = TEST_INITIAL_STATES[0].copy()
        self.engine = WorkflowEngine(_StubLLM(), _StubTokenizer())

    # ------------------------------------------------------------------
    def test_extracts_code_from_python_block(self):
        """Проверяет, что из ответа LLM извлекается код."""
        result = self.engine._generate_code_node(self.sample_state)

        generated = result.get("generated_code", "")
        self.assertIsInstance(generated, str)
        self.assertTrue(len(generated.strip()) > 0, "generated_code пуст")

        # Проверяем, что результат похож на код
        self.assertTrue(
            any(
                kw in generated
                for kw in ("print", "=", "def ", "import", "for ", "return")
            ),
            f"Сгенерированный текст не похож на код:\n{generated}",
        )

        # Проверяем дополнительные поля
        self.assertIn("response_message", result)
        self.assertIn("generate_code_sec", result["timing_info"])
        self.assertGreaterEqual(result["timing_info"]["generate_code_sec"], 0)
        logger.info(f"user_input: {self.sample_state.get('user_input')}")
        logger.info(f"result: {result.get('generated_code')}")

    # ------------------------------------------------------------------
    def test_no_code_block_fallback_to_full_text(self):
        """Если модель не возвращает код в ```python```, результат всё равно должен содержать код."""
        state = self.sample_state.copy()
        # Можно слегка модифицировать prompt, если LLM ожидает вход
        state["input_data"] = {
            "user_message": "Напиши простой код без блока ```python```"
        }

        result = self.engine._generate_code_node(state)
        generated = result.get("generated_code", "").strip()

        self.assertTrue(generated, "generated_code не должен быть пустым")
        self.assertNotIn("```", generated, "ожидался текст без ```")
        logger.info(f"user_input: {state.get('user_input')}")
        logger.info(f"result: {result.get('generated_code')}")

    # ------------------------------------------------------------------
    def test_preserves_existing_timing_info(self):
        """Проверяем, что предыдущие значения в timing_info не затираются."""
        state = self.sample_state.copy()
        state["timing_info"] = {"prev_step_sec": 1.23}

        result = self.engine._generate_code_node(state)
        timing = result.get("timing_info", {})

        self.assertIn("prev_step_sec", timing)
        self.assertIn("generate_code_sec", timing)
        self.assertAlmostEqual(timing["prev_step_sec"], 1.23, places=2)

    # ------------------------------------------------------------------
    def test_timing_value_is_reasonable(self):
        """Проверяем, что generate_code_sec реалистичен по времени."""
        start = time.perf_counter()
        result = self.engine._generate_code_node(self.sample_state)
        elapsed = result["timing_info"]["generate_code_sec"]
        total_elapsed = time.perf_counter() - start

        # Проверяем, что время не отрицательное и меньше общего времени выполнения теста
        self.assertGreaterEqual(elapsed, 0)
        self.assertLessEqual(elapsed, total_elapsed + 1.0)

        logger.info(f"generate_code_sec={elapsed:.3f}s (total {total_elapsed:.3f}s)")

    def test_generated_code_for_all_cases(self):
        """
        Проверяем генерацию кода для всех user_input из TEST_INITIAL_STATES.
        Дополнительно запрещаем использование plt и sns.
        """
        for idx, state in enumerate(TEST_INITIAL_STATES):
            with self.subTest(case=idx, user_input=state.get("user_input", "")):
                state_copy = copy.deepcopy(state)
                start = time.perf_counter()
                result = self.engine._generate_code_node(state_copy)
                elapsed = result["timing_info"]["generate_code_sec"]
                total_elapsed = time.perf_counter() - start

                # ---- Проверяем наличие generated_code ----
                generated = result.get("generated_code", "").strip()

                logger.info(f"user_input: {result.get('user_input')}")
                logger.info(f"result code: {generated}")

                self.assertTrue(len(generated) > 0, f"Case {idx} produced empty code!")

                # ---- Проверяем, что выглядит как Python код ----
                self.assertTrue(
                    any(
                        kw in generated
                        for kw in ("print", "import", "def", "for", "return")
                    ),
                    f"Case {idx} generated text may not be code:\n{generated}",
                )

                # ---- Проверяем отсутствие plt и sns ----
                self.assertNotIn(
                    "plt",
                    generated,
                    f"Case {idx} contains forbidden 'plt':\n{generated}",
                )
                self.assertNotIn(
                    "sns",
                    generated,
                    f"Case {idx} contains forbidden 'sns':\n{generated}",
                )

                # ---- Проверяем, что response_message присутствует ----
                self.assertIn(
                    "response_message", result, f"Case {idx} missing response_message"
                )

                # ---- Проверяем timing_info ----
                self.assertIn("generate_code_sec", result["timing_info"])
                self.assertGreaterEqual(elapsed, 0)
                self.assertLessEqual(elapsed, total_elapsed + 1.0)
