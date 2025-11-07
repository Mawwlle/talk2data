import unittest
import time
from core.tests.tools import TEST_INITIAL_STATES
from core.workflow import generate_code_node, llm_init
import copy


class TestGenerateCode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Инициализируем LLM один раз для всех тестов."""
        print("\nInitializing LLM...")
        llm_init()
        print("LLM initialized\n")

    def setUp(self):
        """Создаём базовое состояние перед каждым тестом."""
        self.sample_state = TEST_INITIAL_STATES[0].copy()

    # ------------------------------------------------------------------
    def test_extracts_code_from_python_block(self):
        """Проверяет, что из ответа LLM извлекается код."""
        result = generate_code_node(self.sample_state)

        generated = result.get("generated_code", "")
        self.assertIsInstance(generated, str)
        self.assertTrue(len(generated.strip()) > 0, "generated_code пуст")

        # Проверяем, что результат похож на код
        self.assertTrue(
            any(kw in generated for kw in ("print", "=", "def ", "import", "for ", "return")),
            f"Сгенерированный текст не похож на код:\n{generated}"
        )

        # Проверяем дополнительные поля
        self.assertIn("response_message", result)
        self.assertIn("generate_code_sec", result["timing_info"])
        self.assertGreaterEqual(result["timing_info"]["generate_code_sec"], 0)
        print("user_input: ", self.sample_state.get("user_input"))
        print("result: ", result.get("generated_code"))
        print()

    # ------------------------------------------------------------------
    def test_no_code_block_fallback_to_full_text(self):
        """Если модель не возвращает код в ```python```, результат всё равно должен содержать код."""
        state = self.sample_state.copy()
        # Можно слегка модифицировать prompt, если LLM ожидает вход
        state["input_data"] = {"user_message": "Напиши простой код без блока ```python```"}

        result = generate_code_node(state)
        generated = result.get("generated_code", "").strip()

        self.assertTrue(generated, "generated_code не должен быть пустым")
        self.assertNotIn("```", generated, "ожидался текст без ```")
        print("user_input: ", state.get("user_input"))
        print("result: ", result.get("generated_code"))
        print()

    # ------------------------------------------------------------------
    def test_preserves_existing_timing_info(self):
        """Проверяем, что предыдущие значения в timing_info не затираются."""
        state = self.sample_state.copy()
        state["timing_info"] = {"prev_step_sec": 1.23}

        result = generate_code_node(state)
        timing = result.get("timing_info", {})

        self.assertIn("prev_step_sec", timing)
        self.assertIn("generate_code_sec", timing)
        self.assertAlmostEqual(timing["prev_step_sec"], 1.23, places=2)

    # ------------------------------------------------------------------
    def test_timing_value_is_reasonable(self):
        """Проверяем, что generate_code_sec реалистичен по времени."""
        start = time.perf_counter()
        result = generate_code_node(self.sample_state)
        elapsed = result["timing_info"]["generate_code_sec"]
        total_elapsed = time.perf_counter() - start

        # Проверяем, что время не отрицательное и меньше общего времени выполнения теста
        self.assertGreaterEqual(elapsed, 0)
        self.assertLessEqual(elapsed, total_elapsed + 1.0)

        print(f"generate_code_sec={elapsed:.3f}s (total {total_elapsed:.3f}s)")
        
    def test_generated_code_for_all_cases(self):
        """
        Проверяем генерацию кода для всех user_input из TEST_INITIAL_STATES.
        Дополнительно запрещаем использование plt и sns.
        """
        for idx, state in enumerate(TEST_INITIAL_STATES):
            with self.subTest(case=idx, user_input=state.get("user_input", "")):
                state_copy = copy.deepcopy(state)
                start = time.perf_counter()
                result = generate_code_node(state_copy)
                elapsed = result["timing_info"]["generate_code_sec"]
                total_elapsed = time.perf_counter() - start

                # ---- Проверяем наличие generated_code ----
                generated = result.get("generated_code", "").strip()
                
                print("user_input: ", result.get("user_input"))
                print("result code: ", generated)
                print()
                
                self.assertTrue(len(generated) > 0, f"Case {idx} produced empty code!")

                # ---- Проверяем, что выглядит как Python код ----
                self.assertTrue(
                    any(kw in generated for kw in ("print", "import", "def", "for", "return")),
                    f"Case {idx} generated text may not be code:\n{generated}"
                )

                # ---- Проверяем отсутствие plt и sns ----
                self.assertNotIn("plt", generated, f"Case {idx} contains forbidden 'plt':\n{generated}")
                self.assertNotIn("sns", generated, f"Case {idx} contains forbidden 'sns':\n{generated}")

                # ---- Проверяем, что response_message присутствует ----
                self.assertIn("response_message", result, f"Case {idx} missing response_message")

                # ---- Проверяем timing_info ----
                self.assertIn("generate_code_sec", result["timing_info"])
                self.assertGreaterEqual(elapsed, 0)
                self.assertLessEqual(elapsed, total_elapsed + 1.0)
                
