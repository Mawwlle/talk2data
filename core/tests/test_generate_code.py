import copy
import logging
import time

import pytest

from core.config import settings
from core.tests.tools import TEST_INITIAL_STATES
from core.workflow import WorkflowEngine

logger = logging.getLogger(__name__)


class _StubLLM:
    def __init__(
        self,
        response_text: str = """```python
print('hello world')
```""",
    ):
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
        return " ".join(message["content"] for message in messages)


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(settings, "REMOTE_LLM", False)
    return WorkflowEngine(_StubLLM(), _StubTokenizer())


@pytest.fixture
def sample_state():
    return TEST_INITIAL_STATES[0].copy()


def test_extracts_code_from_python_block(engine, sample_state):
    """Проверяет, что из ответа LLM извлекается код."""
    result = engine._generate_code_node(sample_state)

    generated = result.get("generated_code", "")
    assert isinstance(generated, str)
    assert len(generated.strip()) > 0, "generated_code пуст"

    # Проверяем, что результат похож на код
    assert any(
        kw in generated for kw in ("print", "=", "def ", "import", "for ", "return")
    ), f"Сгенерированный текст не похож на код:\n{generated}"

    # Проверяем дополнительные поля
    assert "response_message" in result
    assert "generate_code_sec" in result["timing_info"]
    assert result["timing_info"]["generate_code_sec"] >= 0
    logger.info("user_input: %s", sample_state.get("user_input"))
    logger.info("result: %s", result.get("generated_code"))


def test_no_code_block_fallback_to_full_text(engine, sample_state):
    """Если модель не возвращает код в ```python```, результат всё равно
    должен содержать код."""
    state = sample_state.copy()
    state["input_data"] = {"user_message": "Напиши простой код без блока ```python```"}

    engine._llm = _StubLLM("print('no code block')")
    result = engine._generate_code_node(state)
    generated = result.get("generated_code", "").strip()

    assert generated, "generated_code не должен быть пустым"
    assert "```" not in generated, "ожидался текст без ```"
    logger.info("user_input: %s", state.get("user_input"))
    logger.info("result: %s", result.get("generated_code"))


def test_preserves_existing_timing_info(engine, sample_state):
    """Проверяем, что предыдущие значения в timing_info не затираются."""
    state = sample_state.copy()
    state["timing_info"] = {"prev_step_sec": 1.23}

    result = engine._generate_code_node(state)
    timing = result.get("timing_info", {})

    assert "prev_step_sec" in timing
    assert "generate_code_sec" in timing
    assert timing["prev_step_sec"] == pytest.approx(1.23, rel=1e-2)


def test_timing_value_is_reasonable(engine, sample_state):
    """Проверяем, что generate_code_sec реалистичен по времени."""
    start = time.perf_counter()
    result = engine._generate_code_node(sample_state)
    elapsed = result["timing_info"]["generate_code_sec"]
    total_elapsed = time.perf_counter() - start

    assert elapsed >= 0
    assert elapsed <= total_elapsed + 1.0

    logger.info("generate_code_sec=%.3fs (total %.3fs)", elapsed, total_elapsed)


def test_generated_code_for_all_cases(engine):
    """
    Проверяем генерацию кода для всех user_input из TEST_INITIAL_STATES.
    Дополнительно запрещаем использование plt и sns.
    """
    for idx, state in enumerate(TEST_INITIAL_STATES):
        state_copy = copy.deepcopy(state)
        start = time.perf_counter()
        result = engine._generate_code_node(state_copy)
        elapsed = result["timing_info"]["generate_code_sec"]
        total_elapsed = time.perf_counter() - start

        generated = result.get("generated_code", "").strip()

        logger.info("user_input: %s", result.get("user_input"))
        logger.info("result code: %s", generated)

        assert len(generated) > 0, f"Case {idx} produced empty code!"

        assert any(
            kw in generated for kw in ("print", "import", "def", "for", "return")
        ), f"Case {idx} generated text may not be code:\n{generated}"

        assert "plt" not in generated, f"Case {idx} contains forbidden 'plt':\n{generated}"
        assert "sns" not in generated, f"Case {idx} contains forbidden 'sns':\n{generated}"

        assert "response_message" in result, f"Case {idx} missing response_message"

        assert "generate_code_sec" in result["timing_info"]
        assert elapsed >= 0
        assert elapsed <= total_elapsed + 1.0
