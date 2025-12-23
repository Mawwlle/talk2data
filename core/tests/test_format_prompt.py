from core.prompts import CODE_GENERATION_PROMPT
from core.tests.tools import TEST_INITIAL_STATES
from core.workflow import WorkflowEngine


class _DummyLLM:
    def generate(self, prompts, sampling_params):  # pragma: no cover - not used here
        return []


class _DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join([message["content"] for message in messages])


engine = WorkflowEngine(_DummyLLM(), _DummyTokenizer())
result = engine._format_prompt(CODE_GENERATION_PROMPT, TEST_INITIAL_STATES[0])

print("result: ", result)
