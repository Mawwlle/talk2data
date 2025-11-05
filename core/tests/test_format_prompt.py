from core.schemas import AgentState
from core.prompts import DECIDE_ACTION_PROMPT
from string import Template
from tools import TEST_INITIAL_STATES

# Пока сырой тест, в SD-1515 и SD-1516 будет доработано

prompt = "Plot a histogram of sepal_length with 25 bins and add a title."


def format_prompt(messages_template: list, state: AgentState, metadata_fields: dict = {}) -> list[dict[str, str]]:
    """Format chat template using string.Template to avoid conflicts with braces."""
    formatted_messages = []

    mapping = {
        "input": str(state.get("user_input", "")),
        "history": str(state.get("conversation_history", "")),
        "metadata": str(state.get("metadata", {})),
    }
    if metadata_fields:
        mapping.update({k: str(v) for k, v in metadata_fields.items()})

    for msg in messages_template:
        try:
            tmpl = Template(msg["content"])
            content = tmpl.safe_substitute(mapping)
            print('content: ', content)
        except Exception as e:
            print(f"[format_prompt] Template substitution error: {e}")
            # fallback — оставляем оригинал
            content = msg["content"]

        formatted_messages.append({"role": msg["role"], "content": content})

    return formatted_messages


result = format_prompt(DECIDE_ACTION_PROMPT, TEST_INITIAL_STATES[0])

print('result: ', result)
