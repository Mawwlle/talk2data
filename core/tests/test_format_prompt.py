from schemas import AgentState
from prompts import DECIDE_ACTION_PROMPT
from string import Template

# Пока сырой тест, в SD-1515 и SD-1516 будет доработано

prompt = "Plot a histogram of sepal_length with 25 bins and add a title."

initial_state = {
            "user_input": prompt,
            "metadata": {'data': [
                ['', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'], 
                ['count', '150.0', '150.0', '150.0', '150.0', '150'], 
                ['null_count', '0.0', '0.0', '0.0', '0.0', '0'], 
                ['mean', '5.843', '3.054', '3.759', '1.199', 'None'], 
                ['std', '0.828', '0.434', '1.764', '0.763', 'None'], 
                ['min', '4.3', '2.0', '1.0', '0.1', 'setosa'], 
                ['25%', '5.1', '2.8', '1.6', '0.3', 'None'], 
                ['median', '5.8', '3.0', '4.4', '1.3', 'None'], 
                ['75%', '6.4', '3.3', '5.1', '1.8', 'None'], 
                ['max', '7.9', '4.4', '6.9', '2.5', 'virginica'], 
                ['unique', '35', '23', '43', '22', '3'], 
                ['nan_count', '0', '0', '0', '0', '0'], 
                ['dtype', 'Float64', 'Float64', 'Float64', 'Float64', 'String'], 
                ['mode', '5.0', '3.0', '1.5', '0.2', 'nan']
                ]},
            "conversation_history": [],
            "generated_code": None,
            "response_message": None,
            "response_audio": None,
            "decision": None,
            "timing_info": {},
        }

def format_prompt(messages_template: list, state: AgentState, metadata_fields: dict = None) -> str:
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


result = format_prompt(DECIDE_ACTION_PROMPT, initial_state)

print('result: ', result)
