from core.prompts import CODE_GENERATION_PROMPT
from core.workflow import format_prompt, init_tokenizer_only
from core.tests.tools import TEST_INITIAL_STATES

# Пока сырой тест, в SD-1515 и SD-1516 будет доработано

prompt = "Plot a histogram of sepal_length with 25 bins and add a title."
init_tokenizer_only()
result = format_prompt(CODE_GENERATION_PROMPT, TEST_INITIAL_STATES[0])

print('result: ', result)
