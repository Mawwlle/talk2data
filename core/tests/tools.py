import copy
from pathlib import Path

# iris dataset
metadata_default = {
    "data": [
        ["", "sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
        ["unique", "35", "23", "43", "22", "3"],
        ["dtype", "Float64", "Float64", "Float64", "Float64", "String"],
    ]
}

metadata_default = (
    "Columns: sepal_length (float), sepal_width (float), "
    "petal_length (float), petal_width (float), "
    "species (categorical)"
)

INITIAL_STATE = {
    # "user_input": prompt,
    "metadata": metadata_default,
    "conversation_history": [],
    "generated_code": "",
    "response_message": "",
    "response_audio": None,
    "decision": "",
    "timing_info": {},
}


BASE_DIR = Path(__file__).resolve().parent.parent  # это /core
commands_path = BASE_DIR / "benchmarks" / "iris_dataset_benchmark.txt"

with open(commands_path, encoding="utf-8") as lines:
    TEST_COMMANDS = [line.strip() for line in lines if line.strip()]

TEST_INITIAL_STATES = []

for command in TEST_COMMANDS:
    state_copy = copy.deepcopy(INITIAL_STATE)
    state_copy["user_input"] = command
    TEST_INITIAL_STATES.append(state_copy)
