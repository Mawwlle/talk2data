
from pathlib import Path

# iris dataset
metadata_default = {'data': [
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
                ]}

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
commands_path = BASE_DIR / "evaluation_benchmark" / "iris_dataset_benchmark.txt"

with open(commands_path, encoding="utf-8") as lines:
    TEST_COMMANDS = [line.strip() for line in lines if line.strip()]
    
TEST_INITIAL_STATES = []

for command in TEST_COMMANDS:
    INITIAL_STATE["user_input"] = command
    TEST_INITIAL_STATES.append(INITIAL_STATE)