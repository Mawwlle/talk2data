from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_MODEL = (
    "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/"
    "snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
)
RESULT_ID = "infer_2_added_planning_code_stage"
TEST_RESULT_PATH = f"core/evaluation/inference_results/{RESULT_ID}.json"
REPORT_OUTPUT_DIR = "core/evaluation/report_2_added_planning_code_stage"
RANDOM_SEED = 0
SANDBOX_FILENAME = "<sandbox>"
SANDBOX_TIMEOUT_SECONDS = 20
SAFE_BUILTINS = [
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "map",
    "max",
    "min",
    "pow",
    "range",
    "repr",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "zip",
    "__import__",
]

SandboxResult = tuple[Any | None, str | None]
