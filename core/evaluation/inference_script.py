import json
from datetime import datetime
from pathlib import Path

from core.evaluation.constants import RESULT_ID
from core.workflow import create_workflow, llm_init, init_tokenizer_only
from core.config import settings

BENCHMARKS_DIR = Path("core/benchmarks")
RESULTS_DIR = Path("core/evaluation/inference_results")
RESULTS_DIR.mkdir(exist_ok=True)
METADATA = {
    "sepal_length": "float",
    "sepal_width": "float",
    "petal_length": "float",
    "petal_width": "float",
    "species": "category",
}


def load_json(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    return json.loads(text)


def run_single_case(workflow, benchmark: dict) -> dict:
    initial_state = {
        "user_input": benchmark["user_input"],
        "metadata": METADATA,
        "conversation_history": [],
        "generated_code": None,
        "response_message": None,
        "response_audio": None,
        "decision": None,
        "timing_info": {},
    }

    result_state = workflow.invoke(initial_state)

    return {
        "benchmark_id": benchmark["id"],
        "expected_decision": benchmark["expected_decision"],
        "model_decision": result_state.get("decision"),
        "generated_code": result_state.get("generated_code"),
        "response_message": result_state.get("response_message"),
        "timing_info": result_state.get("timing_info", {}),
        "raw_state": result_state,  # можно убрать позже
    }


def main():
    if not settings.REMOTE_LLM:
        llm_init()
    else: 
        init_tokenizer_only()
        
    workflow = create_workflow()
    infer_result_path = f"infer_{RESULT_ID}"

    all_results = {
        "run_id": infer_result_path,
        "timestamp": datetime.utcnow().isoformat(),
        "results": [],
    }

    benchmark_files = sorted(BENCHMARKS_DIR.glob("*.json"))

    for path in benchmark_files:
        benchmarks = load_json(path)

        for benchmark in benchmarks:
            print(f"→ Running {benchmark['id']} ({path.name})")

            try:
                result = run_single_case(workflow, benchmark)
                result["benchmark_file"] = path.name
                all_results["results"].append(result)

            except Exception as e:
                all_results["results"].append(
                    {
                        "benchmark_id": benchmark["id"],
                        "benchmark_file": path.name,
                        "error": str(e),
                    }
                )

    output_path = RESULTS_DIR / f"{infer_result_path}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Eval finished. Results saved to {output_path}")


if __name__ == "__main__":
    main()
