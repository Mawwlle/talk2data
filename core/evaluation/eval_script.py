import json

import ast
from pathlib import Path
import re

from core.evaluation.inference_script import BENCHMARKS_DIR, load_json

def load_benchmarks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def evaluate_decision(model_decision, expected):
    return model_decision == expected

def evaluate_chat(model_output, expected_facts, forbidden_facts=None):
    score = 1.0
    for fact in expected_facts:
        if fact.lower() not in model_output.lower():
            score = 0.0
    if forbidden_facts:
        for fact in forbidden_facts:
            if fact.lower() in model_output.lower():
                score = 0.0
    return score


# ---------- helpers ----------

def extract_imports(code: str) -> set[str]:
    tree = ast.parse(code)
    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.add(name.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)

    return imports


def extract_calls(code: str) -> set[str]:
    tree = ast.parse(code)
    calls = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.add(node.func.id)

    return calls


def has_side_effect(code: str) -> bool:
    patterns = [
        r"\.fit\s*\(",
        r"\.show\s*\(",
        r"print\s*\("
    ]
    return any(re.search(p, code) for p in patterns)


# ---------- main eval ----------

def evaluate_code(model_code: str, benchmark: str) -> float:
    score = 0.0

    # 1️⃣ Syntax check
    try:
        ast.parse(model_code)
        score += 0.3
    except SyntaxError:
        return 0.0

    expected_code = benchmark

    # 2️⃣ Imports
    expected_imports = extract_imports(expected_code)
    model_imports = extract_imports(model_code)

    if expected_imports:
        matched = expected_imports & model_imports
        score += 0.2 * (len(matched) / len(expected_imports))
    else:
        score += 0.2

    # 3️⃣ Key calls
    expected_calls = extract_calls(expected_code)
    model_calls = extract_calls(model_code)

    if expected_calls:
        matched = expected_calls & model_calls
        score += 0.4 * (len(matched) / len(expected_calls))

    # 4️⃣ Side effect
    if has_side_effect(model_code):
        score += 0.1

    return round(min(score, 1.0), 3)

def run_eval(inference_res_path: str):
    outputs = load_json(Path(inference_res_path))
    paths = sorted(BENCHMARKS_DIR.glob("*.json"))
    benchmarks = []
    for path in paths:
        benchmarks += load_json(path)
        
    results = []
    for case in benchmarks:
        model_output = [output for output in outputs if output.get("benchmark_id") == case.get("id")][0]
        decision_score = evaluate_decision(model_output.get("model_decision", {}).get("action"), case["expected_decision"])
        
        chat_score = 1.0
        code_score = 1.0
        
        if case.get("expected_facts"):
            chat_score = evaluate_chat(model_output.get("response_message",""), case["expected_facts"], case.get("forbidden_facts"))
        
        if case.get("expected_code"):
            code_score = evaluate_code(model_output.get("generated_code",""), case["expected_code"])
        
        results.append({
            "id": case["id"],
            "decision_score": decision_score,
            "chat_score": chat_score,
            "code_score": code_score
        })
    return results

# пример использования
if __name__ == "__main__":
    test_path = "core/evaluation/inference_results/eval_2025-12-15T14:04:22.442639_4971634e.json"
    
    res = run_eval(test_path)
    print(res)
