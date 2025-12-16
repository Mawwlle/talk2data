import json

import ast
from pathlib import Path
import re

from core.evaluation.inference_script import BENCHMARKS_DIR, load_json
from sentence_transformers import SentenceTransformer, util

def load_benchmarks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def evaluate_decision(model_decision, expected):
    return model_decision == expected

def evaluate_chat(model_output, expected_facts, forbidden_facts=None, threshold=0.7):
    """
    Оценка текстового ответа модели с использованием семантического совпадения.
    
    score = (доля семантически найденных expected_facts) - (доля семантически найденных forbidden_facts)
    threshold - минимальное косинусное сходство для учета факта как найденного
    """
    # Получаем embedding для модели
    output_emb = embed_model.encode(model_output, convert_to_tensor=True)
    
    # Проверка expected_facts
    expected_score = 0.0
    if expected_facts:
        found_count = 0
        for fact in expected_facts:
            fact_emb = embed_model.encode(fact, convert_to_tensor=True)
            sim = util.cos_sim(output_emb, fact_emb).item()
            if sim >= threshold:
                found_count += 1
        expected_score = found_count / len(expected_facts)
    else:
        expected_score = 1.0
    
    # Проверка forbidden_facts
    forbidden_score = 0.0
    if forbidden_facts:
        found_count = 0
        for fact in forbidden_facts:
            fact_emb = embed_model.encode(fact, convert_to_tensor=True)
            sim = util.cos_sim(output_emb, fact_emb).item()
            if sim >= threshold:
                found_count += 1
        forbidden_score = found_count / len(forbidden_facts)
    
    # Итоговый скор
    score = expected_score - forbidden_score
    score = max(0.0, min(1.0, score))
    
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
    # Загружаем модель для эмбеддингов
    test_path = "core/evaluation/inference_results/eval_0_baseline.json"
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    res = run_eval(test_path)
    print(res)
