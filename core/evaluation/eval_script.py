import json

import ast
from pathlib import Path
import re
import string
from collections import Counter
from functools import lru_cache

import pandas as pd
import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoTokenizer

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


def tokenize(text: str) -> list[str]:
    translator = str.maketrans("", "", string.punctuation)
    return text.lower().translate(translator).split()


def counter_cosine_similarity(vec_a: Counter[str], vec_b: Counter[str]) -> float:
    if not vec_a or not vec_b:
        return 0.0

    shared_keys = set(vec_a.keys()) | set(vec_b.keys())
    dot_product = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in shared_keys)
    norm_a = sum(v * v for v in vec_a.values()) ** 0.5
    norm_b = sum(v * v for v in vec_b.values()) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


@lru_cache(maxsize=1)
def _load_embedding_components(model_name: str = "./all-MiniLM-L6-v2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _compute_embedding(text: str) -> torch.Tensor:
    tokenizer, model = _load_embedding_components()
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        model_output = model(**encoded)
        token_embeddings = model_output.last_hidden_state
        attention_mask = encoded.attention_mask.unsqueeze(-1)
        masked_embeddings = token_embeddings * attention_mask
        summed = masked_embeddings.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        sentence_embedding = summed / counts

    return sentence_embedding.squeeze(0)


def evaluate_text_similarity(baseline_text: str | None, generated_text: str | None) -> float | None:
    if not baseline_text or not generated_text:
        return None

    baseline_embedding = _compute_embedding(baseline_text)
    generated_embedding = _compute_embedding(generated_text)

    similarity = cosine_similarity(
        baseline_embedding.unsqueeze(0),
        generated_embedding.unsqueeze(0),
    ).item()

    return round(similarity, 3)


def normalize_results(results: list[dict] | dict | None) -> list[dict]:
    if isinstance(results, list):
        return results
    if isinstance(results, dict) and "results" in results:
        return results.get("results", [])
    return []


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

def run_eval(inference_res_path: str, baseline_path: str | None = "core/evaluation/inference_results/eval_0_baseline.json"):
    outputs = normalize_results(load_json(Path(inference_res_path)))

    baseline_outputs = []
    if baseline_path:
        baseline_file = Path(baseline_path)
        if baseline_file.exists():
            baseline_outputs = normalize_results(load_json(baseline_file))

    output_by_id = {item.get("benchmark_id"): item for item in outputs}
    baseline_by_id = {item.get("benchmark_id"): item for item in baseline_outputs}

    paths = sorted(BENCHMARKS_DIR.glob("*.json"))
    benchmarks: list[dict] = []
    for path in paths:
        benchmarks += load_json(path)

    results = []
    for case in benchmarks:
        model_output = output_by_id.get(case.get("id"))
        if not model_output:
            results.append({
                "id": case["id"],
                "error": "missing_model_output",
            })
            continue

        decision_score = evaluate_decision(model_output.get("model_decision", {}).get("action"), case["expected_decision"])

        baseline_response = baseline_by_id.get(case.get("id"), {}).get("response_message") if baseline_by_id else None
        baseline_similarity = evaluate_text_similarity(baseline_response, model_output.get("response_message"))

        chat_score = baseline_similarity if baseline_similarity is not None else 1.0
        code_score = 1.0

        if baseline_similarity is None and case.get("expected_facts"):
            chat_score = evaluate_chat(model_output.get("response_message", ""), case["expected_facts"], case.get("forbidden_facts"))

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
    
    res = run_eval(test_path)
    
    res_df = pd.DataFrame(res)
    res_df.to_csv('core/inference_results/eval_0_metrics.csv')