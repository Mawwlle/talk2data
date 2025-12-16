import json

import ast
from pathlib import Path
import re
import string
from collections import Counter
from functools import lru_cache
import io
import contextlib
import math
import builtins
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from sklearn.datasets import load_iris

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
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        if isinstance(func, ast.Attribute):
            calls.add(func.attr)
        elif isinstance(func, ast.Name):
            calls.add(func.id)

    return calls

def _normalize_stdout(text: str) -> str:
    if not text:
        return ""
    lines = [line.rstrip() for line in text.strip().splitlines()]
    return "\n".join(lines)


def _compare_results(expected, actual) -> bool:
    if isinstance(expected, (float, int)) and isinstance(actual, (float, int)):
        return math.isclose(float(expected), float(actual), rel_tol=1e-6, abs_tol=1e-6)

    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip() == actual.strip()

    if isinstance(expected, pd.DataFrame) and isinstance(actual, pd.DataFrame):
        return expected.equals(actual)

    if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
        return np.allclose(expected, actual)

    return expected == actual


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


def _sandbox_globals() -> dict:
    dummy_builtins = {
        "__builtins__": {
            name: getattr(builtins, name)
            for name in [
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
        }
    }

    np.random.seed(0)

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(lambda idx: iris.target_names[idx])
    df = df.drop(columns=["target"]).rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
        }
    )

    return {
        **dummy_builtins,
        "np": np,
        "pd": pd,
        "Path": Path,
        "df": df,
        "math": math,
    }


def _run_code_in_sandbox(code: str) -> tuple[str, object, str | None]:
    sandbox_globals = _sandbox_globals()
    sandbox_locals: dict[str, object] = {}

    try:
        parsed = ast.parse(code)
    except SyntaxError as exc:
        return "", None, f"syntax_error: {exc}"

    stdout_buffer = io.StringIO()
    exec_error: str | None = None
    result_value: object = None

    try:
        with contextlib.redirect_stdout(stdout_buffer):
            if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                body_without_last = ast.Module(body=parsed.body[:-1], type_ignores=[])
                last_expr = ast.Expression(parsed.body[-1].value)

                if body_without_last.body:
                    exec(compile(body_without_last, "<sandbox>", "exec"), sandbox_globals, sandbox_locals)

                result_value = eval(compile(last_expr, "<sandbox>", "eval"), sandbox_globals, sandbox_locals)
            else:
                exec(compile(parsed, "<sandbox>", "exec"), sandbox_globals, sandbox_locals)
    except Exception as exc:  # noqa: BLE001
        exec_error = f"execution_error: {exc}"

    stdout_text = _normalize_stdout(stdout_buffer.getvalue())
    return stdout_text, result_value, exec_error


@lru_cache(maxsize=1)
def _load_embedding_components(
    model_path: str = "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
):
    model_path = str(Path(model_path).expanduser())

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
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


def _mean(values: list[float]) -> float:
    numeric = [v for v in values if isinstance(v, (int, float))]
    if not numeric:
        return 0.0
    return round(float(np.mean(numeric)), 3)


def _collect_metadata(benchmarks: list[dict]) -> dict[str, dict]:
    return {case.get("id"): case for case in benchmarks}


def _summarize_by_group(results: list[dict], benchmarks_by_id: dict[str, dict], key: str) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for row in results:
        case_meta = benchmarks_by_id.get(row["id"], {})
        group_value = None
        if key == "tags":
            group_value = case_meta.get("metadata", {}).get("tags", []) or ["untagged"]
        else:
            group_value = [case_meta.get("metadata", {}).get(key, "unspecified")]

        for value in group_value:
            bucket = summary.setdefault(value, {"count": 0, "decision": [], "chat": [], "code": []})
            bucket["count"] += 1
            bucket["decision"].append(row.get("decision_score"))
            bucket["chat"].append(row.get("chat_score"))
            bucket["code"].append(row.get("code_score"))

    for bucket in summary.values():
        bucket["decision_avg"] = _mean(bucket.pop("decision"))
        bucket["chat_avg"] = _mean(bucket.pop("chat"))
        bucket["code_avg"] = _mean(bucket.pop("code"))

    return summary


def _save_plot(values: dict[str, float], title: str, ylabel: str, output_path: Path) -> str:
    labels = list(values.keys())
    scores = [values[label] for label in labels]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, scores, color="#3b82f6")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")

    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{score:.2f}", ha="center")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return str(output_path)


def _generate_visualizations(report: dict, output_dir: Path) -> dict[str, str]:
    charts: dict[str, str] = {}
    summary = report.get("summary", {})
    chart_dir = output_dir / "charts"

    charts["overall_scores"] = _save_plot(
        {
            "decision": summary.get("decision_accuracy", 0.0),
            "chat": summary.get("chat_score_avg", 0.0),
            "code": summary.get("code_score_avg", 0.0),
        },
        "Средние метрики по всем кейсам",
        "Score",
        chart_dir / "overall_scores.png",
    )

    difficulty = report.get("by_difficulty", {})
    if difficulty:
        charts["difficulty"] = _save_plot(
            {k: v.get("decision_avg", 0.0) for k, v in difficulty.items()},
            "Decision score по уровням сложности",
            "Decision score",
            chart_dir / "decision_by_difficulty.png",
        )

    return charts


# ---------- main eval ----------

def evaluate_code(model_code: str, benchmark: str) -> dict:
    heuristic_score = 0.0

    # 1️⃣ Syntax check
    try:
        ast.parse(model_code)
        heuristic_score += 0.3
    except SyntaxError:
        return {
            "score": 0.0,
            "heuristic_score": 0.0,
            "exec_score": 0.0,
            "stdout_match": False,
            "result_match": False,
            "errors": {
                "model": "syntax_error",
                "expected": None,
            },
        }

    expected_code = benchmark

    # 2️⃣ Imports
    expected_imports = extract_imports(expected_code)
    model_imports = extract_imports(model_code)

    if expected_imports:
        matched = expected_imports & model_imports
        heuristic_score += 0.2 * (len(matched) / len(expected_imports))
    else:
        heuristic_score += 0.2

    # 3️⃣ Key calls
    expected_calls = extract_calls(expected_code)
    model_calls = extract_calls(model_code)

    if expected_calls:
        matched = expected_calls & model_calls
        heuristic_score += 0.4 * (len(matched) / len(expected_calls))

    # ---------- execution-based scoring ----------
    expected_stdout, expected_result, expected_error = _run_code_in_sandbox(expected_code)
    model_stdout, model_result, model_error = _run_code_in_sandbox(model_code)

    stdout_match = expected_error is None and model_error is None and _normalize_stdout(expected_stdout) == _normalize_stdout(model_stdout)
    result_match = expected_error is None and model_error is None and _compare_results(expected_result, model_result)

    exec_score = 0.0
    if expected_error is None:
        if model_error is None:
            if stdout_match:
                exec_score += 0.5
            if result_match:
                exec_score += 0.5
        else:
            exec_score = 0.0
    else:
        exec_score = 0.0

    combined_score = round(min((heuristic_score * 0.5) + (exec_score * 0.5), 1.0), 3)

    return {
        "score": combined_score,
        "heuristic_score": round(min(heuristic_score, 1.0), 3),
        "exec_score": round(exec_score, 3),
        "stdout_match": stdout_match,
        "result_match": result_match,
        "errors": {
            "expected": expected_error,
            "model": model_error,
        },
        "stdout": {
            "expected": expected_stdout,
            "model": model_stdout,
        },
        "results": {
            "expected": expected_result,
            "model": model_result,
        },
    }

def _load_all_benchmarks() -> list[dict]:
    paths = sorted(BENCHMARKS_DIR.glob("*.json"))
    benchmarks: list[dict] = []
    for path in paths:
        benchmarks += load_json(path)
    return benchmarks


def run_eval(inference_res_path: str, baseline_path: str | None = "core/evaluation/inference_results/eval_0_baseline.json"):
    outputs = normalize_results(load_json(Path(inference_res_path)))

    baseline_outputs = []
    if baseline_path:
        baseline_file = Path(baseline_path)
        if baseline_file.exists():
            baseline_outputs = normalize_results(load_json(baseline_file))

    output_by_id = {item.get("benchmark_id"): item for item in outputs}
    baseline_by_id = {item.get("benchmark_id"): item for item in baseline_outputs}

    benchmarks = _load_all_benchmarks()

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

        code_score_details = None
        if case.get("expected_code"):
            code_score_details = evaluate_code(model_output.get("generated_code",""), case["expected_code"])
            code_score = code_score_details.get("score", 0.0)

        results.append({
            "id": case["id"],
            "decision_score": decision_score,
            "chat_score": chat_score,
            "baseline_similarity": baseline_similarity,
            "code_score": code_score,
            "code_details": code_score_details,
        })
    return results, benchmarks


def build_report(results: list[dict], benchmarks: list[dict]) -> dict:
    benchmarks_by_id = _collect_metadata(benchmarks)
    enriched_cases: list[dict] = []

    for row in results:
        meta = benchmarks_by_id.get(row["id"], {})
        meta_info = meta.get("metadata", {})
        enriched_cases.append(
            {
                **row,
                "user_input": meta.get("user_input"),
                "tags": meta_info.get("tags", []),
                "difficulty": meta_info.get("difficulty", "unspecified"),
            }
        )

    summary = {
        "cases_total": len(enriched_cases),
        "missing": len([case for case in enriched_cases if case.get("error")]),
        "decision_accuracy": _mean([case.get("decision_score") for case in enriched_cases if not case.get("error")]),
        "chat_score_avg": _mean([case.get("chat_score") for case in enriched_cases if not case.get("error")]),
        "code_score_avg": _mean([case.get("code_score") for case in enriched_cases if not case.get("error")]),
        "baseline_similarity_avg": _mean([case.get("baseline_similarity") for case in enriched_cases if not case.get("error")]),
    }

    by_difficulty = _summarize_by_group(enriched_cases, benchmarks_by_id, "difficulty")
    by_tag = _summarize_by_group(enriched_cases, benchmarks_by_id, "tags")

    critical_cases = []
    for case in enriched_cases:
        if case.get("error"):
            critical_cases.append({"id": case["id"], "reason": case["error"]})
            continue

        if (case.get("decision_score") == 0) or (case.get("chat_score") is not None and case.get("chat_score") < 0.7) or (case.get("code_score") is not None and case.get("code_score") < 0.7):
            critical_cases.append(
                {
                    "id": case["id"],
                    "decision_score": case.get("decision_score"),
                    "chat_score": case.get("chat_score"),
                    "code_score": case.get("code_score"),
                    "difficulty": case.get("difficulty"),
                }
            )

    return {
        "summary": summary,
        "by_difficulty": by_difficulty,
        "by_tag": by_tag,
        "cases": enriched_cases,
        "focus_cases": critical_cases,
    }


def generate_report(
    inference_res_path: str,
    baseline_path: str | None = "core/evaluation/inference_results/eval_0_baseline.json",
    output_dir: str | Path = "core/evaluation/reports",
) -> dict:
    results, benchmarks = run_eval(inference_res_path, baseline_path)
    report = build_report(results, benchmarks)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    charts = _generate_visualizations(report, output_dir)
    report["charts"] = charts

    cases_df = pd.DataFrame(report["cases"])
    cases_df.to_csv(output_dir / "cases.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {
                "metric": "decision_accuracy",
                "value": report["summary"].get("decision_accuracy", 0.0),
            },
            {
                "metric": "chat_score_avg",
                "value": report["summary"].get("chat_score_avg", 0.0),
            },
            {
                "metric": "code_score_avg",
                "value": report["summary"].get("code_score_avg", 0.0),
            },
            {
                "metric": "baseline_similarity_avg",
                "value": report["summary"].get("baseline_similarity_avg", 0.0),
            },
        ]
    )
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report

# пример использования
if __name__ == "__main__":
    # Пример запуска: формируем полный отчёт и сохраняем метрики и графики
    test_path = "core/evaluation/inference_results/eval_0_baseline.json"
    report = generate_report(test_path)

    print("Отчёт сформирован. Ключевые метрики:")
    print(json.dumps(report.get("summary", {}), ensure_ascii=False, indent=2))
