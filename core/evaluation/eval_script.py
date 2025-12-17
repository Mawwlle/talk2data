from __future__ import annotations

import ast
import builtins
import contextlib
import io
import json
import math
import signal
import string
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_iris
from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from core.evaluation.inference_script import BENCHMARKS_DIR, load_json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL = (
    "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/"
    "snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
)
DEFAULT_BASELINE_PATH = "core/evaluation/inference_results/eval_0_baseline.json"
RANDOM_SEED = 0
SANDBOX_FILENAME = "<sandbox>"
SANDBOX_TIMEOUT_SECONDS = 5
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

SandboxResult = tuple[str, Any | None, str | None]


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------
def load_benchmarks(file_path: str | Path) -> list[dict[str, Any]]:
    """Load benchmarks from a JSONL file."""

    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate_decision(model_decision: Any, expected: Any) -> bool:
    """Return whether the model decision matches the expected decision."""

    return model_decision == expected


# ---------- parsing helpers ----------
def extract_imports(code: str) -> set[str]:
    """Extract imported modules from Python code."""

    tree = ast.parse(code)
    imports: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.add(name.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)

    return imports


def extract_calls(code: str) -> set[str]:
    """Extract called function names from Python code."""

    tree = ast.parse(code)
    calls: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        if isinstance(func, ast.Attribute):
            calls.add(func.attr)
        elif isinstance(func, ast.Name):
            calls.add(func.id)

    return calls


# ---------- execution helpers ----------
def _normalize_stdout(text: str) -> str:
    """Normalize stdout by trimming trailing spaces for stable comparison."""

    if not text:
        return ""
    lines = [line.rstrip() for line in text.strip().splitlines()]
    return "\n".join(lines)


def _compare_results(expected: Any, actual: Any) -> bool:
    """Compare execution results with tolerance for numerics and arrays."""

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
    """Tokenize text by removing punctuation and lowercasing."""

    translator = str.maketrans("", "", string.punctuation)
    return text.lower().translate(translator).split()


def counter_cosine_similarity(vec_a: Counter[str], vec_b: Counter[str]) -> float:
    """Compute cosine similarity between two Counters."""

    if not vec_a or not vec_b:
        return 0.0

    shared_keys = set(vec_a.keys()) | set(vec_b.keys())
    dot_product = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in shared_keys)
    norm_a = sum(v * v for v in vec_a.values()) ** 0.5
    norm_b = sum(v * v for v in vec_b.values()) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def _sandbox_globals() -> dict[str, Any]:
    """Prepare globals for sandboxed execution with restricted builtins."""

    dummy_builtins = {"__builtins__": {name: getattr(builtins, name) for name in SAFE_BUILTINS}}

    np.random.seed(RANDOM_SEED)

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


@contextlib.contextmanager
def _enforce_timeout(seconds: int = SANDBOX_TIMEOUT_SECONDS) -> None: # type: ignore
    """Context manager to enforce a hard execution timeout."""

    def _handler(signum: int, frame: Any) -> None:
        raise TimeoutError("Execution timed out")

    previous_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _run_code_in_sandbox(code: str) -> SandboxResult:
    """Execute code safely with restricted globals and a hard timeout.

    The function parses user code, executes statements, and if the last node is
    an expression, evaluates it to produce a return value. Any exceptions are
    captured as error strings instead of propagating.
    """

    sandbox_globals = _sandbox_globals()
    sandbox_locals: dict[str, Any] = {}

    try:
        parsed = ast.parse(code)
    except SyntaxError as exc:
        return "", None, f"syntax_error: {exc}"

    stdout_buffer = io.StringIO()
    exec_error: str | None = None
    result_value: Any | None = None

    try:
        with _enforce_timeout():
            with contextlib.redirect_stdout(stdout_buffer):
                if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                    body_without_last = ast.Module(body=parsed.body[:-1], type_ignores=[])
                    last_expr = ast.Expression(parsed.body[-1].value)

                    if body_without_last.body:
                        exec(compile(body_without_last, SANDBOX_FILENAME, "exec"), sandbox_globals, sandbox_locals)

                    result_value = eval(compile(last_expr, SANDBOX_FILENAME, "eval"), sandbox_globals, sandbox_locals)
                else:
                    exec(compile(parsed, SANDBOX_FILENAME, "exec"), sandbox_globals, sandbox_locals)
    except TimeoutError as exc:
        exec_error = f"execution_timeout: {exc}"
    except Exception as exc:  # noqa: BLE001
        exec_error = f"execution_error: {exc}"

    stdout_text = _normalize_stdout(stdout_buffer.getvalue())
    return stdout_text, result_value, exec_error


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_embedding_components(model_path: str = DEFAULT_EMBEDDING_MODEL) -> tuple[Any, Any]:
    """Load tokenizer and model for embedding computation."""

    model_path = str(Path(model_path).expanduser())
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    model.eval()
    return tokenizer, model


def _compute_embedding(text: str) -> torch.Tensor:
    """Compute a sentence embedding for the given text."""

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
        sentence_embedding = torch.nn.functional.normalize(
            sentence_embedding, p=2, dim=1
        )

    return sentence_embedding.squeeze(0)


def evaluate_text_similarity(reference_text: str | None, generated_text: str | None) -> float | None:
    """Return cosine similarity between reference text and generated text.

    The reference is expected to be a canonical answer (e.g., concatenated
    expected facts). If no reference is available, the function returns None
    to avoid producing misleading perfect scores.
    """

    if not reference_text or not generated_text:
        return None

    try:
        baseline_embedding = _compute_embedding(reference_text)
        generated_embedding = _compute_embedding(generated_text)

        similarity = cosine_similarity(
            baseline_embedding.unsqueeze(0),
            generated_embedding.unsqueeze(0),
        ).item()

        return round(float(similarity), 3)
    except Exception:  # noqa: BLE001
        baseline_tokens = Counter(tokenize(reference_text))
        generated_tokens = Counter(tokenize(generated_text))
        return round(counter_cosine_similarity(baseline_tokens, generated_tokens), 3)


def evaluate_chat_semantics(
    expected_facts: list[str] | None,
    forbidden_facts: list[str] | None,
    generated_text: str | None,
    fallback_reference: str | None,
) -> dict[str, Any]:
    """Compute semantic score for chat responses with fact coverage and penalties."""

    expected_facts = expected_facts or []
    forbidden_facts = forbidden_facts or []

    if not generated_text:
        return {
            "score": None,
            "similarity": None,
            "expected_coverage": None,
            "forbidden_penalty": None,
            "expected_hits": [],
            "forbidden_hits": [],
        }

    normalized_text = generated_text.lower()
    expected_hits = [fact for fact in expected_facts if fact and fact.lower() in normalized_text]
    forbidden_hits = [fact for fact in forbidden_facts if fact and fact.lower() in normalized_text]

    coverage = len(expected_hits) / len(expected_facts) if expected_facts else 1.0
    penalty = len(forbidden_hits) / len(forbidden_facts) if forbidden_facts else 0.0

    reference_text = ". ".join(expected_facts) if expected_facts else fallback_reference
    similarity = evaluate_text_similarity(reference_text, generated_text)
    similarity = similarity if similarity is not None else coverage

    raw_score = (similarity * 0.6) + (coverage * 0.4) - (penalty * 0.5)
    score = round(max(0.0, min(raw_score, 1.0)), 3)

    return {
        "score": score,
        "similarity": similarity,
        "expected_coverage": round(coverage, 3),
        "forbidden_penalty": round(penalty, 3) if forbidden_facts else 0.0,
        "expected_hits": expected_hits,
        "forbidden_hits": forbidden_hits,
    }


# ---------------------------------------------------------------------------
# Result normalization utilities
# ---------------------------------------------------------------------------
def normalize_results(results: list[dict[str, Any]] | dict[str, Any] | None) -> list[dict[str, Any]]:
    """Normalize inference results to a list of dictionaries."""

    if isinstance(results, list):
        return results
    if isinstance(results, dict) and "results" in results:
        return results.get("results", [])
    return []


def _mean(values: list[float | None]) -> float:
    """Compute a rounded mean ignoring non-numeric entries."""

    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    if not numeric:
        return 0.0
    return round(float(np.mean(numeric)), 3)


def _collect_metadata(benchmarks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Create a mapping from benchmark id to metadata."""

    return {case.get("id"): case for case in benchmarks}


def _summarize_by_group(
    results: list[dict[str, Any]],
    benchmarks_by_id: dict[str, dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    """Aggregate metrics by difficulty or tags."""

    summary: dict[str, dict[str, Any]] = {}
    for row in results:
        case_meta = benchmarks_by_id.get(row["id"], {})
        if key == "tags":
            group_values = case_meta.get("metadata", {}).get("tags", []) or ["untagged"]
        else:
            group_values = [case_meta.get("metadata", {}).get(key, "unspecified")]

        for value in group_values:
            bucket = summary.setdefault(
                value,
                {"count": 0, "decision": [], "semantic_similarity": [], "code": []},
            )
            bucket["count"] += 1
            bucket["decision"].append(row.get("decision_score"))
            bucket["semantic_similarity"].append(row.get("semantic_similarity"))
            bucket["code"].append(row.get("code_score"))

    for bucket in summary.values():
        bucket["decision_avg"] = _mean(bucket.pop("decision"))
        bucket["semantic_similarity_avg"] = _mean(bucket.pop("semantic_similarity"))
        bucket["code_avg"] = _mean(bucket.pop("code"))

    return summary


def _save_plot(values: dict[str, float], title: str, ylabel: str, output_path: Path) -> str:
    """Save a simple bar plot and return its file path."""

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


def _aggregate_code_metrics(cases: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate code-generation metrics across cases."""

    code_cases = [case for case in cases if case.get("expected_decision") == "code_generation" and not case.get("error")]
    if not code_cases:
        return {}

    heuristics = [case.get("code_details", {}).get("heuristic_score") for case in code_cases]
    exec_scores = [case.get("code_details", {}).get("exec_score") for case in code_cases]
    stdout_matches = [1.0 if case.get("code_details", {}).get("stdout_match") else 0.0 for case in code_cases]
    result_matches = [1.0 if case.get("code_details", {}).get("result_match") else 0.0 for case in code_cases]
    code_scores = [case.get("code_score") for case in code_cases]

    return {
        "code_score": _mean(code_scores),
        "heuristic_score": _mean(heuristics),
        "exec_score": _mean(exec_scores),
        "stdout_match_rate": _mean(stdout_matches),
        "result_match_rate": _mean(result_matches),
    }


def _generate_visualizations(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Generate charts for key metrics and return their paths."""

    charts: dict[str, str] = {}
    summary = report.get("summary", {})
    chart_dir = output_dir / "charts"

    charts["overall_scores"] = _save_plot(
        {
            "decision": summary.get("decision_accuracy", 0.0),
            "semantic_similarity": summary.get("semantic_similarity_avg", 0.0),
            "code": summary.get("code_score_avg", 0.0),
        },
        "Средние метрики по всем кейсам",
        "Score",
        chart_dir / "overall_scores.png",
    )

    difficulty = report.get("by_difficulty", {})
    if difficulty:
        charts["decision_by_difficulty"] = _save_plot(
            {k: v.get("decision_avg", 0.0) for k, v in difficulty.items()},
            "Точность decision по уровням сложности",
            "Accuracy",
            chart_dir / "decision_by_difficulty.png",
        )
        charts["semantic_by_difficulty"] = _save_plot(
            {k: v.get("semantic_similarity_avg", 0.0) for k, v in difficulty.items()},
            "Semantic similarity по уровням сложности",
            "Semantic similarity",
            chart_dir / "semantic_similarity_by_difficulty.png",
        )

    if report.get("cases"):
        code_metrics = _aggregate_code_metrics(report["cases"])
        if code_metrics:
            charts["code_metrics"] = _save_plot(
                {
                    "code_score": code_metrics.get("code_score", 0.0),
                    "heuristic": code_metrics.get("heuristic_score", 0.0),
                    "execution": code_metrics.get("exec_score", 0.0),
                    "stdout_match": code_metrics.get("stdout_match_rate", 0.0),
                    "result_match": code_metrics.get("result_match_rate", 0.0),
                },
                "Средние кодовые метрики",
                "Score",
                chart_dir / "code_metrics.png",
            )

    return charts


def _make_json_safe(value: Any) -> Any:
    """Convert non-serializable values into JSON-friendly structures."""

    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, tuple) or isinstance(value, set):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _render_case_markdown(
    cases: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Render a human-friendly markdown report for all cases."""

    report_path = output_dir / "cases_report.md"
    lines: list[str] = []

    lines.append("# Детальный отчёт по кейсам")
    lines.append("")
    lines.append("## Методика расчёта метрик")
    lines.append("### Semantic similarity (только для chat_response)")
    lines.append(
        "- Ищем ожидаемые факты в ответе и считаем долю покрытых фактов (expected_coverage)."
    )
    lines.append(
        "- Считаем попадания запрещённых фактов и превращаем их в штраф (forbidden_penalty)."
    )
    lines.append(
        "- Дополнительно считаем embedding-cosine между эталонным ответом (конкатенация expected_facts) и ответом модели."
    )
    lines.append(
        "- Итоговая semantic_similarity = 0.6 * similarity + 0.4 * expected_coverage - 0.5 * forbidden_penalty, ограниченная от 0 до 1."
    )
    lines.append("")
    lines.append("### Code score (только для code_generation)")
    lines.append("- heuristic_score: синтаксис (+0.3), совпадение импортов (до +0.2), ключевых вызовов (до +0.4).")
    lines.append(
        "- exec_score: проверяем, исполняется ли код без ошибок; +0.5 за совпадение stdout и ещё +0.5 за совпадение результата."
    )
    lines.append("- code_score = 0.5 * heuristic_score + 0.5 * exec_score (от 0 до 1).")
    lines.append("")
    lines.append("## Кейсы")

    for case in cases:
        lines.append(f"### {case.get('id')} ({case.get('expected_decision')}, difficulty: {case.get('difficulty')})")
        lines.append("")
        lines.append(f"**User input:** {case.get('user_input')}")

        expected_facts = case.get("expected_facts") or []
        forbidden_facts = case.get("forbidden_facts") or []

        lines.append("**Ground truth:**")
        if expected_facts:
            lines.append("- expected_facts: " + "; ".join(expected_facts))
        if forbidden_facts:
            lines.append("- forbidden_facts: " + "; ".join(forbidden_facts))
        if case.get("expected_code"):
            lines.append("- expected_code:")
            lines.append("```python")
            lines.append(case.get("expected_code"))
            lines.append("```")

        lines.append("**Model output:**")
        if case.get("response_message"):
            lines.append("> " + case.get("response_message").replace("\n", " "))
        if case.get("model_generated_code"):
            lines.append("```python")
            lines.append(case.get("model_generated_code"))
            lines.append("```")

        lines.append("**Метрики:**")
        lines.append("- decision_score: " + str(case.get("decision_score")))

        if case.get("expected_decision") == "chat_response":
            semantic_details = case.get("semantic_details") or {}
            lines.append(f"- semantic_similarity: {semantic_details.get('score')}")
            lines.append(
                f"  - expected_coverage: {semantic_details.get('expected_coverage')} | forbidden_penalty: {semantic_details.get('forbidden_penalty')}"
            )
            if semantic_details.get("expected_hits"):
                lines.append("  - покрытые факты: " + "; ".join(semantic_details.get("expected_hits", [])))
            if semantic_details.get("forbidden_hits"):
                lines.append("  - упомянутые запрещённые факты: " + "; ".join(semantic_details.get("forbidden_hits", [])))

        if case.get("expected_decision") == "code_generation":
            details = case.get("code_details") or {}
            lines.append(f"- code_score: {case.get('code_score')}")
            lines.append("  - heuristic_score: " + str(details.get("heuristic_score")))
            lines.append("  - exec_score: " + str(details.get("exec_score")))
            lines.append("  - stdout_match: " + str(details.get("stdout_match")))
            lines.append("  - result_match: " + str(details.get("result_match")))

        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------
def evaluate_code(model_code: str, benchmark: str) -> dict[str, Any]:
    """Score generated code using syntax, heuristic, and execution signals."""

    heuristic_score = 0.0

    # 1️⃣ Syntax check: fail fast on invalid Python
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

    # 2️⃣ Imports: ensure required modules are present
    expected_imports = extract_imports(expected_code)
    model_imports = extract_imports(model_code)

    if expected_imports:
        matched = expected_imports & model_imports
        heuristic_score += 0.2 * (len(matched) / len(expected_imports))
    else:
        heuristic_score += 0.2

    # 3️⃣ Key calls: verify important function invocations
    expected_calls = extract_calls(expected_code)
    model_calls = extract_calls(model_code)

    if expected_calls:
        matched = expected_calls & model_calls
        heuristic_score += 0.4 * (len(matched) / len(expected_calls))

    # ---------- execution-based scoring ----------
    expected_stdout, expected_result, expected_error = _run_code_in_sandbox(expected_code)
    model_stdout, model_result, model_error = _run_code_in_sandbox(model_code)

    stdout_match = (
        expected_error is None
        and model_error is None
        and _normalize_stdout(expected_stdout) == _normalize_stdout(model_stdout)
    )
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


def _load_all_benchmarks() -> list[dict[str, Any]]:
    """Load all benchmark cases from the configured directory."""

    paths = sorted(BENCHMARKS_DIR.glob("*.json"))
    benchmarks: list[dict[str, Any]] = []
    for path in paths:
        benchmarks += load_json(path)
    return benchmarks


def run_eval(inference_res_path: str, baseline_path: str | None = DEFAULT_BASELINE_PATH) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run evaluation comparing inference results to benchmarks and baseline."""

    outputs = normalize_results(load_json(Path(inference_res_path)))

    baseline_outputs: list[dict[str, Any]] = []
    if baseline_path:
        baseline_file = Path(baseline_path)
        if baseline_file.exists():
            baseline_outputs = normalize_results(load_json(baseline_file))

    output_by_id = {item.get("benchmark_id"): item for item in outputs}
    baseline_by_id = {item.get("benchmark_id"): item for item in baseline_outputs}

    benchmarks = _load_all_benchmarks()

    results: list[dict[str, Any]] = []
    for case in benchmarks:
        model_output = output_by_id.get(case.get("id"))
        if not model_output:
            results.append({
                "id": case["id"],
                "expected_decision": case.get("expected_decision"),
                "error": "missing_model_output",
            })
            continue

        decision_score = evaluate_decision(model_output.get("model_decision", {}).get("action"), case["expected_decision"])

        semantic_similarity = None
        semantic_details = None
        if case.get("expected_decision") == "chat_response":
            baseline_reference = None
            if baseline_by_id:
                baseline_output = baseline_by_id.get(case.get("id"))
                if baseline_output:
                    baseline_reference = baseline_output.get("response_message")

            semantic_details = evaluate_chat_semantics(
                case.get("expected_facts"),
                case.get("forbidden_facts"),
                model_output.get("response_message"),
                baseline_reference,
            )
            semantic_similarity = semantic_details.get("score")

        code_score_details = None
        code_score = None
        if case.get("expected_decision") == "code_generation" and case.get("expected_code"):
            code_score_details = evaluate_code(model_output.get("generated_code", ""), case["expected_code"])
            code_score = code_score_details.get("score")

        results.append({
            "id": case["id"],
            "expected_decision": case.get("expected_decision"),
            "decision_score": decision_score,
            "semantic_similarity": semantic_similarity,
            "semantic_details": semantic_details,
            "code_score": code_score,
            "code_details": code_score_details,
            "model_generated_code": model_output.get("generated_code"),
            "response_message": model_output.get("response_message"),
        })
    return results, benchmarks


def build_report(results: list[dict[str, Any]], benchmarks: list[dict[str, Any]]) -> dict[str, Any]:
    """Construct aggregated report data from evaluation results."""

    benchmarks_by_id = _collect_metadata(benchmarks)
    enriched_cases: list[dict[str, Any]] = []

    for row in results:
        meta = benchmarks_by_id.get(row["id"], {})
        meta_info = meta.get("metadata", {})
        enriched_cases.append(
            {
                **row,
                "user_input": meta.get("user_input"),
                "tags": meta_info.get("tags", []),
                "difficulty": meta_info.get("difficulty", "unspecified"),
                "expected_decision": meta.get("expected_decision"),
                "expected_facts": meta.get("expected_facts"),
                "forbidden_facts": meta.get("forbidden_facts"),
                "expected_code": meta.get("expected_code"),
            }
        )

    summary = {
        "cases_total": len(enriched_cases),
        "missing": len([case for case in enriched_cases if case.get("error")]),
        "decision_accuracy": _mean([case.get("decision_score") for case in enriched_cases if not case.get("error")]),
        "semantic_similarity_avg": _mean([case.get("semantic_similarity") for case in enriched_cases if not case.get("error")]),
        "code_score_avg": _mean([case.get("code_score") for case in enriched_cases if not case.get("error")]),
    }

    by_difficulty = _summarize_by_group(enriched_cases, benchmarks_by_id, "difficulty")
    by_tag = _summarize_by_group(enriched_cases, benchmarks_by_id, "tags")

    critical_cases = []
    for case in enriched_cases:
        if case.get("error"):
            critical_cases.append({"id": case["id"], "reason": case["error"]})
            continue

        if (
            case.get("decision_score") == 0
            or (case.get("semantic_similarity") is not None and case.get("semantic_similarity") < 0.7)
            or (case.get("code_score") is not None and case.get("code_score") < 0.7)
        ):
            critical_cases.append(
                {
                    "id": case["id"],
                    "decision_score": case.get("decision_score"),
                    "semantic_similarity": case.get("semantic_similarity"),
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
    baseline_path: str | None = DEFAULT_BASELINE_PATH,
    output_dir: str | Path = "core/evaluation/reports",
) -> dict[str, Any] | list[dict[str, Any]]:
    """Generate evaluation report files and return the structured report."""

    results, benchmarks = run_eval(inference_res_path, baseline_path)
    report = build_report(results, benchmarks)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    charts = _generate_visualizations(report, output_dir)
    report["charts"] = charts

    case_report_path = _render_case_markdown(report["cases"], output_dir)
    report["case_report_path"] = str(case_report_path)

    cases_df = pd.DataFrame(report["cases"])
    cases_df["heuristic_score"] = cases_df["code_details"].apply(lambda x: x.get("heuristic_score") if isinstance(x, dict) else None)
    cases_df["exec_score"] = cases_df["code_details"].apply(lambda x: x.get("exec_score") if isinstance(x, dict) else None)
    cases_df["stdout_match"] = cases_df["code_details"].apply(lambda x: x.get("stdout_match") if isinstance(x, dict) else None)
    cases_df["result_match"] = cases_df["code_details"].apply(lambda x: x.get("result_match") if isinstance(x, dict) else None)
    cases_df = cases_df.drop(columns=[
        "model_generated_code",
        "response_message",
        "code_details",
        "semantic_details",
    ], errors="ignore")
    cases_df.to_csv(output_dir / "cases.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {
                "metric": "decision_accuracy",
                "value": report["summary"].get("decision_accuracy", 0.0),
            },
            {
                "metric": "semantic_similarity_avg",
                "value": report["summary"].get("semantic_similarity_avg", 0.0),
            },
            {
                "metric": "code_score_avg",
                "value": report["summary"].get("code_score_avg", 0.0),
            },
        ]
    )
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    return _make_json_safe(report)


if __name__ == "__main__":
    # Пример запуска: формируем полный отчёт и сохраняем метрики и графики
    test_path = "core/evaluation/inference_results/eval_0_baseline.json"
    report = generate_report(test_path)

    print("Отчёт сформирован. Ключевые метрики:")
    print(json.dumps(report.get("summary", {}), ensure_ascii=False, indent=2))
