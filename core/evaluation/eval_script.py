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
SANDBOX_TIMEOUT_SECONDS = 60
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
    """Load benchmarks from a JSON file."""

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


def _fact_presence_score(fact: str, generated_text: str) -> float:
    """Estimate how strongly a fact is present in the generated text.

    The score combines embedding cosine similarity (primary) with a token-overlap
    fallback so we can capture paraphrased mentions instead of requiring exact
    substring matches.
    """

    if not fact or not generated_text:
        return 0.0

    try:
        fact_embedding = _compute_embedding(fact)
        text_embedding = _compute_embedding(generated_text)
        similarity = cosine_similarity(
            fact_embedding.unsqueeze(0), text_embedding.unsqueeze(0)
        ).item()
        return float(similarity)
    except Exception:  # noqa: BLE001
        fact_tokens = Counter(tokenize(fact))
        text_tokens = Counter(tokenize(generated_text))
        return counter_cosine_similarity(fact_tokens, text_tokens)


def _ensure_kaleido() -> bool:
    """Try to ensure kaleido is available for Plotly image export.

    Returns True if kaleido is importable after the check; otherwise False.
    """

    try:
        import importlib.util  # noqa: WPS433 (used only here)

        if importlib.util.find_spec("kaleido"):
            return True

        import subprocess  # noqa: WPS433

        subprocess.run(
            [
                "python",
                "-m",
                "pip",
                "install",
                "--quiet",
                "kaleido",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return importlib.util.find_spec("kaleido") is not None
    except Exception:  # noqa: BLE001
        return False


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


def _infer_language_from_filename(path: Path) -> str:
    """Best-effort language inference from a benchmark filename."""

    name = path.stem.lower()
    if name.endswith("_en"):
        return "en"
    if name.endswith("_ru"):
        return "ru"
    return "unknown"


@contextlib.contextmanager
def _enforce_timeout(seconds: int = SANDBOX_TIMEOUT_SECONDS, message: str = "sandbox_timeout") -> None:
    """Raise ``TimeoutError`` if the block runs longer than ``seconds`` seconds."""

    def _handler(signum: int, frame: Any) -> None:  # noqa: ANN001
        raise TimeoutError(message)

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
    presence_threshold: float = 0.55,
) -> dict[str, Any]:
    """Compute semantic score for chat responses with semantic fact coverage.

    Fact presence is determined semantically (embedding cosine with token-overlap
    fallback) rather than by exact substring search so paraphrases count as
    coverage, and forbidden facts can be penalized even when phrased differently.
    """

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

    expected_hit_scores: list[tuple[str, float]] = []
    forbidden_hit_scores: list[tuple[str, float]] = []

    for fact in expected_facts:
        score = _fact_presence_score(fact, generated_text)
        if score >= presence_threshold:
            expected_hit_scores.append((fact, round(score, 3)))

    for fact in forbidden_facts:
        score = _fact_presence_score(fact, generated_text)
        if score >= presence_threshold:
            forbidden_hit_scores.append((fact, round(score, 3)))

    coverage = len(expected_hit_scores) / len(expected_facts) if expected_facts else 1.0
    penalty = len(forbidden_hit_scores) / len(forbidden_facts) if forbidden_facts else 0.0

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
        "expected_hits": expected_hit_scores,
        "forbidden_hits": forbidden_hit_scores,
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


def _collect_metadata(benchmarks: list[dict[str, Any]]) -> dict[tuple[str | None, str], dict[str, Any]]:
    """Create a mapping from (benchmark id, language) to metadata."""

    mapping: dict[tuple[str | None, str], dict[str, Any]] = {}
    for case in benchmarks:
        language = case.get("metadata", {}).get("language", "unknown")
        mapping[(case.get("id"), language)] = case
    return mapping


def _summarize_by_group(
    results: list[dict[str, Any]],
    benchmarks_by_id: dict[tuple[str | None, str], dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    """Aggregate metrics by difficulty or tags."""

    summary: dict[str, dict[str, Any]] = {}
    for row in results:
        meta_key = (row.get("id"), row.get("language", "unknown"))
        case_meta = benchmarks_by_id.get(meta_key, {})
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


def _summarize_by_language(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate metrics by benchmark language."""

    summary: dict[str, dict[str, Any]] = {}
    for row in results:
        lang = row.get("language", "unknown")
        bucket = summary.setdefault(
            lang,
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
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.2f}",
            ha="center",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return str(output_path)


def _aggregate_code_metrics(cases: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate code-generation metrics across cases."""

    code_cases = [
        case for case in cases if case.get("expected_decision") == "code_generation" and not case.get("error")
    ]
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


def _save_language_comparison(
    language_summary: dict[str, dict[str, float]],
    output_path: Path,
) -> str:
    """Save grouped bar plot comparing metrics across languages."""

    if not language_summary:
        return ""

    languages = sorted(language_summary.keys())
    metrics = ["decision_avg", "semantic_similarity_avg", "code_avg"]
    metric_labels = ["Decision", "Semantic", "Code"]
    x = np.arange(len(languages))
    bar_width = 0.22

    plt.figure(figsize=(10, 5))
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        scores = [language_summary[lang].get(metric, 0.0) for lang in languages]
        plt.bar(x + idx * bar_width, scores, width=bar_width, label=label)

    plt.xticks(x + bar_width, languages)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Сравнение метрик по языкам ввода")
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return str(output_path)


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

    language_summary = report.get("by_language", {})
    if language_summary:
        charts["language_comparison"] = _save_language_comparison(
            language_summary,
            chart_dir / "language_comparison.png",
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


def _describe_visual_output(code_text: str) -> str:
    """Provide a readable hint for visual outputs without raw JSON dumps."""

    lowered = code_text.lower()
    if "plotly" in lowered or "px." in lowered:
        return "Интерактивный график (Plotly)"
    if "plt." in lowered or "matplotlib" in lowered:
        return "Статический график (matplotlib)"
    return "Графический вывод"


def _render_plotly_image(code_text: str, output_path: Path) -> tuple[str | None, str | None]:
    """Execute plotly code to export a PNG preview.

    Returns (path, error). Path is filled on success; otherwise error contains
    the failure reason. Execution uses the sandbox globals with plotly/io.show
    stubbed to avoid opening renderers.
    """

    if "plotly" not in code_text.lower():
        return None, None

    if not _ensure_kaleido():
        return None, "kaleido is not available to export PNG"

    env = _sandbox_globals()

    try:
        import plotly  # noqa: WPS433
        import plotly.express as px  # noqa: WPS433
        import plotly.graph_objects as go  # noqa: WPS433
        import plotly.io as pio  # noqa: WPS433

        def _no_show(*_: Any, **__: Any) -> None:  # noqa: ANN002,ANN003
            return None

        pio.show = _no_show  # type: ignore[assignment]
        env.update({"px": px, "go": go, "pio": pio, "plotly": plotly})
        exec(compile(code_text, SANDBOX_FILENAME, "exec"), env, env)

        fig = env.get("fig")
        if fig is None:
            return None, "no figure named 'fig' was created"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(output_path))
        return str(output_path), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _render_case_markdown(
    cases: list[dict[str, Any]],
    output_dir: Path,
    filename: str = "cases_report.md",
    title: str = "Детальный отчёт по кейсам",
) -> Path:
    """Render a human-friendly markdown report for all cases."""

    report_path = output_dir / filename
    lines: list[str] = []

    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Методика расчёта метрик")
    lines.append("### Semantic similarity (только для chat_response/theoretical_response)")
    lines.append("Формула:")
    lines.append("```text")
    lines.append("semantic_similarity = 0.6 * similarity + 0.4 * expected_coverage - 0.5 * forbidden_penalty")
    lines.append("semantic_similarity = clip(semantic_similarity, 0, 1)")
    lines.append("```")
    lines.append("Где доли: expected_coverage — доля ожидаемых фактов, упомянутых в ответе; forbidden_penalty — доля запрещённых фактов,")
    lines.append("попавших в ответ (штраф). similarity — embedding-cosine между эталонным ответом (конкатенация expected_facts или базовый")
    lines.append("референс) и ответом модели.")
    lines.append("Источники: взято из распространённой практике оценки фактологичности QA (cosine по sentence-transformers, coverage/penalty как")
    lines.append("в rag-as-a-service baseline и open-domain QA leaderboard). Факт считается покрытым, если косинусная близость fact↔ответ ≥ 0.55")
    lines.append("(или высокая токеновая схожесть), что позволяет засчитывать перефраз. Весами (0.6/0.4/0.5) балансируем близость текста и полноту фактов,")
    lines.append("давая штраф за запрещённые факты, чтобы сохранить интерпретируемость (веса суммарно ограничивают метрику в [0, 1]).")
    lines.append("")
    lines.append("### Code score (только для code_generation)")
    lines.append("Формула:")
    lines.append("```text")
    lines.append("code_score = 0.5 * heuristic_score + 0.5 * exec_score")
    lines.append("code_score = clip(code_score, 0, 1)")
    lines.append("```")
    lines.append("Разложение долей:")
    lines.append("- heuristic_score = 0.3 (валидный синтаксис) + до 0.2 (совпадение импортов) + до 0.4 (совпадение ключевых вызовов).")
    lines.append("- exec_score: +0.5 если stdout совпал при успешном выполнении; +0.5 если совпал вычисленный результат.")
    lines.append("Источники: опираемся на принципы автотестов LeetCode/Codeforces (выполнение и сравнение вывода/результата) и на статический")
    lines.append("анализ из pymetrics/ruff (синтаксис, импорты, ключевые вызовы) для интерпретируемого разбиения вклада.")
    lines.append("Пояснения: stdout_match — флаг, что нормализованный вывод программы совпал с бенчмарком при отсутствии ошибок; result_match — флаг, что")
    lines.append("финальное значение выражения совпало. Эти флаги формируют exec_score.")
    lines.append("")
    lines.append("## Кейсы")

    for case in cases:
        lines.append(
            f"### {case.get('id')} ({case.get('expected_decision')}, language: {case.get('language')}, difficulty: {case.get('difficulty')})"
        )
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
            details = case.get("code_details") or {}
            lines.append("- expected_stdout:")
            lines.append("```")
            lines.append(str(details.get("stdout", {}).get("expected")))
            lines.append("```")
            lines.append("- expected_result: " + str(details.get("results", {}).get("expected")))
            if not details.get("stdout", {}).get("expected") and details.get("results", {}).get("expected") is None:
                lines.append("- expected_output: " + _describe_visual_output(case.get("expected_code", "")))
            lines.append("- expected_error: " + str(details.get("errors", {}).get("expected")))
            if case.get("expected_plot_path"):
                lines.append(f"- expected_plot: ![expected plot]({case.get('expected_plot_path')})")
            elif case.get("expected_plot_error"):
                lines.append(f"- expected_plot_error: {case.get('expected_plot_error')}")

        lines.append("**Model output:**")
        if case.get("response_message"):
            lines.append("> " + case.get("response_message").replace("\n", " "))
        if case.get("model_generated_code"):
            lines.append("```python")
            lines.append(case.get("model_generated_code"))
            lines.append("```")
            details = case.get("code_details") or {}
            lines.append("- model_stdout:")
            lines.append("```")
            lines.append(str(details.get("stdout", {}).get("model")))
            lines.append("```")
            lines.append("- model_result: " + str(details.get("results", {}).get("model")))
            lines.append("- model_error: " + str(details.get("errors", {}).get("model")))
            if case.get("model_plot_path"):
                lines.append(f"- model_plot: ![model plot]({case.get('model_plot_path')})")
            elif case.get("model_plot_error"):
                lines.append(f"- model_plot_error: {case.get('model_plot_error')}")

        lines.append("**Метрики:**")
        lines.append("- decision_score: " + str(case.get("decision_score")))

        if case.get("expected_decision") == "chat_response":
            semantic_details = case.get("semantic_details") or {}
            lines.append(f"- semantic_similarity: {semantic_details.get('score')}")
            lines.append(
                f"  - expected_coverage: {semantic_details.get('expected_coverage')} | forbidden_penalty: {semantic_details.get('forbidden_penalty')}"
            )
            if semantic_details.get("expected_hits"):
                formatted = [f"{fact} (score {score})" for fact, score in semantic_details.get("expected_hits", [])]
                lines.append("  - покрытые факты: " + "; ".join(formatted))
            if semantic_details.get("forbidden_hits"):
                formatted = [f"{fact} (score {score})" for fact, score in semantic_details.get("forbidden_hits", [])]
                lines.append("  - упомянутые запрещённые факты: " + "; ".join(formatted))

        if case.get("expected_decision") == "code_generation":
            details = case.get("code_details") or {}
            lines.append(f"- code_score: {case.get('code_score')}")
            lines.append("  - heuristic_score: " + str(details.get("heuristic_score")))
            breakdown = details.get("heuristic_breakdown", {})
            syntax_part = breakdown.get("syntax", {})
            import_part = breakdown.get("imports", {})
            calls_part = breakdown.get("calls", {})
            lines.append(
                "    - syntax_check: "
                + str(syntax_part.get("score"))
                + " (ok="
                + str(syntax_part.get("ok"))
                + ")"
            )
            lines.append(
                "    - imports_score: "
                + str(import_part.get("score"))
                + " | expected: "
                + ", ".join(import_part.get("expected", []))
                + " | model: "
                + ", ".join(import_part.get("model", []))
                + " | matched: "
                + ", ".join(import_part.get("matched", []))
            )
            lines.append(
                "    - calls_score: "
                + str(calls_part.get("score"))
                + " | expected: "
                + ", ".join(calls_part.get("expected", []))
                + " | model: "
                + ", ".join(calls_part.get("model", []))
                + " | matched: "
                + ", ".join(calls_part.get("matched", []))
            )
            lines.append("  - exec_score: " + str(details.get("exec_score")))
            lines.append("  - stdout_match: " + str(details.get("stdout_match")))
            lines.append("  - result_match: " + str(details.get("result_match")))

        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _attach_plot_previews(cases: list[dict[str, Any]], output_dir: Path) -> None:
    """Generate PNG previews for Plotly code in expected/model snippets."""

    plots_dir = output_dir / "plots"

    for case in cases:
        if case.get("expected_code"):
            expected_path, expected_err = _render_plotly_image(
                case["expected_code"],
                plots_dir / f"{case.get('id')}_expected.png",
            )
            if expected_path:
                case["expected_plot_path"] = expected_path
            if expected_err:
                case["expected_plot_error"] = expected_err

        model_code = case.get("model_generated_code")
        if model_code:
            model_path, model_err = _render_plotly_image(
                model_code,
                plots_dir / f"{case.get('id')}_model.png",
            )
            if model_path:
                case["model_plot_path"] = model_path
            if model_err:
                case["model_plot_error"] = model_err


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------
def evaluate_code(model_code: str, benchmark: str) -> dict[str, Any]:
    """Score generated code using syntax, heuristic, and execution signals."""

    heuristic_score = 0.0
    heuristic_breakdown: dict[str, Any] = {
        "syntax": {"score": 0.0, "ok": False},
        "imports": {"score": 0.0, "expected": [], "model": [], "matched": []},
        "calls": {"score": 0.0, "expected": [], "model": [], "matched": []},
    }

    # 1️⃣ Syntax check: fail fast on invalid Python
    try:
        ast.parse(model_code)
        heuristic_score += 0.3
        heuristic_breakdown["syntax"] = {"score": 0.3, "ok": True}
    except SyntaxError:
        return {
            "score": 0.0,
            "heuristic_score": 0.0,
            "exec_score": 0.0,
            "stdout_match": False,
            "result_match": False,
            "heuristic_breakdown": heuristic_breakdown,
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
        import_score = 0.2 * (len(matched) / len(expected_imports))
    else:
        matched = set()
        import_score = 0.2

    heuristic_score += import_score
    heuristic_breakdown["imports"] = {
        "score": round(import_score, 3),
        "expected": sorted(expected_imports),
        "model": sorted(model_imports),
        "matched": sorted(matched),
    }

    # 3️⃣ Key calls: verify important function invocations
    expected_calls = extract_calls(expected_code)
    model_calls = extract_calls(model_code)

    if expected_calls:
        matched = expected_calls & model_calls
        call_score = 0.4 * (len(matched) / len(expected_calls))
    else:
        matched = set()
        call_score = 0.0

    heuristic_score += call_score
    heuristic_breakdown["calls"] = {
        "score": round(call_score, 3),
        "expected": sorted(expected_calls),
        "model": sorted(model_calls),
        "matched": sorted(matched),
    }

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
        "heuristic_breakdown": heuristic_breakdown,
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
    """Load all benchmark cases from the configured directory with language tags."""

    paths = sorted(BENCHMARKS_DIR.glob("*.json"))
    benchmarks: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str]] = set()

    for path in paths:
        language = _infer_language_from_filename(path)
        for case in load_json(path):
            case_id = case.get("id")
            key = (case_id, language)
            if key in seen:
                print(
                    f"[benchmarks] duplicate id '{case_id}' for language {language} in {path.name} ignored",
                )
                continue

            seen.add(key)
            meta = {**case.get("metadata", {}), "language": language}
            benchmarks.append({**case, "metadata": meta})

    return benchmarks


def run_eval(
    inference_res_path: str, baseline_path: str | None = DEFAULT_BASELINE_PATH
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
    total_cases = len(benchmarks)

    for idx, case in enumerate(benchmarks, start=1):
        print(f"[eval] ({idx}/{total_cases}) processing {case.get('id')}")
        language = case.get("metadata", {}).get("language", "unknown")
        model_output = output_by_id.get(case.get("id"))
        if not model_output:
            results.append(
                {
                    "id": case["id"],
                    "expected_decision": case.get("expected_decision"),
                    "language": language,
                    "error": "missing_model_output",
                }
            )
            continue

        decision_score = evaluate_decision(
            model_output.get("model_decision", {}).get("action"), case["expected_decision"]
        )

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

        results.append(
            {
                "id": case["id"],
                "expected_decision": case.get("expected_decision"),
                "language": language,
                "decision_score": decision_score,
                "semantic_similarity": semantic_similarity,
                "semantic_details": semantic_details,
                "code_score": code_score,
                "code_details": code_score_details,
                "model_generated_code": model_output.get("generated_code"),
                "response_message": model_output.get("response_message"),
            }
        )
    return results, benchmarks


def build_report(results: list[dict[str, Any]], benchmarks: list[dict[str, Any]]) -> dict[str, Any]:
    """Construct aggregated report data from evaluation results."""

    benchmarks_by_id = _collect_metadata(benchmarks)
    enriched_cases: list[dict[str, Any]] = []

    for row in results:
        meta_key = (row.get("id"), row.get("language", "unknown"))
        meta = benchmarks_by_id.get(meta_key, {})
        meta_info = meta.get("metadata", {})
        enriched_cases.append(
            {
                **row,
                "user_input": meta.get("user_input"),
                "tags": meta_info.get("tags", []),
                "difficulty": meta_info.get("difficulty", "unspecified"),
                "language": meta_info.get("language", row.get("language", "unknown")),
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
        "semantic_similarity_avg": _mean(
            [case.get("semantic_similarity") for case in enriched_cases if not case.get("error")]
        ),
        "code_score_avg": _mean([case.get("code_score") for case in enriched_cases if not case.get("error")]),
    }

    by_difficulty = _summarize_by_group(enriched_cases, benchmarks_by_id, "difficulty")
    by_tag = _summarize_by_group(enriched_cases, benchmarks_by_id, "tags")
    by_language = _summarize_by_language(enriched_cases)

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
        "by_language": by_language,
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
    benchmarks_by_id = _collect_metadata(benchmarks)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    charts = _generate_visualizations(report, output_dir)
    report["charts"] = charts

    _attach_plot_previews(report["cases"], output_dir)

    report_paths: dict[str, str] = {}
    all_cases_path = _render_case_markdown(report["cases"], output_dir)
    report_paths["all"] = str(all_cases_path)

    languages = sorted({case.get("language", "unknown") for case in report["cases"]})
    for lang in languages:
        lang_cases = [case for case in report["cases"] if case.get("language") == lang]
        if not lang_cases:
            continue
        filename = f"cases_report_{lang}.md"
        title = f"Детальный отчёт по кейсам ({lang})"
        lang_path = _render_case_markdown(lang_cases, output_dir, filename=filename, title=title)
        report_paths[lang] = str(lang_path)

    report["case_report_path"] = report_paths

    cases_df = pd.DataFrame(report["cases"])
    cases_df["heuristic_score"] = cases_df["code_details"].apply(
        lambda x: x.get("heuristic_score") if isinstance(x, dict) else None
    )
    cases_df["exec_score"] = cases_df["code_details"].apply(
        lambda x: x.get("exec_score") if isinstance(x, dict) else None
    )
    cases_df["stdout_match"] = cases_df["code_details"].apply(
        lambda x: x.get("stdout_match") if isinstance(x, dict) else None
    )
    cases_df["result_match"] = cases_df["code_details"].apply(
        lambda x: x.get("result_match") if isinstance(x, dict) else None
    )
    cases_df = cases_df.drop(
        columns=[
            "model_generated_code",
            "response_message",
            "code_details",
            "semantic_details",
        ],
        errors="ignore",
    )
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
    # Пример запуска: формируем полный отчёт и сохраняем метрии и графики
    test_path = "core/evaluation/inference_results/eval_0_baseline.json"
    report = generate_report(test_path)

    print("Отчёт сформирован. Ключевые метрики:")
    print(json.dumps(report.get("summary", {}), ensure_ascii=False, indent=2))
