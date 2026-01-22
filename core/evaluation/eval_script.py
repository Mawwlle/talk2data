from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from torch.nn.functional import cosine_similarity

from core.evaluation.constants import REPORT_OUTPUT_DIR, TEST_RESULT_PATH
from core.evaluation.io_utils import BENCHMARKS_DIR, load_json
from core.evaluation.tools.code_evaluation_tools import (
    _run_code_in_sandbox,
    extract_calls,
    extract_imports,
)
from core.evaluation.tools.report_helpers import (
    _mean,
    attach_plot_previews,
    generate_visualizations_data,
    render_case_markdown,
    summarize_by_group,
    summarize_by_language,
)
from core.evaluation.tools.text_evaluation_tools import (
    _compute_embedding,
    fact_presence_score,
)


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------
def validate_list(
    results: list[dict[str, Any]] | dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Normalize inference results to a list of dictionaries."""

    if isinstance(results, list):
        return results
    if isinstance(results, dict) and "results" in results:
        return results.get("results", [])
    return []


def _collect_benchmark_metadata(
    benchmarks: list[dict[str, Any]],
) -> dict[tuple[str | None, str | None], dict[str, Any]]:
    """Collect metadata by (id, language) for quick lookup."""

    meta: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    for case in benchmarks:
        key = (case.get("id"), case.get("metadata", {}).get("language"))
        meta[key] = case
    return meta


@lru_cache(maxsize=256)
def _infer_language_from_filename(path: Path) -> str:
    """Infer language from filename suffix: *_en.json or *_ru.json."""

    if path.name.endswith("_ru.json"):
        return "ru"
    if path.name.endswith("_en.json"):
        return "en"
    return "unknown"


def _has_plotly_calls(code: str | None) -> bool:
    """Detect Plotly usage to allow implicit display calls."""
    if code is None:
        return False

    lowered = code.lower()
    return "plotly" in lowered or "px." in lowered or "go." in lowered


def _has_possible_code_object(tree: ast.AST, possible_code_objects: list[str] | None) -> bool:
    if not possible_code_objects:
        return False
    allowed = set(possible_code_objects)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in allowed:
                return True
            if isinstance(func, ast.Attribute) and func.attr in allowed:
                return True
        if isinstance(node, ast.Name) and node.id in allowed:
            return True
        if isinstance(node, ast.Attribute) and node.attr in allowed:
            return True
    return False


def evaluate_object_match(
    model_code: str | None,
    possible_code_objects: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate code against AST-based requirements."""
    if model_code is None:
        return {
            "score": 0.0,
            "code_objects_match": False,
            "errors": {"model": "code not parsed"},
        }

    try:
        tree = ast.parse(model_code)
    except SyntaxError:
        return {
            "score": 0.0,
            "code_objects_match": False,
            "errors": {"model": "syntax_error"},
        }

    object_match = _has_possible_code_object(tree, possible_code_objects)

    return {
        "score": 0.4 if object_match else 0.0,
        "code_objects_match": object_match,
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
                message = f"[benchmarks] duplicate id '{case_id}' for language {language} " f"in {path.name} ignored"
                print(message)
                continue

            seen.add(key)
            meta = {**case.get("metadata", {}), "language": language}
            benchmarks.append({**case, "metadata": meta})

    return benchmarks


# --- Evaluations ---


def evaluate_decision(model_decision: Any, expected: Any) -> bool:
    """Return whether the model decision matches the expected decision."""

    return model_decision == expected


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

        similarity = cosine_similarity(baseline_embedding, generated_embedding, dim=0).item()
        return round(similarity, 4)
    except Exception:
        return None


def evaluate_chat_semantics(
    expected_facts: list[str] | None,
    forbidden_facts: list[str] | None,
    generated_text: str | None,
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
        score = fact_presence_score(fact, generated_text)
        if score >= presence_threshold:
            expected_hit_scores.append((fact, round(score, 3)))

    for fact in forbidden_facts:
        score = fact_presence_score(fact, generated_text)
        if score >= presence_threshold:
            forbidden_hit_scores.append((fact, round(score, 3)))

    coverage = len(expected_hit_scores) / len(expected_facts) if expected_facts else 1.0
    penalty = len(forbidden_hit_scores) / len(forbidden_facts) if forbidden_facts else 0.0

    reference_text = ". ".join(expected_facts)
    similarity = evaluate_text_similarity(reference_text, generated_text)
    similarity = similarity if similarity is not None else coverage

    raw_score = (similarity * 0.6) + (coverage * 0.4) - (penalty * 0.5)
    score = round(max(0.0, min(raw_score, 1.0)), 3)

    return {
        "score": score,
        "similarity": similarity,
        "expected_coverage": round(coverage, 3),
        "forbidden_penalty": round(penalty, 3),
        "expected_hits": expected_hit_scores,
        "forbidden_hits": forbidden_hit_scores,
    }


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------
def evaluate_code(
    model_code: str | None,
    benchmark: str | None,
    possible_code_objects: list[str] | None = None,
) -> dict[str, Any]:
    """Score generated code using syntax, heuristic, and execution signals."""

    heuristic_score = 0.0
    syntax_score = 0.0
    import_score = 0.0
    call_score = 0.0
    object_score = 0.0
    heuristic_breakdown: dict[str, Any] = {
        "syntax": {"score": 0.0, "ok": False},
        "imports": {"score": 0.0, "expected": [], "model": [], "matched": []},
        "calls": {"score": 0.0, "expected": [], "model": [], "matched": []},
        "objects": {
            "score": 0.0,
            "possible": possible_code_objects or [],
            "matched": False,
        },
    }

    # 1️⃣ Syntax check: fail fast on invalid Python
    try:
        if model_code:
            ast.parse(model_code)
            syntax_score = 0.3
            heuristic_score += syntax_score
    except SyntaxError:
        return {
            "score": 0.0,
            "heuristic_score": 0.0,
            "result_match": False,
            "heuristic_breakdown": heuristic_breakdown,
            "errors": {
                "model": "syntax_error",
                "expected": None,
            },
        }

    expected_code = benchmark or ""

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
    adjusted_model_calls = set(model_calls)
    if "show" in expected_calls and "show" not in model_calls and _has_plotly_calls(model_code):
        adjusted_model_calls.add("show")

    if expected_calls:
        matched = expected_calls & adjusted_model_calls
        call_score = 0.4 * (len(matched) / len(expected_calls))
    else:
        matched = set()
        call_score = 0.0

    heuristic_score += call_score
    heuristic_breakdown["calls"] = {
        "score": round(call_score, 3),
        "expected": sorted(expected_calls),
        "model": sorted(adjusted_model_calls),
        "matched": sorted(matched),
    }

    # 4️⃣ Optional object match bonus
    object_match_details = evaluate_object_match(model_code, possible_code_objects)
    object_score = object_match_details.get("score", 0.0)

    # ---------- execution-based scoring ----------
    expected_result, expected_error = _run_code_in_sandbox(expected_code)
    model_result, model_error = _run_code_in_sandbox(model_code)

    max_heuristic_score = 0.9
    normalized_heuristic = heuristic_score / max_heuristic_score if max_heuristic_score else 0.0
    normalized_heuristic = min(max(normalized_heuristic, 0.0), 1.0)
    normalization_factor = 1.0 / max_heuristic_score if max_heuristic_score else 0.0
    heuristic_breakdown["syntax"] = {
        "score": round(syntax_score * normalization_factor, 3),
        "ok": True,
    }
    heuristic_breakdown["imports"]["score"] = round(import_score * normalization_factor, 3)
    heuristic_breakdown["calls"]["score"] = round(call_score * normalization_factor, 3)
    heuristic_breakdown["objects"]["score"] = round(object_score * normalization_factor, 3)

    return {
        "heuristic_score": round(normalized_heuristic, 3),
        "heuristic_breakdown": heuristic_breakdown,
        "errors": {
            "expected": expected_error,
            "model": model_error,
        },
        "results": {
            "expected": expected_result,
            "model": model_result,
        },
    }


def run_eval(
    inference_res_path: str, decision_only: bool = False, code_check: bool = True
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run evaluation comparing inference results to benchmarks and baseline."""
    # 1. inference format check
    outputs = validate_list(load_json(Path(inference_res_path)))
    output_by_id = {item.get("benchmark_id"): item for item in outputs}

    # 2. Загружаем все бэнчмарки с ground truth
    benchmarks = _load_all_benchmarks()
    total_cases = len(benchmarks)

    results: list[dict[str, Any]] = []

    # 3. Формируем начальную инфу обо всех кейсах
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

        decision_score = 0
        if model_output.get("model_decision", {}):
            decision_score = evaluate_decision(
                model_output.get("model_decision"),
                case["expected_decision"],
            )

        code_score_details = None
        if case.get("expected_decision") == "code_generation" and not decision_only and code_check:
            possible_code_objects = case.get("possible_code_objects")
            code_score_details = evaluate_code(
                model_output.get("generated_code", ""),
                case.get("expected_code"),
                possible_code_objects=possible_code_objects,
            )

        results.append(
            {
                "id": case["id"],
                "expected_decision": case.get("expected_decision"),
                "language": language,
                "decision_score": decision_score,
                "code_details": code_score_details,
                "model_generated_code": model_output.get("generated_code"),
                "response_message": model_output.get("response_message"),
            }
        )
    return results, benchmarks


def build_report_data(results: list[dict[str, Any]], benchmarks: list[dict[str, Any]]) -> dict[str, Any]:
    """Construct aggregated report data from evaluation results."""

    benchmarks_by_id = _collect_benchmark_metadata(benchmarks)
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
                "possible_code_objects": meta.get("possible_code_objects"),
            }
        )

    summary = {
        "cases_total": len(enriched_cases),
        "missing": sum(1 for case in enriched_cases if case.get("error")),
        "decision_accuracy": _mean(case.get("decision_score") for case in enriched_cases if not case.get("error")),
    }

    by_difficulty = summarize_by_group(enriched_cases, benchmarks_by_id, "difficulty")
    by_language = summarize_by_language(enriched_cases)

    return {
        "summary": summary,
        "by_difficulty": by_difficulty,
        "by_language": by_language,
        "cases": enriched_cases,
    }


def generate_report(
    inference_res_path: str,
    output_dir: str | Path = REPORT_OUTPUT_DIR,
    decision_only: bool = False,
    code_check: bool = True,
) -> dict[str, Any]:
    """Generate evaluation report files and return the structured report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Запускаем Evaluation
    results, benchmarks = run_eval(inference_res_path, decision_only, code_check)

    # 2. Формируем данные для отчёта
    report = build_report_data(results, benchmarks)

    # 3. Добавляем графики
    if not decision_only and code_check:
        charts = generate_visualizations_data(
            report,
            output_dir,
        )
        report["charts"] = charts
        attach_plot_previews(report["cases"], output_dir)

    # 4. Сохраняем отдельно отчёт для русскоязычных и англоязычных инпутов
    report_paths: dict[str, str] = {}
    languages = sorted({case.get("language", "unknown") for case in report["cases"]})
    for lang in languages:
        lang_cases = [case for case in report["cases"] if case.get("language") == lang]
        if not lang_cases:
            continue
        filename = f"cases_report_{lang}.md"
        title = f"Детальный отчёт по кейсам ({lang})"
        lang_path = render_case_markdown(lang_cases, output_dir, filename=filename, title=title)
        report_paths[lang] = str(lang_path)
    report["case_report_path"] = report_paths

    # 5. Формируем csv-отчёт по всем кейсам
    cases_df = pd.DataFrame(report["cases"])
    cases_df["heuristic_score"] = cases_df["code_details"].apply(
        lambda x: x.get("heuristic_score") if isinstance(x, dict) else None
    )
    cases_df["result_match"] = cases_df["code_details"].apply(
        lambda x: x.get("result_match") if isinstance(x, dict) else None
    )
    cases_df = cases_df.drop(
        columns=[
            "model_generated_code",
            "response_message",
        ],
        errors="ignore",
    )
    cases_df.to_csv(output_dir / "cases.csv", index=False)

    # 6. Формируем csv-отчёт по агрегированным метрикам
    summary_df = pd.DataFrame(
        [
            {
                "metric": "decision_accuracy",
                "value": report["summary"].get("decision_accuracy", 0.0),
            },
        ]
    )
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    return report


if __name__ == "__main__":
    # формируем полный отчёт и сохраняем метрии и графики
    report = generate_report(
        TEST_RESULT_PATH, code_check=False
    )  # , decision_only=True если хочется проверить только accuracy для decision

    print("Отчёт сформирован. Ключевые метрики:")
    print(report.get("summary", {}))
