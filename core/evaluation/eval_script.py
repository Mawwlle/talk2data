from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from torch.nn.functional import cosine_similarity

from core.evaluation.constants import TEST_RESULT_PATH
from core.evaluation.inference_script import BENCHMARKS_DIR, load_json
from core.evaluation.tools.code_evaluation_tools import compare_execution_results, _run_code_in_sandbox, extract_calls, extract_imports
from core.evaluation.tools.report_helpers import attach_plot_previews, generate_visualizations_data, _mean, render_case_markdown, summarize_by_group, summarize_by_language
from core.evaluation.tools.text_evaluation_tools import _compute_embedding, fact_presence_score, counter_cosine_similarity, tokenize

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------
def validate_list(results: list[dict[str, Any]] | dict[str, Any] | None) -> list[dict[str, Any]]:
    """Normalize inference results to a list of dictionaries."""

    if isinstance(results, list):
        return results
    if isinstance(results, dict) and "results" in results:
        return results.get("results", [])
    return []

def _collect_benchmark_metadata(benchmarks: list[dict[str, Any]]) -> dict[tuple[str | None, str | None], dict[str, Any]]:
    """Collect metadata by (id, language) for quick lookup."""

    meta: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    for case in benchmarks:
        key = (case.get("id"), case.get("metadata", {}).get("language"))
        meta[key] = case
    return meta

def _infer_language_from_filename(path: Path) -> str:
    """Infer language from filename suffix: *_en.json or *_ru.json."""

    if path.name.endswith("_ru.json"):
        return "ru"
    if path.name.endswith("_en.json"):
        return "en"
    return "unknown"

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
    expected_result, expected_error = _run_code_in_sandbox(expected_code)
    model_result, model_error = _run_code_in_sandbox(model_code)

    result_match = (
        expected_error is None
        and model_error is None
        and compare_execution_results(expected_result, model_result)
    )

    combined_score = round(
        min((heuristic_score * 0.5) + (0.5 if result_match else 0.0), 1.0), 3
    )

    return {
        "score": combined_score,
        "heuristic_score": round(min(heuristic_score, 1.0), 3),
        "result_match": result_match,
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
    inference_res_path: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run evaluation comparing inference results to benchmarks and baseline."""
    # 1. inference format check
    outputs = validate_list(load_json(Path(inference_res_path)))
    output_by_id = {item.get("benchmark_id"): item for item in outputs}
    # baseline_by_id = {item.get("benchmark_id"): item for item in baseline_outputs}

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

        decision_score = evaluate_decision(
            model_output.get("model_decision", {}).get("action"), case["expected_decision"]
        )

        # a - проверяем сгенерённый текст
        semantic_similarity = None
        semantic_details = None
        if case.get("expected_decision") == "chat_response":

            semantic_details = evaluate_chat_semantics(
                case.get("expected_facts"),
                case.get("forbidden_facts"),
                model_output.get("response_message"),
            )
            semantic_similarity = semantic_details.get("score")

        # б - проверяем сгенерённый код
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
    output_dir: str | Path = "core/evaluation/reports",
) -> dict[str, Any]:
    """Generate evaluation report files and return the structured report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Запускаем Evaluation 
    results, benchmarks = run_eval(inference_res_path)
    
    # 2. Формируем данные для отчёта
    report = build_report_data(results, benchmarks)

    # 3. Добавляем графики
    charts = generate_visualizations_data(report, output_dir)
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
            "code_details",
            "semantic_details",
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

    return report


if __name__ == "__main__":
    # Пример запуска: формируем полный отчёт и сохраняем метрии и графики
    report = generate_report(TEST_RESULT_PATH)

    print("Отчёт сформирован. Ключевые метрики:")
    print(report.get("summary", {}))
