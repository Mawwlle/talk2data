import ast
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from core.evaluation.constants import SANDBOX_FILENAME
from core.evaluation.tools.code_evaluation_tools import _sandbox_globals

# --- aggregation ---


def _mean(values: list[float | None]) -> float:
    """Compute a rounded mean ignoring non-numeric entries."""

    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    if not numeric:
        return 0.0
    return round(sum(numeric) / len(numeric), 3)


def summarize_by_group(
    cases: list[dict[str, Any]],
    benchmarks_by_id: dict[tuple[str | None, str | None], dict[str, Any]],
    group_key: str,
) -> dict[str, dict[str, float]]:
    """Summarize metrics by a given metadata group (e.g., difficulty or tag)."""

    summary: dict[str, dict[str, Any]] = {}
    for case in cases:
        meta_key = (case.get("id"), case.get("language", "unknown"))
        meta = benchmarks_by_id.get(meta_key, {})
        values = meta.get("metadata", {}).get(group_key, [])
        if not isinstance(values, list):
            values = [values]

        for value in values:
            if value is None:
                continue
            bucket = summary.setdefault(
                value,
                {"count": 0, "decision": [], "semantic_similarity": [], "code": []},
            )
            if not case.get("error"):
                bucket["count"] += 1
                bucket["decision"].append(case.get("decision_score"))
                bucket["semantic_similarity"].append(case.get("semantic_similarity"))
                bucket["code"].append(case.get("heuristic_score"))

    for value, bucket in summary.items():
        bucket["decision_avg"] = _mean(bucket.pop("decision"))
        bucket["semantic_similarity_avg"] = _mean(bucket.pop("semantic_similarity"))
        bucket["code_avg"] = _mean(bucket.pop("code"))

    return summary


def summarize_by_language(
    cases: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Summarize metrics by language."""

    summary: dict[str, dict[str, Any]] = {}
    for case in cases:
        lang = case.get("language", "unknown")
        bucket = summary.setdefault(
            lang,
            {"count": 0, "decision": [], "semantic_similarity": [], "code": []},
        )
        if not case.get("error"):
            bucket["count"] += 1
            bucket["decision"].append(case.get("decision_score"))
            bucket["semantic_similarity"].append(case.get("semantic_similarity"))
            
            code_score = case.get('code_details', {}).get('heuristic_score') if case.get('code_details', {}) else None
            bucket["code"].append(code_score)

    for lang, bucket in summary.items():
        bucket["decision_avg"] = _mean(bucket.pop("decision"))
        bucket["semantic_similarity_avg"] = _mean(bucket.pop("semantic_similarity"))
        bucket["code_avg"] = _mean(bucket.pop("code"))

    return summary


def _aggregate_code_metrics(cases: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate code-generation metrics across cases."""

    code_cases = [
        case
        for case in cases
        if case.get("expected_decision") == "code_generation" and not case.get("error")
    ]
    if not code_cases:
        return {}

    heuristics = [
        case.get("code_details", {}).get("heuristic_score") for case in code_cases
    ]
    result_matches = [
        1.0 if case.get("code_details", {}).get("result_match") else 0.0
        for case in code_cases
    ]
    code_scores = [case.get("code_score") for case in code_cases]

    return {
        "code_score": _mean(code_scores),
        "heuristic_score": _mean(heuristics),
        "result_match_rate": _mean(result_matches),  # type: ignore
    }


# --- visualization helpers ---


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


def _save_plot(
    values: dict[str, float], title: str, ylabel: str, output_path: Path
) -> str:
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


def generate_visualizations_data(
    report: dict[str, Any], output_dir: Path
) -> dict[str, str]:
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

    language_summary = report.get("by_language", {})
    if language_summary:
        charts["language_comparison"] = _save_language_comparison(
            language_summary,
            chart_dir / "language_comparison.png",
        )

    return charts


def _describe_visual_output(code_text: str) -> str:
    """Provide a readable hint for visual outputs without raw JSON dumps."""

    lowered = code_text.lower()
    if "plotly" in lowered or "px." in lowered:
        return "Интерактивный график (Plotly)"
    if "plt." in lowered or "matplotlib" in lowered:
        return "Статический график (matplotlib)"
    return "Графический вывод"


def _rel_image_path(image_path: str | None, report_path: Path) -> str:
    """Render a stable relative path for images in markdown reports."""

    if not image_path:
        return ""

    resolved_path = Path(image_path)
    try:
        return resolved_path.relative_to(report_path.parent).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def _render_plotly_image(
    code_text: str, output_path: Path
) -> tuple[str | None, str | None]:
    """Execute Plotly code and export figure to PNG."""

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
        parsed = ast.parse(code_text)
        if parsed.body and isinstance(parsed.body[-1], ast.Expr):
            parsed.body[-1] = ast.Assign(
                targets=[ast.Name(id="_plotly_last_expr", ctx=ast.Store())],
                value=parsed.body[-1].value,
            )
            ast.fix_missing_locations(parsed)
        exec(compile(parsed, SANDBOX_FILENAME, "exec"), env, env)

        fig = env.get("fig")
        if not isinstance(fig, go.Figure):
            fig = env.get("_plotly_last_expr")
        if not isinstance(fig, go.Figure):
            fig = next(
                (value for value in env.values() if isinstance(value, go.Figure)),
                None,
            )
        if fig is None:
            return None, "no Plotly figure was created"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(output_path))
        return str(output_path), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def render_case_markdown(
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
    lines.append("### Semantic similarity (только для chat_response)")
    lines.append("Формула:")
    lines.append("```text")
    lines.append(
        "semantic_similarity = 0.6 * similarity + 0.4 * expected_coverage - 0.5 * forbidden_penalty"
    )
    lines.append("```")
    lines.append(
        "Где доли:\n - expected_coverage — доля ожидаемых фактов, упомянутых в ответе;\n - forbidden_penalty — доля запрещённых фактов,"
    )
    lines.append(
        "попавших в ответ (штраф).\n - similarity — embedding-cosine между эталонным ответом (конкатенация expected_facts или базовый"
    )
    lines.append("референс) и ответом модели.")
    lines.append(
        "\n\nФакт считается покрытым, если косинусная близость fact↔ответ ≥ 0.55"
    )
    lines.append(
        "(или высокая токеновая схожесть), что позволяет засчитывать перефраз. Весами (0.6/0.4/0.5) балансируем близость текста и полноту фактов,"
    )
    lines.append(
        "давая штраф за запрещённые факты, чтобы сохранить интерпретируемость (веса суммарно ограничивают метрику в [0, 1])."
    )
    lines.append("")
    lines.append("### Code score (только для code_generation)")
    lines.append(
        "- heuristic_score = 0.3 (валидный синтаксис) + до 0.2 (совпадение импортов) + до 0.4 (совпадение ключевых вызовов)."
    )
    lines.append(
        "- result_match: бинарный флаг (1.0/0.0), что итоговый результат выполнения совпал с эталоном без ошибок исполнения."
    )
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

        lines.append("\n\n**Ground truth:**")
        if expected_facts:
            lines.append("- expected_facts: " + "; ".join(expected_facts))
        if forbidden_facts:
            lines.append("- forbidden_facts: " + "; ".join(forbidden_facts))
        if case.get("expected_code"):
            lines.append("- expected_code:")
            lines.append("```python")
            lines.append(case.get("expected_code"))  # type: ignore
            lines.append("```")
            possible_code_objects = case.get("possible_code_objects") or []
            if possible_code_objects:
                lines.append(
                    "- possible_code_objects: " + ", ".join(possible_code_objects)
                )
            details = case.get("code_details") or {}
            lines.append("- expected_result:\n ")
            lines.append("```python")
            expected_result = details.get("results", {}).get("expected")
            lines.append(str(expected_result))
            lines.append("```")
            if details.get("results", {}).get("expected") is None:
                lines.append(
                    "- expected_output: "
                    + _describe_visual_output(case.get("expected_code", ""))
                )
            lines.append("- expected_error:\n ")
            lines.append("```python")
            expected_error = details.get("errors", {}).get("expected")
            lines.append(str(expected_error))
            lines.append("```")
            if case.get("expected_plot_path"):
                lines.append(
                    f"- expected_plot:\n\n ![expected plot]({_rel_image_path(case.get('expected_plot_path'), report_path)})"
                )
            elif case.get("expected_plot_error"):
                lines.append(
                    f"- expected_plot_error: {case.get('expected_plot_error')}"
                )

        lines.append("\n\n**Model output:**")
        if case.get("response_message"):
            lines.append("> " + case.get("response_message").replace("\n", " "))  # type: ignore
        if case.get("model_generated_code"):
            lines.append("```python")
            lines.append(case.get("model_generated_code"))  # type: ignore
            lines.append("```")
            details = case.get("code_details") or {}
            lines.append("- model_result:\n ")
            lines.append("```python")
            model_result = details.get("results", {}).get("model")
            lines.append(str(model_result))
            lines.append("```")
            lines.append("- model_error:\n ")
            lines.append("```python")
            model_error = details.get("errors", {}).get("model")
            lines.append(str(model_error))
            lines.append("```")
            if case.get("model_plot_path"):
                lines.append(
                    f"- model_plot:\n\n ![model plot]({_rel_image_path(case.get('model_plot_path'), report_path)})"
                )
            elif case.get("model_plot_error"):
                lines.append(f"- model_plot_error: {case.get('model_plot_error')}")

        lines.append("\n\n**Метрики:**")
        lines.append("- decision_score: " + str(case.get("decision_score")))

        if case.get("expected_decision") == "chat_response":
            semantic_details = case.get("semantic_details") or {}
            lines.append(f"- semantic_similarity: {semantic_details.get('score')}")
            lines.append(
                f"  - expected_coverage: {semantic_details.get('expected_coverage')} | forbidden_penalty: {semantic_details.get('forbidden_penalty')}"
            )
            if semantic_details.get("expected_hits"):
                formatted = [
                    f"{fact} (score {score})"
                    for fact, score in semantic_details.get("expected_hits", [])
                ]
                lines.append("  - покрытые факты: " + "; ".join(formatted))
            if semantic_details.get("forbidden_hits"):
                formatted = [
                    f"{fact} (score {score})"
                    for fact, score in semantic_details.get("forbidden_hits", [])
                ]
                lines.append(
                    "  - упомянутые запрещённые факты: " + "; ".join(formatted)
                )

        if case.get("expected_decision") == "code_generation":
            details = case.get("code_details") or {}
            if "heuristic_score" in details:
                lines.append(
                    "  - heuristic_score: " + str(details.get("heuristic_score"))
                )
            if details.get("heuristic_breakdown"):
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
            if "result_match" in details:
                lines.append("  - result_match: " + str(details.get("result_match")))
            if details.get("requirements"):
                lines.append(
                    "  - requirements_match: "
                    + str(details.get("requirements_match"))
                )
                for requirement in details.get("requirements", []):
                    lines.append(
                        "    - "
                        + str(requirement.get("description") or requirement.get("id"))
                        + ": "
                        + str(requirement.get("ok"))
                    )

        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def attach_plot_previews(cases: list[dict[str, Any]], output_dir: Path) -> None:
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
