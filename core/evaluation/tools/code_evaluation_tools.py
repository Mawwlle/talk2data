# ---------- code parsing helpers ----------
import ast
import builtins
import contextlib
import importlib.util
import io
import math
import os
import signal
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from core.evaluation.constants import (
    RANDOM_SEED,
    SAFE_BUILTINS,
    SANDBOX_FILENAME,
    SANDBOX_TIMEOUT_SECONDS,
    SandboxResult,
)


def extract_imports(code: str | None) -> set[str]:
    """Extract imported modules from Python code."""
    if code is None:
        return set()
    
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


def extract_calls(code: str | None) -> set[str]:
    """Extract called function names from Python code."""
    if code is None:
        return set()
    
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


def _is_plotly_figure(value: Any) -> bool:
    """Check whether a value looks like a Plotly Figure without importing Plotly."""

    if value is None:
        return False
    value_type = type(value)
    return (
        value_type.__name__ == "Figure"
        and isinstance(value_type.__module__, str)
        and value_type.__module__.startswith("plotly")
    )


def _plotly_signature(fig: Any) -> list[tuple[str, tuple[str, ...]]]:
    """Build a lightweight signature list for Plotly traces."""

    if fig is None or not hasattr(fig, "to_plotly_json"):
        return []
    payload = fig.to_plotly_json()
    traces = payload.get("data", []) if isinstance(payload, dict) else []
    signatures: list[tuple[str, tuple[str, ...]]] = []
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        trace_type = str(trace.get("type", "unknown"))
        dims = trace.get("dimensions")
        if isinstance(dims, list):
            labels = tuple(
                str(item.get("label"))
                for item in dims
                if isinstance(item, dict) and item.get("label") is not None
            )
            signatures.append((trace_type, labels))
            continue
        x_values = trace.get("x")
        y_values = trace.get("y")
        signature_details = []
        if x_values is not None:
            signature_details.append(f"x:{len(x_values)}")
        if y_values is not None:
            signature_details.append(f"y:{len(y_values)}")
        signatures.append((trace_type, tuple(signature_details)))
    return signatures


def _compare_plotly_figures(expected: Any, actual: Any) -> float | None:
    """Return percentage of matching Plotly trace signatures."""

    expected_sigs = _plotly_signature(expected)
    actual_sigs = _plotly_signature(actual)
    total = max(len(expected_sigs), len(actual_sigs))
    if total == 0:
        return None
    expected_counts: dict[tuple[str, tuple[str, ...]], int] = {}
    for sig in expected_sigs:
        expected_counts[sig] = expected_counts.get(sig, 0) + 1
    matched = 0
    for sig in actual_sigs:
        if expected_counts.get(sig, 0) > 0:
            expected_counts[sig] -= 1
            matched += 1
    return round((matched / total), 3)


def compare_execution_results(expected: Any, actual: Any) -> tuple[bool, float | None]:
    """Compare execution results with tolerance for numerics and arrays."""

    # if _is_plotly_figure(expected) or _is_plotly_figure(actual):
    #     match_percent = _compare_plotly_figures(expected, actual)
    #     if match_percent is not None:
    #         return match_percent > 0, match_percent
    #     return (
    #         (_is_plotly_figure(expected) and actual is None)
    #         or (_is_plotly_figure(actual) and expected is None),
    #         0.0,
    #     )

    if isinstance(expected, (float, int)) and isinstance(actual, (float, int)):
        return (
            math.isclose(float(expected), float(actual), rel_tol=1e-6, abs_tol=1e-6),
            None,
        )

    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip() == actual.strip(), None

    if isinstance(expected, pd.DataFrame) and isinstance(actual, pd.DataFrame):
        return expected.equals(actual), None

    if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
        return np.allclose(expected, actual), None

    return expected == actual, None


# --- code execution ---


def _sandbox_globals() -> dict[str, Any]:
    """Prepare globals for sandboxed execution with restricted builtins."""

    dummy_builtins = {
        "__builtins__": {name: getattr(builtins, name) for name in SAFE_BUILTINS}
    }

    np.random.seed(RANDOM_SEED)

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()  # type: ignore
    df["species"] = df["target"].map(lambda idx: iris.target_names[idx])  # type: ignore
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
    }


def _extract_plotly_figure(code: str) -> Any | None:
    """Execute code and return a Plotly figure if one is created."""

    if importlib.util.find_spec("plotly") is None:
        return None
    import plotly  # noqa: WPS433
    import plotly.express as px  # noqa: WPS433
    import plotly.graph_objects as go  # noqa: WPS433
    import plotly.io as pio  # noqa: WPS433

    env = _sandbox_globals()

    def _no_show(*_: Any, **__: Any) -> None:  # noqa: ANN002,ANN003
        return None

    pio.show = _no_show  # type: ignore[assignment]
    env.update({"px": px, "go": go, "pio": pio, "plotly": plotly})

    try:
        parsed = ast.parse(code)
        if parsed.body and isinstance(parsed.body[-1], ast.Expr):
            parsed.body[-1] = ast.Assign(
                targets=[ast.Name(id="_plotly_last_expr", ctx=ast.Store())],
                value=parsed.body[-1].value,
            )
            ast.fix_missing_locations(parsed)
        exec(compile(parsed, SANDBOX_FILENAME, "exec"), env, env)
    except Exception:  # noqa: BLE001
        return None

    fig = env.get("fig")
    if not isinstance(fig, go.Figure):
        fig = env.get("_plotly_last_expr")
    if not isinstance(fig, go.Figure):
        fig = next(
            (value for value in env.values() if isinstance(value, go.Figure)),
            None,
        )
    return fig


@contextlib.contextmanager
def _enforce_timeout(seconds: int = SANDBOX_TIMEOUT_SECONDS):
    """Context manager that raises TimeoutError if block execution exceeds limit."""

    def _timeout_handler(signum, _):
        raise TimeoutError("sandbox_timeout")

    original_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)

    timer = threading.Timer(seconds, lambda: os.kill(os.getpid(), signal.SIGALRM))
    timer.start()

    try:
        yield
    finally:
        timer.cancel()
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


def _run_code_in_sandbox(code: str | None) -> SandboxResult:
    """Execute code safely with restricted globals and a hard timeout.

    The function parses user code, executes statements, and if the last node is
    an expression, evaluates it to produce a return value. Any exceptions are
    captured as error strings instead of propagating.
    """
    if code is None:
        return None, None

    sandbox_globals = _sandbox_globals()
    sandbox_locals: dict[str, Any] = {}

    try:
        parsed = ast.parse(code)
    except SyntaxError as exc:
        return "", None, f"syntax_error: {exc}"  # type: ignore

    stdout_buffer = io.StringIO()
    exec_error: str | None = None
    result_value: Any | None = None

    try:
        with _enforce_timeout():
            with contextlib.redirect_stdout(stdout_buffer):
                if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                    body_without_last = ast.Module(
                        body=parsed.body[:-1], type_ignores=[]
                    )
                    last_expr = ast.Expression(parsed.body[-1].value)

                    if body_without_last.body:
                        exec(
                            compile(body_without_last, SANDBOX_FILENAME, "exec"),
                            sandbox_globals,
                            sandbox_locals,
                        )

                    result_value = eval(
                        compile(last_expr, SANDBOX_FILENAME, "eval"),
                        sandbox_globals,
                        sandbox_locals,
                    )
                else:
                    exec(
                        compile(parsed, SANDBOX_FILENAME, "exec"),
                        sandbox_globals,
                        sandbox_locals,
                    )
    except TimeoutError as exc:
        exec_error = f"execution_timeout: {exc}"
    except Exception as exc:  # noqa: BLE001
        exec_error = f"execution_error: {exc}"

    return result_value, exec_error