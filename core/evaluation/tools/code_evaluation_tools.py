# ---------- code parsing helpers ----------
import ast
import builtins
import contextlib
import io
import os
import signal
import threading
from functools import lru_cache
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


@lru_cache(maxsize=1024)
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


@lru_cache(maxsize=1024)
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
                if not parsed.body or not isinstance(parsed.body[-1], ast.Expr):
                    exec(
                        compile(parsed, SANDBOX_FILENAME, "exec"),
                        sandbox_globals,
                        sandbox_locals,
                    )
                else:
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
    except TimeoutError as exc:
        exec_error = f"execution_timeout: {exc}"
    except Exception as exc:  # noqa: BLE001
        exec_error = f"execution_error: {exc}"

    return result_value, exec_error
