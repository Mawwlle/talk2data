# ---------- code parsing helpers ----------
import ast
import builtins
import contextlib
import io
import math
import os
import re
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


def compare_execution_results(expected: Any, actual: Any) -> bool:
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

class _DFMethodStub:
    """
    Callable stub returned for any valid DataFrame method.
    Calling it returns a DummyDataFrame again (chainable).
    """
    def __call__(self, *args, **kwargs):
        return _DummyDataFrame()

    def __getattr__(self, name: str):
        # allow chaining: df.groupby(...).mean().reset_index()
        return _DFMethodStub()


class _DummyDataFrame:
    """
    Fast, safe proxy that behaves like a *non-empty* pd.DataFrame
    for attribute/method existence checks.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        import pandas as pd  # local import, negligible cost

        # Attribute exists on real DataFrame → allow
        if hasattr(pd.DataFrame, name):
            return _DFMethodStub()

        # pandas exposes properties like .loc, .iloc, .columns, etc.
        # They are also attributes on DataFrame
        raise AttributeError(name)

    def __getitem__(self, key):
        # df["col"] → Series → allow chaining
        return _DFMethodStub()

    def __bool__(self):
        # DataFrame truthiness is forbidden in pandas, but
        # returning True avoids accidental crashes in conditions
        return True

    def __len__(self):
        # Non-empty
        return 1


_NAME_ERROR_RE = re.compile(r"name '([^']+)' is not defined")
_ATTR_ERROR_RE = re.compile(r"object has no attribute '([^']+)'")
_MODULE_ATTR_ERROR_RE = re.compile(
    r"module\s+'[^']+'\s+has\s+no\s+attribute\s+'([^']+)'"
)
_IMPORT_ERROR_RE = re.compile(r"No module named '([^']+)'")


class CodeValidator:
    def __init__(self, code: str | None):
        self.code: str = code or ""
        self.errors: list[str] = []
        self._bad_words: list[str] = []
        self._checked: bool = False

    def fast_check_runs(self) -> None:
        """
        Fast runtime executability check.

        - compiles code
        - executes it in empty namespace
        - collects *symbol names* that caused failure
        """
        if self._checked:
            return

        self._checked = True
        error = None

        try:
            compiled = compile(
                self.code,
                "<generated>",
                "exec",
                dont_inherit=True,
                optimize=2,
            )

            globals_ns = {
                "df": _DummyDataFrame(),
            }
            locals_ns = {}

            exec(compiled, globals_ns, locals_ns)

        except NameError as e:
            error = type(e).__name__
            name = self._extract_name(_NAME_ERROR_RE, str(e))
            self._bad_words += self.to_bad_words(name or "UNKNOWN_NAME")

        except AttributeError as e:
            error = type(e).__name__
            name = self._extract_name(_ATTR_ERROR_RE, str(e))
            if not name:
                name = self._extract_name(_MODULE_ATTR_ERROR_RE, str(e))
            
            self._bad_words += self.to_bad_words(name or "UNKNOWN_ATTRIBUTE")

        except ImportError as e:
            error = type(e).__name__
            name = self._extract_name(_IMPORT_ERROR_RE, str(e))
            self._bad_words += self.to_bad_words(name or "UNKNOWN_IMPORT")

        except Exception as e:
            error = type(e).__name__
            return
            # fallback: keep exception class only if we cannot extract a name
        
        if error:
            self.errors.append(error)

    @staticmethod
    def _extract_name(regex: re.Pattern, message: str) -> str | None:
        m = regex.search(message)
        return m.group(1) if m else None
    
    def to_bad_words(self, symbol: str) -> list[str]:
        return [
            symbol,
            f"{symbol}(",
            f".{symbol.split('.')[-1]}",
        ]

    @property
    def bad_words(self) -> list[str]:
        self.fast_check_runs()
        return self._bad_words