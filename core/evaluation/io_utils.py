from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

BENCHMARKS_DIR = Path("core/benchmarks")


@lru_cache(maxsize=128)
def load_json(path: Path) -> list[dict[str, Any]]:
    """Load JSON from disk with basic caching to avoid repeated reads."""
    text = path.read_text(encoding="utf-8")
    return json.loads(text)
