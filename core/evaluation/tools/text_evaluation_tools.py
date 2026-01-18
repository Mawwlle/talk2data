import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from core.evaluation.constants import DEFAULT_EMBEDDING_MODEL

_EMBEDDING_AVAILABLE: bool | None = None

# --- text preprocessers and embeddings ---


@lru_cache(maxsize=4096)
def tokenize(text: str) -> list[str]:
    """Return simple whitespace-tokenization with punctuation handling."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return text.split()


@lru_cache(maxsize=1)
def _load_embedding_components(
    model_path: str = DEFAULT_EMBEDDING_MODEL,
) -> tuple[Any, Any]:
    """Load tokenizer and model for embedding computation."""

    global _EMBEDDING_AVAILABLE

    if _EMBEDDING_AVAILABLE is False:
        raise RuntimeError("Embedding model is unavailable.")

    resolved_path = Path(model_path).expanduser()
    if not resolved_path.exists():
        _EMBEDDING_AVAILABLE = False
        raise FileNotFoundError(f"Embedding model not found at {resolved_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(resolved_path), local_files_only=True)
        model = AutoModel.from_pretrained(str(resolved_path), local_files_only=True)
        model.eval()
    except Exception:  # noqa: BLE001
        _EMBEDDING_AVAILABLE = False
        raise

    _EMBEDDING_AVAILABLE = True
    return tokenizer, model


@lru_cache(maxsize=1024)
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
        sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=1)

    return sentence_embedding.squeeze(0)


# --- text metrics ---


def counter_cosine_similarity(counter1: Counter[str], counter2: Counter[str]) -> float:
    """Return cosine similarity between two Counters."""
    if not counter1 or not counter2:
        return 0.0

    common_keys = counter1.keys() & counter2.keys()
    dot_product = sum(counter1[k] * counter2[k] for k in common_keys)
    norm_a = sum(v * v for v in counter1.values()) ** 0.5
    norm_b = sum(v * v for v in counter2.values()) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def fact_presence_score(fact: str, text: str) -> float:
    """Return semantic presence score for a fact in text (0-1)."""

    try:
        fact_embedding = _compute_embedding(fact)
        text_embedding = _compute_embedding(text)
        similarity = torch.cosine_similarity(fact_embedding.unsqueeze(0), text_embedding.unsqueeze(0)).item()
        return float(similarity)
    except Exception:  # noqa: BLE001
        fact_tokens = Counter(tokenize(fact))
        text_tokens = Counter(tokenize(text))
        return counter_cosine_similarity(fact_tokens, text_tokens)
