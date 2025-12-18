import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from core.evaluation.constants import DEFAULT_EMBEDDING_MODEL

# --- text preprocessers and embeddings ---


def tokenize(text: str) -> list[str]:
    """Return simple whitespace-tokenization with punctuation handling."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return text.split()


@lru_cache(maxsize=1)
def _load_embedding_components(
    model_path: str = DEFAULT_EMBEDDING_MODEL,
) -> tuple[Any, Any]:
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


# --- text metrics ---


def counter_cosine_similarity(counter1: Counter[str], counter2: Counter[str]) -> float:
    """Return cosine similarity between two Counters."""
    terms = set(counter1.keys()).union(counter2.keys())
    mag1 = sum(counter1.get(k, 0) ** 2 for k in terms) ** 0.5
    mag2 = sum(counter2.get(k, 0) ** 2 for k in terms) ** 0.5
    if mag1 == 0 or mag2 == 0:
        return 0.0

    shared_keys = set(counter1.keys()) | set(counter2.keys())
    dot_product = sum(counter1.get(k, 0) * counter2.get(k, 0) for k in shared_keys)
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
        similarity = torch.cosine_similarity(
            fact_embedding.unsqueeze(0), text_embedding.unsqueeze(0)
        ).item()
        return float(similarity)
    except Exception:  # noqa: BLE001
        fact_tokens = Counter(tokenize(fact))
        text_tokens = Counter(tokenize(text))
        return counter_cosine_similarity(fact_tokens, text_tokens)
