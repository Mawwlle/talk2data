# models.py
import logging
from pathlib import Path

import environ
from transformers import AutoTokenizer
from vllm import LLM

from core.config import settings

logger = logging.getLogger(__name__)

# === Загрузка .env ===
env = environ.Env()
env.read_env(Path(__file__).resolve().parent.parent / ".env")

if env.bool("LOCAL_RUN", False):
    # Для локального облегчённоего запуска
    VLLM_CONFIG = dict(
        # max_model_len=4096,            # ↓ уменьшаем контекст, если оставить по умолчанию, то init engine займёт около 145 секунд
        gpu_memory_utilization=0.95,  # ↑ разрешаем использовать больше GPU-памяти
        enforce_eager=False,
        dtype="float16",
        tensor_parallel_size=1,  # for evaluation
    )
else:
    LOCAL_CONFIG = {}

_tokenizer = None
_llm = None


def get_tokenizer():
    global _tokenizer
    if not _tokenizer:
        logger.info("Загружаю токенайзер...")
        try:
            _tokenizer = AutoTokenizer.from_pretrained(
                settings.LLM_LOCAL_PATH,
                use_fast=True,
                padding_side="left",
                local_files_only=True,
            )
        except OSError:
            logger.warning("Local weights not found, trying to download from HF...")
            _tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_NAME)
            _tokenizer.save_pretrained(settings.LLM_LOCAL_PATH)
        if _tokenizer.pad_token is None:
            _tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        logger.info("Токенайзер загружен")
    return _tokenizer


def get_llm():
    global _llm
    if not _llm:
        logger.info("Загружаю модель...")
        _llm = LLM(
            model=settings.LLM_LOCAL_PATH,
            load_format=(
                settings.LLM_LOAD_FORMAT
                if hasattr(settings, "LLM_LOAD_FORMAT")
                else "auto"
            ),
            **VLLM_CONFIG,
        )
        logger.info("Модель загружена")
    return _llm
