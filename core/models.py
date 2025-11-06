# models.py
import torch
import logging
import settings
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM
import environ

logger = logging.getLogger(__name__)

# === Загрузка .env ===
env = environ.Env()
env.read_env(Path(__file__).resolve().parent.parent / ".env")


device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

if env.bool("LOCAL_RUN", False):
    # Для локального облегчённоего запуска
    LOCAL_CONFIG = dict(
        # max_model_len=4096,            # ↓ уменьшаем контекст, если оставить по умолчанию, то init engine займёт около 145 секунд
        gpu_memory_utilization=0.95,     # ↑ разрешаем использовать больше GPU-памяти
        enforce_eager=False,
        dtype="float16",
    )
else:
    LOCAL_CONFIG = {}

_tokenizer = None
_llm = None

def get_tokenizer():
    global _tokenizer
    if not _tokenizer:
        logger.info("Загружаю токенайзер...")
        _tokenizer = AutoTokenizer.from_pretrained(
            settings.LLM_LOCAL_PATH,
            use_fast=True,
            padding_side="left",
            local_files_only=True,
        )
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
            load_format="safetensors", # if not is_cached else "npcache",  или "pt"
            **LOCAL_CONFIG
        )
        logger.info("Модель загружена")
    return _llm
