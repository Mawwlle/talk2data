# models.py
import torch
import logging
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM
import environ

logger = logging.getLogger(__name__)

# === Загрузка .env ===
env = environ.Env()
env.read_env(Path(__file__).resolve().parent.parent / ".env")

# === Настройка кэша ===
CACHE_DIR = Path(env.str("TRANSFORMERS_CACHE", "./cache/huggingface"))
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["HF_HOME"] = str(CACHE_DIR)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# === Параметры модели ===
LLM_MODEL_NAME = env.str("LLM_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
DTYPE = getattr(torch, env.str("DTYPE", "float16"))
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

if env.bool("LOCAL_RUN", False):
    # Для локального облегчённоего запуска
    LOCAL_CONFIG = dict(
        max_model_len=4096,              # ↓ уменьшаем контекст
        gpu_memory_utilization=0.95,     # ↑ разрешаем использовать больше GPU-памяти
        enforce_eager=False
    )
else:
    LOCAL_CONFIG = {}

def is_model_and_tokenizer_cached(model_name: str, cache_dir: Path) -> bool:
    model_cache_path = cache_dir / f"models--{model_name.replace('/', '--')}"
    if not model_cache_path.exists():
        return False

    snapshots = list((model_cache_path / "snapshots").glob("*"))
    if not snapshots:
        return False
    snapshot_dir = snapshots[0]

    model_files = ["pytorch_model.bin", "model.safetensors"]
    if not any((snapshot_dir / f).exists() for f in model_files):
        return False

    tokenizer_files = ["tokenizer.json", "vocab.json", "merges.txt"]
    if not any((snapshot_dir / f).exists() for f in tokenizer_files):
        return False

    return True

is_cached = is_model_and_tokenizer_cached(LLM_MODEL_NAME, CACHE_DIR)
logger.info(f"Model cache check: {is_cached}, path={CACHE_DIR}")

_tokenizer = None
_llm = None

def get_tokenizer():
    global _tokenizer
    if not _tokenizer:
        logger.info("Загружаю токенайзер...")
        _tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME,
            use_fast=True,
            padding_side="left",
            cache_dir=CACHE_DIR,
            local_files_only=is_cached
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
            model=LLM_MODEL_NAME,
            dtype=DTYPE,
            download_dir=CACHE_DIR,
            **LOCAL_CONFIG
        )
        logger.info("Модель загружена")
    return _llm
