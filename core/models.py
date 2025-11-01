# models.py
import torch
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM
from pathlib import Path
import environ
import logging
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
# from TTS.api import TTS
# import whisper
env = environ.Env()
env_path = Path(__file__).resolve().parent.parent / ".env"
env.read_env(env_path)

# Настроим локальный кэш (если не указан явно — используем ./cache/huggingface)
# Это необходимо для более быстрого повторного запуска
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "./cache/huggingface")
os.environ["HF_HOME"] = os.getenv("HF_HOME", "./cache/huggingface")

# Создадим каталог, если его нет
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

CACHE_DIR = Path(env.str("TRANSFORMERS_CACHE", "./cache/huggingface"))
LLM_MODEL_NAME = env.str("LLM_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
DTYPE = env.str("DTYPE", "float16")

if env.bool("LOCAL_RUN", False):
    # Для локального облегчённоего запуска
    LOCAL_CONFIG = dict(
        max_model_len=4096,              # ↓ уменьшаем контекст
        gpu_memory_utilization=0.95,     # ↑ разрешаем использовать больше GPU-памяти
        enforce_eager=False
    )
else:
    LOCAL_CONFIG = {}
    
# Создаём структуру папок
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["HF_HOME"] = str(CACHE_DIR)
model_path = CACHE_DIR / f"models--{LLM_MODEL_NAME.replace('/', '--')}" 

import os
from pathlib import Path

def is_model_and_tokenizer_cached(model_name: str, cache_dir: Path) -> bool:
    # Путь к локальному кэшу модели (формат huggingface hub)
    model_cache_path = cache_dir / f"models--{model_name.replace('/', '--')}"
    
    if not model_cache_path.exists():
        return False
    
    # Проверяем наличие файлов модели
    model_files = ["pytorch_model.bin", "model.safetensors"]
    if not any((model_cache_path / f).exists() for f in model_files):
        return False
    
    # Проверяем наличие файлов токенизатора
    tokenizer_files = ["tokenizer.json", "vocab.json", "merges.txt"]
    if not any((model_cache_path / f).exists() for f in tokenizer_files):
        return False
    
    return True

is_cached = is_model_and_tokenizer_cached(LLM_MODEL_NAME, CACHE_DIR)
logger.info(f"Model cache check: {is_cached}, path={CACHE_DIR}")

_tokenizer = None
_llm = None


def get_tokenizer():
    global _tokenizer
    if not _tokenizer:
        logger.info(f"model path {model_path} exists: {is_cached}")
        logger.info("Скачиваю токенайзер...")
        _tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME,
            use_fast=True,
            padding_side="left",
            download_dir=CACHE_DIR,  # используем локальный кэш
            cache_dir=CACHE_DIR,
            local_files_only=is_cached  # запрещаем скачивание, если уже скачано
        )
        logger.info("Токенайзер скачан")
        if _tokenizer.pad_token is None:
            _tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return _tokenizer


def get_llm():
    global _llm
    if not _llm:
        logger.info(f"model path {model_path} exists: {is_cached}")
        logger.info("Скачиваю конфигурацию модели...")
        llm_config = AutoConfig.from_pretrained(
            LLM_MODEL_NAME,
            cache_dir=CACHE_DIR,
            download_dir=CACHE_DIR,
            local_files_only=is_cached  # если уже скачано, то True
        )
        _llm = LLM(
            model=LLM_MODEL_NAME,
            tokenizer=LLM_MODEL_NAME,
            dtype=DTYPE,
            download_dir=CACHE_DIR,   # чтобы vLLM тоже брал кэш локально
            **LOCAL_CONFIG
        )
        logger.info("Конфигурация скачана")
    return _llm


# logger.info("Initializing TTS...")
# tts = TTS(model_name=TTS_MODEL_NAME, progress_bar=True)

# def text_to_speech(text: str) -> bytes:
#     audio_buffer = io.BytesIO()
#     tts.tts_to_file(text=text, file_path=audio_buffer, speaker=TTS_SPEAKER, speed=1.9)
#     audio_buffer.seek(0)
#     return audio_buffer.read()