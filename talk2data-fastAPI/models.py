# models.py
import torch
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM
from pathlib import Path
import environ
# from TTS.api import TTS
# import whisper
env = environ.Env()
env.read_env(Path("../talk2data"), ".env")

import os
from dotenv import load_dotenv


# Настроим локальный кэш (если не указан явно — используем ./cache/huggingface)
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "./cache/huggingface")
os.environ["HF_HOME"] = os.getenv("HF_HOME", "./cache/huggingface")

# Создадим каталог, если его нет
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

CACHE_DIR = Path(env.str("TRANSFORMERS_CACHE", "./cache/huggingface")) # возможно, это стоит делать на s3
LLM_MODEL_NAME = env.str("LLM_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
DTYPE = env.str("DTYPE", "float16")

if env.bool("LOCAL_RUN", False):
    # Для локального запуска
    LOCAL_CONFIG = dict(
        max_model_len=4096,              # ↓ уменьшаем контекст
        gpu_memory_utilization=0.95,     # ↑ разрешаем использовать больше GPU-памяти
    )
else:
    LOCAL_CONFIG = {}
    
# Создаём структуру папок
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["HF_HOME"] = str(CACHE_DIR)
model_path = CACHE_DIR / f"models--{LLM_MODEL_NAME.replace('/', '--')}" 
is_cached = model_path.exists()

_tokenizer = None
_llm = None


def get_tokenizer():
    global _tokenizer
    if not _tokenizer:
        print(f"model path {model_path} exists: {is_cached}")
        print("Скачиваю токенайзер...")
        _tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME,
            use_fast=True,
            padding_side="left",
            download_dir=CACHE_DIR,  # используем локальный кэш
            cache_dir=CACHE_DIR,
            local_files_only=is_cached  # запрещаем скачивание, если уже скачано
        )
        print("Токенайзер скачан")
        if _tokenizer.pad_token is None:
            _tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return _tokenizer


def get_llm():
    global _llm
    if not _llm:
        print(f"model path {model_path} exists: {is_cached}")
        print("⬇️  Скачиваю конфигурацию модели...")
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
            enforce_eager=True,
            download_dir=CACHE_DIR,   # чтобы vLLM тоже брал кэш локально
            **LOCAL_CONFIG
        )
        print("Конфигурация скачана")
    return _llm


# print("Initializing TTS...")
# tts = TTS(model_name=TTS_MODEL_NAME, progress_bar=True)

# def text_to_speech(text: str) -> bytes:
#     audio_buffer = io.BytesIO()
#     tts.tts_to_file(text=text, file_path=audio_buffer, speaker=TTS_SPEAKER, speed=1.9)
#     audio_buffer.seek(0)
#     return audio_buffer.read()