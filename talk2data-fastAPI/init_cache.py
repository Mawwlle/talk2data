"""
init_cache.py
Создаёт локальный кэш моделей Hugging Face для проекта Talk2Data.
Запускается перед запуском uvicorn и сразу же сохраняет все веса в кэше всего один раз
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM
import torch
import environ

env = environ.Env()
env.read_env(Path("../talk2data"), ".env")

# === 1. Настройки ===
load_dotenv()

CACHE_DIR = Path(env.str("TRANSFORMERS_CACHE", "./cache/huggingface")) # возможно, это стоит делать на s3
LLM_MODEL_NAME = env.str("LLM_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = env.str("DTYPE", "float16")

# === 2. Создаём структуру папок ===
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["HF_HOME"] = str(CACHE_DIR)

print(f"✅ Кэш-моделей будет сохранён в: {CACHE_DIR.resolve()}")
print(f"💻 Устройство: {DEVICE}")
print(f"🧠 Модель: {LLM_MODEL_NAME}")

# === 3. Проверяем токенайзер ===
tokenizer_path = CACHE_DIR / "models--" / LLM_MODEL_NAME.replace("/", "--")
if not tokenizer_path.exists():
    print("⬇️  Скачиваю токенайзер...")
AutoTokenizer.from_pretrained(LLM_MODEL_NAME, use_fast=True, padding_side="left")
print("✅ Токенайзер загружен")

# === 4. Проверяем конфиг ===
if not (CACHE_DIR / f"models--{LLM_MODEL_NAME.replace('/', '--')}").exists():
    print("⬇️  Скачиваю конфигурацию модели...")
AutoConfig.from_pretrained(LLM_MODEL_NAME)
print("✅ Конфигурация загружена")

# === 5. Проверяем веса модели ===
print("⬇️  Проверяю наличие весов модели (vLLM)...")
try:
    llm = LLM(
        model=LLM_MODEL_NAME,
        tokenizer=LLM_MODEL_NAME,
        dtype=DTYPE,
        enforce_eager=True,
        download_dir=CACHE_DIR  # вот сюда сохраняем веса
    )
    print("✅ Веса модели загружены и готовы к использованию")
except Exception as e:
    print(f"⚠️ Ошибка при инициализации модели: {e}")

print("\n🎉 Кэш успешно инициализирован. Можно запускать FastAPI:")
print("   $ CUDA_VISIBLE_DEVICES=0 poetry run uvicorn api:app --host 0.0.0.0 --port 6000")
