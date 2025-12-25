# models.py
import logging
from pathlib import Path
from typing import Any
import os
import openai
import environ
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM

from core.config import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, *, env_file: Path | None = None) -> None:
        env = environ.Env()
        env.read_env(env_file or Path(__file__).resolve().parent.parent / ".env")
        self._settings = settings
        self._vllm_config = self._build_vllm_config(env)
        self._tokenizer = None
        self._llm = None

    @staticmethod
    def _build_vllm_config(env: environ.Env) -> dict[str, Any]:
        if env.bool("LOCAL_RUN", False):
            return dict(
                gpu_memory_utilization=0.95,
                enforce_eager=False,
                dtype="float16",
                tensor_parallel_size=1,
            )
        return {}

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            logger.info("Загружаю токенайзер...")
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._settings.LLM_LOCAL_PATH,
                    use_fast=True,
                    padding_side="left",
                    local_files_only=True,
                )
            except OSError:
                logger.warning("Local weights not found, trying to download from HF...")
                self._tokenizer = AutoTokenizer.from_pretrained(self._settings.LLM_MODEL_NAME)
                self._tokenizer.save_pretrained(self._settings.LLM_LOCAL_PATH)
            if self._tokenizer.pad_token is None:
                self._tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            logger.info("Токенайзер загружен")
        return self._tokenizer

    def get_llm(self) -> LLM | openai.OpenAI:
        if settings.REMOTE_LLM:
            if self._llm is None:
                self._llm = openai.OpenAI(
                    api_key=os.getenv("OPEN_AI_API_KEY"),
                    base_url=settings.REMOTE_URL,
                )
            return self._llm

        if self._llm is None:
            logger.info("Загружаю модель...")
            self._llm = LLM(
                model=self._settings.LLM_LOCAL_PATH,
                load_format=(
                    self._settings.LLM_LOAD_FORMAT
                    if hasattr(self._settings, "LLM_LOAD_FORMAT")
                    else "auto"
                ),
                **self._vllm_config,
            )
            logger.info("Модель загружена")
        return self._llm
