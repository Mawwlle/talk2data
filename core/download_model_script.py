# Скачивание весов в локальную папку

from transformers import AutoModelForCausalLM, AutoTokenizer
from core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
LOCAL_PATH = settings.LLM_LOCAL_PATH

logger.info(f"Downloading model {settings.LLM_MODEL_NAME} to {LOCAL_PATH}...")


model = AutoModelForCausalLM.from_pretrained(settings.LLM_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_NAME)
model.save_pretrained(LOCAL_PATH)
tokenizer.save_pretrained(LOCAL_PATH)
