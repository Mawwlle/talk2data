# Скачивание весов в локальную папку

from transformers import AutoModelForCausalLM, AutoTokenizer
import settings

LOCAL_PATH = settings.LLM_LOCAL_PATH

model = AutoModelForCausalLM.from_pretrained(settings.LLM_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_NAME)
model.save_pretrained(LOCAL_PATH)
tokenizer.save_pretrained(LOCAL_PATH)