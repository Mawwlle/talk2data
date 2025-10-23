# models.py
import torch
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM
# from TTS.api import TTS
# import whisper

import os
from dotenv import load_dotenv

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

LLM_MODEL_NAME = os.getenv("LLM_MODEL")

_tokenizer = None
_llm = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        print("Initializing tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME, use_fast=True, padding_side="left"
        )
        if _tokenizer.pad_token is None:
            _tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return _tokenizer


def get_llm():
    global _llm
    if _llm is None:
        print("Initializing vLLM...")
        llm_config = AutoConfig.from_pretrained(LLM_MODEL_NAME)
        _llm = LLM(
            model=LLM_MODEL_NAME,
            tokenizer=LLM_MODEL_NAME,
            dtype="float16",
            enforce_eager=True,
            max_model_len=llm_config.max_position_embeddings,
        )
    return _llm


# print("Initializing TTS...")
# tts = TTS(model_name=TTS_MODEL_NAME, progress_bar=True)

# def text_to_speech(text: str) -> bytes:
#     audio_buffer = io.BytesIO()
#     tts.tts_to_file(text=text, file_path=audio_buffer, speaker=TTS_SPEAKER, speed=1.9)
#     audio_buffer.seek(0)
#     return audio_buffer.read()