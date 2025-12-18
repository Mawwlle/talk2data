from transformers import AutoTokenizer

LLM_MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
LOCAL_PATH = "model_weights_presaved"

_tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_PATH,
    use_fast=True,
    padding_side="left",
    local_files_only=True,
)


from vllm import LLM

llm = LLM(model=LOCAL_PATH, load_format="safetensors")
