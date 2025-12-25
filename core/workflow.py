# workflow.py

import copy
import logging
import time
from string import Template
from typing import cast

import torch
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import END, StateGraph
from openai import OpenAI
from transformers import PreTrainedTokenizerBase
from vllm import LLM, SamplingParams

from core.config import settings
from core.prompts import (
    CHAT_RESPONSE_PROMPT,
    CODE_GENERATION_PROMPT,
    DECIDE_ACTION_PROMPT,
    PromptMessage,
)
from core.schemas import AgentState, Decision
from task_management.domain.conversation.ports import WorkflowInvokerPort

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


DECIDE_ACTION_DEFAULT = "chat_response"


def extract_code_block(generated_text: str) -> str:
    if "```python" in generated_text:
        _, _, remainder = generated_text.partition("```python")
        code, _, _ = remainder.partition("```")
        return code.strip()

    if "```" in generated_text:
        _, _, remainder = generated_text.partition("```")
        code, _, _ = remainder.partition("```")
        return code.strip()

    return generated_text.strip()


class WorkflowEngine:
    def __init__(
        self, llm: LLM | OpenAI | None, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        self._llm = llm
        self._tokenizer = tokenizer

    @property
    def llm(self) -> LLM | OpenAI:
        if self._llm is not None:
            return self._llm

        raise ValueError(
            "You are trying to implement the local llm which is not initialized"
        )

    def _local_llm(self) -> LLM:
        llm = self.llm
        if isinstance(llm, OpenAI):
            raise ValueError("Local generation requires a vLLM instance")
        return llm

    def _format_prompt(
        self,
        messages_template: list[PromptMessage],
        state: AgentState,
        metadata_fields: dict[str, str] | None = None,
    ) -> str:
        metadata_fields = metadata_fields or {}
        local_prompt = copy.deepcopy(messages_template)

        logger.info(f"state: {state}")

        mapping = {
            "input": str(state.get("user_input", "")),
            "history": str(state.get("conversation_history", "")),
            "metadata": str(state.get("metadata", {})),
        }
        if metadata_fields:
            mapping.update({k: str(v) for k, v in metadata_fields.items()})

        formatted_messages = []
        for msg in local_prompt:
            tmpl = Template(msg["content"])
            content = tmpl.safe_substitute(mapping)
            formatted_messages.append({"role": msg["role"], "content": content})

        return cast(
            str,
            self._tokenizer.apply_chat_template(
                formatted_messages, tokenize=False, add_generation_prompt=True
            ),
        )

    def _remote_chat_completion(self, prompt: str) -> str:
        llm = self.llm
        if not isinstance(llm, OpenAI):
            raise ValueError("Remote chat completion requires an OpenAI client")
        completion = llm.chat.completions.create(
            model=settings.REMOTE_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content or ""

    def _decide_action(self, state: AgentState) -> AgentState:
        """Decision node with enhanced logging using print and timing."""
        start = time.perf_counter()
        parser = JsonOutputParser(pydantic_object=Decision)

        try:
            prompt = self._format_prompt(DECIDE_ACTION_PROMPT, state)
            logger.info("[decide_action] Formatted prompt: %s", prompt)

            # Генерация ответа от LLM
            if settings.REMOTE_LLM:
                raw_response = self._remote_chat_completion(prompt).strip()
            else:
                # Настраиваем параметры сэмплирования
                sampling_params = SamplingParams(
                    max_tokens=100,
                    temperature=0.0,  # полная детерминированность
                    top_p=1.0,  # отключает сэмплирование по вероятностям
                    stop=[
                        "</s>",
                        "\n\n",
                        "\nUser:",
                    ],  # можно добавить безопасные стоп-токены
                    # не трогаем (нет смысла для коротких ответов)
                    repetition_penalty=1.0,
                )

                logger.info(f"[decide_action] Sampling parameters: {sampling_params}")

                llm = self._local_llm()
                outputs = llm.generate([prompt], sampling_params)
                raw_response = outputs[0].outputs[0].text.strip()
                logger.info("[decide_action] Raw LLM response: %s", raw_response)

            decision = parser.parse(raw_response)
            logger.info("[decide_action] Parsed decision: %s", decision)

        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("[decide_action] Decision error: %s", exc)
            decision = {"action": DECIDE_ACTION_DEFAULT}

        elapsed = time.perf_counter() - start
        timing_info = state.get("timing_info", {})
        timing_info["decide_action_sec"] = round(elapsed, 4)
        state["timing_info"] = timing_info
        logger.info("[decide_action] Time elapsed: %.4f sec", elapsed)

        state["decision"] = decision
        logger.info("[decide_action] Final state: %s", state)

        return state

    @staticmethod
    def _route_action(state: AgentState) -> str:
        return state.get("decision", {}).get("action", DECIDE_ACTION_DEFAULT)

    def _generate_code_node(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        code_prompt = self._format_prompt(CODE_GENERATION_PROMPT, state)

        if settings.REMOTE_LLM:
            generated_text = self._remote_chat_completion(code_prompt)
        else:
            bad_words = [
                "print",
                "pd.read_csv",
                "pandas.read_csv",
                "df =",
                "df=",
                "df = pd.DataFrame",
                "df=pd.DataFrame",
            ]

            sampling_params = SamplingParams(
                max_tokens=512,
                temperature=0.0,  # 0.7
                top_p=0.95,
                stop=["<|", "</s>"],
                repetition_penalty=1.05,
                presence_penalty=0.5,
                seed=42,
                bad_words=bad_words,
            )

            llm = self._local_llm()
            outputs = llm.generate([code_prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text

        code_block = extract_code_block(generated_text)

        elapsed = time.perf_counter() - start
        timing_info = state.get("timing_info", {})
        timing_info["generate_code_sec"] = round(elapsed, 4)
        state["timing_info"] = timing_info

        logger.info("[generate_code] Generated code: %s...", code_block[:300])

        state.update(
            {
                "generated_code": code_block,
                "response_message": "Here's the generated code:",
            }
        )
        return state

    def _generate_chat_response_node(self, state: AgentState) -> AgentState:
        """Chat response generation with TTS integration, measure time."""
        start = time.perf_counter()
        chat_prompt = self._format_prompt(CHAT_RESPONSE_PROMPT, state)

        if settings.REMOTE_LLM:
            response = self._remote_chat_completion(chat_prompt).strip()
        else:
            sampling_params = SamplingParams(
                max_tokens=200,
                temperature=0.0,  # 0.7
                top_p=0.9,
                stop=["</s>"],
                seed=42,
            )

            llm = self._local_llm()
            outputs = llm.generate([chat_prompt], sampling_params)
            response = outputs[0].outputs[0].text.strip()
        elapsed_llm = time.perf_counter() - start
        logger.info("[test_response] Generated text response: %s", response)

        timing_info = state.get("timing_info", {})
        timing_info["generate_chat_response_sec"] = round(elapsed_llm, 4)
        state["timing_info"] = timing_info

        state["response_message"] = response
        return state

    def create_workflow(self) -> WorkflowInvokerPort:
        builder = StateGraph(AgentState)
        builder.add_node("decide_action", self._decide_action)
        builder.add_node("generate_code", self._generate_code_node)
        builder.add_node("generate_chat_response", self._generate_chat_response_node)
        builder.add_conditional_edges(
            "decide_action",
            self._route_action,
            {
                "code_generation": "generate_code",
                "chat_response": "generate_chat_response",
            },
        )
        builder.add_edge("generate_code", END)
        builder.add_edge("generate_chat_response", END)
        builder.set_entry_point("decide_action")
        return builder.compile()


def safe_destroy_process_group() -> None:
    """
    Безопасный shutdown моделей. Позволяет избежать утечек на GPU
    """
    if torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception as e:
            logger.exception("Error during destroy_process_group: %s", e)
            raise
