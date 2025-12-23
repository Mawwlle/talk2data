# workflow.py

import atexit
import copy
import logging
import time
from string import Template
from typing import Callable

import torch
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import END, StateGraph
from vllm import SamplingParams

from core.prompts import (
    CHAT_RESPONSE_PROMPT,
    CODE_GENERATION_PROMPT,
    DECIDE_ACTION_PROMPT,
)
from core.schemas import AgentState, Decision

logger = logging.getLogger(__name__)

DECIDE_ACTION_DEFAULT = "chat_response"


class WorkflowEngine:
    def __init__(self, llm, tokenizer) -> None:
        self._llm = llm
        self._tokenizer = tokenizer

    def _format_prompt(
        self, messages_template: list, state: AgentState, metadata_fields: dict | None = None
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
            try:
                tmpl = Template(msg["content"])
                content = tmpl.safe_substitute(mapping)
            except Exception as exc:  # pragma: no cover - defensive
                logger.info("[format_prompt] Template substitution error: %s", exc)
                content = msg["content"]

            formatted_messages.append({"role": msg["role"], "content": content})

        return self._tokenizer.apply_chat_template(
            formatted_messages, tokenize=False, add_generation_prompt=True
        )

    def _decide_action(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        logger.info("[decide_action] Starting with state: %s", state)

        parser = JsonOutputParser(pydantic_object=Decision)

        try:
            prompt = self._format_prompt(DECIDE_ACTION_PROMPT, state)
            logger.info("[decide_action] Formatted prompt: %s", prompt)

            sampling_params = SamplingParams(
                max_tokens=100,
                temperature=0.0,
                top_p=1.0,
                stop=["</s>", "\n\n", "\nUser:"],
                repetition_penalty=1.0,
            )

            logger.info("[decide_action] Sampling parameters: %s", sampling_params)

            outputs = self._llm.generate([prompt], sampling_params)
            raw_response = outputs[0].outputs[0].text.strip()
            logger.info("[decide_action] Raw LLM response: %s", raw_response)

            decision = parser.parse(raw_response)
            logger.info("[decide_action] Parsed decision: %s", decision)

        except Exception as exc:  # pragma: no cover - defensive
            logger.info("[decide_action] Decision error: %s", exc)
            decision = {"action": DECIDE_ACTION_DEFAULT}
            logger.info("[decide_action] Defaulting decision to: %s", decision)

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
        try:
            return state.get("decision", {}).get("action", DECIDE_ACTION_DEFAULT)
        except Exception:  # pragma: no cover - defensive
            return DECIDE_ACTION_DEFAULT

    def _generate_code_node(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        code_prompt = self._format_prompt(CODE_GENERATION_PROMPT, state)

        logger.info("[code_prompt] Prompt for code generation: %s", code_prompt)
        sampling_params = SamplingParams(
            max_tokens=512,
            temperature=0.0,
            top_p=0.95,
            stop=["<|", "</s>"],
            repetition_penalty=1.05,
            presence_penalty=0.5,
            seed=42,
        )

        outputs = self._llm.generate([code_prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text

        code_block = generated_text.split("```python")[-1].split("```")[0].strip()

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
        start = time.perf_counter()
        chat_prompt = self._format_prompt(CHAT_RESPONSE_PROMPT, state)
        logger.info("[chat_prompt] Prompt for text generation: %s", chat_prompt)
        sampling_params = SamplingParams(
            max_tokens=200,
            temperature=0.0,
            top_p=0.9,
            stop=["</s>"],
            seed=42,
        )

        outputs = self._llm.generate([chat_prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        elapsed_llm = time.perf_counter() - start
        logger.info("[test_response] Generated text response: %s", response)

        timing_info = state.get("timing_info", {})
        timing_info["generate_chat_response_sec"] = round(elapsed_llm, 4)
        state["timing_info"] = timing_info

        state["response_message"] = response
        return state

    def create_workflow(self) -> Callable[[AgentState], AgentState]:
        builder = StateGraph(AgentState)
        builder.add_node("decide_action", self._decide_action)
        builder.add_node("generate_code", self._generate_code_node)
        builder.add_node("generate_chat_response", self._generate_chat_response_node)
        builder.add_conditional_edges(
            "decide_action",
            self._route_action,
            {"code_generation": "generate_code", "chat_response": "generate_chat_response"},
        )
        builder.add_edge("generate_code", END)
        builder.add_edge("generate_chat_response", END)
        builder.set_entry_point("decide_action")
        return builder.compile()


def safe_destroy_process_group():
    """
    Безопасный shutdown моделей. Позволяет избежать утечек на GPU
    """
    if torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception as e:
            logger.error(f"Error during destroy_process_group: {e}")


atexit.register(safe_destroy_process_group)
