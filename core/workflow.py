# workflow.py

import copy
import logging
import time
from string import Template
from typing import Any, cast

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
from task_management.domain.conversation.streaming import StreamingEmitterPort

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


DECIDE_ACTION_DEFAULT = "chat_response"
DECIDE_ACTION_CHOICES = {"code_generation", "chat_response"}


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
    def __init__(self, llm: LLM | OpenAI | None, tokenizer: PreTrainedTokenizerBase) -> None:
        self._llm = llm
        self._tokenizer = tokenizer

    @property
    def llm(self) -> LLM | OpenAI:
        if self._llm is not None:
            return self._llm

        raise ValueError("You are trying to implement the local llm which is not initialized")

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

        if not settings.REMOTE_LLM:
            return cast(
                str,
                self._tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True),
            )

        # REMOTE_LLM=True:
        # Do NOT use local tokenizer chat template
        # (it will leak template tokens to OpenAI-compatible servers).
        # Instead, pass a plain-text, role-delimited transcript.
        role_map = {
            "system": "SYSTEM",
            "user": "USER",
            "assistant": "ASSISTANT",
            "tool": "TOOL",
        }

        chunks: list[str] = []
        for m in formatted_messages:
            role = role_map.get(m["role"], m["role"].upper())
            content = (m.get("content") or "").strip()
            if not content:
                continue
            chunks.append(f"{role}:\n{content}")

        # add_generation_prompt=True equivalent
        chunks.append("ASSISTANT:\n")

        return "\n\n".join(chunks).strip()

    def _remote_chat_completion(self, prompt: str) -> str:
        llm = self.llm
        if not isinstance(llm, OpenAI):
            raise ValueError("Remote chat completion requires an OpenAI client")
        completion = llm.chat.completions.create(
            model=settings.REMOTE_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content or ""

    def _remote_llm_streaming(
        self,
        prompt: str,
        *,
        emitter: StreamingEmitterPort | None,
        meta: dict[str, Any],
    ) -> str:
        llm = self.llm
        if not isinstance(llm, OpenAI):
            raise ValueError("Remote streaming requires an OpenAI client")

        if emitter is not None:
            emitter.emit_start(meta)

        chunks: list[str] = []
        seq = 0
        finish_reason: str | None = None
        usage: dict[str, Any] | None = None
        logger.info(
            "[streaming] start node=%s request_id=%s project_id=%s",
            meta.get("node"),
            meta.get("request_id"),
            meta.get("project_id"),
        )

        try:
            stream = llm.chat.completions.create(
                model=settings.REMOTE_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )

            for chunk in stream:
                if chunk is None:
                    continue
                choices = getattr(chunk, "choices", []) or []
                if not choices:
                    continue
                choice = choices[0]
                delta_obj = getattr(choice, "delta", None)
                delta = getattr(delta_obj, "content", None)
                finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                usage = getattr(chunk, "usage", None) or usage
                if not delta:
                    continue
                chunks.append(delta)
                if emitter is not None:
                    emitter.emit_delta(delta, seq, meta)
                seq += 1
        except Exception as exc:
            logger.exception(
                "[streaming] error node=%s request_id=%s project_id=%s",
                meta.get("node"),
                meta.get("request_id"),
                meta.get("project_id"),
            )
            if emitter is not None:
                emitter.emit_error(str(exc), meta)
            raise

        final_text = "".join(chunks)
        if emitter is not None:
            emitter.emit_end(final_text, meta, usage=usage, finish_reason=finish_reason)

        logger.info(
            "[streaming] end node=%s request_id=%s project_id=%s chunks=%s chars=%s",
            meta.get("node"),
            meta.get("request_id"),
            meta.get("project_id"),
            seq,
            len(final_text),
        )
        return final_text

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
            decision = self._normalize_decision(decision, raw_response, state)
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

    def _normalize_decision(self, decision: Any, raw_response: str, state: AgentState) -> Decision:
        action: str | None = None
        if isinstance(decision, dict):
            action = decision.get("action")
        elif isinstance(decision, str):
            action = decision

        if action:
            action = action.strip().lower()

        if action in DECIDE_ACTION_CHOICES:
            return {"action": action}

        raw_lower = raw_response.lower()
        if "code_generation" in raw_lower:
            return {"action": "code_generation"}
        if "chat_response" in raw_lower:
            return {"action": "chat_response"}

        return {"action": DECIDE_ACTION_DEFAULT}

    @staticmethod
    def _route_action(state: AgentState) -> str:
        return state.get("decision", {}).get("action", DECIDE_ACTION_DEFAULT)

    def _generate_code_node(self, state: AgentState) -> AgentState:
        start = time.perf_counter()
        code_prompt = self._format_prompt(CODE_GENERATION_PROMPT, state)

        if settings.REMOTE_LLM:
            streaming_emitter = state.get("streaming_emitter")
            streaming_meta = dict(state.get("streaming_meta") or {})
            streaming_meta.update(
                {
                    "node": "generate_code",
                    "project_id": streaming_meta.get("project_id") or state.get("metadata", {}).get("project_id"),
                }
            )
            if settings.REMOTE_LLM_STREAMING:
                emitter = streaming_emitter
                if emitter is not None:
                    emitter = _CodeStreamingEmitter(emitter)
                generated_text = self._remote_llm_streaming(
                    code_prompt,
                    emitter=emitter,
                    meta=streaming_meta,
                )
            else:
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
            streaming_emitter = state.get("streaming_emitter")
            streaming_meta = dict(state.get("streaming_meta") or {})
            streaming_meta.update(
                {
                    "node": "generate_chat_response",
                    "project_id": streaming_meta.get("project_id") or state.get("metadata", {}).get("project_id"),
                }
            )
            if settings.REMOTE_LLM_STREAMING:
                response = self._remote_llm_streaming(
                    chat_prompt,
                    emitter=streaming_emitter,
                    meta=streaming_meta,
                ).strip()
            else:
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
        compiled = builder.compile()
        return _WorkflowInvoker(compiled)


class _CodeStreamingEmitter:
    def __init__(self, emitter: StreamingEmitterPort) -> None:
        self._emitter = emitter

    def emit_start(self, meta: dict[str, Any]) -> None:
        self._emitter.emit_start(meta)

    def emit_delta(self, delta_text: str, seq: int, meta: dict[str, Any]) -> None:
        self._emitter.emit_delta(delta_text, seq, meta)

    def emit_end(
        self,
        final_text: str,
        meta: dict[str, Any],
        usage: dict[str, Any] | None = None,
        finish_reason: str | None = None,
    ) -> None:
        code_block = extract_code_block(final_text)
        self._emitter.emit_end(code_block, meta, usage=usage, finish_reason=finish_reason)

    def emit_error(self, error: str, meta: dict[str, Any]) -> None:
        self._emitter.emit_error(error, meta)


class _WorkflowInvoker:
    def __init__(self, compiled: Any) -> None:
        self._compiled = compiled

    def invoke(self, state: dict) -> dict:
        return cast(dict, self._compiled.invoke(state))


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
