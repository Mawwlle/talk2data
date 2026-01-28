import json
import logging
from typing import Any

import pika  # type: ignore[import-untyped]
from pika.exceptions import (  # type: ignore[import-untyped]
    AMQPChannelError,
    AMQPConnectionError,
)

from task_management.domain.conversation.streaming import StreamingEmitterPort

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class RabbitMQStreamingEmitter(StreamingEmitterPort):
    def __init__(
        self,
        *,
        host: str,
        user: str,
        password: str,
        exchange: str,
        routing_key: str,
        task: str,
        project_id: str | None,
        base_meta: dict[str, Any] | None = None,
    ) -> None:
        self._project_id = project_id
        self._task = task
        self._exchange = exchange
        self._routing_key = routing_key
        self._base_meta = base_meta or {}
        self._closed = False

        credentials = pika.PlainCredentials(user, password)
        try:
            self._connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=host, credentials=credentials, heartbeat=60)
            )
            if not self._connection.is_open:
                raise ConnectionError("RabbitMQ connection failed silently")
            self._channel = self._connection.channel()
            logger.info("Streaming emitter connected to RabbitMQ")
        except AMQPConnectionError as conn_err:
            logger.error("Streaming emitter connection failed: %s", conn_err)
            raise
        except AMQPChannelError as channel_err:
            logger.error("Streaming emitter channel error: %s", channel_err)
            raise
        except Exception as exc:
            logger.error("Streaming emitter unexpected error: %s", exc)
            raise

    def emit_start(self, meta: dict[str, Any]) -> None:
        payload = {
            "status": "stream",
            "task": self._task,
            "result": {
                "type": "start",
                "stream": True,
            },
            "project_id": self._project_id,
            "meta": self._merge_meta(meta),
        }
        self._publish(payload)

    def emit_delta(self, delta_text: str, seq: int, meta: dict[str, Any]) -> None:
        payload = {
            "status": "stream",
            "task": self._task,
            "result": {
                "type": "delta",
                "message": delta_text,
                "seq": seq,
                "stream": True,
            },
            "project_id": self._project_id,
            "meta": self._merge_meta(meta),
        }
        self._publish(payload)

    def emit_end(
        self,
        final_text: str,
        meta: dict[str, Any],
        usage: dict[str, Any] | None = None,
        finish_reason: str | None = None,
    ) -> None:
        result_payload: dict[str, Any] = {
            "type": "final",
        }
        node = meta.get("node")
        if node == "generate_code":
            result_payload["code"] = final_text
        else:
            result_payload["message"] = final_text
        if usage is not None:
            result_payload["usage"] = usage
        if finish_reason is not None:
            result_payload["finish_reason"] = finish_reason

        payload = {
            "status": "stream",
            "task": self._task,
            "result": result_payload,
            "project_id": self._project_id,
            "meta": self._merge_meta(meta),
        }
        timing_info = meta.get("timing_info")
        if timing_info is not None:
            payload["timing_info"] = timing_info
        self._publish(payload)
        self._close()

    def emit_error(self, error: str, meta: dict[str, Any]) -> None:
        payload = {
            "status": "error",
            "task": self._task,
            "result": {
                "type": "error",
                "message": error,
                "stream": True,
            },
            "error": error,
            "project_id": self._project_id,
            "meta": self._merge_meta(meta),
        }
        self._publish(payload)
        self._close()

    def _merge_meta(self, meta: dict[str, Any]) -> dict[str, Any]:
        return {**self._base_meta, **meta}

    def _publish(self, payload: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            self._channel.basic_publish(
                exchange=self._exchange,
                routing_key=self._routing_key,
                body=json.dumps(payload, ensure_ascii=False),
                mandatory=True,
            )
        except Exception:
            logger.exception("Streaming emitter publish failed")

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._channel.close()
        except Exception:
            logger.exception("Streaming emitter failed to close channel")
        try:
            self._connection.close()
        except Exception:
            logger.exception("Streaming emitter failed to close connection")
