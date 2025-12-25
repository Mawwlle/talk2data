import json
import logging
from dataclasses import replace
from typing import Any, Callable, Mapping

import pika
from pika.exceptions import AMQPChannelError, AMQPConnectionError

from task_management.domain.task_queue.models import TaskResponse
from task_management.domain.task_queue.ports import TaskQueuePort

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class RabbitMQAdapter(TaskQueuePort):
    def __init__(
        self,
        *,
        host: str,
        user: str,
        password: str,
        task_queue: str,
        response_queue: str,
        exchange: str,
        routing_key: str,
    ):
        credentials = pika.PlainCredentials(user, password)
        try:
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=host, credentials=credentials, heartbeat=60
                )
            )
            if not self.connection.is_open:
                raise ConnectionError("RabbitMQ connection failed silently")

            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=task_queue, durable=True)
            self.channel.queue_declare(queue=response_queue, durable=True)
            logger.info("Successfully connected to RabbitMQ")
        except AMQPConnectionError as conn_err:
            logger.error("RabbitMQ connection failed: %s", conn_err)
            raise
        except AMQPChannelError as channel_err:
            logger.error("Channel error: %s", channel_err)
            raise
        except Exception as exc:
            logger.error("Unexpected error: %s", exc)
            raise

        self._task_queue = task_queue
        self._response_queue = response_queue
        self._exchange = exchange
        self._routing_key = routing_key

    def _publish(self, response: TaskResponse) -> None:
        payload = {
            "status": response.status,
            "task": response.task,
            "result": response.result,
            "error": response.error,
            "project_id": response.project_id,
        }
        self.channel.basic_publish(
            exchange=self._exchange,
            routing_key=self._routing_key,
            body=json.dumps(payload),
            mandatory=True,
        )
        logger.info(
            "Sent response for %s: %s %s",
            response.task,
            response.status,
            response.error or "",
        )

    def start(self, handler: Callable[[Mapping[str, Any]], TaskResponse]) -> None:
        def _callback(ch, method, properties, body):
            try:
                message: Mapping[str, Any] = json.loads(body)
                logger.info("Received task: %s", message.get("task"))
                response = handler(message)
                resolved_project_id = response.project_id or message.get("data", {}).get("project_id")
                response_with_project = (
                    replace(response, project_id=resolved_project_id)
                    if resolved_project_id != response.project_id
                    else response
                )
                self._publish(response_with_project)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Error handling message: %s", exc)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=self._task_queue, on_message_callback=_callback)
        logger.info("Worker started. Waiting for tasks...")
        logger.info("Listening to queue: %s", self._task_queue)
        self.channel.start_consuming()
