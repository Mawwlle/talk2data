import json
import logging
from typing import Callable

import pika
from pika.exceptions import AMQPChannelError, AMQPConnectionError

from core.config import settings
from domain.entities import TaskResponse
from domain.interfaces import TaskQueue

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class RabbitMQAdapter(TaskQueue):
    def __init__(self):
        credentials = pika.PlainCredentials(settings.RABBITMQ_USER, settings.RABBITMQ_PASS)
        try:
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=settings.RABBITMQ_HOST, credentials=credentials, heartbeat=60
                )
            )
            if not self.connection.is_open:
                raise ConnectionError("RabbitMQ connection failed silently")

            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=settings.TASK_QUEUE, durable=True)
            self.channel.queue_declare(queue=settings.RESPONSE_QUEUE, durable=True)
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

    def _publish(self, response: TaskResponse) -> None:
        payload = {
            "status": response.status,
            "task": response.task,
            "result": response.result,
            "error": response.error,
            "project_id": response.project_id,
        }
        self.channel.basic_publish(
            exchange=settings.EXCHANGE,
            routing_key=settings.ROUTING_KEY,
            body=json.dumps(payload),
            mandatory=True,
        )
        logger.info(
            "Sent response for %s: %s %s",
            response.task,
            response.status,
            response.error or "",
        )

    def start(self, handler: Callable[[dict], TaskResponse]) -> None:
        def _callback(ch, method, properties, body):
            try:
                message = json.loads(body)
                logger.info("Received task: %s", message.get("task"))
                response = handler(message)
                if response.project_id is None:
                    response.project_id = message.get("data", {}).get("project_id")
                self._publish(response)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Error handling message: %s", exc)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=settings.TASK_QUEUE, on_message_callback=_callback)
        logger.info("Worker started. Waiting for tasks...")
        logger.info("Listening to queue: %s", settings.TASK_QUEUE)
        self.channel.start_consuming()
