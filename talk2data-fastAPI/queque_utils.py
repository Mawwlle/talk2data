# queue_utils.py
import pika
import json
import settings
import logging

logger = logging.getLogger(__name__)

def get_connection():
    return pika.BlockingConnection(pika.ConnectionParameters(settings.RABBITMQ_HOST))

def send_task(task_name: str, data: dict):
    """Отправить задачу в очередь"""
    connection = get_connection()
    channel = connection.channel()
    channel.queue_declare(queue=settings.TASK_QUEUE)
    message = json.dumps({"task": task_name, "data": data})
    channel.basic_publish(exchange='', routing_key=settings.TASK_QUEUE, body=message)
    connection.close()
    logger.info(f"Sent task: {task_name}")
