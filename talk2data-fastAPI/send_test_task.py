import pika
import json

RABBITMQ_HOST = "localhost"  # или "gateway-rabbitmq", если нужно
TASK_QUEUE = "talk2data_tasks"

connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
channel = connection.channel()
channel.queue_declare(queue=TASK_QUEUE)

task = {"task": "converse", "data": {"user_input": "Hi! WTF?", "metadata": {}}}
channel.basic_publish(exchange="", routing_key=TASK_QUEUE, body=json.dumps(task))
print("✅ Test task sent!")
connection.close()