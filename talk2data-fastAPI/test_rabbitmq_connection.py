# test_rabbitmq_connection.py
import pika

RABBITMQ_HOST = "localhost"  # если воркер снаружи Docker
connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
channel = connection.channel()
channel.queue_declare(queue="test_queue")
print("✅ RabbitMQ доступен и очередь test_queue создана.")
connection.close()
