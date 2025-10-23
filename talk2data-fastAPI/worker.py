# worker.py
import json
import time
import pika
# from models import whisper_model
from workflow import create_workflow
from models import get_llm, get_tokenizer
from schemas import ConversationRequest
import logging

logger = logging.getLogger(__name__)

# Настройки RabbitMQ
RABBITMQ_HOST = "localhost"
TASK_QUEUE = "talk2data_tasks"
RESPONSE_QUEUE = "talk2data_responses"

# Инициализация соединения
connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
channel = connection.channel()
channel.queue_declare(queue=TASK_QUEUE)
channel.queue_declare(queue=RESPONSE_QUEUE)

logger.info("Worker connected to RabbitMQ")

# Прогрев моделей
logger.info("Loading models...")
llm = get_llm()
tokenizer = get_tokenizer()
logger.info("Models ready!")


def handle_converse(data: dict):
    """Обработка текстового запроса (LLM pipeline)."""
    start = time.perf_counter()
    try:
        # Собираем состояние
        req = ConversationRequest(**data)
        workflow = create_workflow()
        initial_state = {
            "user_input": req.user_input,
            "metadata": req.metadata,
            "conversation_history": req.conversation_history,
            "generated_code": None,
            "response_message": None,
            "response_audio": None,
            "decision": None,
            "timing_info": {},
        }

        result = workflow.invoke(initial_state)
        total_time = round(time.perf_counter() - start, 3)

        # Формируем финальный ответ
        response = {
            "status": "done",
            "task": "converse",
            "result": {
                "code": result.get("generated_code"),
                "message": result.get("response_message"),
                "audio": result.get("response_audio"),
                "updated_history": result["conversation_history"] + [
                    {
                        "user": req.user_input,
                        "system": result.get("generated_code") or result.get("response_message"),
                    }
                ],
                "timing": {**result.get("timing_info", {}), "total_time": total_time},
            },
        }
        return response
    except Exception as e:
        return {"status": "error", "task": "converse", "error": str(e)}


def handle_transcribe(data: dict):
    """Обработка аудио (Whisper)."""
    start = time.perf_counter()
    try:
        # Преобразуем байты обратно
        import tempfile
        from models import whisper_model

        audio_bytes = bytes.fromhex(data["file_bytes"])
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        result = whisper_model.transcribe(tmp_path)
        total_time = round(time.perf_counter() - start, 3)

        return {
            "status": "done",
            "task": "transcribe",
            "result": {"text": result["text"], "timing": {"total_time": total_time}},
        }
    except Exception as e:
        return {"status": "error", "task": "transcribe", "error": str(e)}

task_mapping = {
    "converse": process_converse,
    # "transcribe": process_transcribe # будет добавлено позже
}

def callback(ch, method, properties, body):
    """Основная функция обработки входящих задач."""
    try:
        msg = json.loads(body)
        task_type = msg.get("task")
        data = msg.get("data", {})
        handler = task_mapping.get(task)
        
        logger.info(f"Received task: {task_type}")
        if handler is None:
            err_msg = f"Unknown task type: {task_type}"
            logger.warning(err_msg)
            response = {"status": "error", "error": err_msg}
        else:
            response = handler(data)

        # Отправляем результат обратно
        channel.basic_publish(
            exchange="",
            routing_key=RESPONSE_QUEUE,
            body=json.dumps(response)
        )

        logger.info(f"Sent response for {task_type}: {response['status']}")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        logger.info(f"Error handling message: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


# Подписываемся на очередь задач
channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=TASK_QUEUE, on_message_callback=callback)

logger.info("Worker started. Waiting for tasks...")
channel.start_consuming()
