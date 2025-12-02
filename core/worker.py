# worker.py
import base64
import json
from pathlib import Path
import time
import pika

# from models import whisper_model
from core.workflow import create_workflow, llm_init
from core.schemas import ConversationRequest
import tempfile

import logging
from settings import RABBITMQ_HOST, TASK_QUEUE, RESPONSE_QUEUE, EXCHANGE, ROUTING_KEY
from voice2text.whisper_model import Voice2Text

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Инициализация соединения
connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
channel = connection.channel()
channel.queue_declare(queue=TASK_QUEUE, durable=True)
channel.queue_declare(queue=RESPONSE_QUEUE, durable=True)

logger.info("Worker connected to RabbitMQ")

llm_init()

whisper_model = Voice2Text()


def save_result_locally(result: dict, filename: str = "result.json") -> None:
    """Сохраняет result в локальный JSON-файл рядом с текущим модулем."""
    try:
        file_path = Path(__file__).with_name(filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        logger.info("Result saved to %s", file_path)
    except Exception as e:
        logger.exception("Failed to save result: %s", e)


def load_result_locally(filename: str = "result.json") -> dict:
    """Загружает сохранённый result из JSON-файла."""
    file_path = Path(__file__).with_name(filename)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.exception("Failed to load result: %s", e)
        return {}


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

        # для дебага вместо workflow, если нет времени разворачивать llm:
        # result = load_result_locally()

        total_time = round(time.perf_counter() - start, 3)

        # Формируем финальный ответ
        response = {
            "status": "done",
            "task": "llm_agent_response",
            "result": {
                "code": result.get("generated_code"),
                "message": result.get("response_message"),
                "updated_history": result["conversation_history"]
                + [
                    {
                        "user": req.user_input,
                        "system": result.get("generated_code")
                        or result.get("response_message"),
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
    try:
        audio_bytes = base64.b64decode(data["file_bytes"])
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        result = whisper_model.transcribe(tmp_path)

        return {
            "status": "done",
            "task": "transcribe",
            "result": {"text": result["text"]},
        }
    except Exception as e:
        return {"status": "error", "task": "transcribe", "error": str(e)}


task_mapping = {"converse": handle_converse, "transcribe": handle_transcribe}


def callback(ch, method, properties, body):
    """Основная функция обработки входящих задач."""
    try:
        msg = json.loads(body)
        task_type = msg.get("task")
        data = msg.get("data", {})
        handler = task_mapping.get(task_type)

        logger.info(f"Received task: {task_type}")
        if handler is None:
            err_msg = f"Unknown task type: {task_type}"
            logger.warning(err_msg)
            response = {"status": "error", "error": err_msg}
        else:
            response = handler(data)

        response["project_id"] = data.get("project_id")

        # Отправляем результат обратно
        ch.basic_publish(
            exchange=EXCHANGE,
            routing_key=ROUTING_KEY,
            body=json.dumps(response),
            mandatory=True,
        )

        logger.info(
            f"Sent response for {task_type}: {response.get('status')} {response.get('error', '')}"
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        logger.info(f"Error handling message: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


# Подписываемся на очередь задач
channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=TASK_QUEUE, on_message_callback=callback)

logger.info("Worker started. Waiting for tasks...")
logger.info(f"Listening to queue: {TASK_QUEUE}")
channel.start_consuming()
