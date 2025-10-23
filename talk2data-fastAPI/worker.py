# worker.py
import json
import pika
# from models import whisper_model
from workflow import create_workflow
from settings import TASK_QUEUE, RABBITMQ_HOST
import logging
import tempfile

logger = logging.getLogger(__name__)

def process_converse(data: dict):
    workflow = create_workflow()
    state = {
        "user_input": data.get("user_input"),
        "metadata": data.get("metadata"),
        "conversation_history": data.get("conversation_history", []),
        "generated_code": None,
        "response_message": None,
        "response_audio": None,
        "decision": None,
        "timing_info": {}
    }
    result = workflow.invoke(state)
    return {
        "code": result.get("generated_code"),
        "message": result.get("response_message"),
        "audio": result.get("response_audio"),
        "updated_history": result["conversation_history"]
    }

def process_transcribe(data: dict):
    file_bytes = data.get("file_bytes")
    if not file_bytes:
        return {"error": "No audio bytes provided"}
    
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        result = whisper_model.transcribe(tmp.name)
    return {"text": result.get("text", "")}
    
task_mapping = {
    "converse": process_converse,
    # "transcribe": process_transcribe # будет добавлено позже
}
        
def callback(ch, method, properties, body):
    message = json.loads(body)
    task = message.get("task")
    task_id = message.get("task_id")
    data = message.get("data", {})

    handler = task_mapping.get(task)
    
    if not handler:
        logger.warning(f"Unknown or missing task: {task}")
        return
    
    try:
        result = handler(data)
        response = {
            "task": task,
            "task_id": task_id,
            "result": result
        }
        ch.basic_publish(
            exchange="",
            routing_key="talk2data_responses",
            body=json.dumps(response)
        )
    except Exception as e:
        logger.error(f"Error processing task {task}: {e}")
        ch.basic_publish(
            exchange="",
            routing_key="talk2data_responses",
            body=json.dumps({"task": task, "task_id": task_id, "error": str(e)})
        )


def start_worker():
    connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=TASK_QUEUE)
    channel.basic_consume(queue=TASK_QUEUE, on_message_callback=callback, auto_ack=True)
    logger.info(" Worker started. Waiting for tasks...")
    channel.start_consuming()

if __name__ == "__main__":
    start_worker()
