# api.py
import json
import pika
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from settings import RABBITMQ_HOST, TASK_QUEUE

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def send_task(task: str, data: dict):
    """Отправляем задачу в RabbitMQ"""
    connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=TASK_QUEUE)
    channel.basic_publish(
        exchange="",
        routing_key=TASK_QUEUE,
        body=json.dumps({"task": task, "data": data})
    )
    connection.close()

@app.post("/converse")
async def converse(request: Request):
    try:
        body = await request.json()
        send_task("converse", body)
        return {"status": "queued", "task": "converse"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        data = {"filename": file.filename, "file_bytes": file_bytes.hex()}
        send_task("transcribe", data)
        return {"status": "queued", "task": "transcribe"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))