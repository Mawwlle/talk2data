from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_USER: str = "guest"
    RABBITMQ_PASS: str = "guest"
    TASK_QUEUE: str = "talk2data.requests"
    RESPONSE_QUEUE: str = "smile_queue_gateway"
    EXCHANGE: str = "smile_exchange"
    ROUTING_KEY: str = "talk2data_response"
    REMOTE_LLM: bool = True
    # доступны: qwen3-instruct-30b, gigachat-20b-a3b или gpt-oss-20b
    REMOTE_MODEL_NAME: str = "qwen3-instruct-30b"
    REMOTE_URL: str = "http://10.32.15.88:4000/v1"

    # LLM
    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    LLM_LOCAL_PATH: str = "models/model_weights_presaved"
    LLM_LOAD_FORMAT: str = "safetensors"

    # Speech To Text model
    STT_MODEL: str = "medium"
    METRICS_ENABLED: bool = True
    METRICS_PORT: int = 8003

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = AppSettings()
