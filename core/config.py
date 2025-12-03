from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_USER: str = "guest"
    RABBITMQ_PASS: str = "guest"
    TASK_QUEUE: str = "talk2data.requests"
    RESPONSE_QUEUE: str = "smile_queue_gateway"
    EXCHANGE: str = "smile_exchange"
    ROUTING_KEY: str = "talk2data_response"

    # LLM
    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    LLM_LOCAL_PATH: str = "models/model_weights_presaved"
    LLM_LOAD_FORMAT: str = "safetensors"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = AppSettings()
