
# Talk2Data

## 🚀 Quick start (local, upd for SMILE)

### 1. Клонируем и устанавливаем среду

```bash
git clone https://github.com/mohammad-nour-alawad/talk2data.git && cd talk2data
```

### 2. Для локального запуска в .env копируем содержание .env.local, для прода - .env.prod

### 3. Перед самым первым запуском скачайте llm веса локально:
```bash
CUDA_VISIBLE_DEVICES=0 poetry run python -m core.download_model_script
```

### 4. Запускаем worker (посылает сообщения через rabbitmq) + инициализируем LLM
```bash
CUDA_VISIBLE_DEVICES=0 poetry run python -m core.worker
```
Когда всё загрузится, в логах получите сообщение вроде "Listening to queque..."

### 5. Запускаем контейнеры, (если они ещё не запущены)

### 6. Запускаем django на gateway и другие необходимые сервисы смайла (если они ещё не запущены)

### 7. (Optional) Если нужно протестить голосовой ввод, то запускаем следующее (локально или в контейнере):
```bash
sudo apt-get update && apt-get install -y ffmpeg
```

---

## 🗂 Repository layout

```
├── core/                  # LLM логика, конфиги, тесты и вспомогательные скрипты
│   ├── benchmarks/        # Наборы данных и скрипты для бенчмарков
│   ├── evaluation/        # Отчёты и инструменты оценки качества
│   ├── tests/             # Юнит-тесты
│   ├── config.py          # Настройки проекта
│   ├── download_model_script.py  # Скачивание весов моделей
│   ├── open_model_script.py      # Инициализация/запуск моделей
│   ├── prompts.py         # Подготовленные промпты
│   ├── schemas.py         # Pydantic-схемы
│   ├── workflow.py        # Основные пайплайны
│   └── worker.py          # Обработчик очереди сообщений
├── voice2text/            # Голосовой ввод и распознавание речи
│   └── whisper_model.py   # Обёртка над Whisper
├── Dockerfile.worker      # Образ worker-сервиса
├── docker-compose.yml     # Композиция сервисов
├── pyproject.toml         # Настройки poetry и зависимостей
├── poetry.lock            # Зафиксированные версии зависимостей
├── LICENSE                # Лицензия
└── README.md              # You are here
```
