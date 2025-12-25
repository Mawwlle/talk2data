
# Talk2Data

## 🚀 Quick start (local, upd for SMILE)

### 1. Клонируем и устанавливаем среду

```bash
git clone https://github.com/mohammad-nour-alawad/talk2data.git && cd talk2data
```

### 2. Для локального запуска в .env копируем содержание .env.local, для прода - .env.prod

### 3. Перед самым первым запуском скачайте llm веса локально (если запускаете через vllm):
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
├── .github/                                     # Конфигурация GitHub
│   └── workflows/                               # CI/CD пайплайны
│       └── ci.yml                               # Workflow для проверок
├── core/                                        # LLM логика, конфиги и вспомогательные скрипты
│   ├── benchmarks/                              # Наборы данных для бенчмарков
│   │   ├── iris_code_benchmark_en.json          # Кодовый бенчмарк (EN)
│   │   ├── iris_code_benchmark_ru.json          # Кодовый бенчмарк (RU)
│   │   ├── iris_dataset_benchmark_en.txt        # Датасет бенчмарка (EN)
│   │   ├── iris_text_benchmark_en.json          # Текстовый бенчмарк (EN)
│   │   └── iris_text_benchmark_ru.json          # Текстовый бенчмарк (RU)
│   ├── evaluation/                              # Оценка качества и отчёты
│   │   ├── inference_results/                   # Результаты инференса
│   │   │   ├── infer_0_baseline.json             # Выход базовой модели
│   │   │   ├── infer_1_local_llm_with_modifications.json # Выход локальной LLM
│   │   │   ├── infer_2_remote_llm_gigachat-20b-a3b.json # Выход GigaChat 20B
│   │   │   ├── infer_2_remote_llm_gpt-oss-20b.json      # Выход GPT-OSS 20B
│   │   │   ├── infer_2_remote_llm_qwen3-instruct-30b.json # Выход Qwen3 30B
│   │   ├── report_0_baseline/                    # Отчёт по baseline
│   │   │   ├── charts/                           # Диаграммы и метрики
│   │   │   │   ├── decision_by_difficulty.png    # Решения по сложности
│   │   │   │   ├── language_comparison.png       # Сравнение языков
│   │   │   │   ├── overall_scores.png            # Сводные оценки
│   │   │   │   └── semantic_similarity_by_difficulty.png # Семантическая близость
│   │   │   ├── plots/                            # Графики по кейсам
│   │   │   ├── cases.csv                         # Таблица кейсов
│   │   │   ├── cases_report_en.md                # Отчёт по кейсам (EN)
│   │   │   ├── cases_report_ru.md                # Отчёт по кейсам (RU)
│   │   │   └── summary.csv                       # Сводка метрик
│   │   ├── report_1_local_llm_with_modifications/ # Отчёт по локальной LLM
│   │   ├── report_2_remote_llm_gigachat-20b-a3b/  # Отчёт по GigaChat
│   │   ├── report_2_remote_llm_gpt-oss-20b/       # Отчёт по GPT-OSS
│   │   ├── report_2_remote_llm_qwen3-instruct-30b/ # Отчёт по Qwen3
│   │   ├── tools/                                # Инструменты оценки
│   │   │   ├── code_evaluation_tools.py          # Метрики для кода
│   │   │   ├── report_helpers.py                 # Помощники для отчётов
│   │   │   └── text_evaluation_tools.py          # Метрики для текста
│   │   ├── constants.py                          # Константы для оценки
│   │   ├── eval_script.py                        # Скрипт запуска оценивания
│   │   ├── inference_script.py                   # Скрипт инференса
│   │   ├── io_utils.py                           # Ввод/вывод утилиты
│   │   └── ports.py                              # Порты и интерфейсы
│   ├── observability/                            # Логирование и наблюдаемость
│   │   └── logging.py                            # Настройки логирования
│   ├── tests/                                    # Юнит-тесты core
│   │   ├── test_extract_code_block.py            # Тесты извлечения блоков кода
│   │   ├── test_format_prompt.py                 # Тесты форматирования промптов
│   │   ├── test_generate_code.py                 # Тесты генерации кода
│   │   ├── test_remote_chat_integration.py       # Интеграционные тесты чата
│   │   └── tools.py                              # Тестовые утилиты
│   ├── config.py                                 # Конфигурация LLM
│   ├── download_model_script.py                  # Загрузка весов моделей
│   ├── models.py                                 # Описания моделей
│   ├── prompts.py                                # Подготовленные промпты
│   ├── result.json                               # Пример результата
│   ├── schemas.py                                # Pydantic-схемы
│   ├── worker.py                                 # Worker для очереди
│   └── workflow.py                               # Основные пайплайны
├── task_management/                              # Слой управления задачами
│   ├── adapters/                                 # Внешние адаптеры
│   │   ├── conversation/                         # Адаптер диалога
│   │   │   └── result_persister.py                # Сохранение результатов
│   │   ├── task_queue/                            # Адаптер очередей
│   │   │   └── rabbitmq.py                        # Интеграция с RabbitMQ
│   │   ├── transcription/                        # Адаптер транскрипции
│   │   │   └── whisper_transcriber.py             # Обёртка Whisper
│   ├── app/                                      # Прикладной слой
│   │   ├── conversation/                          # API диалога
│   │   │   ├── api.py                             # Точки входа API
│   │   │   ├── schema.py                          # Схемы запросов/ответов
│   │   │   └── service.py                         # Сервис диалога
│   │   ├── transcription/                         # API транскрипции
│   │   │   ├── api.py                             # Точки входа API
│   │   │   ├── schemas.py                         # Схемы запросов/ответов
│   │   │   └── service.py                         # Сервис транскрипции
│   ├── domain/                                   # Доменные модели и порты
│   │   ├── conversation/                          # Домены диалога
│   │   │   ├── exceptions.py                      # Исключения домена
│   │   │   ├── models.py                          # Доменные модели
│   │   │   ├── ports.py                           # Абстракции портов
│   │   │   └── workflow.py                        # Доменный workflow
│   │   ├── task_queue/                            # Домены очередей задач
│   │   │   ├── models.py                          # Доменные модели
│   │   │   └── ports.py                           # Абстракции портов
│   │   ├── transcription/                         # Домены транскрипции
│   │   │   ├── exceptions.py                      # Исключения домена
│   │   │   ├── models.py                          # Доменные модели
│   │   │   └── ports.py                           # Абстракции портов
│   │   └── exceptions.py                          # Общие исключения домена
│   ├── observability/                            # Метрики и мониторинг
│   │   └── metrics.py                             # Метрики и экспорт
├── voice2text/                                   # Голосовой ввод
│   └── whisper_model.py                           # Обёртка над Whisper
├── .env.local                                    # Локальные переменные окружения
├── .env.prod                                     # Переменные окружения для прода
├── .gitignore                                    # Исключения Git
├── .pre-commit-config.yaml                       # Настройки pre-commit
├── docker-compose.yml                            # Композиция сервисов
├── Dockerfile.worker                             # Dockerfile worker-сервиса
├── LICENSE                                       # Лицензия проекта
├── Makefile                                      # Команды разработки
├── poetry.lock                                   # Зафиксированные зависимости
├── pyproject.toml                                # Настройки poetry и пакета
├── pytest.ini                                    # Конфигурация тестов
└── README.md                                     # Этот файл
```
