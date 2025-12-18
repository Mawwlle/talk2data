
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

## 🗂 Repository layout (upd for SMILE)

```
.
├── core/                  # Всё, что используем в SMILE
└── README.md              # You are here
```
