
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

### 4. Запускаем worker (посылает сообщения через rabbitmq)
```bash
CUDA_VISIBLE_DEVICES=0 poetry run python -m core.worker
```

### 5. Запускаем контейнеры, (если они ещё не запущены)

### 6. Запускаем django на gateway и другие необходимые сервисы смайла (если они ещё не запущены)

---

## 🗂 Repository layout (uod for SMILE)

```
.
├── core/                  # Всё, что используем в SMILE
├── talk2data-django/      # Частично перенесён на gateway
├── talk2data-fastAPI/     # FastAPI service running
├── demo.mp4               # 1-min demo.
├── LICENSE                # Apache‑2.0
└── README.md              # You are here
```

---

> **Status:** Public artifact accompanying the IEEE ICDM 2025 BIGIS Workshop paper *“A Multimodal Conversational Agent for Tabular Data Analysis”.*

https://github.com/user-attachments/assets/71d15e7b-1998-41f9-a689-a624aad7acb2

https://github.com/mohammad-nour-alawad/talk2data

---


## ✨ What is Talk2Data?

Talk2Data is an end‑to‑end **voice → code → visual + speech** agent that lets anyone explore tabular datasets conversationally.  It combines:

* **ASR** (OpenAI Whisper) to transcribe user speech.
* **LangGraph‑routed LLM** (Qwen‑2.5‑Coder) that decides whether to answer directly or generate Python.
* **Secure sandbox** that executes LLM‑generated Pandas / Matplotlib code with zero network / FS writes.
* **TTS** (Coqui VITS) so the agent can read results aloud.
* **Memory** to support multi‑turn follow‑ups like “Now colour by gender”.

While our evaluation uses public benchmarks, the framework is designed to generalize to intelligence and security contexts where analysts need rapid, conversational access to large structured datasets.

A 1‑min demo video (📽 `demo.mp4`) shows the full workflow.

---

## 📥 Datasets

We evaluate on three publicly available tabular datasets (not included in the repo). but below are the Kaggle links to the datasets:

```bash
# Otto Group Product Classification (≈ 27 MB)
[kaggle competitions download -c otto-group-product-classification-challenge -p data/otto && unzip data/otto/*.zip -d talk2data-fastAPI/evaluation_benchmark
](https://www.kaggle.com/competitions/otto-group-product-classification-challenge/data)

# US Flights 2008 (≈ 673 MB)
[kaggle datasets download -d vikalpdongre/us-flights-data-2008 -p data/flights2008 && unzip data/flights2008/*.zip -d talk2data-fastAPI/evaluation_benchmark
](https://www.kaggle.com/datasets/vikalpdongre/us-flights-data-2008)

# Students Performance in Exams (≈ 70 KB)
[kaggle datasets download -d spscientist/students-performance-in-exams -p data/students && unzip data/students/*.zip -d talk2data-fastAPI/evaluation_benchmark
](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
```

And for each dataset we created 16 natural language commands, a total of 48 commands can be found under `talk2data-fastAPI/evaluation_benchmark`.

---

## 📊 Reproducing paper benchmarks

The code to run the evaluation is `talk2data-fastAPI/experiment.py`, it will generate the output inside `talk2data-fastAPI/evaluation_results` along with all metrics.

---

## 📝 Associated publication

If you use Talk2Data in academic work, please cite:

**A Multimodal Conversational Agent for Tabular Data Analysis**  
_Mohammad Nour Al Awad, Sergey Ivanov, Olga Tikhonova, Ivan Khodnenko_

In: **IEEE International Conference on Data Mining (ICDM) – BIGIS Workshop, 2025.**

[📄 Paper link (IEEE Xplore)](https://doi.org/...) — to be added once DOI is live.

---

## 📊 Citation
```bibtex
@misc{talk2data2025,
  author       = {Mohammad Nour Al Awad and Sergey Ivanov and Olga Tikhonova and Ivan Khodnenko},
  title        = {A Multimodal Conversational Agent for Tabular Data Analysis},
  booktitle    = {IEEE International Conference on Data Mining (ICDM), BIGIS Workshop},
  year         = {2025},
  publisher    = {IEEE}
}
```

---

## 🔐 License

Apache License 2.0 — see `LICENSE`.  Demo datasets retain their original Kaggle licences.
