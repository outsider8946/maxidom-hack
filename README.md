# Maxidom

Приложение для подбора строительных материалов и расчёта сметы ремонта.

## Архитектура

```
┌─────────────┐     HTTP      ┌──────────────────────────────┐
│   UI (8501) │ ──────────── │        Backend (8000)        │
│  Streamlit  │              │  FastAPI: RAG + Калькулятор  │
└─────────────┘              │  + Матчинг (FRIDA-Decisions) │
                             └──────────┬───────────────────┘
                                        │ HTTP
                                        ▼
                             ┌──────────────────┐
                             │  Ollama (11434)  │
                             │  LLM: qwen3.5:4b │
                             └──────────────────┘
```

**Сервисы:**
- **UI** — Streamlit-приложение, тонкий клиент
- **Backend** — FastAPI-сервер: семантический поиск (ChromaDB + embeddings), калькулятор смет, матчинг (FRIDA-Decisions)
- **Ollama** — LLM для выбора товаров (опционально)

---

## Требования
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Ollama (опционально, для LLM-выбора товаров)

## Установка

```bash
# MacOS (Apple Silicon)
uv sync --extra frida-mlx

# Windows/Linux с NVIDIA
uv sync --extra frida-torch
```

## Запуск

Нужно запустить **оба процесса** в разных терминалах:

**Терминал 1 — Backend:**
```bash
uv run uvicorn server.main:app --host 0.0.0.0 --port 8000
```

**Терминал 2 — UI:**
```bash
uv run streamlit run app.py
```

После этого откроется http://localhost:8501

## Данные

Каталог товаров (`data/ann_data_v1.json`, ~170 МБ) скачивается автоматически при первом запуске `DBLoader` с Google Drive. Если файл уже есть — загрузка пропускается.

ChromaDB находится в `src/maxidom_hack/db/chroma_db/` и загружается автоматически.

## Ollama (опционально)

Для включения LLM-выбора товаров:
```bash
ollama pull qwen3.5:4b
```

## Конфигурация

| Переменная | Описание | По умолчанию |
|---|---|---|
| `EMBED` | Модель эмбеддингов | `intfloat/multilingual-e5-small` |
| `LLM` | Модель LLM | `qwen3.5:4b` |
| `API_BASE` | URL backend для UI | `http://localhost:8000` |
| `CHROMA_DB_PATH` | Путь к ChromaDB | `src/maxidom_hack/db/chroma_db` |
| `OLLAMA_HOST` | URL Ollama | `http://localhost:11434` |

## API

Backend предоставляет REST API на `http://localhost:8000`:

| Метод | Путь | Описание |
|---|---|---|
| GET | `/api/v1/health` | Проверка доступности |
| POST | `/api/v1/pipeline` | Полный пайплайн: поиск → парсинг → LLM → смета |
| POST | `/api/v1/estimate` | Расчёт сметы по товарам |
| POST | `/api/v1/search` | Поиск товаров |
| POST | `/api/v1/match` | Матчинг товаров (поверхность + влагостойкость) |
| POST | `/api/v1/select-llm` | Выбор товара через LLM |
