# Food Tracker — MCP-агент для трекинга питания

AI-агент на базе Claude Vision + LangGraph, который помогает отслеживать питание:
распознаёт блюда по фото, считает КБЖУ, ведёт дневник и даёт рекомендации.

## Стек

- **LLM**: Anthropic Claude (vision)
- **Агент**: LangGraph (нелинейный граф с ветвлениями)
- **MCP**: собственный MCP-сервер
- **RAG**: ChromaDB + база продуктов
- **UI**: Streamlit
- **БД**: SQLite
- **Мониторинг**: structlog + LangSmith

## Быстрый старт

```bash
# 1. Клонировать репозиторий
git clone <repo-url> && cd food-tracker

# 2. Создать виртуальное окружение
python -m venv .venv && source .venv/bin/activate

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Настроить переменные окружения
cp .env.example .env
# Отредактировать .env — указать ANTHROPIC_API_KEY

# 5. Проиндексировать базу продуктов
python -m src.rag.indexer

# 6. Запустить
streamlit run src/ui/app.py
```

## Docker

```bash
docker-compose up --build
```

Приложение будет доступно на `http://localhost:8501`.

## Структура проекта

```
src/
├── agent/        # LangGraph — граф агента
├── mcp_server/   # MCP-сервер с инструментами
├── rag/          # RAG: индексация и поиск продуктов
├── db/           # SQLAlchemy модели и сессии
├── ui/           # Streamlit интерфейс
├── validation/   # Валидация входных данных
└── monitoring/   # Логирование и мониторинг
```
