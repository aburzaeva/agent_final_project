"""Tests for MCP tools (database and RAG operations)."""

import os
import tempfile
import pytest

from src.db.session_store import SessionStore
from src.rag.indexer import build_index
from src.rag.retriever import ProductRetriever


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield f"sqlite:///{db_path}"
    os.unlink(db_path)


@pytest.fixture
def store(temp_db):
    return SessionStore(temp_db)


@pytest.fixture
def retriever(tmp_path):
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "products.csv")
    chroma_dir = str(tmp_path / "chroma")
    build_index(csv_path, chroma_dir)
    return ProductRetriever(chroma_dir)


class TestSessionStore:
    def test_create_user(self, store):
        user = store.get_or_create_user("test-session-1")
        assert user["session_id"] == "test-session-1"
        assert user["calorie_goal"] == 2000.0

    def test_get_existing_user(self, store):
        store.get_or_create_user("test-session-2")
        user = store.get_or_create_user("test-session-2")
        assert user["session_id"] == "test-session-2"

    def test_update_goals(self, store):
        store.get_or_create_user("test-session-3")
        result = store.update_user_goals("test-session-3", calorie_goal=1800, protein_goal=120)
        assert result["calorie_goal"] == 1800
        assert result["protein_goal"] == 120

    def test_log_meal(self, store):
        store.get_or_create_user("test-session-4")
        result = store.log_meal(
            session_id="test-session-4",
            description="Куриная грудка",
            calories=137,
            protein=29.8,
            fat=1.8,
            carbs=0.5,
            meal_type="lunch",
        )
        assert result["description"] == "Куриная грудка"
        assert result["calories"] == 137

    def test_daily_summary(self, store):
        store.get_or_create_user("test-session-5")
        store.log_meal("test-session-5", "Завтрак", 300, 20, 10, 40)
        store.log_meal("test-session-5", "Обед", 500, 30, 20, 60)

        summary = store.get_daily_summary("test-session-5")
        assert summary["totals"]["calories"] == 800
        assert summary["totals"]["protein"] == 50
        assert len(summary["meals"]) == 2

    def test_daily_summary_empty(self, store):
        summary = store.get_daily_summary("nonexistent")
        assert summary["meals"] == []

    def test_chat_history(self, store):
        store.save_chat_message("test-session-6", "user", "Hello")
        store.save_chat_message("test-session-6", "assistant", "Hi there!")
        history = store.get_chat_history("test-session-6")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["content"] == "Hi there!"


class TestProductRetriever:
    def test_search_chicken(self, retriever):
        results = retriever.search("куриная грудка")
        assert len(results) > 0
        names = [r["name"].lower() for r in results]
        assert any("куриная" in n or "курин" in n for n in names)

    def test_search_returns_nutrition(self, retriever):
        results = retriever.search("рис")
        assert len(results) > 0
        product = results[0]
        assert "calories" in product
        assert "protein" in product
        assert product["calories"] > 0

    def test_get_product(self, retriever):
        product = retriever.get_product("яблоко")
        assert product is not None
        assert product["calories"] > 0

    def test_calculate_nutrition(self, retriever):
        ingredients = [
            {"name": "куриная грудка", "grams": 200},
            {"name": "рис", "grams": 150},
        ]
        result = retriever.calculate_nutrition(ingredients)
        assert "total" in result
        assert "details" in result
        assert result["total"]["calories"] > 0
        assert len(result["details"]) == 2

    def test_search_no_results(self, retriever):
        results = retriever.search("xyznonexistentproduct123")
        assert isinstance(results, list)
