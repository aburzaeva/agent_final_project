"""Tool implementations for the Food Tracker MCP server."""

import base64
import json
import os
from typing import Optional

import anthropic

from ..db.session_store import SessionStore
from ..rag.retriever import ProductRetriever
from ..validation.validators import (
    validate_image,
    validate_nutrition_values,
    validate_meal_type,
    validate_goals,
    ValidationError,
)
from ..monitoring.logging_config import get_logger

logger = get_logger(__name__)

_store: SessionStore | None = None
_retriever: ProductRetriever | None = None
_anthropic_client: anthropic.Anthropic | None = None


def _get_store() -> SessionStore:
    global _store
    if _store is None:
        db_url = os.environ.get("DATABASE_URL", "sqlite:///./data/food_tracker.db")
        _store = SessionStore(db_url)
    return _store


def _get_retriever() -> ProductRetriever:
    global _retriever
    if _retriever is None:
        _retriever = ProductRetriever()
    return _retriever


def _get_anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def analyze_food_photo(image_base64: str, media_type: str = "image/jpeg") -> dict:
    """Analyze a food photo using Claude Vision and estimate the dish and nutrition."""
    logger.info("analyze_food_photo called", media_type=media_type)

    image_bytes = base64.b64decode(image_base64)
    validate_image(image_bytes, media_type)

    client = _get_anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Ты эксперт по питанию. Проанализируй это фото еды.\n"
                            "Определи блюдо/продукты и оцени примерный состав на 100г или на порцию.\n"
                            "Ответь строго в JSON:\n"
                            '{"dish_name": "название", "confidence": "high/medium/low", '
                            '"estimated_portion_grams": число, '
                            '"nutrition_per_portion": {"calories": число, "protein": число, "fat": число, "carbs": число}, '
                            '"ingredients": [{"name": "продукт", "grams": число}], '
                            '"needs_clarification": true/false, '
                            '"clarification_question": "вопрос если нужно"}'
                        ),
                    },
                ],
            }
        ],
    )

    text = response.content[0].text
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"dish_name": "unknown", "raw_response": text, "needs_clarification": True}


def parse_ingredient_table(image_base64: str, media_type: str = "image/jpeg") -> dict:
    """Parse a photo of an ingredient/nutrition table and extract structured data."""
    logger.info("parse_ingredient_table called")

    image_bytes = base64.b64decode(image_base64)
    validate_image(image_bytes, media_type)

    client = _get_anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Извлеки данные из этой таблицы ингредиентов/пищевой ценности.\n"
                            "Ответь строго в JSON:\n"
                            '{"ingredients": [{"name": "продукт", "grams": число}], '
                            '"nutrition_label": {"calories": число, "protein": число, '
                            '"fat": число, "carbs": число, "fiber": число, "sugar": число}, '
                            '"serving_size_grams": число, '
                            '"notes": "доп. информация"}'
                        ),
                    },
                ],
            }
        ],
    )

    text = response.content[0].text
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"error": "could not parse table", "raw_response": text}


def calculate_nutrition(ingredients: list[dict]) -> dict:
    """
    Calculate total KBZHU for a list of ingredients.
    Each ingredient: {"name": str, "grams": float}
    """
    logger.info("calculate_nutrition called", ingredient_count=len(ingredients))
    retriever = _get_retriever()
    return retriever.calculate_nutrition(ingredients)


def search_product(query: str, n_results: int = 5) -> list[dict]:
    """Search the product database by name."""
    logger.info("search_product called", query=query)
    retriever = _get_retriever()
    return retriever.search(query, n_results=n_results)


def log_meal(
    session_id: str,
    description: str,
    calories: float,
    protein: float,
    fat: float,
    carbs: float,
    meal_type: Optional[str] = None,
    fiber: float = 0.0,
    sugar: float = 0.0,
    sodium: float = 0.0,
) -> dict:
    """Log a meal entry for the user's daily diary."""
    logger.info("log_meal called", session_id=session_id, description=description)

    validate_nutrition_values(
        calories=calories, protein=protein, fat=fat,
        carbs=carbs, fiber=fiber, sugar=sugar, sodium=sodium,
    )
    meal_type = validate_meal_type(meal_type)

    store = _get_store()
    return store.log_meal(
        session_id=session_id,
        description=description,
        calories=calories,
        protein=protein,
        fat=fat,
        carbs=carbs,
        meal_type=meal_type,
        fiber=fiber,
        sugar=sugar,
        sodium=sodium,
    )


def get_daily_summary(session_id: str) -> dict:
    """Get the daily nutrition summary for the user."""
    logger.info("get_daily_summary called", session_id=session_id)
    store = _get_store()
    return store.get_daily_summary(session_id)


def get_weekly_stats(session_id: str) -> dict:
    """Get weekly nutrition statistics."""
    logger.info("get_weekly_stats called", session_id=session_id)
    store = _get_store()
    return store.get_weekly_stats(session_id)


def set_user_goals(
    session_id: str,
    calorie_goal: Optional[float] = None,
    protein_goal: Optional[float] = None,
    fat_goal: Optional[float] = None,
    carb_goal: Optional[float] = None,
    name: Optional[str] = None,
) -> dict:
    """Set or update user's daily nutrition goals."""
    logger.info("set_user_goals called", session_id=session_id)

    validated = validate_goals(calorie_goal, protein_goal, fat_goal, carb_goal)
    store = _get_store()
    return store.update_user_goals(
        session_id=session_id,
        name=name,
        **validated,
    )


def get_recommendations(session_id: str) -> dict:
    """Get nutrition recommendations based on today's remaining balance."""
    logger.info("get_recommendations called", session_id=session_id)

    store = _get_store()
    summary = store.get_daily_summary(session_id)
    remaining = summary.get("remaining", {})
    totals = summary.get("totals", {})
    goals = summary.get("goals", {})

    retriever = _get_retriever()

    recommendations = []
    if remaining.get("protein", 0) > 30:
        products = retriever.search("белок курица рыба творог", n_results=3)
        recommendations.append({
            "type": "protein_deficit",
            "message": f"Не хватает {remaining['protein']:.0f}г белка. Рекомендуемые продукты:",
            "products": [p["name"] for p in products],
        })

    if remaining.get("calories", 0) > 500:
        recommendations.append({
            "type": "calorie_deficit",
            "message": f"Осталось {remaining['calories']:.0f} ккал до нормы.",
        })

    cal_goal = goals.get("calorie_goal", 2000)
    if totals.get("calories", 0) > cal_goal:
        recommendations.append({
            "type": "calorie_excess",
            "message": f"Превышена норма калорий на {totals['calories'] - cal_goal:.0f} ккал.",
        })

    if not recommendations:
        recommendations.append({
            "type": "on_track",
            "message": "Вы на правильном пути! Питание сбалансировано.",
        })

    return {"summary": summary, "recommendations": recommendations}
