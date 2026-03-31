"""MCP Server exposing food tracking tools."""

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from .tools import (
    analyze_food_photo,
    parse_ingredient_table,
    calculate_nutrition,
    search_product,
    log_meal,
    get_daily_summary,
    get_weekly_stats,
    set_user_goals,
    get_recommendations,
)
from ..monitoring.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

mcp = FastMCP("food-tracker", description="MCP сервер для трекинга питания и расчёта КБЖУ")


@mcp.tool()
def tool_analyze_food_photo(image_base64: str, media_type: str = "image/jpeg") -> str:
    """Анализирует фото еды и определяет блюдо, примерный состав КБЖУ и ингредиенты.
    Принимает base64-encoded изображение."""
    try:
        result = analyze_food_photo(image_base64, media_type)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("tool_analyze_food_photo error", error=str(e))
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def tool_parse_ingredient_table(image_base64: str, media_type: str = "image/jpeg") -> str:
    """Парсит фото таблицы ингредиентов или пищевой ценности, извлекает продукты и граммовку."""
    try:
        result = parse_ingredient_table(image_base64, media_type)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("tool_parse_ingredient_table error", error=str(e))
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def tool_calculate_nutrition(ingredients_json: str) -> str:
    """Рассчитывает КБЖУ по списку ингредиентов.
    Вход: JSON-строка вида [{"name": "продукт", "grams": 100}]"""
    try:
        ingredients = json.loads(ingredients_json)
        result = calculate_nutrition(ingredients)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("tool_calculate_nutrition error", error=str(e))
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def tool_search_product(query: str, n_results: int = 5) -> str:
    """Поиск продукта в базе данных питательной ценности по названию."""
    try:
        result = search_product(query, n_results)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("tool_search_product error", error=str(e))
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def tool_log_meal(
    session_id: str,
    description: str,
    calories: float,
    protein: float,
    fat: float,
    carbs: float,
    meal_type: str = "",
    fiber: float = 0.0,
    sugar: float = 0.0,
    sodium: float = 0.0,
) -> str:
    """Записывает приём пищи в дневник питания пользователя."""
    try:
        result = log_meal(
            session_id=session_id,
            description=description,
            calories=calories,
            protein=protein,
            fat=fat,
            carbs=carbs,
            meal_type=meal_type or None,
            fiber=fiber,
            sugar=sugar,
            sodium=sodium,
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("tool_log_meal error", error=str(e))
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def tool_get_daily_summary(session_id: str) -> str:
    """Возвращает сводку КБЖУ за текущий день: все приёмы пищи, итоги и остаток до нормы."""
    try:
        result = get_daily_summary(session_id)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("tool_get_daily_summary error", error=str(e))
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def tool_get_weekly_stats(session_id: str) -> str:
    """Возвращает статистику питания за последние 7 дней."""
    try:
        result = get_weekly_stats(session_id)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("tool_get_weekly_stats error", error=str(e))
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def tool_set_user_goals(
    session_id: str,
    calorie_goal: float = 0,
    protein_goal: float = 0,
    fat_goal: float = 0,
    carb_goal: float = 0,
    name: str = "",
) -> str:
    """Устанавливает или обновляет дневную норму КБЖУ пользователя."""
    try:
        result = set_user_goals(
            session_id=session_id,
            calorie_goal=calorie_goal if calorie_goal > 0 else None,
            protein_goal=protein_goal if protein_goal > 0 else None,
            fat_goal=fat_goal if fat_goal > 0 else None,
            carb_goal=carb_goal if carb_goal > 0 else None,
            name=name or None,
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("tool_set_user_goals error", error=str(e))
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def tool_get_recommendations(session_id: str) -> str:
    """Даёт рекомендации по питанию на основе текущего дневного баланса КБЖУ."""
    try:
        result = get_recommendations(session_id)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("tool_get_recommendations error", error=str(e))
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def main():
    logger.info("Starting Food Tracker MCP server")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
