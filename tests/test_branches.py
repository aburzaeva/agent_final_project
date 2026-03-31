"""Integration tests verifying correct responses on each graph branch."""

import json
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage

from src.agent.state import AgentState
from src.agent.nodes import (
    route_input,
    analyze_photo_node,
    needs_clarification_check,
    ask_clarification,
    log_from_photo,
    parse_table_node,
    classify_text_intent,
    get_text_intent,
    handle_log_meal,
    handle_daily_summary,
    handle_weekly_stats,
    handle_recommendations,
    handle_settings,
    handle_search_product,
    handle_general_chat,
)


class TestBranch1_InputRouting:
    """Branch 1: route_input determines photo_food / photo_table / text."""

    def test_text_input_routes_to_text(self):
        state = AgentState(
            session_id="test-1",
            messages=[HumanMessage(content="Я съел яблоко")],
        )
        result = route_input(state)
        assert result.input_type == "text"

    @patch("src.agent.nodes._get_llm")
    def test_photo_food_routes_to_photo_food(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = "food"
        mock_llm.return_value.invoke.return_value = mock_response

        state = AgentState(
            session_id="test-2",
            image_base64="fake_base64",
            image_media_type="image/jpeg",
            messages=[HumanMessage(content="Фото")],
        )
        result = route_input(state)
        assert result.input_type == "photo_food"

    @patch("src.agent.nodes._get_llm")
    def test_photo_table_routes_to_photo_table(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = "table"
        mock_llm.return_value.invoke.return_value = mock_response

        state = AgentState(
            session_id="test-3",
            image_base64="fake_base64",
            image_media_type="image/jpeg",
            messages=[HumanMessage(content="Фото")],
        )
        result = route_input(state)
        assert result.input_type == "photo_table"


class TestBranch2_ClarificationDecision:
    """Branch 2: after analyze_photo, decides needs_clarification or not."""

    def test_no_clarification_needed(self):
        state = AgentState(
            session_id="test-4",
            needs_clarification=False,
            clarification_question=None,
        )
        assert needs_clarification_check(state) == "no_clarification"

    def test_clarification_needed(self):
        state = AgentState(
            session_id="test-5",
            needs_clarification=True,
            clarification_question="Какой размер порции?",
        )
        assert needs_clarification_check(state) == "needs_clarification"

    def test_ask_clarification_produces_response(self):
        state = AgentState(
            session_id="test-6",
            needs_clarification=True,
            clarification_question="Какой размер порции?",
            nutrition_result={
                "dish_name": "Борщ",
                "confidence": "medium",
                "calories": 150,
                "protein": 8,
                "fat": 5,
                "carbs": 18,
                "portion_grams": 300,
            },
            messages=[HumanMessage(content="Фото")],
        )
        result = ask_clarification(state)
        assert result.final_response is not None
        assert "Борщ" in result.final_response
        assert len(result.messages) > 1

    @patch("src.agent.nodes.log_meal")
    @patch("src.agent.nodes.get_daily_summary")
    def test_log_from_photo_produces_response(self, mock_summary, mock_log):
        mock_log.return_value = {"id": "1"}
        mock_summary.return_value = {
            "totals": {"calories": 150},
            "remaining": {"calories": 1850},
        }

        state = AgentState(
            session_id="test-7",
            nutrition_result={
                "dish_name": "Салат",
                "calories": 150,
                "protein": 5,
                "fat": 8,
                "carbs": 12,
                "portion_grams": 200,
            },
            messages=[HumanMessage(content="Фото")],
        )
        result = log_from_photo(state)
        assert result.final_response is not None
        assert "Салат" in result.final_response
        assert "150" in result.final_response


class TestBranch3_TextIntentRouting:
    """Branch 3: classify_text_intent routes to different handlers."""

    @patch("src.agent.nodes._get_llm")
    def test_classify_log_meal(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = "log_meal"
        mock_llm.return_value.invoke.return_value = mock_response

        state = AgentState(
            session_id="test-10",
            messages=[HumanMessage(content="Я съел куриную грудку 200г")],
        )
        result = classify_text_intent(state)
        assert result.text_intent == "log_meal"

    @patch("src.agent.nodes._get_llm")
    def test_classify_daily_summary(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = "daily_summary"
        mock_llm.return_value.invoke.return_value = mock_response

        state = AgentState(
            session_id="test-11",
            messages=[HumanMessage(content="Покажи сводку за день")],
        )
        result = classify_text_intent(state)
        assert result.text_intent == "daily_summary"

    @patch("src.agent.nodes._get_llm")
    def test_classify_search_product(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = "search_product"
        mock_llm.return_value.invoke.return_value = mock_response

        state = AgentState(
            session_id="test-12",
            messages=[HumanMessage(content="Сколько калорий в рисе?")],
        )
        result = classify_text_intent(state)
        assert result.text_intent == "search_product"


class TestTextHandlers:
    """Each text intent handler produces a valid response."""

    @patch("src.agent.nodes.get_daily_summary")
    @patch("src.agent.nodes.log_meal")
    @patch("src.agent.nodes._get_llm")
    def test_handle_log_meal(self, mock_llm, mock_log, mock_summary):
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "description": "Куриная грудка",
            "calories": 274,
            "protein": 58,
            "fat": 3.6,
            "carbs": 0,
            "meal_type": "lunch",
        })
        mock_llm.return_value.invoke.return_value = mock_response
        mock_log.return_value = {"id": "1"}
        mock_summary.return_value = {
            "totals": {"calories": 274},
            "goals": {"calorie_goal": 2000},
            "remaining": {"calories": 1726},
        }

        state = AgentState(
            session_id="test-20",
            messages=[HumanMessage(content="Я съел куриную грудку 200г")],
        )
        result = handle_log_meal(state)
        assert result.final_response is not None
        assert "Куриная грудка" in result.final_response

    @patch("src.agent.nodes.get_daily_summary")
    def test_handle_daily_summary_empty(self, mock_summary):
        mock_summary.return_value = {"meals": [], "totals": {}, "goals": {}, "remaining": {}}

        state = AgentState(
            session_id="test-21",
            messages=[HumanMessage(content="Покажи сводку")],
        )
        result = handle_daily_summary(state)
        assert result.final_response is not None
        assert "нет записей" in result.final_response.lower()

    @patch("src.agent.nodes.get_daily_summary")
    def test_handle_daily_summary_with_meals(self, mock_summary):
        mock_summary.return_value = {
            "date": "2026-03-31",
            "meals": [
                {"description": "Овсянка", "calories": 300, "protein": 10, "fat": 5, "carbs": 50},
            ],
            "totals": {"calories": 300, "protein": 10, "fat": 5, "carbs": 50},
            "goals": {"calorie_goal": 2000},
            "remaining": {"calories": 1700},
        }

        state = AgentState(
            session_id="test-22",
            messages=[HumanMessage(content="Покажи сводку за день")],
        )
        result = handle_daily_summary(state)
        assert result.final_response is not None
        assert "Овсянка" in result.final_response
        assert "300" in result.final_response

    @patch("src.agent.nodes.get_weekly_stats")
    def test_handle_weekly_stats_empty(self, mock_stats):
        mock_stats.return_value = {"days": []}

        state = AgentState(
            session_id="test-23",
            messages=[HumanMessage(content="Статистика за неделю")],
        )
        result = handle_weekly_stats(state)
        assert result.final_response is not None
        assert "нет данных" in result.final_response.lower()

    @patch("src.agent.nodes.get_weekly_stats")
    def test_handle_weekly_stats_with_data(self, mock_stats):
        mock_stats.return_value = {
            "days": [
                {"date": "2026-03-30", "calories": 1800, "protein": 120, "fat": 60, "carbs": 200},
                {"date": "2026-03-31", "calories": 2100, "protein": 140, "fat": 70, "carbs": 250},
            ]
        }

        state = AgentState(
            session_id="test-24",
            messages=[HumanMessage(content="Статистика за неделю")],
        )
        result = handle_weekly_stats(state)
        assert result.final_response is not None
        assert "Среднее" in result.final_response

    @patch("src.agent.nodes.get_recommendations")
    def test_handle_recommendations(self, mock_recs):
        mock_recs.return_value = {
            "recommendations": [
                {"type": "protein_deficit", "message": "Не хватает 50г белка.", "products": ["Курица", "Творог"]},
            ]
        }

        state = AgentState(
            session_id="test-25",
            messages=[HumanMessage(content="Дай рекомендации")],
        )
        result = handle_recommendations(state)
        assert result.final_response is not None
        assert "Рекомендации" in result.final_response
        assert "белка" in result.final_response

    @patch("src.agent.nodes.set_user_goals")
    @patch("src.agent.nodes._get_llm")
    def test_handle_settings(self, mock_llm, mock_goals):
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "calorie_goal": 1800,
            "protein_goal": 120,
            "fat_goal": None,
            "carb_goal": None,
            "name": "Алекс",
        })
        mock_llm.return_value.invoke.return_value = mock_response
        mock_goals.return_value = {
            "calorie_goal": 1800,
            "protein_goal": 120,
            "fat_goal": 70,
            "carb_goal": 250,
            "name": "Алекс",
        }

        state = AgentState(
            session_id="test-26",
            messages=[HumanMessage(content="Поставь цель 1800 калорий, 120г белка")],
        )
        result = handle_settings(state)
        assert result.final_response is not None
        assert "1800" in result.final_response
        assert "Алекс" in result.final_response

    @patch("src.agent.nodes.search_product")
    def test_handle_search_product_found(self, mock_search):
        mock_search.return_value = [
            {"name": "Рис белый", "calories": 344, "protein": 6.7, "fat": 0.7, "carbs": 78.9},
        ]

        state = AgentState(
            session_id="test-27",
            messages=[HumanMessage(content="Рис")],
        )
        result = handle_search_product(state)
        assert result.final_response is not None
        assert "Рис" in result.final_response

    @patch("src.agent.nodes.search_product")
    def test_handle_search_product_not_found(self, mock_search):
        mock_search.return_value = []

        state = AgentState(
            session_id="test-28",
            messages=[HumanMessage(content="xyzunknownproduct")],
        )
        result = handle_search_product(state)
        assert result.final_response is not None
        assert "нашла" in result.final_response.lower() or "Не" in result.final_response

    @patch("src.agent.nodes._get_llm")
    def test_handle_general_chat(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = "Здравствуйте! Я помогу вам с питанием."
        mock_llm.return_value.invoke.return_value = mock_response

        state = AgentState(
            session_id="test-29",
            messages=[HumanMessage(content="Привет")],
        )
        result = handle_general_chat(state)
        assert result.final_response is not None
        assert len(result.final_response) > 0


class TestMetrics:
    """Test that metrics collector works correctly."""

    def test_metrics_collection(self):
        from src.monitoring.logging_config import MetricsCollector

        m = MetricsCollector()
        m.record_request_start()
        m.record_request_end(0.5, "test")

        result = m.get_metrics()
        assert result["total_requests"] == 1
        assert result["active_requests"] == 0
        assert result["total_errors"] == 0
        assert result["status"] == "healthy"
        assert result["avg_response_time_ms"] == 500.0

    def test_error_tracking(self):
        from src.monitoring.logging_config import MetricsCollector

        m = MetricsCollector()
        m.record_request_start()
        m.record_error()
        m.record_request_end(0.1, "test")

        result = m.get_metrics()
        assert result["total_errors"] == 1
        assert result["error_rate"] > 0
