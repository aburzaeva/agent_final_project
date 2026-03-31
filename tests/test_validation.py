"""Tests for input validation."""

import pytest
from src.validation.validators import (
    validate_text_input,
    validate_nutrition_values,
    validate_meal_type,
    validate_goals,
    sanitize_text,
    ValidationError,
)


class TestValidateTextInput:
    def test_valid_text(self):
        assert validate_text_input("Я съел яблоко") == "Я съел яблоко"

    def test_strips_whitespace(self):
        assert validate_text_input("  hello  ") == "hello"

    def test_empty_text_raises(self):
        with pytest.raises(ValidationError, match="Пустое сообщение"):
            validate_text_input("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValidationError, match="Пустое сообщение"):
            validate_text_input("   ")

    def test_too_long_text_raises(self):
        with pytest.raises(ValidationError, match="слишком длинное"):
            validate_text_input("a" * 2001)


class TestSanitizeText:
    def test_removes_script_tags(self):
        result = sanitize_text("hello <script>alert('xss')</script> world")
        assert "<script>" not in result
        assert "hello" in result

    def test_normal_text_unchanged(self):
        text = "Я съел 200г куриной грудки на обед"
        assert sanitize_text(text) == text


class TestValidateNutritionValues:
    def test_valid_values(self):
        result = validate_nutrition_values(calories=500, protein=30, fat=20, carbs=50)
        assert result == {"calories": 500.0, "protein": 30.0, "fat": 20.0, "carbs": 50.0}

    def test_negative_calories_raises(self):
        with pytest.raises(ValidationError):
            validate_nutrition_values(calories=-10)

    def test_excessive_calories_raises(self):
        with pytest.raises(ValidationError):
            validate_nutrition_values(calories=6000)

    def test_none_values_skipped(self):
        result = validate_nutrition_values(calories=100, protein=None)
        assert result == {"calories": 100.0}


class TestValidateMealType:
    def test_english_types(self):
        assert validate_meal_type("breakfast") == "breakfast"
        assert validate_meal_type("lunch") == "lunch"
        assert validate_meal_type("dinner") == "dinner"
        assert validate_meal_type("snack") == "snack"

    def test_russian_types(self):
        assert validate_meal_type("завтрак") == "breakfast"
        assert validate_meal_type("обед") == "lunch"
        assert validate_meal_type("ужин") == "dinner"
        assert validate_meal_type("перекус") == "snack"

    def test_none_returns_none(self):
        assert validate_meal_type(None) is None

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError, match="Неизвестный тип"):
            validate_meal_type("полдник")


class TestValidateGoals:
    def test_valid_goals(self):
        result = validate_goals(calorie_goal=1800, protein_goal=120)
        assert result == {"calorie_goal": 1800, "protein_goal": 120}

    def test_too_low_calories_raises(self):
        with pytest.raises(ValidationError):
            validate_goals(calorie_goal=500)

    def test_none_values_skipped(self):
        result = validate_goals(calorie_goal=2000)
        assert result == {"calorie_goal": 2000}
