"""Input validation and safety checks for the food tracker agent."""

import io
import re
from PIL import Image

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_IMAGE_SIZE_MB = 10
MAX_TEXT_LENGTH = 2000
NUTRITION_RANGES = {
    "calories": (0, 5000),
    "protein": (0, 500),
    "fat": (0, 500),
    "carbs": (0, 1000),
    "fiber": (0, 200),
    "sugar": (0, 500),
    "sodium": (0, 10000),
}


class ValidationError(Exception):
    pass


def validate_image(image_bytes: bytes, content_type: str | None = None) -> bool:
    if content_type and content_type not in ALLOWED_IMAGE_TYPES:
        raise ValidationError(f"Недопустимый тип изображения: {content_type}. Допустимые: {ALLOWED_IMAGE_TYPES}")

    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise ValidationError(f"Изображение слишком большое: {size_mb:.1f} МБ. Максимум: {MAX_IMAGE_SIZE_MB} МБ")

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
    except Exception:
        raise ValidationError("Файл повреждён или не является изображением.")

    return True


def validate_text_input(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValidationError("Пустое сообщение. Пожалуйста, введите текст или загрузите фото.")

    if len(text) > MAX_TEXT_LENGTH:
        raise ValidationError(f"Сообщение слишком длинное ({len(text)} символов). Максимум: {MAX_TEXT_LENGTH}")

    return sanitize_text(text)


def sanitize_text(text: str) -> str:
    """Remove potentially dangerous patterns while keeping food-related content intact."""
    dangerous_patterns = [
        r"<script.*?>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
    ]
    cleaned = text
    for pattern in dangerous_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def validate_nutrition_values(**kwargs) -> dict:
    validated = {}
    for key, value in kwargs.items():
        if key in NUTRITION_RANGES and value is not None:
            min_val, max_val = NUTRITION_RANGES[key]
            if not (min_val <= value <= max_val):
                raise ValidationError(
                    f"Некорректное значение {key}: {value}. "
                    f"Допустимый диапазон: {min_val}–{max_val}"
                )
            validated[key] = round(value, 1)
    return validated


def validate_meal_type(meal_type: str | None) -> str | None:
    if meal_type is None:
        return None
    allowed = {"breakfast", "lunch", "dinner", "snack", "завтрак", "обед", "ужин", "перекус"}
    mapping = {"завтрак": "breakfast", "обед": "lunch", "ужин": "dinner", "перекус": "snack"}
    lower = meal_type.lower().strip()
    if lower in mapping:
        return mapping[lower]
    if lower in allowed:
        return lower
    raise ValidationError(f"Неизвестный тип приёма пищи: {meal_type}. Допустимые: завтрак, обед, ужин, перекус")


def validate_goals(calorie_goal=None, protein_goal=None, fat_goal=None, carb_goal=None) -> dict:
    goals = {}
    for name, value, range_ in [
        ("calorie_goal", calorie_goal, (800, 8000)),
        ("protein_goal", protein_goal, (10, 500)),
        ("fat_goal", fat_goal, (10, 400)),
        ("carb_goal", carb_goal, (20, 1000)),
    ]:
        if value is not None:
            min_v, max_v = range_
            if not (min_v <= value <= max_v):
                raise ValidationError(f"Некорректная цель {name}: {value}. Диапазон: {min_v}–{max_v}")
            goals[name] = value
    return goals
