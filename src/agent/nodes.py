"""Graph node implementations for the LangGraph food tracker agent."""

import json
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .state import AgentState
from .prompts import SYSTEM_PROMPT, ROUTER_PROMPT, CLARIFICATION_PROMPT
from ..mcp_server.tools import (
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
from ..monitoring.logging_config import get_logger

logger = get_logger(__name__)

_llm = None


def _get_llm() -> ChatAnthropic:
    global _llm
    if _llm is None:
        _llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=2048)
    return _llm


# ─── BRANCH 1: Input Router ───────────────────────────────────────────

def route_input(state: AgentState) -> AgentState:
    """Determines the input type: photo of food, photo of table, or text."""
    logger.info("route_input", session_id=state.session_id)

    if state.image_base64:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content="Определи, что на фото: еда/блюдо или таблица ингредиентов/пищевой ценности. Ответь ОДНИМ словом: 'food' или 'table'."),
            HumanMessage(content=[
                {"type": "image", "source": {"type": "base64", "media_type": state.image_media_type, "data": state.image_base64}},
                {"type": "text", "text": "Что на этом фото?"},
            ]),
        ])
        answer = response.content.strip().lower()
        if "table" in answer or "таблиц" in answer:
            state.input_type = "photo_table"
        else:
            state.input_type = "photo_food"
    else:
        state.input_type = "text"

    logger.info("route_input result", input_type=state.input_type)
    return state


def get_input_type(state: AgentState) -> str:
    """Conditional edge: returns input_type for routing."""
    return state.input_type


# ─── Photo Food Branch ────────────────────────────────────────────────

def analyze_photo_node(state: AgentState) -> AgentState:
    """Analyzes a food photo using Claude Vision."""
    logger.info("analyze_photo_node")

    result = analyze_food_photo(state.image_base64, state.image_media_type)
    state.photo_analysis = result
    state.needs_clarification = result.get("needs_clarification", False)
    state.clarification_question = result.get("clarification_question")

    if result.get("ingredients"):
        state.ingredients = result["ingredients"]

    nutrition = result.get("nutrition_per_portion", {})
    state.nutrition_result = {
        "dish_name": result.get("dish_name", "Неизвестное блюдо"),
        "calories": nutrition.get("calories", 0),
        "protein": nutrition.get("protein", 0),
        "fat": nutrition.get("fat", 0),
        "carbs": nutrition.get("carbs", 0),
        "portion_grams": result.get("estimated_portion_grams", 0),
        "confidence": result.get("confidence", "low"),
    }

    return state


# ─── BRANCH 2: Clarification Decision ─────────────────────────────────

def needs_clarification_check(state: AgentState) -> str:
    """Conditional edge: decides if clarification is needed."""
    if state.needs_clarification and state.clarification_question:
        return "needs_clarification"
    return "no_clarification"


def ask_clarification(state: AgentState) -> AgentState:
    """Asks the user a clarifying question about the food photo."""
    logger.info("ask_clarification")

    nr = state.nutrition_result or {}
    msg = CLARIFICATION_PROMPT.format(
        dish_name=nr.get("dish_name", "блюдо"),
        confidence=nr.get("confidence", "low"),
        calories=nr.get("calories", 0),
        protein=nr.get("protein", 0),
        fat=nr.get("fat", 0),
        carbs=nr.get("carbs", 0),
        portion=nr.get("portion_grams", 0),
        clarification_question=state.clarification_question or "Подтвердите, пожалуйста, размер порции.",
    )
    state.final_response = msg
    state.messages = state.messages + [AIMessage(content=msg)]
    return state


def log_from_photo(state: AgentState) -> AgentState:
    """Logs the meal from photo analysis results."""
    logger.info("log_from_photo")

    nr = state.nutrition_result or {}
    result = log_meal(
        session_id=state.session_id,
        description=nr.get("dish_name", "Блюдо с фото"),
        calories=nr.get("calories", 0),
        protein=nr.get("protein", 0),
        fat=nr.get("fat", 0),
        carbs=nr.get("carbs", 0),
    )
    state.tool_result = result

    summary = get_daily_summary(state.session_id)
    totals = summary.get("totals", {})
    remaining = summary.get("remaining", {})

    msg = (
        f"Записано: **{nr.get('dish_name', 'Блюдо')}** (~{nr.get('portion_grams', '?')}г)\n\n"
        f"| | Значение |\n|---|---|\n"
        f"| Калории | {nr.get('calories', 0)} ккал |\n"
        f"| Белки | {nr.get('protein', 0)}г |\n"
        f"| Жиры | {nr.get('fat', 0)}г |\n"
        f"| Углеводы | {nr.get('carbs', 0)}г |\n\n"
        f"**Итого за день:** {totals.get('calories', 0):.0f} ккал "
        f"(осталось {remaining.get('calories', 0):.0f} ккал)"
    )
    state.final_response = msg
    state.messages = state.messages + [AIMessage(content=msg)]
    return state


# ─── Photo Table Branch ───────────────────────────────────────────────

def parse_table_node(state: AgentState) -> AgentState:
    """Parses an ingredient/nutrition table from photo."""
    logger.info("parse_table_node")

    result = parse_ingredient_table(state.image_base64, state.image_media_type)

    if result.get("ingredients"):
        state.ingredients = result["ingredients"]
        nutrition = calculate_nutrition(result["ingredients"])
        state.nutrition_result = nutrition.get("total", {})
        state.tool_result = nutrition

        total = nutrition.get("total", {})
        details = nutrition.get("details", [])
        detail_lines = "\n".join(
            f"- {d.get('name', '?')}: {d.get('grams', '?')}г → {d.get('calories', '?')} ккал"
            for d in details if "error" not in d
        )
        msg = (
            f"Рассчитано по таблице ингредиентов:\n\n{detail_lines}\n\n"
            f"**Итого:** {total.get('calories', 0)} ккал, "
            f"Б: {total.get('protein', 0)}г, "
            f"Ж: {total.get('fat', 0)}г, "
            f"У: {total.get('carbs', 0)}г\n\n"
            f"Хотите записать это в дневник?"
        )
    elif result.get("nutrition_label"):
        label = result["nutrition_label"]
        msg = (
            f"Данные с этикетки (на порцию {result.get('serving_size_grams', '?')}г):\n\n"
            f"- Калории: {label.get('calories', '?')} ккал\n"
            f"- Белки: {label.get('protein', '?')}г\n"
            f"- Жиры: {label.get('fat', '?')}г\n"
            f"- Углеводы: {label.get('carbs', '?')}г\n\n"
            f"Хотите записать это в дневник?"
        )
        state.nutrition_result = label
    else:
        msg = "Не удалось распознать таблицу. Попробуйте сделать фото чётче."

    state.final_response = msg
    state.messages = state.messages + [AIMessage(content=msg)]
    return state


# ─── BRANCH 3: Text Intent Router ─────────────────────────────────────

def classify_text_intent(state: AgentState) -> AgentState:
    """Classifies the user's text message intent."""
    logger.info("classify_text_intent")

    last_human = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_human = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if not last_human:
        state.text_intent = "general_chat"
        return state

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content="Ты классификатор запросов пользователя по питанию."),
        HumanMessage(content=ROUTER_PROMPT.format(user_message=last_human)),
    ])
    intent = response.content.strip().lower().strip('"').strip("'")

    intent_map = {
        "log_meal": "log_meal",
        "daily_summary": "daily_summary",
        "weekly_stats": "weekly_stats",
        "recommendations": "recommendations",
        "settings": "settings",
        "search_product": "search_product",
        "general_chat": "general_chat",
    }
    state.text_intent = intent_map.get(intent, "general_chat")
    logger.info("classify_text_intent result", intent=state.text_intent)
    return state


def get_text_intent(state: AgentState) -> str:
    """Conditional edge: returns text_intent for routing."""
    return state.text_intent


# ─── Text Intent Handlers ─────────────────────────────────────────────

def handle_log_meal(state: AgentState) -> AgentState:
    """Handles logging a meal from text description using LLM to extract nutrition."""
    logger.info("handle_log_meal")

    last_human = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_human = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if not last_human:
        state.final_response = "Опишите, что вы ели, чтобы я мог записать приём пищи."
        state.messages = state.messages + [AIMessage(content=state.final_response)]
        return state

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "Пользователь описывает что он съел. Извлеки данные и ответь JSON:\n"
            '{"description": "название блюда", "calories": число, "protein": число, '
            '"fat": число, "carbs": число, "meal_type": "breakfast/lunch/dinner/snack или null"}\n'
            "Если не можешь точно определить — оцени примерно на основе стандартных порций."
        )),
        HumanMessage(content=last_human),
    ])

    text = response.content
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        meal_data = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        state.final_response = "Не смог распознать приём пищи. Опишите подробнее, что вы ели и примерный размер порции."
        state.messages = state.messages + [AIMessage(content=state.final_response)]
        return state

    result = log_meal(
        session_id=state.session_id,
        description=meal_data.get("description", last_human),
        calories=meal_data.get("calories", 0),
        protein=meal_data.get("protein", 0),
        fat=meal_data.get("fat", 0),
        carbs=meal_data.get("carbs", 0),
        meal_type=meal_data.get("meal_type"),
    )
    state.tool_result = result

    summary = get_daily_summary(state.session_id)
    totals = summary.get("totals", {})
    remaining = summary.get("remaining", {})

    msg = (
        f"Записано: **{meal_data.get('description', 'Приём пищи')}**\n\n"
        f"- Калории: {meal_data.get('calories', 0)} ккал\n"
        f"- Белки: {meal_data.get('protein', 0)}г\n"
        f"- Жиры: {meal_data.get('fat', 0)}г\n"
        f"- Углеводы: {meal_data.get('carbs', 0)}г\n\n"
        f"**За день:** {totals.get('calories', 0):.0f}/{summary.get('goals', {}).get('calorie_goal', 2000)} ккал "
        f"(осталось {remaining.get('calories', 0):.0f})"
    )
    state.final_response = msg
    state.messages = state.messages + [AIMessage(content=msg)]
    return state


def handle_daily_summary(state: AgentState) -> AgentState:
    """Shows daily nutrition summary."""
    logger.info("handle_daily_summary")

    summary = get_daily_summary(state.session_id)
    meals = summary.get("meals", [])
    totals = summary.get("totals", {})
    goals = summary.get("goals", {})
    remaining = summary.get("remaining", {})

    if not meals:
        msg = "За сегодня пока нет записей. Расскажите, что вы ели!"
    else:
        meal_lines = "\n".join(
            f"- {m['description']}: {m['calories']} ккал "
            f"(Б:{m['protein']}г Ж:{m['fat']}г У:{m['carbs']}г)"
            for m in meals
        )
        msg = (
            f"**Сводка за {summary.get('date', 'сегодня')}:**\n\n"
            f"{meal_lines}\n\n"
            f"**Итого:** {totals['calories']:.0f} ккал | "
            f"Б: {totals['protein']:.0f}г | "
            f"Ж: {totals['fat']:.0f}г | "
            f"У: {totals['carbs']:.0f}г\n\n"
            f"**Цель:** {goals.get('calorie_goal', 2000)} ккал | "
            f"Осталось: {remaining.get('calories', 0):.0f} ккал"
        )

    state.final_response = msg
    state.messages = state.messages + [AIMessage(content=msg)]
    return state


def handle_weekly_stats(state: AgentState) -> AgentState:
    """Shows weekly nutrition statistics."""
    logger.info("handle_weekly_stats")

    stats = get_weekly_stats(state.session_id)
    days = stats.get("days", [])

    if not days:
        msg = "Нет данных за последнюю неделю. Начните записывать приёмы пищи!"
    else:
        lines = "\n".join(
            f"- {d['date']}: {d['calories']:.0f} ккал "
            f"(Б:{d['protein']:.0f}г Ж:{d['fat']:.0f}г У:{d['carbs']:.0f}г)"
            for d in days
        )
        avg_cal = sum(d["calories"] for d in days) / len(days)
        msg = (
            f"**Статистика за неделю:**\n\n{lines}\n\n"
            f"**Среднее:** {avg_cal:.0f} ккал/день"
        )

    state.final_response = msg
    state.messages = state.messages + [AIMessage(content=msg)]
    return state


def handle_recommendations(state: AgentState) -> AgentState:
    """Provides nutrition recommendations."""
    logger.info("handle_recommendations")

    result = get_recommendations(state.session_id)
    recs = result.get("recommendations", [])

    lines = "\n".join(f"- {r['message']}" for r in recs)
    msg = f"**Рекомендации:**\n\n{lines}"

    if any(r.get("products") for r in recs):
        for r in recs:
            if r.get("products"):
                msg += "\n\nРекомендуемые продукты: " + ", ".join(r["products"])

    state.final_response = msg
    state.messages = state.messages + [AIMessage(content=msg)]
    return state


def handle_settings(state: AgentState) -> AgentState:
    """Handles user settings/goals changes via LLM extraction."""
    logger.info("handle_settings")

    last_human = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_human = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if not last_human:
        state.final_response = "Укажите, какие цели по питанию вы хотите установить."
        state.messages = state.messages + [AIMessage(content=state.final_response)]
        return state

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "Пользователь хочет изменить свои цели по питанию. Извлеки значения и ответь JSON:\n"
            '{"calorie_goal": число_или_null, "protein_goal": число_или_null, '
            '"fat_goal": число_или_null, "carb_goal": число_или_null, "name": "имя_или_null"}'
        )),
        HumanMessage(content=last_human),
    ])

    text = response.content
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        goals_data = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        state.final_response = "Не удалось распознать цели. Укажите, например: «Поставь цель 1800 ккал, 120г белка»."
        state.messages = state.messages + [AIMessage(content=state.final_response)]
        return state

    result = set_user_goals(
        session_id=state.session_id,
        calorie_goal=goals_data.get("calorie_goal"),
        protein_goal=goals_data.get("protein_goal"),
        fat_goal=goals_data.get("fat_goal"),
        carb_goal=goals_data.get("carb_goal"),
        name=goals_data.get("name"),
    )

    msg = (
        f"Цели обновлены:\n\n"
        f"- Калории: {result.get('calorie_goal', '?')} ккал\n"
        f"- Белки: {result.get('protein_goal', '?')}г\n"
        f"- Жиры: {result.get('fat_goal', '?')}г\n"
        f"- Углеводы: {result.get('carb_goal', '?')}г"
    )
    if result.get("name"):
        msg = f"Привет, {result['name']}! " + msg

    state.final_response = msg
    state.messages = state.messages + [AIMessage(content=msg)]
    return state


def handle_search_product(state: AgentState) -> AgentState:
    """Searches for a product in the nutrition database."""
    logger.info("handle_search_product")

    last_human = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_human = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if not last_human:
        state.final_response = "Укажите название продукта для поиска."
        state.messages = state.messages + [AIMessage(content=state.final_response)]
        return state

    results = search_product(last_human, n_results=5)
    if not results:
        msg = f"Не нашла «{last_human}» в базе продуктов. Попробуйте другое название."
    else:
        lines = "\n".join(
            f"- **{p['name']}**: {p['calories']} ккал | "
            f"Б:{p['protein']}г Ж:{p['fat']}г У:{p['carbs']}г (на 100г)"
            for p in results
        )
        msg = f"**Найдено в базе:**\n\n{lines}"

    state.final_response = msg
    state.messages = state.messages + [AIMessage(content=msg)]
    return state


def handle_general_chat(state: AgentState) -> AgentState:
    """Handles general conversation about food/nutrition."""
    logger.info("handle_general_chat")

    last_human = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_human = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    if not last_human:
        state.final_response = "Пожалуйста, задайте вопрос о питании."
        state.messages = state.messages + [AIMessage(content=state.final_response)]
        return state

    llm = _get_llm()
    msgs = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=last_human)]
    response = llm.invoke(msgs)

    state.final_response = response.content
    state.messages = state.messages + [AIMessage(content=response.content)]
    return state
