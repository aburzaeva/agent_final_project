"""Streamlit UI for the Food Tracker agent."""

import base64
import os
import sys
import uuid

import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

sys.path.insert(0, PROJECT_ROOT)

from src.monitoring.logging_config import setup_logging, setup_langsmith, get_logger, metrics, track_request
from src.db.session_store import SessionStore
from src.agent.graph import compile_graph
from src.agent.state import AgentState
from src.validation.validators import validate_text_input, validate_image, ValidationError

setup_logging()
setup_langsmith()
logger = get_logger(__name__)

from langchain_core.messages import HumanMessage, AIMessage


# ─── Session init ──────────────────────────────────────────────────────

def init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "store" not in st.session_state:
        db_url = os.environ.get("DATABASE_URL", "sqlite:///./data/food_tracker.db")
        st.session_state.store = SessionStore(db_url)
    if "messages" not in st.session_state:
        store: SessionStore = st.session_state.store
        history = store.get_chat_history(st.session_state.session_id)
        st.session_state.messages = history if history else []
    if "agent" not in st.session_state:
        st.session_state.agent = compile_graph()


# ─── Dashboard ─────────────────────────────────────────────────────────

def render_dashboard():
    store: SessionStore = st.session_state.store
    summary = store.get_daily_summary(st.session_state.session_id)
    totals = summary.get("totals", {})
    goals = summary.get("goals", {})

    col1, col2, col3, col4 = st.columns(4)

    cal_pct = min(totals.get("calories", 0) / max(goals.get("calorie_goal", 2000), 1) * 100, 150)
    prot_pct = min(totals.get("protein", 0) / max(goals.get("protein_goal", 150), 1) * 100, 150)
    fat_pct = min(totals.get("fat", 0) / max(goals.get("fat_goal", 70), 1) * 100, 150)
    carb_pct = min(totals.get("carbs", 0) / max(goals.get("carb_goal", 250), 1) * 100, 150)

    with col1:
        st.metric(
            "Калории",
            f"{totals.get('calories', 0):.0f}",
            f"из {goals.get('calorie_goal', 2000):.0f}",
        )
        st.progress(min(cal_pct / 100, 1.0))

    with col2:
        st.metric(
            "Белки",
            f"{totals.get('protein', 0):.0f}г",
            f"из {goals.get('protein_goal', 150):.0f}г",
        )
        st.progress(min(prot_pct / 100, 1.0))

    with col3:
        st.metric(
            "Жиры",
            f"{totals.get('fat', 0):.0f}г",
            f"из {goals.get('fat_goal', 70):.0f}г",
        )
        st.progress(min(fat_pct / 100, 1.0))

    with col4:
        st.metric(
            "Углеводы",
            f"{totals.get('carbs', 0):.0f}г",
            f"из {goals.get('carb_goal', 250):.0f}г",
        )
        st.progress(min(carb_pct / 100, 1.0))


def render_weekly_chart():
    store: SessionStore = st.session_state.store
    stats = store.get_weekly_stats(st.session_state.session_id)
    days = stats.get("days", [])
    goals = stats.get("goals", {})

    if not days:
        st.info("Нет данных за неделю. Начните записывать приёмы пищи!")
        return

    dates = [d["date"] for d in days]
    calories = [d["calories"] for d in days]
    protein = [d["protein"] for d in days]
    fat = [d["fat"] for d in days]
    carbs = [d["carbs"] for d in days]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Калории", x=dates, y=calories, marker_color="#FF6B6B"))
    if goals.get("calorie_goal"):
        fig.add_hline(
            y=goals["calorie_goal"],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Цель: {goals['calorie_goal']} ккал",
        )
    fig.update_layout(title="Калории за неделю", xaxis_title="Дата", yaxis_title="ккал", height=300)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Белки", x=dates, y=protein, marker_color="#4ECDC4"))
    fig2.add_trace(go.Bar(name="Жиры", x=dates, y=fat, marker_color="#FFE66D"))
    fig2.add_trace(go.Bar(name="Углеводы", x=dates, y=carbs, marker_color="#A8E6CF"))
    fig2.update_layout(
        title="БЖУ за неделю",
        xaxis_title="Дата",
        yaxis_title="граммы",
        barmode="group",
        height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)


# ─── Agent interaction ─────────────────────────────────────────────────

@track_request(request_type="agent")
def run_agent(user_text: str, image_data: tuple | None = None):
    """Run the LangGraph agent and return the response."""
    agent = st.session_state.agent

    history_messages = []
    for msg in st.session_state.messages[-10:]:
        if msg["role"] == "user":
            history_messages.append(HumanMessage(content=msg["content"]))
        else:
            history_messages.append(AIMessage(content=msg["content"]))
    history_messages.append(HumanMessage(content=user_text))

    state = AgentState(
        session_id=st.session_state.session_id,
        messages=history_messages,
    )

    if image_data:
        img_base64, media_type = image_data
        state.image_base64 = img_base64
        state.image_media_type = media_type

    result = agent.invoke(state)

    response = result.get("final_response", "")
    if not response and result.get("messages"):
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content") and msg.content and not isinstance(msg, HumanMessage):
                response = msg.content
                break

    return response or "Не удалось обработать запрос. Попробуйте ещё раз."


# ─── Main App ──────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Food Tracker",
        page_icon="🥗",
        layout="wide",
    )

    init_session()

    st.title("🥗 Food Tracker")
    st.caption("AI-ассистент по питанию — отслеживайте КБЖУ, загружайте фото еды, получайте рекомендации")

    # Sidebar
    with st.sidebar:
        st.header("📊 Дашборд")
        render_dashboard()

        st.divider()

        with st.expander("📈 Статистика за неделю"):
            render_weekly_chart()

        st.divider()

        with st.expander("⚙️ Метрики и состояние"):
            m = metrics.get_metrics()
            st.write(f"**Статус:** {m['status']}")
            st.write(f"**Аптайм:** {m['uptime_seconds']:.0f}с")
            st.write(f"**Запросов:** {m['total_requests']}")
            st.write(f"**Ошибок:** {m['total_errors']} ({m['error_rate']:.1%})")
            st.write(f"**Среднее время ответа:** {m['avg_response_time_ms']:.0f}мс")
            st.write(f"**P95 время ответа:** {m['p95_response_time_ms']:.0f}мс")

        st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
        if st.button("🔄 Новая сессия"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

    # Chat area
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Photo upload
    uploaded_file = st.file_uploader(
        "📸 Загрузите фото еды или таблицы ингредиентов",
        type=["jpg", "jpeg", "png", "webp"],
        key="photo_upload",
    )

    image_data = None
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        media_type = uploaded_file.type or "image/jpeg"

        try:
            validate_image(image_bytes, media_type)
        except ValidationError as e:
            st.error(str(e))
            st.stop()

        img_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_data = (img_base64, media_type)

        st.image(image_bytes, caption="Загруженное фото", width=300)

        if st.button("🔍 Анализировать фото"):
            st.session_state.messages.append({"role": "user", "content": "📸 [Фото загружено]"})

            with st.chat_message("user"):
                st.markdown("📸 [Фото загружено]")

            with st.chat_message("assistant"):
                with st.spinner("Анализирую фото..."):
                    response = run_agent("Проанализируй это фото еды", image_data)
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

            store: SessionStore = st.session_state.store
            store.save_chat_message(st.session_state.session_id, "user", "[Фото]")
            store.save_chat_message(st.session_state.session_id, "assistant", response)
            st.rerun()

    # Text input
    if user_input := st.chat_input("Напишите что вы ели, спросите о продукте или попросите сводку..."):
        try:
            user_input = validate_text_input(user_input)
        except ValidationError as e:
            st.error(str(e))
            st.stop()

        logger.info("user_input", session_id=st.session_state.session_id, text_length=len(user_input))

        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                response = run_agent(user_input)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        store: SessionStore = st.session_state.store
        store.save_chat_message(st.session_state.session_id, "user", user_input)
        store.save_chat_message(st.session_state.session_id, "assistant", response)
        st.rerun()


if __name__ == "__main__":
    main()
