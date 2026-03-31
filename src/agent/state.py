"""Agent state definition for the LangGraph food tracker."""

from typing import Annotated, Literal, Optional
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """State that flows through the LangGraph agent."""

    messages: Annotated[list, add_messages] = Field(default_factory=list)
    session_id: str = ""

    input_type: Literal["photo_food", "photo_table", "text", "unknown"] = "unknown"
    image_base64: Optional[str] = None
    image_media_type: str = "image/jpeg"

    photo_analysis: Optional[dict] = None
    needs_clarification: bool = False
    clarification_question: Optional[str] = None

    ingredients: Optional[list[dict]] = None
    nutrition_result: Optional[dict] = None

    text_intent: Literal[
        "log_meal", "daily_summary", "weekly_stats",
        "recommendations", "settings", "search_product",
        "general_chat", "unknown",
    ] = "unknown"

    tool_result: Optional[dict] = None
    final_response: Optional[str] = None
