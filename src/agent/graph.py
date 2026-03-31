"""LangGraph definition for the Food Tracker agent with branching logic."""

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import (
    route_input,
    get_input_type,
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


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # ── Nodes ──
    graph.add_node("route_input", route_input)

    # Photo food branch
    graph.add_node("analyze_photo", analyze_photo_node)
    graph.add_node("ask_clarification", ask_clarification)
    graph.add_node("log_from_photo", log_from_photo)

    # Photo table branch
    graph.add_node("parse_table", parse_table_node)

    # Text branch
    graph.add_node("classify_text", classify_text_intent)
    graph.add_node("handle_log_meal", handle_log_meal)
    graph.add_node("handle_daily_summary", handle_daily_summary)
    graph.add_node("handle_weekly_stats", handle_weekly_stats)
    graph.add_node("handle_recommendations", handle_recommendations)
    graph.add_node("handle_settings", handle_settings)
    graph.add_node("handle_search_product", handle_search_product)
    graph.add_node("handle_general_chat", handle_general_chat)

    # ── Entry point ──
    graph.set_entry_point("route_input")

    # ── BRANCH 1: Input type routing ──
    graph.add_conditional_edges(
        "route_input",
        get_input_type,
        {
            "photo_food": "analyze_photo",
            "photo_table": "parse_table",
            "text": "classify_text",
            "unknown": "classify_text",
        },
    )

    # ── BRANCH 2: Clarification decision ──
    graph.add_conditional_edges(
        "analyze_photo",
        needs_clarification_check,
        {
            "needs_clarification": "ask_clarification",
            "no_clarification": "log_from_photo",
        },
    )

    # Clarification and log_from_photo → END
    graph.add_edge("ask_clarification", END)
    graph.add_edge("log_from_photo", END)

    # Table parsing → END
    graph.add_edge("parse_table", END)

    # ── BRANCH 3: Text intent routing ──
    graph.add_conditional_edges(
        "classify_text",
        get_text_intent,
        {
            "log_meal": "handle_log_meal",
            "daily_summary": "handle_daily_summary",
            "weekly_stats": "handle_weekly_stats",
            "recommendations": "handle_recommendations",
            "settings": "handle_settings",
            "search_product": "handle_search_product",
            "general_chat": "handle_general_chat",
            "unknown": "handle_general_chat",
        },
    )

    # All text handlers → END
    graph.add_edge("handle_log_meal", END)
    graph.add_edge("handle_daily_summary", END)
    graph.add_edge("handle_weekly_stats", END)
    graph.add_edge("handle_recommendations", END)
    graph.add_edge("handle_settings", END)
    graph.add_edge("handle_search_product", END)
    graph.add_edge("handle_general_chat", END)

    return graph


def compile_graph():
    """Compile the graph into a runnable."""
    graph = build_graph()
    return graph.compile()


agent = compile_graph()
