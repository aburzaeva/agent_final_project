"""Tests for agent state and graph structure."""

import pytest
from src.agent.state import AgentState
from src.agent.graph import build_graph


class TestAgentState:
    def test_default_state(self):
        state = AgentState()
        assert state.input_type == "unknown"
        assert state.text_intent == "unknown"
        assert state.session_id == ""
        assert state.messages == []
        assert state.image_base64 is None
        assert state.needs_clarification is False

    def test_state_with_session(self):
        state = AgentState(session_id="test-123")
        assert state.session_id == "test-123"

    def test_state_with_image(self):
        state = AgentState(
            image_base64="abc123",
            image_media_type="image/png",
        )
        assert state.image_base64 == "abc123"
        assert state.image_media_type == "image/png"


class TestGraphStructure:
    def test_graph_builds(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_has_nodes(self):
        graph = build_graph()
        node_names = set(graph.nodes.keys())

        expected_nodes = {
            "route_input",
            "analyze_photo",
            "ask_clarification",
            "log_from_photo",
            "parse_table",
            "classify_text",
            "handle_log_meal",
            "handle_daily_summary",
            "handle_weekly_stats",
            "handle_recommendations",
            "handle_settings",
            "handle_search_product",
            "handle_general_chat",
        }
        assert expected_nodes.issubset(node_names)

    def test_graph_compiles(self):
        graph = build_graph()
        compiled = graph.compile()
        assert compiled is not None
