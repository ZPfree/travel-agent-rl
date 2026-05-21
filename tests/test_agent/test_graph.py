"""Tests for the travel agent LangGraph state graph.

Verifies that create_travel_agent() produces a compiled graph with all
required nodes and correct routing logic for the Long Horizon workflow.
"""

import pytest

from agent.graph import (
    create_travel_agent,
    analyze_node,
    think_node,
    tool_node,
    plan_node,
    check_node,
    compress_node,
    finalize_node,
    should_continue_v2,
    should_use_tool,
    loop_decision,
)
from agent.state import ProgressTracker, TravelState


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_state(**overrides) -> TravelState:
    """Return a minimal valid TravelState with optional overrides."""
    base: TravelState = {
        "query": "Plan a 5-day trip to Tokyo",
        "constraints": [],
        "messages": [],
        "tool_calls": [],
        "current_plan": None,
        "step_count": 0,
        "max_steps": 50,
        "constraint_satisfaction": 0.0,
        "final_plan": None,
        "is_complete": False,
        "progress_tracker": ProgressTracker(),
    }
    base.update(overrides)
    return base


# ── Graph compilation ────────────────────────────────────────────────────


class TestGraphCompiles:
    """Tests for create_travel_agent()."""

    def test_graph_compiles(self):
        """create_travel_agent() should return a non-None compiled graph."""
        graph = create_travel_agent()
        assert graph is not None

    def test_graph_is_compiled(self):
        """The returned graph should have an invoke method (i.e. is compiled)."""
        graph = create_travel_agent()
        assert hasattr(graph, "invoke")


# ── Graph nodes ──────────────────────────────────────────────────────────


class TestGraphNodes:
    """Tests that the graph contains all required nodes."""

    def test_graph_has_required_nodes(self):
        """Graph must contain analyze, think, use_tool, and finalize nodes."""
        graph = create_travel_agent()
        node_names = set(graph.nodes)
        required = {"analyze", "think", "use_tool", "finalize"}
        assert required.issubset(node_names), (
            f"Missing nodes: {required - node_names}"
        )

    def test_graph_has_long_horizon_nodes(self):
        """Graph must also contain update_plan, check_constraints, compress_context."""
        graph = create_travel_agent()
        node_names = set(graph.nodes)
        long_horizon = {"update_plan", "check_constraints", "compress_context"}
        assert long_horizon.issubset(node_names), (
            f"Missing Long Horizon nodes: {long_horizon - node_names}"
        )

    def test_graph_has_all_seven_nodes(self):
        """Graph should have exactly 7 user-defined nodes (plus internal __start__)."""
        graph = create_travel_agent()
        node_names = set(graph.nodes)
        expected = {
            "analyze",
            "think",
            "use_tool",
            "update_plan",
            "check_constraints",
            "compress_context",
            "finalize",
        }
        # LangGraph compiled graphs add __start__; verify our 7 are all present
        assert expected.issubset(node_names)
        # Verify no unexpected user nodes beyond our 7
        assert node_names - {"__start__"} == expected


# ── Node functions ───────────────────────────────────────────────────────


class TestNodeFunctions:
    """Each node function should return a dict that updates TravelState."""

    def test_analyze_node_returns_dict(self):
        state = _make_state()
        result = analyze_node(state)
        assert isinstance(result, dict)

    def test_think_node_returns_dict(self):
        state = _make_state()
        result = think_node(state)
        assert isinstance(result, dict)

    def test_tool_node_returns_dict(self):
        state = _make_state()
        result = tool_node(state)
        assert isinstance(result, dict)

    def test_plan_node_returns_dict(self):
        state = _make_state()
        result = plan_node(state)
        assert isinstance(result, dict)

    def test_check_node_returns_dict(self):
        state = _make_state()
        result = check_node(state)
        assert isinstance(result, dict)

    def test_compress_node_returns_dict(self):
        state = _make_state()
        result = compress_node(state)
        assert isinstance(result, dict)

    def test_finalize_node_returns_dict(self):
        state = _make_state()
        result = finalize_node(state)
        assert isinstance(result, dict)

    def test_analyze_node_increments_step(self):
        state = _make_state(step_count=0)
        result = analyze_node(state)
        assert result.get("step_count", state["step_count"]) == 1

    def test_finalize_node_sets_complete(self):
        state = _make_state(current_plan="Visit Tokyo Tower")
        result = finalize_node(state)
        assert result.get("is_complete") is True


# ── Routing functions ────────────────────────────────────────────────────


class TestRoutingFunctions:
    """Tests for should_use_tool, should_continue_v2, loop_decision."""

    def test_should_use_tool_returns_use_tool(self):
        state = _make_state()
        assert should_use_tool(state) == "use_tool"

    def test_should_continue_v2_normal_returns_continue(self):
        state = _make_state(step_count=5, max_steps=50)
        assert should_continue_v2(state) == "continue"

    def test_should_continue_v2_near_limit_returns_finalize(self):
        """At >= 80% of max_steps, should route to finalize."""
        state = _make_state(step_count=40, max_steps=50)
        assert should_continue_v2(state) == "finalize"

    def test_should_continue_v2_at_limit_returns_end(self):
        """At >= max_steps, should route to end."""
        state = _make_state(step_count=50, max_steps=50)
        assert should_continue_v2(state) == "end"

    def test_should_continue_v2_over_limit_returns_end(self):
        state = _make_state(step_count=55, max_steps=50)
        assert should_continue_v2(state) == "end"

    def test_loop_decision_returns_normal(self):
        state = _make_state()
        assert loop_decision(state) == "normal"
