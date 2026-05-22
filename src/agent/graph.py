"""LangGraph state graph for the travel agent RL system.

Defines the agent's control flow as a compiled ``StateGraph`` with nodes for
analysis, reasoning, tool use, plan updates, constraint checking, context
compression, and finalization.  Supports Long Horizon episodes (50 steps)
via the ``compress_context`` node.

Typical usage::

    graph = create_travel_agent()
    result = graph.invoke(initial_state)
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from src.agent.state import TravelState


# ── Node functions ───────────────────────────────────────────────────────
#
# Each node receives the current ``TravelState`` and returns a *dict* of
# fields to merge back into the state.  The actual LLM calls will be added
# later; these stubs establish the control-flow skeleton.


def analyze_node(state: TravelState) -> dict[str, Any]:
    """Parse the user query and extract constraints.

    Increments ``step_count`` and returns an update dict.
    """
    return {
        "step_count": state["step_count"] + 1,
    }


def think_node(state: TravelState) -> dict[str, Any]:
    """Reason about the current state and decide the next action.

    Increments ``step_count`` and returns an update dict.
    """
    return {
        "step_count": state["step_count"] + 1,
    }


def tool_node(state: TravelState) -> dict[str, Any]:
    """Execute a tool call and record the result.

    Increments ``step_count`` and returns an update dict.
    """
    return {
        "step_count": state["step_count"] + 1,
    }


def plan_node(state: TravelState) -> dict[str, Any]:
    """Update the current working plan based on gathered information.

    Increments ``step_count`` and returns an update dict.
    """
    return {
        "step_count": state["step_count"] + 1,
    }


def check_node(state: TravelState) -> dict[str, Any]:
    """Check whether all constraints are satisfied.

    Increments ``step_count`` and returns an update dict.
    """
    return {
        "step_count": state["step_count"] + 1,
    }


def compress_node(state: TravelState) -> dict[str, Any]:
    """Compress context window to stay within token budget.

    Does *not* increment step_count (compression is a housekeeping action).
    Returns an update dict.
    """
    return {}


def finalize_node(state: TravelState) -> dict[str, Any]:
    """Produce the final travel plan and mark the episode as complete.

    Sets ``is_complete`` to True and copies ``current_plan`` to
    ``final_plan``.
    """
    return {
        "final_plan": state.get("current_plan"),
        "is_complete": True,
    }


# ── Routing functions ────────────────────────────────────────────────────


def should_use_tool(state: TravelState) -> str:
    """Conditional edge: always routes to ``use_tool``.

    In a full implementation this would inspect the LLM's ``think`` output
    to decide whether a tool call is needed.  For the skeleton we always
    route to tool use.
    """
    return "use_tool"


def should_continue_v2(state: TravelState) -> str:
    """Decide whether to continue the main loop or finalize.

    Rules:
      - ``step_count >= max_steps`` --> ``"end"``  (hard stop)
      - ``step_count >= 0.8 * max_steps`` --> ``"finalize"`` (wrap up)
      - otherwise --> ``"continue"`` (keep planning)
    """
    step = state["step_count"]
    max_s = state["max_steps"]

    if step >= max_s:
        return "end"
    if step >= 0.8 * max_s:
        return "finalize"
    return "continue"


def loop_decision(state: TravelState) -> str:
    """Routing after ``check_constraints``.

    Always returns ``"normal"`` in the skeleton.  A full implementation
    could return ``"compress"`` when context is too large.
    """
    return "normal"


# ── Graph construction ───────────────────────────────────────────────────


def create_travel_agent() -> Any:
    """Build and compile the travel agent LangGraph.

    Returns a compiled ``StateGraph`` with the following nodes and edges:

    Nodes:
        analyze, think, use_tool, update_plan, check_constraints,
        compress_context, finalize

    Edges:
        analyze --> think
        think --> (use_tool | update_plan)   via should_use_tool
        use_tool --> update_plan
        update_plan --> check_constraints
        check_constraints --> (think | finalize | END)  via should_continue_v2
        compress_context --> think
        finalize --> END
    """
    graph = StateGraph(TravelState)

    # Register nodes
    graph.add_node("analyze", analyze_node)
    graph.add_node("think", think_node)
    graph.add_node("use_tool", tool_node)
    graph.add_node("update_plan", plan_node)
    graph.add_node("check_constraints", check_node)
    graph.add_node("compress_context", compress_node)
    graph.add_node("finalize", finalize_node)

    # Entry point
    graph.set_entry_point("analyze")

    # Static edges
    graph.add_edge("analyze", "think")
    graph.add_edge("use_tool", "update_plan")
    graph.add_edge("update_plan", "check_constraints")
    graph.add_edge("compress_context", "think")
    graph.add_edge("finalize", END)

    # Conditional edges
    # think -> use_tool or update_plan
    graph.add_conditional_edges(
        "think",
        should_use_tool,
        {
            "use_tool": "use_tool",
        },
    )

    # check_constraints -> think | finalize | END
    graph.add_conditional_edges(
        "check_constraints",
        should_continue_v2,
        {
            "continue": "think",
            "finalize": "finalize",
            "end": END,
        },
    )

    return graph.compile()
