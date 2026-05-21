"""Agent state definitions for the travel agent RL system.

- Constraint:   TypedDict describing a user constraint (budget, date, ...).
- ToolCall:     TypedDict recording a single tool invocation and its result.
- ProgressTracker: Pydantic model that tracks key information independently
                   of the context window.  Its ``to_summary()`` output is
                   injected into prompts so the agent always knows what it
                   has already accomplished.
- TravelState:  TypedDict used as the LangGraph state schema.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ── TypedDicts ───────────────────────────────────────────────────────────


class Constraint(TypedDict):
    """A single user constraint.

    Attributes:
        type:   Constraint category (e.g. "budget", "date", "preference").
        value:  Human-readable value (e.g. "5000 CNY", "2026-06-01").
        weight: Importance weight in [0, 1].
    """

    type: str
    value: str
    weight: float


class ToolCall(TypedDict):
    """Record of a single tool invocation.

    Attributes:
        tool_name:  Name of the tool that was called.
        parameters: Arguments passed to the tool.
        result:     Tool output (may be None on failure).
        step:       Agent step number when the call was made.
    """

    tool_name: str
    parameters: dict[str, Any]
    result: Any
    step: int


# ── ProgressTracker ──────────────────────────────────────────────────────


class ProgressTracker(BaseModel):
    """Tracks key information independently of the context window.

    The agent calls methods on this object as it gathers data.  Its
    ``to_summary()`` is injected into prompts so the LLM can see what
    has already been accomplished without consuming context tokens for
    every raw detail.
    """

    collected_pois: list[dict[str, Any]] = Field(default_factory=list)
    route_segments: list[dict[str, Any]] = Field(default_factory=list)
    constraint_status: dict[str, str] = Field(default_factory=dict)
    weather_cache: dict[str, dict[str, Any]] = Field(default_factory=dict)
    calendar_events: list[dict[str, Any]] = Field(default_factory=list)
    decisions_made: list[str] = Field(default_factory=list)

    # -- mutation helpers --------------------------------------------------

    def add_poi(self, poi: dict[str, Any]) -> None:
        """Record a point of interest collected during planning."""
        self.collected_pois.append(poi)

    def add_route(self, segment: dict[str, Any]) -> None:
        """Record a route segment."""
        self.route_segments.append(segment)

    def update_constraint(self, constraint_name: str, status: str) -> None:
        """Update the satisfaction status of a named constraint."""
        self.constraint_status[constraint_name] = status

    def cache_weather(self, city: str, weather: dict[str, Any]) -> None:
        """Cache weather data for a city."""
        self.weather_cache[city] = weather

    def add_decision(self, decision: str) -> None:
        """Record a planning decision."""
        self.decisions_made.append(decision)

    # -- summary -----------------------------------------------------------

    def to_summary(self) -> str:
        """Return a compact, human-readable summary of current progress.

        Designed to be injected into agent prompts so the LLM can see at a
        glance what has been accomplished without consuming context tokens
        for every raw detail.
        """
        # Constraint satisfaction stats
        total_constraints = len(self.constraint_status)
        satisfied = sum(
            1 for s in self.constraint_status.values() if s == "satisfied"
        )

        lines = [
            "=== Progress Summary ===",
            f"POIs collected: {len(self.collected_pois)}",
            f"Route segments: {len(self.route_segments)}",
            f"Constraints: {satisfied}/{total_constraints} satisfied",
            f"Weather cached: {len(self.weather_cache)} cities",
            f"Calendar events: {len(self.calendar_events)}",
            f"Decisions made: {len(self.decisions_made)}",
        ]

        # Show individual constraint status if any exist
        if self.constraint_status:
            lines.append("Constraint details:")
            for name, status in self.constraint_status.items():
                lines.append(f"  - {name}: {status}")

        return "\n".join(lines)


# ── TravelState (LangGraph state schema) ─────────────────────────────────


class TravelState(TypedDict):
    """LangGraph state schema for the travel planning agent.

    Attributes:
        query:        The user's original travel query.
        constraints:  Parsed list of user constraints.
        messages:     Chat history; annotated with ``add_messages`` so
                      LangGraph appends rather than overwrites.
        tool_calls:   Record of all tool invocations made so far.
        current_plan: The agent's current working plan (string or None).
        step_count:   Number of agent steps taken so far.
        max_steps:    Hard limit on steps (for safety / episode length).
        constraint_satisfaction: Aggregate satisfaction score [0, 1].
        final_plan:   The completed plan (set when is_complete=True).
        is_complete:  Whether the agent has finished planning.
        progress_tracker: Independent tracking of key planning progress.
    """

    query: str
    constraints: list[Constraint]
    messages: Annotated[list[BaseMessage], add_messages]
    tool_calls: list[ToolCall]
    current_plan: str | None
    step_count: int
    max_steps: int
    constraint_satisfaction: float
    final_plan: str | None
    is_complete: bool
    progress_tracker: ProgressTracker
