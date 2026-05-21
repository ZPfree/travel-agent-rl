"""Dead-loop detector for the travel agent.

Detects when the agent is stuck in a repetitive tool-calling pattern and
provides a break strategy that suggests an alternative tool to try instead.

Two detection methods:
  1. Consecutive duplicates: the last *threshold* calls have identical
     tool_name **and** parameters.
  2. Window frequency: within the last *window* calls, any single tool
     appears more than *threshold* times (regardless of parameters).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.agent.state import ToolCall

# ── Loop-break mapping ──────────────────────────────────────────────────

LOOP_BREAK_MAP: dict[str, dict[str, str]] = {
    "amap_poi_search": {
        "new_tool": "google_search",
        "reason": "Repeated AMap POI search detected; switching to Google Search for broader coverage.",
    },
    "amap_nearby_search": {
        "new_tool": "amap_poi_search",
        "reason": "Repeated AMap nearby search detected; switching to AMap POI search for a wider query.",
    },
    "amap_route_plan": {
        "new_tool": "google_search",
        "reason": "Repeated AMap route planning detected; switching to Google Search for alternative routes.",
    },
    "google_search": {
        "new_tool": "tavily_search",
        "reason": "Repeated Google search detected; switching to Tavily for a different search perspective.",
    },
    "tavily_search": {
        "new_tool": "google_search",
        "reason": "Repeated Tavily search detected; switching to Google for a different search perspective.",
    },
}

_DEFAULT_STRATEGY: dict[str, str] = {
    "new_tool": "google_search",
    "reason": "Tool loop detected for an unmapped tool; falling back to Google Search.",
}


class LoopDetector:
    """Detects repetitive tool-call loops and suggests alternative tools.

    Args:
        window: Number of recent calls to consider for frequency-based
            detection (default 5).
        threshold: Minimum consecutive duplicates **or** maximum frequency
            within *window* that triggers a loop flag (default 3).
    """

    def __init__(self, window: int = 5, threshold: int = 3) -> None:
        self.window = window
        self.threshold = threshold

    # ── public API ───────────────────────────────────────────────────────

    def detect_loop(self, recent_calls: list[ToolCall]) -> bool:
        """Return True if the agent appears to be in a loop.

        Two heuristics are applied in order:
          1. **Consecutive duplicates** — the last *threshold* calls share
             the same ``tool_name`` **and** ``parameters``.
          2. **Window frequency** — within the last *window* calls, any
             single ``tool_name`` appears more than *threshold* times.
        """
        if len(recent_calls) < self.threshold:
            return False

        # Method 1: consecutive duplicates
        tail = recent_calls[-self.threshold :]
        first = tail[0]
        if all(
            c["tool_name"] == first["tool_name"] and c["parameters"] == first["parameters"]
            for c in tail
        ):
            return True

        # Method 2: window frequency
        window_calls = recent_calls[-self.window :]
        freq: Counter[str] = Counter(c["tool_name"] for c in window_calls)
        if any(count > self.threshold for count in freq.values()):
            return True

        return False

    def get_break_strategy(self, tool_name: str) -> dict[str, Any]:
        """Return a strategy dict suggesting an alternative tool.

        Keys:
            ``new_tool`` — the alternative tool to try.
            ``reason``   — human-readable explanation.
        """
        return dict(LOOP_BREAK_MAP.get(tool_name, _DEFAULT_STRATEGY))
