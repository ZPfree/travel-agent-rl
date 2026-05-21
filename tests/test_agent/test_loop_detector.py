"""Tests for LoopDetector — dead-loop detection and breaking strategies."""

from __future__ import annotations

import pytest

from src.agent.loop_detector import LoopDetector
from src.agent.state import ToolCall


# ── helpers ──────────────────────────────────────────────────────────────


def _make_call(tool_name: str, params: dict, step: int, result: str = "ok") -> ToolCall:
    return ToolCall(tool_name=tool_name, parameters=params, result=result, step=step)


# ── test: consecutive duplicates detected ────────────────────────────────


class TestDetectConsecutiveDuplicates:
    """3 consecutive calls with same tool + same params should be flagged."""

    def test_detect_consecutive_duplicates(self) -> None:
        detector = LoopDetector(window=5, threshold=3)
        calls = [
            _make_call("amap_poi_search", {"city": "Beijing"}, step=1),
            _make_call("amap_poi_search", {"city": "Beijing"}, step=2),
            _make_call("amap_poi_search", {"city": "Beijing"}, step=3),
        ]
        assert detector.detect_loop(calls) is True

    def test_threshold_not_met_returns_false(self) -> None:
        """Only 2 consecutive duplicates with threshold=3 → no loop."""
        detector = LoopDetector(window=5, threshold=3)
        calls = [
            _make_call("amap_poi_search", {"city": "Beijing"}, step=1),
            _make_call("amap_poi_search", {"city": "Beijing"}, step=2),
        ]
        assert detector.detect_loop(calls) is False


# ── test: same tool but different params → no loop ───────────────────────


class TestNoLoopDifferentParams:
    """Same tool name but varying parameters should NOT trigger loop detection."""

    def test_no_loop_different_params(self) -> None:
        detector = LoopDetector(window=5, threshold=3)
        calls = [
            _make_call("amap_poi_search", {"city": "Beijing"}, step=1),
            _make_call("amap_poi_search", {"city": "Shanghai"}, step=2),
            _make_call("amap_poi_search", {"city": "Guangzhou"}, step=3),
        ]
        assert detector.detect_loop(calls) is False


# ── test: window-based frequency detection ───────────────────────────────


class TestWindowFrequencyDetection:
    """Within a sliding window, too many calls to the same tool → loop."""

    def test_frequency_loop_in_window(self) -> None:
        """4 out of 5 calls to the same tool exceeds the frequency threshold."""
        detector = LoopDetector(window=5, threshold=3)
        calls = [
            _make_call("amap_poi_search", {"city": "A"}, step=1),
            _make_call("google_search", {"q": "hotels"}, step=2),
            _make_call("amap_poi_search", {"city": "B"}, step=3),
            _make_call("amap_poi_search", {"city": "C"}, step=4),
            _make_call("amap_poi_search", {"city": "D"}, step=5),
        ]
        # 4 calls to amap_poi_search within window of 5 → loop
        assert detector.detect_loop(calls) is True

    def test_no_loop_when_frequencies_spread(self) -> None:
        """Each tool appears ≤ threshold times → no loop."""
        detector = LoopDetector(window=5, threshold=3)
        calls = [
            _make_call("amap_poi_search", {"city": "A"}, step=1),
            _make_call("google_search", {"q": "x"}, step=2),
            _make_call("amap_route_plan", {"from": "A", "to": "B"}, step=3),
            _make_call("amap_poi_search", {"city": "B"}, step=4),
            _make_call("google_search", {"q": "y"}, step=5),
        ]
        assert detector.detect_loop(calls) is False


# ── test: get_break_strategy ────────────────────────────────────────────


class TestBreakStrategy:
    """get_break_strategy should return a dict with new_tool and reason."""

    def test_break_strategy_amap_poi_search(self) -> None:
        detector = LoopDetector()
        strategy = detector.get_break_strategy("amap_poi_search")
        assert strategy["new_tool"] == "google_search"
        assert "reason" in strategy

    def test_break_strategy_amap_nearby_search(self) -> None:
        detector = LoopDetector()
        strategy = detector.get_break_strategy("amap_nearby_search")
        assert strategy["new_tool"] == "amap_poi_search"

    def test_break_strategy_amap_route_plan(self) -> None:
        detector = LoopDetector()
        strategy = detector.get_break_strategy("amap_route_plan")
        assert strategy["new_tool"] == "google_search"

    def test_break_strategy_google_search(self) -> None:
        detector = LoopDetector()
        strategy = detector.get_break_strategy("google_search")
        assert strategy["new_tool"] == "tavily_search"

    def test_break_strategy_tavily_search(self) -> None:
        detector = LoopDetector()
        strategy = detector.get_break_strategy("tavily_search")
        assert strategy["new_tool"] == "google_search"

    def test_break_strategy_unknown_tool(self) -> None:
        """Unknown tools should return a fallback strategy, not raise."""
        detector = LoopDetector()
        strategy = detector.get_break_strategy("some_unknown_tool")
        assert "new_tool" in strategy
        assert "reason" in strategy
