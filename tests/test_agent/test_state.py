"""Tests for agent state definitions: Constraint, ToolCall, ProgressTracker, TravelState."""

import pytest
from pydantic import ValidationError

from agent.state import Constraint, ToolCall, ProgressTracker, TravelState


# ── Constraint TypedDict ─────────────────────────────────────────────────


class TestConstraint:
    """Tests for the Constraint TypedDict."""

    def test_create_constraint(self):
        c: Constraint = {"type": "budget", "value": "5000 CNY", "weight": 0.8}
        assert c["type"] == "budget"
        assert c["value"] == "5000 CNY"
        assert c["weight"] == 0.8

    def test_constraint_fields(self):
        c: Constraint = {"type": "date", "value": "2026-06-01", "weight": 1.0}
        assert set(c.keys()) == {"type", "value", "weight"}


# ── ToolCall TypedDict ───────────────────────────────────────────────────


class TestToolCall:
    """Tests for the ToolCall TypedDict."""

    def test_create_tool_call(self):
        tc: ToolCall = {
            "tool_name": "search_poi",
            "parameters": {"query": "Tokyo Tower"},
            "result": {"name": "Tokyo Tower", "rating": 4.5},
            "step": 1,
        }
        assert tc["tool_name"] == "search_poi"
        assert tc["step"] == 1

    def test_tool_call_with_none_result(self):
        tc: ToolCall = {
            "tool_name": "search_flight",
            "parameters": {"from": "PEK", "to": "NRT"},
            "result": None,
            "step": 3,
        }
        assert tc["result"] is None


# ── ProgressTracker ──────────────────────────────────────────────────────


class TestProgressTrackerAddPOI:
    """Tests for ProgressTracker.add_poi."""

    def test_add_single_poi(self):
        tracker = ProgressTracker()
        tracker.add_poi({"name": "Tokyo Tower", "lat": 35.6586, "lng": 139.7454})
        assert len(tracker.collected_pois) == 1
        assert tracker.collected_pois[0]["name"] == "Tokyo Tower"

    def test_add_multiple_pois(self):
        tracker = ProgressTracker()
        tracker.add_poi({"name": "Tokyo Tower"})
        tracker.add_poi({"name": "Senso-ji"})
        tracker.add_poi({"name": "Meiji Shrine"})
        assert len(tracker.collected_pois) == 3

    def test_add_poi_returns_none(self):
        """add_poi should not return a value."""
        tracker = ProgressTracker()
        result = tracker.add_poi({"name": "Test"})
        assert result is None


class TestProgressTrackerUpdateConstraint:
    """Tests for ProgressTracker.update_constraint."""

    def test_update_constraint_new(self):
        tracker = ProgressTracker()
        tracker.update_constraint("budget", "satisfied")
        assert tracker.constraint_status["budget"] == "satisfied"

    def test_update_constraint_overwrite(self):
        tracker = ProgressTracker()
        tracker.update_constraint("budget", "unsatisfied")
        tracker.update_constraint("budget", "satisfied")
        assert tracker.constraint_status["budget"] == "satisfied"

    def test_update_multiple_constraints(self):
        tracker = ProgressTracker()
        tracker.update_constraint("budget", "satisfied")
        tracker.update_constraint("date", "partial")
        tracker.update_constraint("preference", "satisfied")
        assert len(tracker.constraint_status) == 3
        assert tracker.constraint_status["date"] == "partial"


class TestProgressTrackerAddRoute:
    """Tests for ProgressTracker.add_route."""

    def test_add_route(self):
        tracker = ProgressTracker()
        tracker.add_route({"from": "Tokyo", "to": "Osaka", "mode": "shinkansen"})
        assert len(tracker.route_segments) == 1
        assert tracker.route_segments[0]["mode"] == "shinkansen"

    def test_add_multiple_routes(self):
        tracker = ProgressTracker()
        tracker.add_route({"from": "A", "to": "B"})
        tracker.add_route({"from": "B", "to": "C"})
        assert len(tracker.route_segments) == 2


class TestProgressTrackerCacheWeather:
    """Tests for ProgressTracker.cache_weather."""

    def test_cache_weather(self):
        tracker = ProgressTracker()
        tracker.cache_weather("Tokyo", {"temp": 25, "condition": "sunny"})
        assert "Tokyo" in tracker.weather_cache
        assert tracker.weather_cache["Tokyo"]["temp"] == 25

    def test_cache_weather_overwrite(self):
        tracker = ProgressTracker()
        tracker.cache_weather("Tokyo", {"temp": 25})
        tracker.cache_weather("Tokyo", {"temp": 30})
        assert tracker.weather_cache["Tokyo"]["temp"] == 30


class TestProgressTrackerAddDecision:
    """Tests for ProgressTracker.add_decision."""

    def test_add_decision(self):
        tracker = ProgressTracker()
        tracker.add_decision("Choose direct flight over layover")
        assert len(tracker.decisions_made) == 1
        assert tracker.decisions_made[0] == "Choose direct flight over layover"

    def test_add_multiple_decisions(self):
        tracker = ProgressTracker()
        tracker.add_decision("Decision A")
        tracker.add_decision("Decision B")
        assert len(tracker.decisions_made) == 2


class TestProgressTrackerToSummary:
    """Tests for ProgressTracker.to_summary."""

    def test_empty_tracker_summary(self):
        tracker = ProgressTracker()
        summary = tracker.to_summary()
        assert isinstance(summary, str)
        assert "POI" in summary or "poi" in summary.lower()
        assert "0" in summary

    def test_summary_with_data(self):
        tracker = ProgressTracker()
        tracker.add_poi({"name": "A"})
        tracker.add_poi({"name": "B"})
        tracker.add_route({"from": "A", "to": "B"})
        tracker.update_constraint("budget", "satisfied")
        tracker.cache_weather("Tokyo", {"temp": 25})
        tracker.add_decision("Pick flight A")
        summary = tracker.to_summary()
        assert isinstance(summary, str)
        # Should contain counts
        assert "2" in summary  # 2 POIs
        assert "1" in summary  # 1 route, 1 decision, etc.

    def test_summary_shows_constraint_status(self):
        tracker = ProgressTracker()
        tracker.update_constraint("budget", "satisfied")
        tracker.update_constraint("date", "unsatisfied")
        summary = tracker.to_summary()
        assert "budget" in summary.lower() or "constraint" in summary.lower()
        assert "satisfied" in summary.lower() or "1/2" in summary or "2" in summary

    def test_summary_returns_non_empty_string(self):
        tracker = ProgressTracker()
        tracker.add_poi({"name": "X"})
        tracker.add_decision("Y")
        summary = tracker.to_summary()
        assert len(summary) > 0


class TestProgressTrackerDefaults:
    """Tests for ProgressTracker default values."""

    def test_default_collected_pois(self):
        tracker = ProgressTracker()
        assert tracker.collected_pois == []

    def test_default_route_segments(self):
        tracker = ProgressTracker()
        assert tracker.route_segments == []

    def test_default_constraint_status(self):
        tracker = ProgressTracker()
        assert tracker.constraint_status == {}

    def test_default_weather_cache(self):
        tracker = ProgressTracker()
        assert tracker.weather_cache == {}

    def test_default_calendar_events(self):
        tracker = ProgressTracker()
        assert tracker.calendar_events == []

    def test_default_decisions_made(self):
        tracker = ProgressTracker()
        assert tracker.decisions_made == []


class TestProgressTrackerSerialization:
    """Tests for ProgressTracker serialization."""

    def test_roundtrip(self):
        tracker = ProgressTracker()
        tracker.add_poi({"name": "A"})
        tracker.update_constraint("budget", "satisfied")
        tracker.add_decision("Pick A")
        d = tracker.model_dump()
        tracker2 = ProgressTracker.model_validate(d)
        assert tracker2.collected_pois == tracker.collected_pois
        assert tracker2.constraint_status == tracker.constraint_status
        assert tracker2.decisions_made == tracker.decisions_made


# ── TravelState TypedDict ────────────────────────────────────────────────


class TestTravelState:
    """Tests for the TravelState TypedDict."""

    def test_create_minimal_state(self):
        state: TravelState = {
            "query": "Plan a trip to Tokyo",
            "constraints": [],
            "messages": [],
            "tool_calls": [],
            "current_plan": None,
            "step_count": 0,
            "max_steps": 10,
            "constraint_satisfaction": 0.0,
            "final_plan": None,
            "is_complete": False,
            "progress_tracker": ProgressTracker(),
        }
        assert state["query"] == "Plan a trip to Tokyo"
        assert state["step_count"] == 0
        assert state["max_steps"] == 10
        assert state["is_complete"] is False
        assert isinstance(state["progress_tracker"], ProgressTracker)

    def test_state_with_constraints(self):
        state: TravelState = {
            "query": "Trip to Kyoto",
            "constraints": [
                {"type": "budget", "value": "3000 CNY", "weight": 0.9},
                {"type": "date", "value": "2026-07-01", "weight": 1.0},
            ],
            "messages": [],
            "tool_calls": [],
            "current_plan": None,
            "step_count": 0,
            "max_steps": 15,
            "constraint_satisfaction": 0.0,
            "final_plan": None,
            "is_complete": False,
            "progress_tracker": ProgressTracker(),
        }
        assert len(state["constraints"]) == 2
        assert state["constraints"][0]["type"] == "budget"

    def test_state_with_tool_calls(self):
        tc: ToolCall = {
            "tool_name": "search_poi",
            "parameters": {"query": "temple"},
            "result": {"name": "Kinkaku-ji"},
            "step": 1,
        }
        state: TravelState = {
            "query": "Trip",
            "constraints": [],
            "messages": [],
            "tool_calls": [tc],
            "current_plan": "Visit Kinkaku-ji",
            "step_count": 1,
            "max_steps": 10,
            "constraint_satisfaction": 0.5,
            "final_plan": None,
            "is_complete": False,
            "progress_tracker": ProgressTracker(),
        }
        assert len(state["tool_calls"]) == 1
        assert state["current_plan"] == "Visit Kinkaku-ji"
