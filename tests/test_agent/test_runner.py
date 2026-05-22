"""Tests for AgentRunner (episode collector).

Tests:
- StepData and Episode dataclasses
- AgentRunner.collect_episode returns episode with steps and total_reward
- AgentRunner respects max_steps limit
- _rule_based_action produces valid ReAct output
- _compute_step_reward scoring logic
- _update_tracker state mutation
"""

import pytest
from unittest.mock import patch, MagicMock

from src.agent.runner import StepData, Episode, AgentRunner
from tools.base import ToolResult
from agent.state import ProgressTracker


# ── StepData dataclass ────────────────────────────────────────────────────


class TestStepData:
    """Tests for the StepData dataclass."""

    def test_create_step_data(self):
        step = StepData(
            prompt="test prompt",
            action="search_poi",
            action_input={"query": "Tokyo Tower"},
            observation="Found Tokyo Tower",
            reward=0.1,
            logprob=-0.5,
            step=1,
        )
        assert step.prompt == "test prompt"
        assert step.action == "search_poi"
        assert step.action_input == {"query": "Tokyo Tower"}
        assert step.observation == "Found Tokyo Tower"
        assert step.reward == 0.1
        assert step.logprob == -0.5
        assert step.step == 1

    def test_step_data_defaults(self):
        step = StepData(
            prompt="p",
            action="a",
            action_input={},
            observation="o",
            reward=0.0,
            logprob=0.0,
            step=0,
        )
        assert step.reward == 0.0
        assert step.logprob == 0.0


# ── Episode dataclass ─────────────────────────────────────────────────────


class TestEpisode:
    """Tests for the Episode dataclass."""

    def test_create_episode(self):
        ep = Episode(
            query="Plan a trip to Tokyo",
            constraints=[{"type": "budget", "value": "5000 CNY", "weight": 0.8}],
            steps=[],
            total_reward=0.0,
            is_complete=False,
        )
        assert ep.query == "Plan a trip to Tokyo"
        assert len(ep.constraints) == 1
        assert ep.total_reward == 0.0
        assert ep.is_complete is False

    def test_episode_with_steps(self):
        s1 = StepData("p1", "a1", {}, "o1", 0.1, -0.3, 0)
        s2 = StepData("p2", "a2", {}, "o2", 0.05, -0.2, 1)
        ep = Episode(
            query="q",
            constraints=[],
            steps=[s1, s2],
            total_reward=0.15,
            is_complete=True,
        )
        assert len(ep.steps) == 2
        assert ep.total_reward == pytest.approx(0.15)


# ── AgentRunner initialization ────────────────────────────────────────────


class TestAgentRunnerInit:
    """Tests for AgentRunner.__init__."""

    def test_default_init(self):
        runner = AgentRunner()
        assert runner.max_steps == 50
        assert runner.context_limit == 20000

    def test_custom_init(self):
        runner = AgentRunner(max_steps=10, context_limit=5000)
        assert runner.max_steps == 10
        assert runner.context_limit == 5000

    def test_init_creates_registry(self):
        runner = AgentRunner()
        assert runner.registry is not None
        assert len(runner.registry.tools) > 0

    def test_init_creates_compressor(self):
        runner = AgentRunner()
        assert runner.compressor is not None


# ── AgentRunner._rule_based_action ────────────────────────────────────────


class TestRuleBasedAction:
    """Tests for AgentRunner._rule_based_action."""

    def test_returns_react_format(self):
        runner = AgentRunner()
        tracker = ProgressTracker()
        result = runner._rule_based_action(
            query="Plan a trip to Tokyo",
            constraints=[{"type": "budget", "value": "5000 CNY", "weight": 0.8}],
            step=0,
            tracker=tracker,
        )
        assert "Thought:" in result
        assert "Action:" in result
        assert "Action Input:" in result

    def test_first_step_uses_search(self):
        runner = AgentRunner()
        tracker = ProgressTracker()
        result = runner._rule_based_action(
            query="Plan a trip to Tokyo",
            constraints=[],
            step=0,
            tracker=tracker,
        )
        # First step should search for POIs
        assert "search" in result.lower() or "poi" in result.lower()

    def test_later_steps_vary_action(self):
        runner = AgentRunner()
        tracker = ProgressTracker()
        results = set()
        for step in range(5):
            result = runner._rule_based_action(
                query="Plan a trip",
                constraints=[],
                step=step,
                tracker=tracker,
            )
            # Extract action name
            for line in result.split("\n"):
                if line.startswith("Action:"):
                    results.add(line.strip())
                    break
        # Should use at least 2 different actions across 5 steps
        assert len(results) >= 2

    def test_contains_valid_json_action_input(self):
        import json
        runner = AgentRunner()
        tracker = ProgressTracker()
        result = runner._rule_based_action(
            query="Plan a trip to Tokyo",
            constraints=[],
            step=0,
            tracker=tracker,
        )
        # Extract Action Input line
        for line in result.split("\n"):
            if line.startswith("Action Input:"):
                json_str = line[len("Action Input:"):].strip()
                parsed = json.loads(json_str)
                assert isinstance(parsed, dict)
                return
        pytest.fail("No Action Input line found")


# ── AgentRunner._compute_step_reward ──────────────────────────────────────


class TestComputeStepReward:
    """Tests for AgentRunner._compute_step_reward."""

    def test_success_reward(self):
        runner = AgentRunner()
        tracker = ProgressTracker()
        parsed = {"is_valid": True, "is_final": False, "action": "search_poi", "action_input": {}}
        result = ToolResult(success=True, data={"name": "Tokyo Tower"}, confidence=0.5)
        reward = runner._compute_step_reward(parsed, result, tracker)
        assert reward == pytest.approx(0.1)

    def test_high_confidence_bonus(self):
        runner = AgentRunner()
        tracker = ProgressTracker()
        parsed = {"is_valid": True, "is_final": False, "action": "search_poi", "action_input": {}}
        result = ToolResult(success=True, data={"name": "Tokyo Tower"}, confidence=0.9)
        reward = runner._compute_step_reward(parsed, result, tracker)
        # 0.1 (success) + 0.05 (high confidence) = 0.15
        assert reward == pytest.approx(0.15)

    def test_failure_penalty(self):
        runner = AgentRunner()
        tracker = ProgressTracker()
        parsed = {"is_valid": True, "is_final": False, "action": "search_poi", "action_input": {}}
        result = ToolResult(success=False, error="Tool not found")
        reward = runner._compute_step_reward(parsed, result, tracker)
        assert reward == pytest.approx(-0.1)

    def test_final_answer_bonus(self):
        runner = AgentRunner()
        tracker = ProgressTracker()
        parsed = {"is_valid": True, "is_final": True, "final_answer": "Here is your plan"}
        result = ToolResult(success=True)
        reward = runner._compute_step_reward(parsed, result, tracker)
        # Final answer should give some reward (implementation-defined)
        assert reward >= 0.0


# ── AgentRunner._update_tracker ───────────────────────────────────────────


class TestUpdateTracker:
    """Tests for AgentRunner._update_tracker."""

    def test_adds_poi_on_search_result(self):
        runner = AgentRunner()
        tracker = ProgressTracker()
        result = ToolResult(
            success=True,
            data=[{"name": "Tokyo Tower", "lat": 35.6586}],
        )
        runner._update_tracker(tracker, "search_poi", {"query": "tower"}, result)
        assert len(tracker.collected_pois) > 0

    def test_adds_route_on_route_result(self):
        runner = AgentRunner()
        tracker = ProgressTracker()
        result = ToolResult(
            success=True,
            data={"distance": "500km", "duration": "2h"},
        )
        runner._update_tracker(tracker, "plan_route", {"from": "Tokyo", "to": "Osaka"}, result)
        assert len(tracker.route_segments) > 0

    def test_caches_weather(self):
        runner = AgentRunner()
        tracker = ProgressTracker()
        result = ToolResult(
            success=True,
            data={"temp": 25, "condition": "sunny"},
        )
        runner._update_tracker(tracker, "query_weather", {"city": "Tokyo"}, result)
        assert len(tracker.weather_cache) > 0

    def test_no_update_on_failure(self):
        runner = AgentRunner()
        tracker = ProgressTracker()
        result = ToolResult(success=False, error="fail")
        runner._update_tracker(tracker, "search_poi", {"query": "x"}, result)
        assert len(tracker.collected_pois) == 0


# ── AgentRunner.collect_episode ───────────────────────────────────────────


class TestCollectEpisode:
    """Tests for AgentRunner.collect_episode."""

    def test_returns_episode(self):
        runner = AgentRunner(max_steps=3)
        episode = runner.collect_episode(
            query="Plan a trip to Tokyo",
            constraints=[{"type": "budget", "value": "5000 CNY", "weight": 0.8}],
        )
        assert isinstance(episode, Episode)
        assert episode.query == "Plan a trip to Tokyo"

    def test_episode_has_steps(self):
        runner = AgentRunner(max_steps=3)
        episode = runner.collect_episode(
            query="Plan a trip to Tokyo",
            constraints=[],
        )
        assert len(episode.steps) > 0
        for step in episode.steps:
            assert isinstance(step, StepData)

    def test_episode_total_reward(self):
        runner = AgentRunner(max_steps=3)
        episode = runner.collect_episode(
            query="Plan a trip to Tokyo",
            constraints=[],
        )
        expected_reward = sum(s.reward for s in episode.steps)
        assert episode.total_reward == pytest.approx(expected_reward)

    def test_runner_respects_max_steps(self):
        max_steps = 5
        runner = AgentRunner(max_steps=max_steps)
        episode = runner.collect_episode(
            query="Plan a complex multi-city trip across Japan with many activities",
            constraints=[{"type": "budget", "value": "10000 CNY", "weight": 0.5}],
        )
        assert len(episode.steps) <= max_steps

    def test_runner_respects_small_max_steps(self):
        max_steps = 2
        runner = AgentRunner(max_steps=max_steps)
        episode = runner.collect_episode(
            query="Plan a trip",
            constraints=[],
        )
        assert len(episode.steps) <= max_steps

    def test_episode_stores_constraints(self):
        constraints = [
            {"type": "budget", "value": "5000 CNY", "weight": 0.8},
            {"type": "date", "value": "2026-06-01", "weight": 1.0},
        ]
        runner = AgentRunner(max_steps=3)
        episode = runner.collect_episode(
            query="Trip to Tokyo",
            constraints=constraints,
        )
        assert episode.constraints == constraints

    def test_step_data_has_all_fields(self):
        runner = AgentRunner(max_steps=3)
        episode = runner.collect_episode(
            query="Plan a trip to Tokyo",
            constraints=[],
        )
        for step in episode.steps:
            assert hasattr(step, "prompt")
            assert hasattr(step, "action")
            assert hasattr(step, "action_input")
            assert hasattr(step, "observation")
            assert hasattr(step, "reward")
            assert hasattr(step, "logprob")
            assert hasattr(step, "step")
            assert isinstance(step.prompt, str)
            assert isinstance(step.action, str)
            assert isinstance(step.action_input, dict)
            assert isinstance(step.observation, str)
            assert isinstance(step.reward, float)
            assert isinstance(step.logprob, float)
            assert isinstance(step.step, int)
