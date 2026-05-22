"""Integration tests for end-to-end validation of the travel agent RL system.

These tests verify that all major components work together correctly:
- AgentRunner episode collection
- PPO trainer full training cycle
- Curriculum manager integration with TrainingConfig
"""

from __future__ import annotations

from src.agent.runner import AgentRunner, Episode
from src.training.config import TrainingConfig
from src.training.ppo_trainer import AgentPPOTrainer, PPOUpdate
from src.training.curriculum import CurriculumManager


class TestEndToEndEpisodeCollection:
    """Test that AgentRunner can collect complete episodes."""

    def test_end_to_end_episode_collection(self) -> None:
        """AgentRunner(max_steps=5).collect_episode should produce a valid episode."""
        runner = AgentRunner(max_steps=5)
        constraints = [{"type": "time", "value": "两天", "weight": 0.3}]
        episode = runner.collect_episode(
            query="周末去北京玩",
            constraints=constraints,
        )

        # Episode should have at least one step
        assert len(episode.steps) > 0, "Episode should have at least one step"

        # Total reward should be non-zero (tools produce rewards)
        assert episode.total_reward != 0, "Total reward should not be zero"

        # Episode should preserve the query
        assert episode.query == "周末去北京玩"

        # Episode should preserve constraints
        assert episode.constraints == constraints

        # Each step should have valid data
        for i, step in enumerate(episode.steps):
            assert step.step == i, f"Step {i} should have step={i}"
            assert isinstance(step.action, str), f"Step {i} action should be a string"
            assert isinstance(step.reward, float), f"Step {i} reward should be a float"
            assert isinstance(step.prompt, str), f"Step {i} prompt should be a string"

    def test_episode_with_multiple_constraints(self) -> None:
        """Episode collection should handle multiple constraints."""
        runner = AgentRunner(max_steps=3)
        constraints = [
            {"type": "time", "value": "两天", "weight": 0.3},
            {"type": "budget", "value": "moderate", "weight": 0.7},
        ]
        episode = runner.collect_episode(
            query="周末去北京玩",
            constraints=constraints,
        )

        assert len(episode.steps) > 0
        assert episode.constraints == constraints

    def test_episode_max_steps_respected(self) -> None:
        """AgentRunner should not exceed max_steps."""
        max_steps = 3
        runner = AgentRunner(max_steps=max_steps)
        episode = runner.collect_episode(
            query="周末去北京玩",
            constraints=[{"type": "time", "value": "两天", "weight": 0.3}],
        )

        # If not complete, steps should be <= max_steps
        # (if complete, final answer step counts toward max_steps)
        assert len(episode.steps) <= max_steps


class TestPPOTrainerFullCycle:
    """Test that the PPO trainer can collect batches and perform updates."""

    def test_ppo_trainer_full_cycle(self) -> None:
        """Full PPO cycle: collect_batch → ppo_update should work end-to-end."""
        config = TrainingConfig(max_steps=5, batch_size=2)
        trainer = AgentPPOTrainer(config)

        queries = ["测试1", "测试2"]
        constraints_list: list[list[dict]] = [[], []]

        # Collect a batch of episodes
        episodes = trainer.collect_batch(queries, constraints_list)
        assert len(episodes) == 2, "Should collect exactly 2 episodes"

        # Each episode should have steps
        for ep in episodes:
            assert len(ep.steps) > 0, "Each episode should have at least one step"

        # Perform PPO update
        update = trainer.ppo_update(model=None, episodes=episodes)

        # PPOUpdate should be returned with valid policy_loss
        assert isinstance(update, PPOUpdate)
        assert update.policy_loss >= 0, "Policy loss should be >= 0"
        assert update.value_loss == 0.0, "Value loss should be 0.0 (no value network yet)"
        assert update.kl_divergence == 0.0, "KL divergence should be 0.0 (simplified)"

    def test_ppo_trainer_collect_single_episode(self) -> None:
        """PPO trainer should collect a single episode correctly."""
        config = TrainingConfig(max_steps=3)
        trainer = AgentPPOTrainer(config)

        episode = trainer.collect_episode(
            query="周末去北京玩",
            constraints=[{"type": "time", "value": "两天", "weight": 0.3}],
        )

        assert isinstance(episode, Episode)
        assert len(episode.steps) > 0

    def test_ppo_trainer_compute_advantages(self) -> None:
        """PPO trainer should compute advantages for an episode."""
        config = TrainingConfig(max_steps=3, gamma=0.99, lam=0.95)
        trainer = AgentPPOTrainer(config)

        episode = trainer.collect_episode(
            query="周末去北京玩",
            constraints=[],
        )

        advantages = trainer.compute_advantages(episode)
        assert len(advantages) == len(episode.steps), "One advantage per step"
        assert all(isinstance(a, float) for a in advantages)


class TestCurriculumIntegration:
    """Test that CurriculumManager integrates with TrainingConfig."""

    def test_curriculum_default_stage(self) -> None:
        """CurriculumManager should start at the 'short' stage with max_steps=10."""
        cm = CurriculumManager()
        assert cm.current_max_steps == 10

    def test_curriculum_config_integration(self) -> None:
        """TrainingConfig(max_steps=cm.current_max_steps) should work."""
        cm = CurriculumManager()
        config = TrainingConfig(max_steps=cm.current_max_steps)
        assert config.max_steps == 10

    def test_curriculum_advance_changes_max_steps(self) -> None:
        """Advancing curriculum stages should change max_steps."""
        cm = CurriculumManager()

        # Stage 0: short (max_steps=10)
        assert cm.current_max_steps == 10

        # Record high performance and advance
        for _ in range(cm._window_size):
            cm.record_performance(0.9)
        assert cm.should_advance()
        cm.advance()

        # Stage 1: medium (max_steps=20)
        assert cm.current_max_steps == 20

    def test_curriculum_full_training_flow(self) -> None:
        """Simulate a full training flow using curriculum + PPO trainer."""
        cm = CurriculumManager()
        config = TrainingConfig(max_steps=cm.current_max_steps)
        trainer = AgentPPOTrainer(config)

        # The trainer should use the curriculum's max_steps
        assert trainer.config.max_steps == 10

        # Collect a batch and update
        episodes = trainer.collect_batch(
            ["周末去北京玩", "周末去上海玩"],
            [[{"type": "time", "value": "两天", "weight": 0.3}], []],
        )
        assert len(episodes) == 2

        update = trainer.ppo_update(model=None, episodes=episodes)
        assert isinstance(update, PPOUpdate)
        assert update.policy_loss >= 0
