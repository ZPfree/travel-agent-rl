"""Tests for AgentPPOTrainer and PPOUpdate."""

import pytest

from src.training.config import TrainingConfig
from src.training.ppo_trainer import PPOUpdate, AgentPPOTrainer


class TestAgentPPOTrainerInit:
    """Tests for AgentPPOTrainer initialization."""

    def test_ppo_trainer_init(self):
        """AgentPPOTrainer(TrainingConfig()) is not None."""
        config = TrainingConfig()
        trainer = AgentPPOTrainer(config)
        assert trainer is not None

    def test_ppo_trainer_has_config(self):
        """Trainer stores the config."""
        config = TrainingConfig()
        trainer = AgentPPOTrainer(config)
        assert trainer.config is config

    def test_ppo_trainer_has_reward_model(self):
        """Trainer initializes a StepRewardModel."""
        config = TrainingConfig()
        trainer = AgentPPOTrainer(config)
        assert trainer.reward_model is not None

    def test_ppo_trainer_has_runner(self):
        """Trainer initializes an AgentRunner."""
        config = TrainingConfig()
        trainer = AgentPPOTrainer(config)
        assert trainer.runner is not None


class TestCollectEpisode:
    """Tests for AgentPPOTrainer.collect_episode."""

    def test_collect_episode_returns_episode(self):
        """collect_episode returns an Episode with steps."""
        config = TrainingConfig()
        trainer = AgentPPOTrainer(config)
        episode = trainer.collect_episode("测试旅行", [])
        assert episode is not None
        assert len(episode.steps) > 0

    def test_collect_episode_has_query(self):
        """Returned episode records the original query."""
        config = TrainingConfig()
        trainer = AgentPPOTrainer(config)
        episode = trainer.collect_episode("北京三日游", [{"type": "budget", "value": "5000", "weight": 1.0}])
        assert episode.query == "北京三日游"

    def test_collect_episode_with_model(self):
        """collect_episode accepts an optional model parameter."""
        config = TrainingConfig()
        trainer = AgentPPOTrainer(config)
        episode = trainer.collect_episode("测试", [], model=None)
        assert len(episode.steps) > 0


class TestCollectBatch:
    """Tests for AgentPPOTrainer.collect_batch."""

    def test_collect_batch_returns_list(self):
        """collect_batch returns a list of episodes matching input length."""
        config = TrainingConfig()
        trainer = AgentPPOTrainer(config)
        queries = ["北京游", "上海游"]
        constraints_list = [[], []]
        episodes = trainer.collect_batch(queries, constraints_list)
        assert isinstance(episodes, list)
        assert len(episodes) == 2

    def test_collect_batch_each_has_steps(self):
        """Each episode in the batch has at least one step."""
        config = TrainingConfig()
        trainer = AgentPPOTrainer(config)
        queries = ["北京游", "上海游", "广州游"]
        constraints_list = [[], [], []]
        episodes = trainer.collect_batch(queries, constraints_list)
        for ep in episodes:
            assert len(ep.steps) > 0


class TestComputeAdvantages:
    """Tests for AgentPPOTrainer.compute_advantages."""

    def test_compute_advantages_length(self):
        """Advantages list has same length as episode steps."""
        config = TrainingConfig()
        trainer = AgentPPOTrainer(config)
        episode = trainer.collect_episode("测试", [])
        advantages = trainer.compute_advantages(episode)
        assert len(advantages) == len(episode.steps)

    def test_compute_advantages_are_floats(self):
        """Each advantage is a float."""
        config = TrainingConfig()
        trainer = AgentPPOTrainer(config)
        episode = trainer.collect_episode("测试", [])
        advantages = trainer.compute_advantages(episode)
        for a in advantages:
            assert isinstance(a, float)


class TestPPOUpdate:
    """Tests for PPOUpdate dataclass and AgentPPOTrainer.ppo_update."""

    def test_ppo_update_dataclass(self):
        """PPOUpdate can be created with expected fields."""
        update = PPOUpdate(policy_loss=1.5, value_loss=0.3, kl_divergence=0.01)
        assert update.policy_loss == 1.5
        assert update.value_loss == 0.3
        assert update.kl_divergence == 0.01

    def test_ppo_update_returns_ppo_update(self):
        """ppo_update returns a PPOUpdate instance."""
        config = TrainingConfig()
        trainer = AgentPPOTrainer(config)
        episode = trainer.collect_episode("测试", [])
        update = trainer.ppo_update(None, [episode])
        assert isinstance(update, PPOUpdate)

    def test_ppo_update_policy_loss_nonnegative(self):
        """policy_loss is non-negative (sum of abs(advantages))."""
        config = TrainingConfig()
        trainer = AgentPPOTrainer(config)
        episode = trainer.collect_episode("测试", [])
        update = trainer.ppo_update(None, [episode])
        assert update.policy_loss >= 0.0

    def test_ppo_update_multiple_episodes(self):
        """ppo_update works with multiple episodes."""
        config = TrainingConfig()
        trainer = AgentPPOTrainer(config)
        episodes = trainer.collect_batch(["测试1", "测试2"], [[], []])
        update = trainer.ppo_update(None, episodes)
        assert isinstance(update, PPOUpdate)
        assert update.policy_loss >= 0.0
