"""Tests for step-level reward model and GAE computation."""

import pytest

from src.training.reward import StepRewardModel, compute_gae


class TestStepReward:
    """Tests for StepRewardModel.compute_step_reward."""

    def test_step_reward_information_gain(self):
        """Successful, high-confidence, non-redundant step with some progress → positive reward."""
        model = StepRewardModel(step_cost=0.01, redundant_penalty=0.1)
        reward = model.compute_step_reward(
            action="search_flights",
            result_success=True,
            result_confidence=0.9,
            is_redundant=False,
            constraint_progress=0.1,
        )
        assert reward > 0

    def test_step_reward_redundant_penalty(self):
        """Successful but redundant step with no progress → negative reward."""
        model = StepRewardModel(step_cost=0.01, redundant_penalty=0.1)
        reward = model.compute_step_reward(
            action="search_flights",
            result_success=True,
            result_confidence=0.5,
            is_redundant=True,
            constraint_progress=0.0,
        )
        assert reward < 0

    def test_step_reward_failure(self):
        """Failed step → negative reward (at least the step cost)."""
        model = StepRewardModel(step_cost=0.01, redundant_penalty=0.1)
        reward = model.compute_step_reward(
            action="search_flights",
            result_success=False,
            result_confidence=0.0,
            is_redundant=False,
            constraint_progress=0.0,
        )
        assert reward < 0

    def test_step_cost_applied(self):
        """Step cost is always subtracted from the reward."""
        model = StepRewardModel(step_cost=0.05, redundant_penalty=0.1)
        reward = model.compute_step_reward(
            action="search_flights",
            result_success=True,
            result_confidence=1.0,
            is_redundant=False,
            constraint_progress=0.0,
        )
        # Even with max confidence, step cost reduces the reward
        assert reward <= 1.0 - 0.05


class TestCompletionReward:
    """Tests for StepRewardModel.compute_completion_reward."""

    def test_completion_reward_fully_satisfied(self):
        """Complete plan with full constraint satisfaction → large positive reward."""
        model = StepRewardModel()
        reward = model.compute_completion_reward(
            is_complete=True, constraint_satisfaction=1.0
        )
        assert reward > 0

    def test_completion_reward_incomplete(self):
        """Incomplete plan → zero or negative reward."""
        model = StepRewardModel()
        reward = model.compute_completion_reward(
            is_complete=False, constraint_satisfaction=0.5
        )
        assert reward <= 0


class TestGAE:
    """Tests for compute_gae (Generalized Advantage Estimation)."""

    def test_gae_computation(self):
        """GAE returns correct number of elements and last element matches."""
        rewards = [0.1, 0.2, 0.3, 0.4, 0.5]
        values = [0.15, 0.25, 0.35, 0.45, 0.0]
        advantages = compute_gae(rewards, values)
        assert len(advantages) == 5
        # Last advantage = last_reward - last_value (since next_value=0 at terminal)
        assert advantages[-1] == pytest.approx(rewards[-1] - values[-1])

    def test_gae_single_step(self):
        """GAE with single step: advantage = reward - value."""
        rewards = [1.0]
        values = [0.5]
        advantages = compute_gae(rewards, values)
        assert len(advantages) == 1
        assert advantages[0] == pytest.approx(0.5)

    def test_gae_length_mismatch_raises(self):
        """GAE requires rewards and values to have the same length."""
        with pytest.raises(ValueError):
            compute_gae([1.0, 2.0], [0.5])
