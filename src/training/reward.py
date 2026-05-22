"""Step-level reward model and GAE computation for PPO training.

- StepRewardModel: computes per-step and completion rewards for travel planning episodes.
- compute_gae: standard Generalized Advantage Estimation for PPO updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StepRewardModel:
    """Computes step-level rewards for travel planning episodes.

    Reward components:
    - Information gain: confidence * (1 - redundancy) for successful steps
    - Constraint progress: bonus for advancing constraint satisfaction
    - Redundant penalty: fixed penalty for duplicate/redundant actions
    - Step cost: small constant subtracted every step to encourage efficiency
    - Completion bonus: large reward when all constraints are satisfied
    """

    step_cost: float = 0.01
    redundant_penalty: float = 0.1
    completion_bonus: float = 5.0

    def compute_step_reward(
        self,
        action: Any,
        result_success: bool,
        result_confidence: float,
        is_redundant: bool,
        constraint_progress: float,
    ) -> float:
        """Compute reward for a single agent step.

        Args:
            action: The action taken (unused in basic model, available for extensions).
            result_success: Whether the tool call succeeded.
            result_confidence: Confidence of the result [0, 1].
            is_redundant: Whether this action duplicates a previous one.
            constraint_progress: Incremental constraint satisfaction progress [0, 1].

        Returns:
            Scalar reward for this step.
        """
        reward = 0.0

        # Step cost (always applied)
        reward -= self.step_cost

        if not result_success:
            # Failed step: penalty beyond the step cost
            reward -= 0.05
            return reward

        # Information gain from successful step (weighted by novelty)
        novelty = 0.0 if is_redundant else 1.0
        reward += result_confidence * 0.2 * novelty

        # Constraint progress bonus
        reward += constraint_progress

        # Redundant action penalty
        if is_redundant:
            reward -= self.redundant_penalty

        return reward

    def compute_completion_reward(
        self, is_complete: bool, constraint_satisfaction: float
    ) -> float:
        """Compute terminal reward at episode end.

        Args:
            is_complete: Whether the agent produced a complete plan.
            constraint_satisfaction: Fraction of constraints satisfied [0, 1].

        Returns:
            Terminal reward.
        """
        if not is_complete:
            return -1.0 * (1.0 - constraint_satisfaction)

        return self.completion_bonus * constraint_satisfaction


def compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> list[float]:
    """Compute Generalized Advantage Estimation (GAE).

    Standard GAE computation for PPO:
        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        A_t = sum_{l=0}^{T-t-1} (gamma * lambda)^l * delta_{t+l}

    For the terminal state, V(s_{T+1}) = 0.

    Args:
        rewards: List of rewards for each timestep.
        values: List of value estimates for each timestep.
        gamma: Discount factor.
        lam: GAE lambda parameter (bias-variance tradeoff).

    Returns:
        List of advantage estimates, one per timestep.

    Raises:
        ValueError: If rewards and values have different lengths.
    """
    if len(rewards) != len(values):
        raise ValueError(
            f"rewards and values must have same length, "
            f"got {len(rewards)} and {len(values)}"
        )

    T = len(rewards)
    advantages = [0.0] * T

    # Compute advantages in reverse order
    # For terminal step: next_value = 0
    next_value = 0.0
    gae = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]

    return advantages
