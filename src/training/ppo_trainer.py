"""Custom PPO training loop for the travel agent RL system.

Provides AgentPPOTrainer which orchestrates episode collection, advantage
computation, and simplified PPO updates.  The full model forward/backward
pass will be added later; for now ppo_update computes policy_loss from
advantages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.training.config import TrainingConfig
from src.training.reward import StepRewardModel, compute_gae
from src.agent.runner import AgentRunner, Episode


@dataclass
class PPOUpdate:
    """Result of a PPO update step.

    Attributes:
        policy_loss:  Simplified policy loss (sum of absolute advantages).
        value_loss:   Placeholder value loss (0.0 for now).
        kl_divergence: Placeholder KL divergence (0.0 for now).
    """

    policy_loss: float
    value_loss: float
    kl_divergence: float


class AgentPPOTrainer:
    """PPO trainer for the travel planning agent.

    Coordinates episode collection (via AgentRunner) and advantage
    estimation (via GAE).  The ppo_update method currently produces a
    simplified policy loss; a full gradient-based update will be added
    once the model forward/backward integration is ready.

    Parameters:
        config: TrainingConfig with hyperparameters.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.reward_model = StepRewardModel()
        self.runner = AgentRunner(
            max_steps=config.max_steps,
            context_limit=config.context_limit,
        )

    def collect_episode(
        self,
        query: str,
        constraints: list[dict[str, Any]],
        model: Any | None = None,
    ) -> Episode:
        """Collect a single episode by running the agent loop.

        Delegates to AgentRunner.collect_episode.

        Args:
            query:       The user's travel planning query.
            constraints: List of constraint dicts.
            model:       Optional LLM model for action generation.

        Returns:
            An Episode with all steps, rewards, and completion status.
        """
        return self.runner.collect_episode(
            query=query,
            constraints=constraints,
            model=model,
        )

    def collect_batch(
        self,
        queries: list[str],
        constraints_list: list[list[dict[str, Any]]],
        model: Any | None = None,
    ) -> list[Episode]:
        """Collect a batch of episodes.

        Args:
            queries:         List of user queries.
            constraints_list: Per-query constraint lists.
            model:           Optional LLM model.

        Returns:
            List of Episode objects, one per query.
        """
        episodes: list[Episode] = []
        for query, constraints in zip(queries, constraints_list):
            episode = self.collect_episode(
                query=query,
                constraints=constraints,
                model=model,
            )
            episodes.append(episode)
        return episodes

    def compute_advantages(self, episode: Episode) -> list[float]:
        """Compute GAE advantages for an episode.

        Extracts per-step rewards from the episode and runs compute_gae
        with zero value estimates (since we don't have a value network yet).

        Args:
            episode: An Episode with step-level rewards.

        Returns:
            List of advantage estimates, one per step.
        """
        rewards = [step.reward for step in episode.steps]
        # No value network yet, so use zeros as value estimates
        values = [0.0] * len(rewards)
        return compute_gae(
            rewards=rewards,
            values=values,
            gamma=self.config.gamma,
            lam=self.config.lam,
        )

    def ppo_update(
        self,
        model: Any | None,
        episodes: list[Episode],
    ) -> PPOUpdate:
        """Perform a simplified PPO update.

        Computes advantages for all episodes and returns a PPOUpdate with
        policy_loss = sum of absolute advantages.  The full gradient-based
        update will be integrated later.

        Args:
            model:   The model being trained (unused in simplified version).
            episodes: Batch of episodes for this update.

        Returns:
            PPOUpdate with policy_loss, value_loss, and kl_divergence.
        """
        total_abs_advantage = 0.0

        for episode in episodes:
            advantages = self.compute_advantages(episode)
            total_abs_advantage += sum(abs(a) for a in advantages)

        return PPOUpdate(
            policy_loss=total_abs_advantage,
            value_loss=0.0,
            kl_divergence=0.0,
        )
