"""Training configuration for custom PPO training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for PPO training of the travel planning agent.

    Attributes:
        model_name: HuggingFace model identifier or local path.
        learning_rate: PPO learning rate.
        batch_size: Number of episodes per PPO update.
        mini_batch_size: Mini-batch size for PPO epochs.
        ppo_epochs: Number of PPO update epochs per batch.
        max_grad_norm: Gradient clipping norm.
        target_kl: Early stopping threshold for KL divergence.
        gamma: Discount factor for returns.
        lam: GAE lambda parameter.
        max_steps: Maximum agent steps per episode.
        context_limit: Maximum context tokens before compression.
    """

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    learning_rate: float = 1e-5
    batch_size: int = 8
    mini_batch_size: int = 2
    ppo_epochs: int = 4
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    gamma: float = 0.99
    lam: float = 0.95
    max_steps: int = 50
    context_limit: int = 16000
