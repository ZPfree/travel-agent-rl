"""Training entry point for the travel agent RL system.

Usage:
    python scripts/train.py --config configs/training_config.yaml --stage ppo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.config import TrainingConfig
from src.training.ppo_trainer import AgentPPOTrainer
from src.training.curriculum import CurriculumManager


# ── Sample data for training ─────────────────────────────────────────────

SAMPLE_QUERIES = [
    "Plan a 3-day trip to Tokyo",
    "Find restaurants near the Eiffel Tower",
    "What's the weather like in London this week?",
    "Plan a road trip from New York to Boston",
    "Find hotels in downtown San Francisco",
    "What are the top attractions in Rome?",
    "Plan a weekend getaway to Barcelona",
    "Find hiking trails near Denver",
    "What's the best time to visit Bali?",
    "Plan a family vacation to Orlando",
]

SAMPLE_CONSTRAINTS = [
    [{"type": "budget", "value": "moderate", "weight": 1.0}],
    [{"type": "distance", "value": "5km", "weight": 0.8}],
    [{"type": "preference", "value": "indoor", "weight": 0.6}],
    [{"type": "budget", "value": "low", "weight": 1.0}],
    [{"type": "budget", "value": "high", "weight": 0.5}],
    [{"type": "preference", "value": "cultural", "weight": 0.9}],
    [{"type": "duration", "value": "2 days", "weight": 0.7}],
    [{"type": "preference", "value": "outdoor", "weight": 0.8}],
    [{"type": "season", "value": "summer", "weight": 0.6}],
    [{"type": "group", "value": "family", "weight": 1.0}],
]


# ── Config loading ───────────────────────────────────────────────────────

def load_config(config_path: str) -> TrainingConfig:
    """Load YAML config and create a TrainingConfig dataclass.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A TrainingConfig populated from the YAML values.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    ppo = raw.get("ppo", {})
    agent = raw.get("agent", {})
    model_cfg = raw.get("model", {})

    return TrainingConfig(
        model_name=model_cfg.get("name", TrainingConfig.model_name),
        learning_rate=ppo.get("learning_rate", TrainingConfig.learning_rate),
        batch_size=ppo.get("batch_size", TrainingConfig.batch_size),
        mini_batch_size=ppo.get("mini_batch_size", TrainingConfig.mini_batch_size),
        ppo_epochs=ppo.get("epochs", TrainingConfig.ppo_epochs),
        gamma=ppo.get("gamma", TrainingConfig.gamma),
        lam=ppo.get("lam", TrainingConfig.lam),
        max_steps=agent.get("max_steps", TrainingConfig.max_steps),
        context_limit=agent.get("context_limit", TrainingConfig.context_limit),
    )


# ── Training loops ───────────────────────────────────────────────────────

def train_ppo(
    trainer: AgentPPOTrainer,
    curriculum: CurriculumManager,
    num_epochs: int = 100,
) -> None:
    """Run the PPO training loop.

    Each epoch:
      1. Collect a batch of episodes using sample queries.
      2. Run a PPO update on the collected batch.
      3. Record performance in the curriculum manager.
      4. Check whether to advance the curriculum stage.

    Args:
        trainer:    The PPO trainer instance.
        curriculum: The curriculum manager.
        num_epochs: Total number of training epochs.
    """
    batch_size = trainer.config.batch_size

    for epoch in range(1, num_epochs + 1):
        # Select queries for this batch (cycle through samples)
        start_idx = ((epoch - 1) * batch_size) % len(SAMPLE_QUERIES)
        queries = [
            SAMPLE_QUERIES[(start_idx + i) % len(SAMPLE_QUERIES)]
            for i in range(batch_size)
        ]
        constraints_list = [
            SAMPLE_CONSTRAINTS[(start_idx + i) % len(SAMPLE_CONSTRAINTS)]
            for i in range(batch_size)
        ]

        # 1. Collect batch
        episodes = trainer.collect_batch(
            queries=queries,
            constraints_list=constraints_list,
        )

        # 2. PPO update
        update = trainer.ppo_update(model=None, episodes=episodes)

        # 3. Compute average reward as performance score
        avg_reward = (
            sum(ep.total_reward for ep in episodes) / len(episodes)
            if episodes
            else 0.0
        )
        completed = sum(1 for ep in episodes if ep.is_complete)

        # 4. Record and check curriculum
        curriculum.record_performance(avg_reward)
        stage_info = curriculum.get_config()

        # Print epoch info
        print(
            f"[Epoch {epoch:3d}/{num_epochs}] "
            f"stage={stage_info['stage_name']:8s} | "
            f"avg_reward={avg_reward:+.3f} | "
            f"completed={completed}/{len(episodes)} | "
            f"policy_loss={update.policy_loss:.4f} | "
            f"max_steps={stage_info['max_steps']}"
        )

        if curriculum.should_advance():
            old_stage = stage_info["stage_name"]
            curriculum.advance()
            new_stage = curriculum.get_config()["stage_name"]
            print(f"  >> Curriculum advanced: {old_stage} -> {new_stage}")


def train_sft(config: TrainingConfig) -> None:
    """Placeholder for SFT training stage."""
    print("[SFT] Supervised fine-tuning not yet implemented. Skipping.")


def train_dpo(config: TrainingConfig) -> None:
    """Placeholder for DPO training stage."""
    print("[DPO] Direct preference optimization not yet implemented. Skipping.")


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the travel agent with PPO / SFT / DPO."
    )
    parser.add_argument(
        "--config",
        default="configs/training_config.yaml",
        help="Path to the YAML training config (default: configs/training_config.yaml)",
    )
    parser.add_argument(
        "--stage",
        choices=["sft", "ppo", "dpo", "all"],
        default="ppo",
        help="Training stage to run (default: ppo)",
    )
    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config} ...")
    config = load_config(args.config)
    print(f"  model={config.model_name}  lr={config.learning_rate}  "
          f"batch_size={config.batch_size}  max_steps={config.max_steps}")

    # Run selected stage(s)
    if args.stage in ("sft", "all"):
        train_sft(config)

    if args.stage in ("ppo", "all"):
        trainer = AgentPPOTrainer(config)
        curriculum = CurriculumManager()
        print(f"\nStarting PPO training (curriculum stage: "
              f"{curriculum.current_stage['name']}) ...\n")
        train_ppo(trainer, curriculum, num_epochs=100)

    if args.stage in ("dpo", "all"):
        train_dpo(config)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
