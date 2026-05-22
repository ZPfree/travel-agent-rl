"""Generate training episode data using the rule-based agent runner.

Writes episodes as JSON lines (JSONL) for downstream training use.

Usage:
    python scripts/generate_data.py --num_episodes 3 --max_steps 3
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.runner import AgentRunner


# ── Sample queries ───────────────────────────────────────────────────────

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


def episode_to_dict(episode) -> dict:
    """Convert an Episode dataclass to a plain dict suitable for JSON serialisation.

    Nested StepData objects are converted via dataclasses.asdict as well.
    """
    return asdict(episode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training episode data with the rule-based agent."
    )
    parser.add_argument(
        "--output",
        default="data/episodes.jsonl",
        help="Output JSONL file path (default: data/episodes.jsonl)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes to generate (default: 100)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10,
        help="Maximum agent steps per episode (default: 10)",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create runner
    runner = AgentRunner(max_steps=args.max_steps)

    print(f"Generating {args.num_episodes} episodes (max_steps={args.max_steps}) ...")

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(args.num_episodes):
            query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
            constraints = SAMPLE_CONSTRAINTS[i % len(SAMPLE_CONSTRAINTS)]

            episode = runner.collect_episode(query=query, constraints=constraints)
            record = episode_to_dict(episode)

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Done. Wrote {count} episodes to {output_path}")


if __name__ == "__main__":
    main()
