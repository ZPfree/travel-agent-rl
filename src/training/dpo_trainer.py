"""DPO (Direct Preference Optimization) trainer for the travel agent RL system.

Provides a simplified DPOTrainer that builds preference pairs from episodes.
The full DPO training integration with transformers will be added later.
max_length is set to 32768 to handle 50-step episodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreferencePair:
    """A single preference pair for DPO training.

    Attributes:
        chosen:     The preferred response/plan.
        rejected:   The non-preferred response/plan.
        query:      The original user query.
        constraints: Optional list of constraint dicts.
    """

    chosen: str
    rejected: str
    query: str = ""
    constraints: list[dict[str, Any]] | None = None


class DPOTrainer:
    """Simplified DPO trainer for the travel planning agent.

    Builds preference pairs from chosen/rejected responses and episodes.
    The full DPO loss computation and gradient-based training will be
    integrated with transformers later.

    Parameters:
        max_length: Maximum sequence length for tokenization (default 32768
                    to handle 50-step episodes).
        beta:       DPO temperature parameter (default 0.1).
    """

    def __init__(self, max_length: int = 32768, beta: float = 0.1) -> None:
        self.max_length = max_length
        self.beta = beta

    def build_preference_pair(
        self,
        chosen: str,
        rejected: str,
        query: str = "",
        constraints: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build a preference pair dict from chosen and rejected responses.

        Args:
            chosen:     The preferred response.
            rejected:   The non-preferred response.
            query:      Optional user query.
            constraints: Optional list of constraint dicts.

        Returns:
            Dict with keys: chosen, rejected, query, constraints.
        """
        return {
            "chosen": chosen,
            "rejected": rejected,
            "query": query,
            "constraints": constraints,
        }

    def generate_preference_data(
        self,
        episodes: list[Any],
    ) -> list[PreferencePair]:
        """Generate preference pairs from collected episodes.

        Placeholder implementation — returns an empty list.  Full
        preference-pair generation from episode rewards and trajectories
        will be added later.

        Args:
            episodes: List of Episode objects.

        Returns:
            List of PreferencePair objects.
        """
        # Placeholder: full implementation will compare high-reward vs
        # low-reward trajectories within each episode.
        return []
