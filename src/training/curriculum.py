"""Curriculum learning manager for gradual difficulty increase during training.

Manages a sequence of training stages that progressively increase max_steps,
allowing the agent to master simpler tasks before tackling longer episodes.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional


DEFAULT_STAGES: List[Dict[str, Any]] = [
    {"name": "short", "max_steps": 10, "advance_threshold": 0.7},
    {"name": "medium", "max_steps": 20, "advance_threshold": 0.6},
    {"name": "long", "max_steps": 35, "advance_threshold": 0.5},
    {"name": "full", "max_steps": 50, "advance_threshold": 0.0},
]


class CurriculumManager:
    """Manages curriculum stages for progressive training difficulty.

    Tracks recent performance scores and automatically determines when
    the agent is ready to advance to a more challenging stage.

    Args:
        stages: List of stage dicts with keys 'name', 'max_steps',
            'advance_threshold'. Defaults to DEFAULT_STAGES.
        window_size: Number of recent performances to consider for
            advancement decisions.
    """

    def __init__(
        self,
        stages: Optional[List[Dict[str, Any]]] = None,
        window_size: int = 10,
    ) -> None:
        self._stages = stages if stages is not None else DEFAULT_STAGES
        self._stage_index = 0
        self._window_size = window_size
        self._performances: deque[float] = deque(maxlen=window_size)

    @property
    def current_stage(self) -> Dict[str, Any]:
        """Return the current stage configuration dict."""
        return self._stages[self._stage_index]

    @property
    def current_max_steps(self) -> int:
        """Return the max_steps for the current stage."""
        return self.current_stage["max_steps"]

    def record_performance(self, score: float) -> None:
        """Record a performance score for the current stage.

        Args:
            score: Performance score between 0 and 1.
        """
        self._performances.append(score)

    def should_advance(self) -> bool:
        """Check whether average performance meets the advance threshold.

        Returns:
            True if the average of recent performances (within window)
            meets or exceeds the current stage's advance_threshold.
            Returns False if no performances have been recorded.
        """
        if not self._performances:
            return False
        threshold = self.current_stage["advance_threshold"]
        # Last stage always has threshold 0.0 and should not auto-advance
        # unless explicitly advanced to (it's the terminal stage)
        if self._stage_index >= len(self._stages) - 1:
            return False
        avg = sum(self._performances) / len(self._performances)
        # Small tolerance for floating point arithmetic
        return avg >= threshold - 1e-9

    def advance(self) -> None:
        """Advance to the next curriculum stage.

        Resets the performance window. Does nothing if already at the
        last stage.
        """
        if self._stage_index < len(self._stages) - 1:
            self._stage_index += 1
            self._performances.clear()

    def get_config(self) -> Dict[str, Any]:
        """Return current stage configuration as a dictionary.

        Returns:
            Dict with keys 'stage_name', 'max_steps', 'advance_threshold'.
        """
        stage = self.current_stage
        return {
            "stage_name": stage["name"],
            "max_steps": stage["max_steps"],
            "advance_threshold": stage["advance_threshold"],
        }
