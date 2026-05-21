"""Tool base classes for the travel agent RL system.

- ToolResult: Pydantic model returned by every tool execution.
- BaseTool:   Abstract base class all tools must implement.
- ToolRetryPolicy: Configurable retry logic with exponential backoff.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Standardised result returned by every tool execution.

    Attributes:
        success: Whether the tool call succeeded.
        data:     Arbitrary payload (dict, list, str, None, ...).
        confidence: 0.0-1.0 score indicating result reliability.
                    The agent uses this to decide whether to cross-validate.
        error:    Human-readable error message when success=False.
        metadata: Arbitrary extra info (latency, source, cache hit, ...).
    """

    success: bool
    data: Any = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseTool(ABC):
    """Abstract base class for all agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique identifier for the tool (e.g. 'flight_search')."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line human-readable description shown to the agent."""

    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema describing the tool's input parameters."""

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool and return a ToolResult."""


class ToolRetryPolicy(BaseModel):
    """Retry configuration for transient tool failures.

    Designed for training: base_delay defaults to 0.0 so there are no
    simulated waits during RL rollouts.
    """

    max_retries: int = 3
    base_delay: float = 0.0
    backoff_factor: float = 2.0
    retryable_errors: set[str] = Field(default_factory=set)

    def should_retry(self, error_name: str, attempt: int) -> bool:
        """Return True if the error is retryable and attempts remain."""
        if error_name not in self.retryable_errors:
            return False
        return attempt < self.max_retries

    def delay(self, attempt: int) -> float:
        """Calculate backoff delay: base_delay * backoff_factor ** attempt."""
        return self.base_delay * (self.backoff_factor ** attempt)
