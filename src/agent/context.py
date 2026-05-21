"""Context compressor for the travel agent RL system.

Provides a rule-based context compression mechanism that keeps the message
history under a token limit.  Used in the training loop to manage context
for 50-step episodes without exceeding the 20 K token budget.

Old messages are replaced with a summary message, while the most recent
``keep_recent`` messages are always preserved intact.  A
``ProgressTracker`` summary is prepended so the agent retains awareness of
accumulated planning state.
"""

from __future__ import annotations

from typing import Any

from src.agent.state import ProgressTracker


class ContextCompressor:
    """Rule-based context window compressor.

    Parameters
    ----------
    token_limit:
        Maximum number of tokens the context may consume.  Messages are
        compressed when the estimate exceeds this value.
    keep_recent:
        Number of most-recent messages to always keep intact, regardless
        of total token count.
    """

    def __init__(self, token_limit: int = 20_000, keep_recent: int = 5) -> None:
        self.token_limit = token_limit
        self.keep_recent = keep_recent

    # -- public API --------------------------------------------------------

    def estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Return a rough token estimate for *messages*.

        Uses the simple heuristic ``len(content) // 2`` per message.  This
        is intentionally fast and approximate -- good enough for deciding
        when to compress.
        """
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
        return total_chars // 2

    def compress(
        self,
        messages: list[dict[str, Any]],
        tracker: ProgressTracker,
    ) -> list[dict[str, Any]]:
        """Compress *messages* so that the result fits within ``token_limit``.

        If the estimated token count is already at or below the limit the
        original list is returned unchanged.

        Otherwise, messages older than the last ``keep_recent`` entries are
        collapsed into a single summary message and the ``ProgressTracker``
        summary is prepended as a system message.

        Parameters
        ----------
        messages:
            The full conversation history (list of dicts with at least
            ``role`` and ``content`` keys).
        tracker:
            The current ``ProgressTracker`` whose ``to_summary()`` output
            is injected into the compressed context.

        Returns
        -------
        list[dict[str, Any]]
            A (possibly compressed) message list.
        """
        estimated = self.estimate_tokens(messages)

        # Fast path: nothing to do.
        if estimated <= self.token_limit:
            return messages

        # Guard: if keep_recent >= len(messages) we can't compress anything
        # meaningful, so just return as-is.
        if self.keep_recent >= len(messages):
            return messages

        # Split into old (to be summarised) and recent (kept intact).
        split_point = len(messages) - self.keep_recent
        old_messages = messages[:split_point]
        recent_messages = messages[split_point:]

        # Build summary of old messages.
        summary_parts: list[str] = []
        for msg in old_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate very long individual messages in the summary.
            if isinstance(content, str) and len(content) > 200:
                content = content[:200] + "..."
            summary_parts.append(f"[{role}]: {content}")

        old_summary = (
            "[Compressed conversation history]\n"
            + "\n".join(summary_parts)
        )

        # Build the result: progress summary + old summary + recent messages.
        result: list[dict[str, Any]] = []

        # ProgressTracker summary as a system message.
        result.append({
            "role": "system",
            "content": tracker.to_summary(),
        })

        # Compressed old messages as a system message.
        result.append({
            "role": "system",
            "content": old_summary,
        })

        # Recent messages kept intact.
        result.extend(recent_messages)

        return result
