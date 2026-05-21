"""Tests for the context compressor.

These tests verify that the ContextCompressor correctly manages the context
window by keeping messages under a token limit while preserving recent
messages and maintaining progress information.
"""

from src.agent.context import ContextCompressor
from src.agent.state import ProgressTracker


def test_compress_within_limit():
    """Messages under token_limit should be returned unchanged."""
    compressor = ContextCompressor(token_limit=1000, keep_recent=5)

    # Create messages that are well under the limit
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "Help me plan a trip"},
    ]

    tracker = ProgressTracker()
    result = compressor.compress(messages, tracker)

    # Should return the same messages unchanged
    assert result == messages


def test_compress_exceeds_limit():
    """Messages over limit should have old messages compressed, recent preserved."""
    compressor = ContextCompressor(token_limit=100, keep_recent=2)

    # Create messages that exceed the limit
    messages = [
        {"role": "user", "content": "A" * 50},  # 25 tokens
        {"role": "assistant", "content": "B" * 50},  # 25 tokens
        {"role": "user", "content": "C" * 50},  # 25 tokens
        {"role": "assistant", "content": "D" * 50},  # 25 tokens
        {"role": "user", "content": "E" * 50},  # 25 tokens
    ]

    tracker = ProgressTracker()
    result = compressor.compress(messages, tracker)

    # Should have fewer messages than original
    assert len(result) < len(messages)

    # Last keep_recent messages should be preserved
    assert result[-1] == messages[-1]
    assert result[-2] == messages[-2]


def test_compress_preserves_recent():
    """Last N messages should always be kept intact, old ones get summary."""
    compressor = ContextCompressor(token_limit=100, keep_recent=3)

    # Create messages with varying content
    messages = [
        {"role": "user", "content": "Old message 1 " * 10},
        {"role": "assistant", "content": "Old response 1 " * 10},
        {"role": "user", "content": "Old message 2 " * 10},
        {"role": "assistant", "content": "Old response 2 " * 10},
        {"role": "user", "content": "Recent message 1"},
        {"role": "assistant", "content": "Recent response 1"},
        {"role": "user", "content": "Recent message 2"},
    ]

    tracker = ProgressTracker()
    result = compressor.compress(messages, tracker)

    # The last keep_recent messages should be exactly preserved
    assert result[-1] == messages[-1]
    assert result[-2] == messages[-2]
    assert result[-3] == messages[-3]

    # First message should be a summary
    assert result[0]["role"] == "system"
    assert "summary" in result[0]["content"].lower() or "compressed" in result[0]["content"].lower()


def test_compress_with_progress_tracker():
    """Compression should include ProgressTracker summary."""
    compressor = ContextCompressor(token_limit=100, keep_recent=2)

    messages = [
        {"role": "user", "content": "Old message " * 20},
        {"role": "assistant", "content": "Old response " * 20},
        {"role": "user", "content": "Recent 1"},
        {"role": "assistant", "content": "Recent 2"},
    ]

    tracker = ProgressTracker()
    tracker.add_poi({"name": "Great Wall", "type": "landmark"})
    tracker.update_constraint("budget", "satisfied")

    result = compressor.compress(messages, tracker)

    # Should include progress summary
    progress_found = any("progress" in msg.get("content", "").lower()
                        for msg in result if msg.get("role") == "system")
    assert progress_found


def test_estimate_tokens():
    """Token estimation should be roughly len(content)//2."""
    compressor = ContextCompressor()

    messages = [
        {"role": "user", "content": "Hello world"},  # 11 chars -> ~5 tokens
        {"role": "assistant", "content": "Hi"},  # 2 chars -> ~1 token
    ]

    estimated = compressor.estimate_tokens(messages)
    expected = (11 + 2) // 2

    assert estimated == expected
