"""Tests for tool base classes: ToolResult, BaseTool, ToolRetryPolicy."""

import pytest
from pydantic import ValidationError

from src.tools.base import BaseTool, ToolResult, ToolRetryPolicy


# ── ToolResult ────────────────────────────────────────────────────────────


class TestToolResult:
    """Tests for the ToolResult model."""

    def test_success_result(self):
        r = ToolResult(success=True, data={"city": "Tokyo"})
        assert r.success is True
        assert r.data == {"city": "Tokyo"}
        assert r.confidence == 1.0
        assert r.error is None
        assert r.metadata == {}

    def test_failure_result(self):
        r = ToolResult(success=False, data=None, error="API timeout")
        assert r.success is False
        assert r.data is None
        assert r.error == "API timeout"

    def test_confidence_range(self):
        """Confidence must be between 0.0 and 1.0 inclusive."""
        ToolResult(success=True, data="ok", confidence=0.0)
        ToolResult(success=True, data="ok", confidence=0.5)
        ToolResult(success=True, data="ok", confidence=1.0)

    def test_confidence_out_of_range_low(self):
        with pytest.raises(ValidationError):
            ToolResult(success=True, data="ok", confidence=-0.1)

    def test_confidence_out_of_range_high(self):
        with pytest.raises(ValidationError):
            ToolResult(success=True, data="ok", confidence=1.1)

    def test_metadata_default_is_empty_dict(self):
        r = ToolResult(success=True, data="x")
        assert r.metadata == {}
        # Mutating default should not affect new instances
        r.metadata["key"] = "val"
        r2 = ToolResult(success=True, data="y")
        assert r2.metadata == {}

    def test_serialization_roundtrip(self):
        r = ToolResult(
            success=True,
            data={"flights": ["AA101", "DL202"]},
            confidence=0.85,
            metadata={"source": "amadeus"},
        )
        d = r.model_dump()
        assert d["success"] is True
        assert d["confidence"] == 0.85
        assert d["data"] == {"flights": ["AA101", "DL202"]}
        assert d["metadata"] == {"source": "amadeus"}
        assert d["error"] is None

        r2 = ToolResult.model_validate(d)
        assert r2 == r

    def test_serialization_failure_case(self):
        r = ToolResult(success=False, data=None, error="not found", confidence=0.0)
        d = r.model_dump()
        assert d["success"] is False
        assert d["error"] == "not found"
        assert d["confidence"] == 0.0

        r2 = ToolResult.model_validate(d)
        assert r2 == r


# ── BaseTool ──────────────────────────────────────────────────────────────


class DummyTool(BaseTool):
    """Concrete tool for testing the abstract base."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "A dummy tool for tests."

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, data=kwargs.get("query"))


class TestBaseTool:
    def test_cannot_instantiate_abstract(self):
        """BaseTool should not be instantiable directly."""
        with pytest.raises(TypeError):
            BaseTool()  # type: ignore[abstract]

    def test_concrete_tool_properties(self):
        t = DummyTool()
        assert t.name == "dummy"
        assert t.description == "A dummy tool for tests."
        assert "query" in t.parameters()["properties"]

    def test_concrete_tool_execute(self):
        t = DummyTool()
        r = t.execute(query="hello")
        assert r.success is True
        assert r.data == "hello"


# ── ToolRetryPolicy ──────────────────────────────────────────────────────


class TestToolRetryPolicy:
    def test_defaults(self):
        p = ToolRetryPolicy()
        assert p.max_retries == 3
        assert p.base_delay == 0.0
        assert p.backoff_factor == 2.0
        assert p.retryable_errors == set()

    def test_should_retry_no_error_type(self):
        """When retryable_errors is empty, nothing is retryable."""
        p = ToolRetryPolicy()
        assert p.should_retry("SomeError", attempt=0) is False

    def test_should_retry_matching_error(self):
        p = ToolRetryPolicy(retryable_errors={"TimeoutError", "ConnectionError"})
        assert p.should_retry("TimeoutError", attempt=0) is True
        assert p.should_retry("ConnectionError", attempt=1) is True

    def test_should_retry_exceeds_max(self):
        p = ToolRetryPolicy(max_retries=2, retryable_errors={"TimeoutError"})
        assert p.should_retry("TimeoutError", attempt=0) is True
        assert p.should_retry("TimeoutError", attempt=1) is True
        assert p.should_retry("TimeoutError", attempt=2) is False  # at max_retries

    def test_should_retry_non_matching_error(self):
        p = ToolRetryPolicy(retryable_errors={"TimeoutError"})
        assert p.should_retry("ValueError", attempt=0) is False

    def test_delay_calculation(self):
        """delay = base_delay * backoff_factor ** attempt"""
        p = ToolRetryPolicy(base_delay=1.0, backoff_factor=2.0)
        assert p.delay(0) == 1.0
        assert p.delay(1) == 2.0
        assert p.delay(2) == 4.0

    def test_delay_zero_base(self):
        """Default base_delay=0.0 means no delay (training mode)."""
        p = ToolRetryPolicy()
        assert p.delay(0) == 0.0
        assert p.delay(5) == 0.0
