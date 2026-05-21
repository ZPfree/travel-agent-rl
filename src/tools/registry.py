"""Tool registry for the travel agent RL system.

Registers all 8 tools and provides execute (with validation+retry) and
get_tool_descriptions for the agent.
"""

from typing import Any

from tools.base import BaseTool, ToolResult, ToolRetryPolicy
from tools.amap_tools import AmapGeoCode, AmapNearbySearch, AmapPoiSearch, AmapRoutePlan
from tools.calendar_tools import CalendarQuery
from tools.search_tools import GoogleSearch, TavilySearch
from tools.weather_tools import WeatherQuery


class ToolCallValidator:
    """Validates tool call parameters against required fields per tool."""

    def __init__(self):
        self.required_fields: dict[str, list[str]] = {}

    def register(self, tool_name: str, required: list[str]):
        """Register required fields for a tool."""
        self.required_fields[tool_name] = required

    def validate(self, tool_name: str, **kwargs: Any) -> tuple[bool, str | None]:
        """Return (is_valid, error_message)."""
        required = self.required_fields.get(tool_name, [])
        missing = [f for f in required if f not in kwargs or kwargs[f] is None]
        if missing:
            return False, f"Missing required fields: {missing}"
        return True, None


class ToolRegistry:
    """Registry that holds all tools, validates calls, and provides descriptions."""

    def __init__(self):
        self.tools: dict[str, BaseTool] = {}
        self.validator = ToolCallValidator()
        self.retry_policy = ToolRetryPolicy(max_retries=3, base_delay=0.0)
        self._register_all()

    def _register_all(self):
        """Register all 8 concrete tools."""
        tool_instances: list[BaseTool] = [
            AmapPoiSearch(),
            AmapNearbySearch(),
            AmapRoutePlan(),
            AmapGeoCode(),
            GoogleSearch(),
            TavilySearch(),
            WeatherQuery(),
            CalendarQuery(),
        ]
        for tool in tool_instances:
            self.tools[tool.name] = tool
            # Extract required fields from the tool's JSON schema
            params = tool.parameters()
            required = params.get("required", [])
            self.validator.register(tool.name, required)

    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name with validation and retry.

        Returns a failure ToolResult if the tool is not found or validation fails.
        """
        # Check tool exists
        tool = self.tools.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown tool: '{tool_name}' not found in registry.",
            )

        # Validate required fields
        is_valid, error_msg = self.validator.validate(tool_name, **kwargs)
        if not is_valid:
            return ToolResult(
                success=False,
                data=None,
                error=f"Validation error for '{tool_name}': {error_msg}",
            )

        # Execute with retry policy
        last_result: ToolResult | None = None
        for attempt in range(self.retry_policy.max_retries + 1):
            result = tool.execute(**kwargs)
            if result.success:
                return result
            last_result = result
            error_name = type(result.error).__name__ if result.error else "UnknownError"
            if not self.retry_policy.should_retry(error_name, attempt):
                break

        return last_result or ToolResult(
            success=False, data=None, error="Execution failed after retries."
        )

    def get_tool_descriptions(self) -> list[dict]:
        """Return a list of tool descriptors for the agent."""
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters(),
                }
            )
        return descriptions
