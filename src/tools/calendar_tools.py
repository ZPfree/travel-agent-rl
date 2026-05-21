"""Calendar tool implementation."""

from typing import Any

from data.mock_data import MOCK_CALENDAR
from tools.base import BaseTool, ToolResult


class CalendarQuery(BaseTool):
    """Query calendar events for a user on a given date."""

    @property
    def name(self) -> str:
        return "calendar_query"

    @property
    def description(self) -> str:
        return "查询指定用户在指定日期的日程安排"

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "日期(YYYY-MM-DD)"},
                "user_id": {"type": "string", "description": "用户ID"},
            },
            "required": ["date", "user_id"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        date = kwargs.get("date", "")
        user_id = kwargs.get("user_id", "")

        events = MOCK_CALENDAR.get((user_id, date), [])
        return ToolResult(success=True, data=events, confidence=0.95)
