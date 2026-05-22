"""Weather tool implementation."""

from typing import Any

from src.data.mock_data import MOCK_WEATHER
from src.tools.base import BaseTool, ToolResult


class WeatherQuery(BaseTool):
    """Query weather for a city on a given date."""

    @property
    def name(self) -> str:
        return "weather_query"

    @property
    def description(self) -> str:
        return "查询指定城市和日期的天气信息"

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
                "date": {"type": "string", "description": "日期(YYYY-MM-DD)"},
            },
            "required": ["city", "date"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        city = kwargs.get("city", "")
        date = kwargs.get("date", "")

        weather = MOCK_WEATHER.get((city, date))
        if weather:
            return ToolResult(success=True, data=weather, confidence=0.9)

        # Fallback: return generic weather
        fallback = {
            "city": city,
            "date": date,
            "weather": "晴",
            "temperature": {"high": 25, "low": 15},
            "humidity": 50,
            "wind": {"direction": "北风", "speed": 2},
            "aqi": 60,
            "aqi_level": "良",
            "suggestion": "天气适宜出行。",
        }
        return ToolResult(success=True, data=fallback, confidence=0.5)
