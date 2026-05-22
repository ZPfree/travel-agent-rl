"""Search tool implementations (Google, Tavily)."""

from typing import Any

from src.data.mock_data import MOCK_SEARCH_RESULTS
from src.tools.base import BaseTool, ToolResult


class GoogleSearch(BaseTool):
    """Simulated Google web search."""

    @property
    def name(self) -> str:
        return "google_search"

    @property
    def description(self) -> str:
        return "使用Google搜索网页信息"

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "num": {
                    "type": "integer",
                    "description": "返回结果数量",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "")
        num = kwargs.get("num", 5)

        # Find matching results
        results = []
        for key, items in MOCK_SEARCH_RESULTS.items():
            if key in query or query in key:
                results.extend(items)

        # If no exact match, return first available results
        if not results:
            for items in MOCK_SEARCH_RESULTS.values():
                results.extend(items)
                if len(results) >= num:
                    break

        results = results[:num]

        return ToolResult(success=True, data=results, confidence=0.8)


class TavilySearch(BaseTool):
    """Simulated Tavily deep search."""

    @property
    def name(self) -> str:
        return "tavily_search"

    @property
    def description(self) -> str:
        return "使用Tavily进行深度语义搜索"

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询"},
                "search_depth": {
                    "type": "string",
                    "description": "搜索深度(basic/advanced)",
                    "default": "basic",
                },
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "")
        search_depth = kwargs.get("search_depth", "basic")

        # Find matching results
        results = []
        for key, items in MOCK_SEARCH_RESULTS.items():
            if key in query or query in key:
                results.extend(items)

        # If no exact match, return first available results
        if not results:
            for items in MOCK_SEARCH_RESULTS.values():
                results.extend(items)

        confidence = 0.85 if search_depth == "advanced" else 0.7

        return ToolResult(
            success=True,
            data=results,
            confidence=confidence,
            metadata={"search_depth": search_depth, "source": "tavily"},
        )
