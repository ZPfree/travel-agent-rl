"""Tests for concrete tool implementations and ToolRegistry."""

import pytest

from tools.base import ToolResult
from tools.registry import ToolRegistry


class TestToolRegistry:
    """Tests for the ToolRegistry."""

    def setup_method(self):
        self.registry = ToolRegistry()

    def test_all_8_tools_registered(self):
        """Registry should contain exactly 8 tools."""
        assert len(self.registry.tools) == 8

    def test_expected_tool_names(self):
        expected = {
            "amap_poi_search",
            "amap_nearby_search",
            "amap_route_plan",
            "amap_geo_code",
            "google_search",
            "tavily_search",
            "weather_query",
            "calendar_query",
        }
        assert set(self.registry.tools.keys()) == expected

    def test_amap_poi_search_returns_pois(self):
        result = self.registry.execute(
            "amap_poi_search", keyword="故宫", city="北京", type="景点"
        )
        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) > 0
        assert all("name" in poi for poi in result.data)
        assert result.confidence == 0.9

    def test_weather_query_returns_weather(self):
        result = self.registry.execute("weather_query", city="北京", date="2026-05-20")
        assert result.success is True
        assert "city" in result.data
        assert "temperature" in result.data
        assert result.confidence == 0.9

    def test_unknown_tool_returns_failure(self):
        result = self.registry.execute("nonexistent_tool", foo="bar")
        assert result.success is False
        assert "not found" in result.error.lower() or "unknown" in result.error.lower()

    def test_amap_nearby_search(self):
        result = self.registry.execute(
            "amap_nearby_search",
            location="116.397428,39.90923",
            radius=1000,
            type="餐饮",
        )
        assert result.success is True
        assert isinstance(result.data, list)
        assert result.confidence == 0.85

    def test_amap_route_plan(self):
        result = self.registry.execute(
            "amap_route_plan",
            origin="116.397428,39.90923",
            destination="116.403414,39.914714",
            mode="walking",
        )
        assert result.success is True
        assert "distance" in result.data
        assert "duration" in result.data
        assert result.confidence == 0.9

    def test_amap_geo_code(self):
        result = self.registry.execute("amap_geo_code", address="北京市天安门广场")
        assert result.success is True
        assert "location" in result.data
        assert result.confidence == 0.95

    def test_google_search(self):
        result = self.registry.execute("google_search", query="北京旅游攻略", num=3)
        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) <= 3
        assert result.confidence == 0.8

    def test_tavily_search(self):
        result = self.registry.execute(
            "tavily_search", query="北京故宫门票", search_depth="basic"
        )
        assert result.success is True
        assert isinstance(result.data, list)
        assert result.confidence == 0.85

    def test_calendar_query(self):
        result = self.registry.execute(
            "calendar_query", date="2026-05-21", user_id="user_001"
        )
        assert result.success is True
        assert isinstance(result.data, list)
        assert result.confidence == 0.95

    def test_get_tool_descriptions(self):
        descs = self.registry.get_tool_descriptions()
        assert isinstance(descs, list)
        assert len(descs) == 8
        for d in descs:
            assert "name" in d
            assert "description" in d
            assert "parameters" in d

    def test_execute_validates_required_fields(self):
        """Missing required fields should return failure."""
        result = self.registry.execute("amap_poi_search", city="北京")
        # keyword is required
        assert result.success is False

    def test_amap_poi_search_filters_by_keyword(self):
        result = self.registry.execute(
            "amap_poi_search", keyword="故宫", city="北京", type="景点"
        )
        assert result.success is True
        for poi in result.data:
            assert "故宫" in poi["name"]
