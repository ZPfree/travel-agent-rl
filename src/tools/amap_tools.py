"""Amap (高德地图) tool implementations."""

from typing import Any

from src.data.mock_data import MOCK_POIS, MOCK_ROUTES
from src.tools.base import BaseTool, ToolResult


class AmapPoiSearch(BaseTool):
    """Search for POIs (Points of Interest) by keyword and city."""

    @property
    def name(self) -> str:
        return "amap_poi_search"

    @property
    def description(self) -> str:
        return "搜索指定城市的兴趣点(POI)，支持按关键词和类型过滤"

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "搜索关键词"},
                "city": {"type": "string", "description": "城市名称"},
                "type": {"type": "string", "description": "POI类型(景点/餐饮/住宿/交通)"},
            },
            "required": ["keyword", "city", "type"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        keyword = kwargs.get("keyword", "")
        city = kwargs.get("city", "")
        poi_type = kwargs.get("type", "")

        results = [
            poi
            for poi in MOCK_POIS
            if keyword in poi["name"]
            and poi["city"] == city
            and (not poi_type or poi["type"] == poi_type)
        ]

        return ToolResult(success=True, data=results, confidence=0.9)


class AmapNearbySearch(BaseTool):
    """Search for nearby POIs given a location and radius."""

    @property
    def name(self) -> str:
        return "amap_nearby_search"

    @property
    def description(self) -> str:
        return "根据坐标和半径搜索附近的兴趣点"

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "中心坐标(经度,纬度)"},
                "radius": {"type": "integer", "description": "搜索半径(米)"},
                "type": {"type": "string", "description": "POI类型"},
            },
            "required": ["location", "radius", "type"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        poi_type = kwargs.get("type", "")

        results = [
            poi for poi in MOCK_POIS if not poi_type or poi["type"] == poi_type
        ]

        return ToolResult(success=True, data=results, confidence=0.85)


class AmapRoutePlan(BaseTool):
    """Plan a route between two locations."""

    @property
    def name(self) -> str:
        return "amap_route_plan"

    @property
    def description(self) -> str:
        return "规划两点之间的路线，支持步行/驾车/公交模式"

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "起点坐标(经度,纬度)"},
                "destination": {"type": "string", "description": "终点坐标(经度,纬度)"},
                "mode": {
                    "type": "string",
                    "description": "出行方式(walking/driving/transit)",
                },
            },
            "required": ["origin", "destination", "mode"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        origin = kwargs.get("origin", "")
        destination = kwargs.get("destination", "")
        mode = kwargs.get("mode", "walking")

        route = MOCK_ROUTES.get((origin, destination, mode))
        if route:
            return ToolResult(success=True, data=route, confidence=0.9)

        # Fallback: return a generic route
        fallback = {
            "distance": "5.0公里",
            "duration": "30分钟",
            "mode": mode,
            "steps": ["从起点出发", "沿主要道路行进", "到达目的地"],
        }
        return ToolResult(success=True, data=fallback, confidence=0.7)


class AmapGeoCode(BaseTool):
    """Geocode an address to coordinates."""

    @property
    def name(self) -> str:
        return "amap_geo_code"

    @property
    def description(self) -> str:
        return "将地址转换为地理坐标(经纬度)"

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "要编码的地址"},
            },
            "required": ["address"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        address = kwargs.get("address", "")

        # Look up address in mock POIs (bidirectional substring match)
        for poi in MOCK_POIS:
            if (
                address in poi["address"]
                or address in poi["name"]
                or poi["name"] in address
            ):
                return ToolResult(
                    success=True,
                    data={
                        "address": poi["address"],
                        "location": poi["location"],
                        "level": "兴趣点",
                    },
                    confidence=0.95,
                )

        # Fallback geocode for unknown addresses
        return ToolResult(
            success=True,
            data={
                "address": address,
                "location": "116.397428,39.90923",
                "level": "城市",
            },
            confidence=0.6,
        )
