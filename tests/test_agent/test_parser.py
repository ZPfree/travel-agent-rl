"""Tests for the ReAct output parser.

Tests parse_react_output with three scenarios:
1. parse_thought_and_action: Tool call format
2. parse_final_answer: Final answer format
3. parse_invalid: Invalid/random text
"""

import pytest

from src.agent.parser import parse_react_output


class TestParseThoughtAndAction:
    """Test parsing Thought + Action + Action Input format."""

    def test_basic_action_parse(self):
        """Basic Action/Action Input should parse correctly."""
        text = (
            "Thought: I need to search for scenic spots in Beijing.\n"
            'Action: amap_poi_search\n'
            'Action Input: {"keyword": "景点"}'
        )
        result = parse_react_output(text)

        assert result["is_valid"] is True
        assert result["is_final"] is False
        assert "search for scenic spots" in result["thought"]
        assert result["action"] == "amap_poi_search"
        assert result["action_input"] == {"keyword": "景点"}
        assert result["final_answer"] is None

    def test_action_with_complex_input(self):
        """Action Input with multiple fields should parse correctly."""
        text = (
            "Thought: I should check the weather for the trip dates.\n"
            "Action: get_weather\n"
            'Action Input: {"city": "Beijing", "date": "2026-06-01"}'
        )
        result = parse_react_output(text)

        assert result["is_valid"] is True
        assert result["is_final"] is False
        assert result["action"] == "get_weather"
        assert result["action_input"]["city"] == "Beijing"
        assert result["action_input"]["date"] == "2026-06-01"

    def test_action_with_multiline_thought(self):
        """Multi-line thought before action should be captured."""
        text = (
            "Thought: The user wants a 3-day trip.\n"
            "I need to find hotels near the scenic area.\n"
            "Action: amap_poi_search\n"
            'Action Input: {"keyword": "酒店", "city": "北京"}'
        )
        result = parse_react_output(text)

        assert result["is_valid"] is True
        assert result["is_final"] is False
        assert "3-day trip" in result["thought"]
        assert "hotels near" in result["thought"]
        assert result["action"] == "amap_poi_search"

    def test_action_no_input(self):
        """Action without Action Input should still parse (empty input)."""
        text = (
            "Thought: I have enough information.\n"
            "Action: finalize_plan"
        )
        result = parse_react_output(text)

        assert result["is_valid"] is True
        assert result["is_final"] is False
        assert result["action"] == "finalize_plan"
        # action_input should be empty dict or None
        assert result["action_input"] in ({}, None)


class TestParseFinalAnswer:
    """Test parsing Thought + Final Answer format."""

    def test_basic_final_answer(self):
        """Basic Final Answer should parse correctly."""
        text = (
            "Thought: I have gathered all the information needed.\n"
            "Final Answer: Here is your 3-day Beijing itinerary..."
        )
        result = parse_react_output(text)

        assert result["is_valid"] is True
        assert result["is_final"] is True
        assert "gathered all" in result["thought"]
        assert "3-day Beijing itinerary" in result["final_answer"]
        assert result["action"] is None
        assert result["action_input"] is None

    def test_final_answer_with_newlines(self):
        """Final Answer spanning multiple lines should be captured."""
        text = (
            "Thought: All constraints are satisfied.\n"
            "Final Answer: Day 1: Visit the Great Wall.\n"
            "Day 2: Explore the Forbidden City.\n"
            "Day 3: Summer Palace and departure."
        )
        result = parse_react_output(text)

        assert result["is_valid"] is True
        assert result["is_final"] is True
        assert "Great Wall" in result["final_answer"]
        assert "Forbidden City" in result["final_answer"]
        assert "Summer Palace" in result["final_answer"]

    def test_final_answer_without_thought(self):
        """Final Answer without explicit Thought should still parse."""
        text = "Final Answer: Your travel plan is ready."
        result = parse_react_output(text)

        assert result["is_valid"] is True
        assert result["is_final"] is True
        assert "travel plan is ready" in result["final_answer"]


class TestParseInvalid:
    """Test parsing invalid/random text."""

    def test_random_text(self):
        """Random text without ReAct markers is invalid."""
        text = "some random text that doesn't follow the format"
        result = parse_react_output(text)

        assert result["is_valid"] is False
        assert result["is_final"] is False
        assert result["thought"] is None
        assert result["action"] is None
        assert result["action_input"] is None
        assert result["final_answer"] is None

    def test_empty_string(self):
        """Empty string is invalid."""
        text = ""
        result = parse_react_output(text)

        assert result["is_valid"] is False

    def test_partial_format_thought_only(self):
        """Thought without Action or Final Answer is invalid."""
        text = "Thought: I am thinking about what to do next."
        result = parse_react_output(text)

        # This should be invalid since there's no Action or Final Answer
        assert result["is_valid"] is False

    def test_partial_format_action_only(self):
        """Action without Thought is invalid."""
        text = 'Action: amap_poi_search\nAction Input: {"keyword": "景点"}'
        result = parse_react_output(text)

        assert result["is_valid"] is False
