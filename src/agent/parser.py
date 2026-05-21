"""ReAct output parser for the travel agent RL system.

Parses LLM output in the ReAct format:

    Thought: <reasoning>
    Action: <tool_name>
    Action Input: <json_args>

or:

    Thought: <reasoning>
    Final Answer: <answer>
"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_react_output(text: str) -> dict[str, Any]:
    """Parse a ReAct-format string into structured fields.

    Args:
        text: Raw LLM output to parse.

    Returns:
        Dict with keys:
            is_valid (bool):      Whether the text matched a valid ReAct format.
            is_final (bool):      True when the output contains a Final Answer.
            thought (str | None): The Thought portion, if present.
            action (str | None):  The Action name, if present.
            action_input (dict | None): Parsed Action Input JSON, if present.
            final_answer (str | None):  The Final Answer text, if present.
    """
    result: dict[str, Any] = {
        "is_valid": False,
        "is_final": False,
        "thought": None,
        "action": None,
        "action_input": None,
        "final_answer": None,
    }

    if not text or not text.strip():
        return result

    # --- Extract Thought (optional prefix before Action / Final Answer) ------
    thought_match = re.search(
        r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|\Z)",
        text,
        re.DOTALL,
    )
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    # --- Try Final Answer first (higher priority) ---------------------------
    final_match = re.search(
        r"Final Answer:\s*(.*)",
        text,
        re.DOTALL,
    )
    if final_match:
        result["is_final"] = True
        result["is_valid"] = True
        result["final_answer"] = final_match.group(1).strip()
        return result

    # --- Try Action + Action Input ------------------------------------------
    action_match = re.search(r"Action:\s*(\S+)", text)
    action_input_match = re.search(
        r"Action Input:\s*(\{.*\})",
        text,
        re.DOTALL,
    )

    if action_match:
        result["action"] = action_match.group(1).strip()

        if action_input_match:
            try:
                result["action_input"] = json.loads(
                    action_input_match.group(1)
                )
            except json.JSONDecodeError:
                result["action_input"] = {}
        else:
            result["action_input"] = {}

        # An action is only valid if there is also a thought
        if result["thought"]:
            result["is_valid"] = True
        else:
            # Action without Thought is invalid
            result["is_valid"] = False

    return result
