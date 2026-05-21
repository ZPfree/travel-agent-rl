"""ReAct prompt templates for the travel agent RL system.

Provides helpers to build the system prompt and per-step prompt that guide the
LLM through Thought / Action / Action-Input / Final-Answer reasoning.
"""

from __future__ import annotations

# ── System Prompt Template ─────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a travel planning assistant. You help users plan trips by using \
available tools to search for points of interest, routes, weather, and \
other travel-related information.

You must respond in the following ReAct format:

When you need to use a tool:
Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <JSON arguments for the tool>

When you have gathered enough information and are ready to give the final plan:
Thought: <your final reasoning>
Final Answer: <the complete travel plan for the user>

## Available Tools
{tool_descriptions}

## Constraints
{constraints}

## Current Progress
{progress_summary}
"""


def build_system_prompt(
    tool_descriptions: str,
    constraints: str,
    progress_summary: str,
) -> str:
    """Build the system prompt by filling in the template variables.

    Args:
        tool_descriptions: Formatted string describing available tools.
        constraints:       Formatted string describing user constraints.
        progress_summary:  Formatted string from ProgressTracker.to_summary().

    Returns:
        The fully populated system prompt string.
    """
    return SYSTEM_PROMPT.format(
        tool_descriptions=tool_descriptions,
        constraints=constraints,
        progress_summary=progress_summary,
    )


def build_step_prompt(
    system_prompt: str,
    history: str,
    current_observation: str,
) -> str:
    """Build the prompt for a single agent step.

    Concatenates the system prompt, conversation history, and the current
    observation, then appends ``Thought:`` at the end to cue the LLM to
    continue reasoning.

    Args:
        system_prompt:       The full system prompt (from build_system_prompt).
        history:             Formatted conversation / action history so far.
        current_observation: The observation from the most recent tool call,
                             or an empty string on the first step.

    Returns:
        The complete prompt string for this step.
    """
    parts = [system_prompt]

    if history:
        parts.append(history)

    if current_observation:
        parts.append(f"Observation: {current_observation}")

    # Cue the LLM to start its next reasoning step
    parts.append("Thought:")

    return "\n\n".join(parts)
