"""AgentRunner - Episode collector for the travel agent RL system.

Collects complete episodes (sequences of ReAct steps) for PPO training.
Each episode consists of multiple StepData records and an Episode summary
with total reward and completion status.

The runner uses a rule-based strategy by default (model=None) for cold
start data collection, or can accept a model to generate actions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.tools.registry import ToolRegistry
from src.tools.base import ToolResult
from src.agent.prompt import build_system_prompt, build_step_prompt
from src.agent.parser import parse_react_output
from src.agent.context import ContextCompressor
from src.agent.state import ProgressTracker


# ── Dataclasses ───────────────────────────────────────────────────────────


@dataclass
class StepData:
    """Records a single agent step within an episode.

    Attributes:
        prompt:       The full prompt sent to the LLM (or rule-based engine).
        action:       The action name extracted from ReAct output.
        action_input: The parsed JSON arguments for the action.
        observation:  The tool execution result as a string.
        reward:       Step-level reward computed by _compute_step_reward.
        logprob:      Log probability of the action (0.0 for rule-based).
        step:         Zero-based step index within the episode.
    """

    prompt: str
    action: str
    action_input: dict[str, Any]
    observation: str
    reward: float
    logprob: float
    step: int


@dataclass
class Episode:
    """Complete episode record for PPO training.

    Attributes:
        query:        The user's original travel query.
        constraints:  List of user constraint dicts.
        steps:        Ordered list of StepData records.
        total_reward: Sum of all step rewards.
        is_complete:  Whether the agent issued a Final Answer.
    """

    query: str
    constraints: list[dict[str, Any]]
    steps: list[StepData]
    total_reward: float
    is_complete: bool


# ── AgentRunner ───────────────────────────────────────────────────────────


class AgentRunner:
    """Collects episodes by running the agent in a loop.

    Parameters:
        max_steps:     Maximum number of ReAct steps per episode.
        context_limit: Token budget for context compression.
    """

    def __init__(
        self,
        max_steps: int = 50,
        context_limit: int = 20000,
    ) -> None:
        self.max_steps = max_steps
        self.context_limit = context_limit
        self.registry = ToolRegistry()
        self.compressor = ContextCompressor(token_limit=context_limit)

    # ── public API ────────────────────────────────────────────────────────

    def collect_episode(
        self,
        query: str,
        constraints: list[dict[str, Any]],
        model: Any | None = None,
    ) -> Episode:
        """Run the agent loop and collect one full episode.

        Args:
            query:       The user's travel planning query.
            constraints: List of constraint dicts (type, value, weight).
            model:       Optional LLM model for action generation.
                         When None, uses rule-based strategy (cold start).

        Returns:
            An Episode with all steps, rewards, and completion status.
        """
        tracker = ProgressTracker()
        messages: list[dict[str, Any]] = []
        steps: list[StepData] = []
        is_complete = False

        # Format tool descriptions and constraints for the system prompt
        tool_descriptions = self._format_tool_descriptions()
        constraints_str = self._format_constraints(constraints)

        for step_idx in range(self.max_steps):
            # Build progress summary from tracker
            progress_summary = tracker.to_summary()

            # Build system prompt
            system_prompt = build_system_prompt(
                tool_descriptions=tool_descriptions,
                constraints=constraints_str,
                progress_summary=progress_summary,
            )

            # Build conversation history string
            history = self._format_history(messages)

            # Current observation from previous step
            current_observation = ""
            if messages:
                last = messages[-1]
                if last.get("role") == "observation":
                    current_observation = last.get("content", "")

            # Build the full step prompt
            prompt = build_step_prompt(
                system_prompt=system_prompt,
                history=history,
                current_observation=current_observation,
            )

            # Generate ReAct output
            if model is not None:
                react_output = model.generate(prompt)
            else:
                react_output = self._rule_based_action(
                    query=query,
                    constraints=constraints,
                    step=step_idx,
                    tracker=tracker,
                )

            # Parse the ReAct output
            parsed = parse_react_output(react_output)

            # Handle invalid parse
            if not parsed["is_valid"]:
                # Record a step with failure reward
                step_data = StepData(
                    prompt=prompt,
                    action="invalid",
                    action_input={},
                    observation="Invalid ReAct output",
                    reward=-0.1,
                    logprob=0.0,
                    step=step_idx,
                )
                steps.append(step_data)
                messages.append({
                    "role": "assistant",
                    "content": react_output,
                })
                messages.append({
                    "role": "observation",
                    "content": "Invalid ReAct output. Please use the correct format.",
                })
                continue

            # Handle Final Answer
            if parsed["is_final"]:
                step_data = StepData(
                    prompt=prompt,
                    action="final_answer",
                    action_input={},
                    observation=parsed["final_answer"] or "",
                    reward=0.5,
                    logprob=0.0,
                    step=step_idx,
                )
                steps.append(step_data)
                is_complete = True
                break

            # Execute the tool
            action_name = parsed["action"]
            action_input = parsed["action_input"] or {}
            result = self.registry.execute(action_name, **action_input)

            # Format observation string
            observation = self._format_observation(result)

            # Compute step reward
            reward = self._compute_step_reward(parsed, result, tracker)

            # Update tracker
            self._update_tracker(tracker, action_name, action_input, result)

            # Record step
            step_data = StepData(
                prompt=prompt,
                action=action_name,
                action_input=action_input,
                observation=observation,
                reward=reward,
                logprob=0.0,
                step=step_idx,
            )
            steps.append(step_data)

            # Update message history
            messages.append({
                "role": "assistant",
                "content": react_output,
            })
            messages.append({
                "role": "observation",
                "content": observation,
            })

            # Compress context if needed
            messages = self.compressor.compress(messages, tracker)

        # Compute total reward
        total_reward = sum(s.reward for s in steps)

        return Episode(
            query=query,
            constraints=constraints,
            steps=steps,
            total_reward=total_reward,
            is_complete=is_complete,
        )

    # ── rule-based strategy ───────────────────────────────────────────────

    def _rule_based_action(
        self,
        query: str,
        constraints: list[dict[str, Any]],
        step: int,
        tracker: ProgressTracker,
    ) -> str:
        """Generate a ReAct-formatted action string using rule-based logic.

        Used for cold start data collection when no trained model is
        available.  Produces deterministic, structured actions that
        cover different tool types across steps.

        Args:
            query:       The user's travel query.
            constraints: User constraints.
            step:        Current step index (0-based).
            tracker:     Current progress tracker.

        Returns:
            A ReAct-formatted string with Thought/Action/Action Input.
        """
        # Define a rotation of tool actions for diverse coverage
        actions = [
            ("search_poi", self._make_poi_action(query)),
            ("query_weather", self._make_weather_action(query, constraints)),
            ("plan_route", self._make_route_action(query)),
            ("search_nearby", self._make_nearby_action(query)),
            ("search_poi", self._make_poi_action(query, variant="food")),
        ]

        # Select action based on step index (cycle through)
        action_name, action_input = actions[step % len(actions)]

        # Build thought based on step and tracker state
        thought = self._build_thought(query, step, tracker, action_name)

        return (
            f"Thought: {thought}\n"
            f"Action: {action_name}\n"
            f"Action Input: {json.dumps(action_input, ensure_ascii=False)}"
        )

    def _build_thought(
        self,
        query: str,
        step: int,
        tracker: ProgressTracker,
        action_name: str,
    ) -> str:
        """Generate a context-appropriate thought string."""
        n_pois = len(tracker.collected_pois)
        n_routes = len(tracker.route_segments)
        n_weather = len(tracker.weather_cache)

        if step == 0:
            return f"I need to search for points of interest for: {query}"
        elif action_name == "search_poi":
            if n_pois == 0:
                return "I should start by finding relevant points of interest."
            return f"I have {n_pois} POIs so far, searching for more options."
        elif action_name == "query_weather":
            return "I should check the weather to plan activities appropriately."
        elif action_name == "plan_route":
            if n_routes == 0:
                return "Now I need to plan routes between the destinations."
            return f"I have {n_routes} route segments, checking for alternatives."
        elif action_name == "search_nearby":
            return "Let me search for nearby attractions and facilities."
        return f"Step {step}: Continuing to gather information."

    def _make_poi_action(self, query: str, variant: str = "attraction") -> dict[str, Any]:
        """Build a search_poi action input."""
        # Extract a city hint from the query
        keywords = query.split()
        city = keywords[-1] if keywords else "destination"
        search_term = f"{variant} in {city}" if variant != "attraction" else city
        return {"query": search_term, "city": city}

    def _make_weather_action(
        self, query: str, constraints: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Build a query_weather action input."""
        keywords = query.split()
        city = keywords[-1] if keywords else "destination"
        return {"city": city}

    def _make_route_action(self, query: str) -> dict[str, Any]:
        """Build a plan_route action input."""
        keywords = query.split()
        city = keywords[-1] if keywords else "destination"
        return {
            "origin": f"{city} center",
            "destination": f"{city} airport",
            "mode": "driving",
        }

    def _make_nearby_action(self, query: str) -> dict[str, Any]:
        """Build a search_nearby action input."""
        keywords = query.split()
        city = keywords[-1] if keywords else "destination"
        return {
            "location": city,
            "keyword": "restaurant",
            "radius": 1000,
        }

    # ── reward computation ────────────────────────────────────────────────

    def _compute_step_reward(
        self,
        parsed: dict[str, Any],
        result: ToolResult,
        tracker: ProgressTracker,
    ) -> float:
        """Compute the reward for a single step.

        Reward scheme:
            +0.1   Successful tool execution
            +0.05  High confidence result (confidence >= 0.8)
            -0.1   Tool execution failure

        Args:
            parsed:  Parsed ReAct output dict.
            result:  ToolResult from tool execution.
            tracker: Current progress tracker.

        Returns:
            The computed step reward.
        """
        reward = 0.0

        if result.success:
            reward += 0.1
            # High confidence bonus
            if result.confidence >= 0.8:
                reward += 0.05
        else:
            reward -= 0.1

        return reward

    # ── tracker updates ───────────────────────────────────────────────────

    def _update_tracker(
        self,
        tracker: ProgressTracker,
        action: str,
        params: dict[str, Any],
        result: ToolResult,
    ) -> None:
        """Update the ProgressTracker based on tool execution result.

        Maps tool actions to tracker methods:
            search_poi / search_nearby -> add_poi
            plan_route -> add_route
            query_weather -> cache_weather

        Args:
            tracker: The ProgressTracker to update.
            action:  The tool action name.
            params:  The tool call parameters.
            result:  The ToolResult from execution.
        """
        if not result.success:
            return

        data = result.data

        if action in ("search_poi", "search_nearby"):
            # Handle list of POIs or single POI
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        tracker.add_poi(item)
            elif isinstance(data, dict):
                tracker.add_poi(data)

        elif action == "plan_route":
            if isinstance(data, dict):
                route_info = {**params, **data}
                tracker.add_route(route_info)

        elif action == "query_weather":
            city = params.get("city", "unknown")
            if isinstance(data, dict):
                tracker.cache_weather(city, data)

    # ── formatting helpers ────────────────────────────────────────────────

    def _format_tool_descriptions(self) -> str:
        """Format tool descriptions for the system prompt."""
        descriptions = self.registry.get_tool_descriptions()
        lines = []
        for desc in descriptions:
            name = desc.get("name", "unknown")
            description = desc.get("description", "No description")
            params = desc.get("parameters", {})
            lines.append(f"- {name}: {description}")
            if params:
                props = params.get("properties", {})
                if props:
                    param_str = ", ".join(
                        f"{k} ({v.get('type', 'any')})" for k, v in props.items()
                    )
                    lines.append(f"  Parameters: {param_str}")
        return "\n".join(lines)

    def _format_constraints(self, constraints: list[dict[str, Any]]) -> str:
        """Format constraints for the system prompt."""
        if not constraints:
            return "No specific constraints."
        lines = []
        for c in constraints:
            c_type = c.get("type", "unknown")
            c_value = c.get("value", "N/A")
            c_weight = c.get("weight", 1.0)
            lines.append(f"- {c_type}: {c_value} (weight: {c_weight})")
        return "\n".join(lines)

    def _format_history(self, messages: list[dict[str, Any]]) -> str:
        """Format conversation history for the step prompt."""
        if not messages:
            return ""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"[{role}]: {content}")
        return "\n\n".join(parts)

    def _format_observation(self, result: ToolResult) -> str:
        """Format a ToolResult as an observation string."""
        if result.success:
            if isinstance(result.data, (dict, list)):
                return json.dumps(result.data, ensure_ascii=False)
            return str(result.data) if result.data is not None else "Success (no data)"
        return f"Error: {result.error}"
