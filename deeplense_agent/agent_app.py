from __future__ import annotations

import os
from pathlib import Path

from pydantic_ai import Agent, RunContext

from deeplense_agent.models import (
    ParameterValidationReport,
    SimulationRequest,
    SimulationRunResult,
)
from deeplense_agent.tools_runtime import (
    AgentDeps,
    execute_deeplense_simulation,
    human_confirm_plan,
    supported_configurations_summary,
    validate_simulation_parameters,
)


SYSTEM_PROMPT = """You are a scientific assistant for gravitational lensing simulations (DeepLenseSim).

Goals:
1. If the user is vague (missing tier, substructure, counts, or redshifts), ask concise clarifying questions before proposing parameters.
2. When you have a concrete plan, build a single JSON object that matches the `SimulationRequest` schema and pass it to `validate_simulation_parameters`.
3. If validation fails, explain gaps to the user and iterate.
4. Before any expensive run, call `human_confirm_plan` with a short human-readable summary plus the validated `SimulationRequest` object (interactive sessions only).
5. After approval, call `execute_deeplense_simulation` with the same `SimulationRequest` instance.

Tier hints from user language:
- \"high resolution cutout\", \"150 pixels\", \"classic DeepLense\"/\"model 1\" -> model_i
- \"Euclid\", \"space survey\", \"model 2\" -> model_ii

Never claim you produced files until `execute_deeplense_simulation` returns paths."""


def build_agent(model: str | None = None) -> Agent[AgentDeps, str]:
    m = model or os.environ.get("DEEPLENSE_AGENT_MODEL", "openai:gpt-4o-mini")

    agent: Agent[AgentDeps, str] = Agent(
        m,
        deps_type=AgentDeps,
        system_prompt=SYSTEM_PROMPT,
        output_type=str,
        defer_model_check=True,
    )

    @agent.tool_plain
    def list_supported_configurations() -> str:
        """Return which model tiers and substructure modes the agent can run locally."""
        return supported_configurations_summary()

    @agent.tool
    def validate_simulation_parameters_tool(
        ctx: RunContext[AgentDeps], params: dict[str, object]
    ) -> ParameterValidationReport:
        """Validate a dict against SimulationRequest. Prefer calling this before confirmation/execution."""
        return validate_simulation_parameters(params)

    @agent.tool
    def human_confirm_plan_tool(ctx: RunContext[AgentDeps], summary: str, spec: SimulationRequest) -> str:
        """Pause for a human decision on the exact simulation parameters (stdin in CLI). Records approval digest."""
        return human_confirm_plan(ctx.deps, summary, spec)

    @agent.tool
    def execute_deeplense_simulation_tool(ctx: RunContext[AgentDeps], spec: SimulationRequest) -> SimulationRunResult:
        """Run DeepLenseSim via the shared runner and return structured metadata + artifact paths."""
        return execute_deeplense_simulation(ctx.deps, spec)

    return agent
