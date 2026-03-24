"""Deterministic agent driver for notebooks (no API keys).

Uses pydantic-ai's ``FunctionModel`` to issue the same tool sequence a real LLM would:
validate → human_confirm → execute.
"""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from deeplense_agent.agent_app import build_agent
from deeplense_agent.models import SimulationRequest


def _completed_tool_names(messages: list[ModelMessage]) -> set[str]:
    done: set[str] = set()
    for m in messages:
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, ToolReturnPart):
                    done.add(p.tool_name)
    return done


def build_notebook_scripted_agent(
    spec: SimulationRequest,
    *,
    summary: str = "Notebook scripted demo (FunctionModel; no LLM).",
):
    """Return an :class:`~pydantic_ai.Agent` that drives the standard tool pipeline in order."""

    spec_json = spec.model_dump(mode="json")

    def drive(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
        done = _completed_tool_names(messages)
        if "validate_simulation_parameters_tool" not in done:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "validate_simulation_parameters_tool",
                        {"params": spec_json},
                        tool_call_id="nb_validate",
                    )
                ]
            )
        if "human_confirm_plan_tool" not in done:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "human_confirm_plan_tool",
                        {"summary": summary, "spec": spec_json},
                        tool_call_id="nb_confirm",
                    )
                ]
            )
        if "execute_deeplense_simulation_tool" not in done:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "execute_deeplense_simulation_tool",
                        dict(spec_json),
                        tool_call_id="nb_execute",
                    )
                ]
            )
        return ModelResponse(
            parts=[
                TextPart(
                    "Scripted run finished. See tool returns for SimulationRunResult (paths to .npy / .png)."
                )
            ]
        )

    return build_agent(model=FunctionModel(drive))
