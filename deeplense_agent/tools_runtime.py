from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from deeplense_agent.models import (
    ParameterValidationReport,
    SimulationRequest,
    SimulationRunResult,
)
from deeplense_agent.runner import run_simulation_batch


def _digest(spec: SimulationRequest) -> str:
    return hashlib.sha256(spec.model_dump_json().encode()).hexdigest()


@dataclass
class SessionState:
    """Mutable per-run state for human-in-the-loop gating."""

    approved_digest: str | None = None


@dataclass
class AgentDeps:
    """Dependencies injected into each agent run."""

    interactive_hitl: bool = True
    output_dir: Path | None = None
    session: SessionState = field(default_factory=SessionState)


def supported_configurations_summary() -> str:
    return """\
Supported DeepLenseSim-style tiers (this package):
- **model_i**: Lenstronomy `simple_sim` class of pipelines (see DeepLenseSim `Model_I/`; `Model_III/` uses the same code path in the upstream repo).
- **model_ii**: Euclid `SimAPI` single-band workflow (`Model_II/` scripts: `set_instrument('Euclid')`, `simple_sim_2`).

Substructure modes (all tiers): **cdm** (power-law point-mass field), **axion** (vortex / fuzzy DM sketch), **no_sub**.

Notes:
- **model_iv** in the upstream repository couples pyHalo line-of-sight halos to real galaxy cutouts (`Galaxy10_DECals.h5`) and is not bundled here; use model_i or model_ii for reproducible agent runs without that dataset.

Use `validate_simulation_parameters` with a JSON object before `execute_deeplense_simulation`. In interactive mode, call `human_confirm_plan` after validation and before execution.
"""


def validate_simulation_parameters(raw: dict[str, Any]) -> ParameterValidationReport:
    try:
        spec = SimulationRequest.model_validate(raw)
    except ValidationError as e:
        missing = [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]
        return ParameterValidationReport(
            ok=False,
            message="Simulation parameters failed validation. Ask the user targeted follow-up questions, then retry.",
            missing_or_ambiguous=missing,
            normalized=None,
        )
    followups: list[str] = []
    if spec.substructure == "axion" and raw.get("axion_mass_eV") in (None, "", "default"):
        followups.append("Confirm axion_mass_eV (positive float, eV); fuzzy models are sensitive to this scale.")
    if spec.num_images > 16:
        followups.append(f"Planned batch size is {spec.num_images}; confirm this runtime is acceptable.")
    msg = "Parameters are valid and ready for confirmation."
    if followups:
        msg += " Optional refinements: " + "; ".join(followups)
    return ParameterValidationReport(
        ok=True,
        message=msg,
        missing_or_ambiguous=followups,
        normalized=spec,
    )


def human_confirm_plan(deps: AgentDeps, summary: str, spec: SimulationRequest) -> str:
    if not deps.interactive_hitl or os.environ.get("DEEPLENSE_AUTO_APPROVE", "").lower() in ("1", "true", "yes"):
        deps.session.approved_digest = _digest(spec)
        return "approved_non_interactive"
    print("\n--- Simulation plan (human review) ---\n")
    print(summary.strip())
    print("\nJSON:", spec.model_dump_json(indent=2))
    choice = input("\nApprove and run? [y/N/edit]: ").strip().lower()
    if choice in ("y", "yes"):
        deps.session.approved_digest = _digest(spec)
        return "approved"
    if choice == "edit":
        return "user_requests_edit"
    return "rejected"


def execute_deeplense_simulation(deps: AgentDeps, spec: SimulationRequest) -> SimulationRunResult:
    if deps.interactive_hitl and os.environ.get("DEEPLENSE_AUTO_APPROVE", "").lower() not in ("1", "true", "yes"):
        digest = _digest(spec)
        if deps.session.approved_digest != digest:
            raise RuntimeError(
                "Execution blocked: run `human_confirm_plan` successfully for this exact parameter set first, "
                "or set DEEPLENSE_AUTO_APPROVE=1 for scripted runs."
            )
    return run_simulation_batch(spec, output_dir=deps.output_dir)
