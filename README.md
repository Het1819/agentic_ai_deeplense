# DeepLense — Agentic simulation workflow

This repository combines **[DeepLenseSim](https://github.com/mwt5345/DeepLenseSim)** (strong gravitational lensing simulations) with a **Pydantic AI** agent that maps natural language to validated parameters, optional human approval, and batch image generation (`.npy` / `.png` + JSON metadata).

## Layout

| Path | Role |
|------|------|
| `DeepLenseSim/` | Upstream-style package; `deeplense.lens.DeepLens` and model scripts (`Model_I/` … `Model_IV/`) |
| `deeplense_agent/` | Pydantic models, simulation runner, tools, CLI, notebook helper |
| `agent_demo.ipynb` | Submission-style demo: scripted agent run **without API keys**, matplotlib figures |
| `deeplense_agent_outputs/` | Default output directory for generated images (created at runtime) |

## Quick start

### Environment

Use Python 3.11+ and a virtual environment. Install agent + simulation dependencies:

```bash
pip install -r deeplense_agent/requirements.txt
pip install -e DeepLenseSim   # optional; or add DeepLenseSim to PYTHONPATH
```

You also need a compatible **pyHalo** checkout (this repo may include one). `DeepLenseSim/deeplense/lens.py` falls back to `pyHalo.PresetModels.cdm` if `preset_models.CDM` is unavailable.

### CLI (live LLM)

Requires `OPENAI_API_KEY` (unless you change the model in code).

```bash
# Windows cmd (from repo root — CLI adds DeepLenseSim to sys.path automatically)
set OPENAI_API_KEY=sk-...
python -m deeplense_agent --no-hitl "Generate 1 Model II CDM lens with z_lens=0.5 z_source=1.0"
```

- `--no-hitl` skips stdin confirmation and aligns with `DEEPLENSE_AUTO_APPROVE=1` for scripted runs.
- Omit `--no-hitl` for interactive human confirmation before execution.

### Notebook (no API key)

Open **`agent_demo.ipynb`** and run all cells. It uses **`FunctionModel`** to drive the same tools as a real LLM, so evaluators can verify the pipeline without configuring keys.

---

## Strategy discussion

### Why Pydantic AI

- **Shared schema with the rest of the stack**: Simulation parameters and results are **Pydantic v2** models (`SimulationRequest`, `SimulationRunResult`, `ParameterValidationReport`). Pydantic AI uses those types for **tool arguments and return values**, so tool calls are validated the same way as any other structured payload—no parallel ad hoc JSON schemas to keep in sync.

- **Structured tool calling**: The agent exposes small, typed tools (`validate_simulation_parameters_tool`, `human_confirm_plan_tool`, `execute_deeplense_simulation_tool`). The heavy physics stays in **`deeplense_agent/runner.py`**; the LLM only orchestrates. That separation improves testability and makes it obvious what ran (tool traces in logs or `all_messages_json()`).

- **Deterministic demos**: Pydantic AI’s **`FunctionModel`** lets a plain Python function emit `ModelResponse` / `ToolCallPart` sequences. The notebook uses that to run **`Agent.run()`** end-to-end **without** `OPENAI_API_KEY`, while still exercising the real tool implementations.

### Human-in-the-loop: cryptographic hash checkpoint

Interactive runs enforce a **digest gate** so execution cannot proceed on a different parameter set than the one the human approved:

1. **`SessionState`** holds `approved_digest: str | None` (see `deeplense_agent/tools_runtime.py`).
2. On approval, **`human_confirm_plan`** (or `human_confirm_plan_tool`) sets  
   `approved_digest = SHA256(spec.model_dump_json())`  
   using **`hashlib.sha256`** over the canonical JSON serialization of the validated **`SimulationRequest`**.
3. **`execute_deeplense_simulation`** checks (when `interactive_hitl=True` and `DEEPLENSE_AUTO_APPROVE` is not set):  
   `approved_digest == SHA256(current spec)`  
   If the model or user changes any field after approval, the digest mismatches and execution raises with an explicit error.

**Automation / notebooks**: `AgentDeps(interactive_hitl=False` or `DEEPLENSE_AUTO_APPROVE=1` skips stdin and the digest requirement so CI and headless notebooks can run without blocking.

---

## Supported simulation modes

- **`model_i`**: Lenstronomy-style cutouts (configurable `num_pix`, plate scale, PSF), aligned with `Model_I/` / `Model_III/` style scripts in the upstream repo.
- **`model_ii`**: Euclid `SimAPI` workflow (`Model_II/` scripts).
- **Substructure**: `cdm`, `axion`, `no_sub`.

**Model IV** in upstream depends on external galaxy HDF5 data; it is not wrapped here. Call `list_supported_configurations` from the agent for the in-repo summary.

---

## Development notes

- **Imports**: When not using the CLI, add `DeepLenseSim` to `PYTHONPATH` or install the package so `from deeplense.lens import DeepLens` resolves.
- **Optional live model**: Set `DEEPLENSE_AGENT_MODEL` to override the default OpenAI model string passed to `build_agent()`.

## License

See `DeepLenseSim/LICENSE` for DeepLenseSim; add or adjust a top-level license if you redistribute the combined project.
