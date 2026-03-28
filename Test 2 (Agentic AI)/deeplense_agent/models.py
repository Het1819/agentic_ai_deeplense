from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelTier(str, Enum):
    """Which published DeepLenseSim configuration to emulate.

    - **model_i**: Lenstronomy `simple_sim`-style cutouts (see `Model_I/` scripts).
    - **model_ii**: Euclid band `SimAPI` / `simple_sim_2`-style (see `Model_II/` scripts).
    """

    MODEL_I = "model_i"
    MODEL_II = "model_ii"


class SubstructureType(str, Enum):
    CDM = "cdm"
    AXION = "axion"
    NO_SUB = "no_sub"


class SimulationRequest(BaseModel):
    """Validated simulation parameters the agent maps from natural language."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    tier: ModelTier
    substructure: SubstructureType
    num_images: int = Field(default=1, ge=1, le=64)
    z_lens: float = Field(default=0.5, gt=0.0)
    z_source: float = Field(default=1.0, gt=0.0)
    main_halo_mass_solar: float = Field(default=1e12, gt=0)
    H0: float = Field(default=70.0, gt=0)
    Om0: float = Field(default=0.3, gt=0, lt=1)
    Ob0: float = Field(default=0.05, gt=0, lt=1)

    # Geometry / instrument (Model I)
    num_pix: int = Field(default=150, ge=32, le=512)
    pixel_scale_arcsec: float = Field(default=0.05, gt=0)
    psf_fwhm_arcsec: float = Field(default=0.087, gt=0)
    background_rms: float = Field(default=1e-2, gt=0)

    # Axion / vortex (only if substructure == axion)
    axion_mass_eV: float | None = Field(
        default=None,
        description="Axion mass in eV; maps to de Broglie scale in DeepLens.make_vortex pipeline.",
    )
    vortex_mass_solar: float | None = Field(
        default=3e10,
        description="Total substructure mass budget for vortex discretization (solar masses).",
    )
    vortex_resolution: int = Field(default=100, ge=8, le=500)

    # CDM point-mass swarm (only if substructure == cdm); forwarded to draw_old_cdm_sub_masses
    n_sub_mean: int = Field(default=25, ge=0)
    m_sub_min_solar: float = Field(default=1e6, gt=0)
    m_sub_max_solar: float = Field(default=1e10, gt=0)
    cdm_mass_slope_beta: float = Field(default=-0.9, description="Negative dN/dM power-law slope.")

    # Model II: SimAPI uses fixed numpix in upstream; we keep 64 to match scripts but allow override for small grids.
    euclid_num_pix: int = Field(default=64, ge=32, le=256)

    seed: int | None = Field(default=None, description="RNG seed for reproducible batches.")

    @model_validator(mode="after")
    def _z_order(self) -> SimulationRequest:
        if self.z_source <= self.z_lens:
            raise ValueError("Source redshift must be greater than lens redshift.")
        return self

    @model_validator(mode="after")
    def _axion_fields(self) -> SimulationRequest:
        if self.substructure == SubstructureType.AXION:
            if self.axion_mass_eV is None or self.axion_mass_eV <= 0:
                raise ValueError("axion_mass_eV is required and must be positive for axion substructure.")
            if self.vortex_mass_solar is None:
                raise ValueError("vortex_mass_solar is required for axion substructure.")
        return self

    @model_validator(mode="after")
    def _cdm_mass_range(self) -> SimulationRequest:
        if self.m_sub_min_solar >= self.m_sub_max_solar:
            raise ValueError("m_sub_min_solar must be less than m_sub_max_solar.")
        return self


class ImageArtifact(BaseModel):
    """One generated image plus machine-readable metadata."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    format: str = Field(description="npy | png")
    shape: tuple[int, ...]
    dtype: str
    tier: ModelTier
    substructure: SubstructureType
    z_lens: float
    z_source: float
    index: int = Field(ge=0, description="Index within this batch.")


class SimulationRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: SimulationRequest
    artifacts: list[ImageArtifact]
    notes: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class ParameterValidationReport(BaseModel):
    """Returned by the validation tool for structured agent reasoning."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    message: str
    missing_or_ambiguous: list[str] = Field(default_factory=list)
    normalized: SimulationRequest | None = None
