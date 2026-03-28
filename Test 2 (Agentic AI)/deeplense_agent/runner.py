from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from deeplense.lens import DeepLens

from deeplense_agent.models import (
    ImageArtifact,
    ModelTier,
    SimulationRequest,
    SimulationRunResult,
    SubstructureType,
)


def _ensure_output_dir(base: Path | None) -> Path:
    root = base or Path.cwd() / "deeplense_agent_outputs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _simple_sim_custom(
    lens: DeepLens,
    *,
    num_pix: int,
    delta_pix: float,
    fwhm: float,
    background_rms: float,
) -> np.ndarray:
    """Lenstronomy single-band simulation mirroring `DeepLens.simple_sim` with tunable plate scale."""
    from lenstronomy.Data.imaging_data import ImageData
    from lenstronomy.Data.psf import PSF
    from lenstronomy.ImSim.image_model import ImageModel
    from lenstronomy.Util import util
    import lenstronomy.Util.image_util as image_util

    exp_time = 10 ** np.random.uniform(3, 3.5)
    _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ = util.make_grid_with_coordtransform(
        numPix=num_pix,
        deltapix=delta_pix,
        center_ra=0,
        center_dec=0,
        subgrid_res=1,
        inverse=False,
    )
    kwargs_data = {
        "background_rms": background_rms,
        "exposure_time": exp_time,
        "ra_at_xy_0": ra_at_xy_0,
        "dec_at_xy_0": dec_at_xy_0,
        "transform_pix2angle": Mpix2coord,
        "image_data": np.zeros((num_pix, num_pix)),
    }
    data_class = ImageData(**kwargs_data)
    kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": fwhm, "pixel_size": delta_pix, "truncation": 3}
    psf_class = PSF(**kwargs_psf)
    kwargs_numerics = {"supersampling_factor": 1, "supersampling_convolution": False}
    image_model = ImageModel(
        data_class,
        psf_class,
        lens_model_class=lens.lens_model_class,
        source_model_class=lens.source_model_class,
        kwargs_numerics=kwargs_numerics,
        lens_light_model_class=None,
    )
    image_clean = image_model.image(lens.kwargs_lens_list, lens.kwargs_source, kwargs_lens_light=None, kwargs_ps=None)
    poisson = image_util.add_poisson(image_clean, exp_time=exp_time)
    bkg = image_util.add_background(image_clean, sigma_bkd=background_rms)
    image_real = exp_time * (image_clean + poisson + bkg)
    return np.random.poisson(np.clip(image_real, 0, None))


def _apply_substructure(lens: DeepLens, req: SimulationRequest) -> None:
    if req.substructure == SubstructureType.CDM:
        masses = lens.draw_old_cdm_sub_masses(
            m_sub_min=req.m_sub_min_solar,
            m_sub_max=req.m_sub_max_solar,
            n_sub=req.n_sub_mean,
            beta=req.cdm_mass_slope_beta,
        )
        e_list = lens.mass_to_radius(masses, lens.z_halo, lens.z_gal)
        subhalo_type = "POINT_MASS"
        for i in range(len(e_list)):
            lens.lens_model_list.append(subhalo_type)
            r, th = np.random.uniform(0.25, 2.0), np.random.uniform(0, 2 * np.pi)
            x1, x2 = r * np.sin(th), r * np.cos(th)
            lens.kwargs_lens_list.append({"theta_E": e_list[i], "center_x": x1, "center_y": x2})
            lens.lens_redshift_list.append(lens.z_halo)
        from lenstronomy.LensModel.lens_model import LensModel

        lens.lens_model_class = LensModel(lens.lens_model_list)
    elif req.substructure == SubstructureType.NO_SUB:
        lens.make_no_sub()
    elif req.substructure == SubstructureType.AXION:
        assert req.axion_mass_eV is not None and req.vortex_mass_solar is not None
        lens.axion_mass = req.axion_mass_eV
        lens.make_vortex(req.vortex_mass_solar, res=req.vortex_resolution)
    else:
        raise ValueError(f"Unknown substructure {req.substructure!r}")


def _run_model_i_one(req: SimulationRequest) -> np.ndarray:
    lens = DeepLens(
        axion_mass=req.axion_mass_eV if req.substructure == SubstructureType.AXION else None,
        H0=req.H0,
        Om0=req.Om0,
        Ob0=req.Ob0,
        z_halo=req.z_lens,
        z_gal=req.z_source,
    )
    lens.make_single_halo(req.main_halo_mass_solar)
    _apply_substructure(lens, req)
    lens.make_source_light()
    return _simple_sim_custom(
        lens,
        num_pix=req.num_pix,
        delta_pix=req.pixel_scale_arcsec,
        fwhm=req.psf_fwhm_arcsec,
        background_rms=req.background_rms,
    )


def _run_model_ii_one(req: SimulationRequest) -> np.ndarray:
    from lenstronomy.SimulationAPI.sim_api import SimAPI
    from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid

    lens = DeepLens(
        axion_mass=req.axion_mass_eV if req.substructure == SubstructureType.AXION else None,
        H0=req.H0,
        Om0=req.Om0,
        Ob0=req.Ob0,
        z_halo=req.z_lens,
        z_gal=req.z_source,
    )
    lens.make_single_halo(req.main_halo_mass_solar)
    _apply_substructure(lens, req)
    lens.set_instrument("Euclid")
    lens.make_source_light_mag()

    kwargs_model_physical = {
        "lens_model_list": lens.lens_model_list,
        "lens_redshift_list": lens.lens_redshift_list,
        "source_light_model_list": lens.source_model_list,
        "source_redshift_list": lens.source_redshift_list,
        "cosmo": lens.astropy_instance,
        "z_source_convention": 2.5,
        "z_source": req.z_source,
    }
    kwargs_numerics = {"point_source_supersampling_factor": 1}
    euclid = Euclid(band="VIS", psf_type="GAUSSIAN", coadd_years=6)
    sim = SimAPI(
        numpix=req.euclid_num_pix,
        kwargs_single_band=euclid.kwargs_single_band(),
        kwargs_model=kwargs_model_physical,
    )
    im_sim = sim.image_model_class(kwargs_numerics)
    _, kwargs_source, _ = sim.magnitude2amplitude(None, lens.kwargs_source)
    image = im_sim.image(lens.kwargs_lens_list, kwargs_source, None)
    noise = sim.noise_for_model(model=image)
    return image + noise


def run_simulation_batch(
    req: SimulationRequest,
    *,
    output_dir: Path | None = None,
    save_png: bool = True,
    run_id: str | None = None,
) -> SimulationRunResult:
    """Execute DeepLenseSim-style pipeline for Model I or Model II."""
    if req.seed is not None:
        np.random.seed(req.seed)

    out = _ensure_output_dir(output_dir)
    rid = run_id or "run"
    sub = rid.replace(" ", "_")

    artifacts: list[ImageArtifact] = []
    notes: list[str] = []

    for i in range(req.num_images):
        if req.tier == ModelTier.MODEL_I:
            img = _run_model_i_one(req)
        elif req.tier == ModelTier.MODEL_II:
            img = _run_model_ii_one(req)
        else:
            raise ValueError(f"Unsupported tier {req.tier}")

        stem = f"{sub}_img_{i:03d}"
        np_path = out / f"{stem}.npy"
        np.save(np_path, img)
        artifacts.append(
            ImageArtifact(
                path=np_path.resolve(),
                format="npy",
                shape=img.shape,
                dtype=str(img.dtype),
                tier=req.tier,
                substructure=req.substructure,
                z_lens=req.z_lens,
                z_source=req.z_source,
                index=i,
            )
        )
        if save_png:
            png_path = out / f"{stem}.png"
            img_f = np.asarray(img, dtype=float)
            if img_f.max() > 0:
                img_u8 = (255 * (img_f / img_f.max())).clip(0, 255).astype(np.uint8)
            else:
                img_u8 = np.zeros_like(img_f, dtype=np.uint8)
            Image.fromarray(img_u8).save(png_path)
            artifacts.append(
                ImageArtifact(
                    path=png_path.resolve(),
                    format="png",
                    shape=img_u8.shape,
                    dtype=str(img_u8.dtype),
                    tier=req.tier,
                    substructure=req.substructure,
                    z_lens=req.z_lens,
                    z_source=req.z_source,
                    index=i,
                )
            )

    meta_path = out / f"{sub}_metadata.json"
    payload: dict[str, Any] = {
        "request": json.loads(req.model_dump_json()),
        "artifacts": [json.loads(a.model_dump_json()) for a in artifacts],
        "notes": notes,
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    notes.append(f"Wrote aggregated metadata to {meta_path}")

    return SimulationRunResult(request=req, artifacts=artifacts, notes=notes, extra={"metadata_path": str(meta_path)})
