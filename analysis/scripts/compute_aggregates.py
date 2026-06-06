"""
Compute binned matrix aggregates for covariance / ADMM ghost imaging.

For each GMD bin, accumulates over all kept shots:

    <A>   = (1/N) Σ Aᵢ                 mean VLS spectrum
    <AtA> = (1/N) Σ Aᵢ Aᵢᵀ
    <D>   = (1/N) Σ Dᵢ                 mean binned eTOF spectrum
    <DtD> = (1/N) Σ Dᵢ Dᵢᵀ
    <AtD> = (1/N) Σ Aᵢ Dᵢᵀ

The H5 file is iterated in chunks of trains so the full
``(n_shots, n_pixels)`` and ``(n_shots, n_tof)`` matrices never live in
memory; only the chunk and the small per-bin aggregates do.

Config file
-----------
A Python module exposing the module-level names below. Example file:
``analysis/configs/aggregates_example.py``.

    CROP_ROI    : (roi_min, roi_max) half-open VLS pixel ROI
    BACKGROUND  : dict
        {"type": "auto",  "roi": (bunch_start, bunch_end)}   or
        {"type": "array", "path": "background.npy"}          or
        {"type": "none"}
    TOF_EDGES   : 1D array-like of TOF bin edges (100 ps)
    GMD_EDGES   : 1D array-like of GMD bin edges (µJ); shots outside
                  the first/last edges are dropped
    CONFIG      : 1 or 2  (H5 layout — picks tofs_e vs liq_tofs_e)
    CHUNK_SIZE  : trains per chunk         (optional, default 200)
    TRIM_START  : trains trimmed from the start (optional, 0)
    TRIM_END    : trains trimmed from the end   (optional, 0)
    OUTPUT_SUFFIX : str appended to the default output filename
                    (optional, default ""). Inserted before ".h5", so
                    a value of "_v2" turns the default
                    "<input>_aggregates.h5" into
                    "<input>_aggregates_v2.h5". Ignored when ``-o`` is
                    passed on the CLI.

CLI
---
    python compute_aggregates.py CONFIG.py INPUT.h5 [-o OUT.h5]

Output H5 layout (means per bin):
    /A          (N_GMD, n_pixels)
    /AtA        (N_GMD, n_pixels, n_pixels)
    /D          (N_GMD, n_tof)
    /DtD        (N_GMD, n_tof, n_tof)
    /AtD        (N_GMD, n_pixels, n_tof)
    /n_per_bin  (N_GMD,)              int64
    /gmd_edges  (N_GMD+1,)
    /tof_edges  (n_tof+1,)
    /vls_pixels (n_pixels,)           source-pixel indices
    /background (n_pixels,)           background that was subtracted
    attrs:      provenance + summary counts
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np


__all__ = [
    "AggregatesData",
    "compute_aggregates",
    "load_aggregates",
    "load_python_config",
    "main",
]


# ----------------------------------------------------------------------
# Public data container + loader
# ----------------------------------------------------------------------

@dataclass
class AggregatesData:
    """
    Container for the binned aggregates produced by
    :func:`compute_aggregates`. All matrix fields are *means* over the
    shots that fell into each bin (empty bins give NaN).

    The shapes and which fields are populated depend on the
    experiment configuration (``config`` attribute in :attr:`metadata`):

    Config 2 — VLS + liquid-jet eTOF  (A, D, plus G aggregates):
        A    (N_GMD[, N_Z], n_pixels)
        AtA  (N_GMD[, N_Z], n_pixels, n_pixels)
        D    (N_GMD[, N_Z], n_tof)
        DtD  (N_GMD[, N_Z], n_tof, n_tof)
        AtD  (N_GMD[, N_Z], n_pixels, n_tof)
        AtG  (N_GMD[, N_Z], n_pixels)
        DtG  (N_GMD[, N_Z], n_tof)

    Config 1 — electron + ion TOF (D = eTOF, C = ion TOF):
        D    (N_GMD, n_tof)
        DtD  (N_GMD, n_tof, n_tof)
        C    (N_GMD, n_tof_i)
        CtC  (N_GMD, n_tof_i, n_tof_i)
        DtC  (N_GMD, n_tof, n_tof_i)
        DtG  (N_GMD, n_tof)
        CtG  (N_GMD, n_tof_i)

    Both configs always populate the GMD aggregates and bookkeeping:
        G          (per-bin mean GMD, scalar per bin)
        GtG        (per-bin mean GMD², for Var(G))
        n_per_bin  (shot counts per bin)
        gmd_edges, tof_edges, metadata

    Time-resolved mode (config 2 only): an extra ``N_Z`` axis is
    inserted before the per-shot dimensions; ``z_edges`` is populated
    and ``tof_edges`` collapses to ``[tof_roi_min, tof_roi_max]``
    (``n_tof = 1``).

    XAS-scan mode (config 2, written by ``compute_xas_aggregates.py``):
    bins are ``(N_E, N_GMD)`` indexed by per-section nominal photon
    energy and GMD. Only the VLS aggregates (``A``, ``AtA``, ``AtG``)
    and GMD aggregates (``G``, ``GtG``, ``n_per_bin``) are populated;
    eTOF fields (``D``, ``DtD``, ``DtG``, ``tof_edges``) are ``None``.
    ``nominal_energies`` carries the (N_E,) energy axis in eV.
    """
    # Bookkeeping (always present).
    G: np.ndarray
    GtG: np.ndarray
    n_per_bin: np.ndarray
    gmd_edges: np.ndarray
    # eTOF (spectral / time-resolved modes; None in xas_scan).
    D: Optional[np.ndarray] = None
    DtD: Optional[np.ndarray] = None
    DtG: Optional[np.ndarray] = None
    tof_edges: Optional[np.ndarray] = None
    # Config 2 only.
    A: Optional[np.ndarray] = None
    AtA: Optional[np.ndarray] = None
    AtD: Optional[np.ndarray] = None
    AtG: Optional[np.ndarray] = None
    vls_pixels: Optional[np.ndarray] = None
    background: Optional[np.ndarray] = None
    # Config 1 only.
    C: Optional[np.ndarray] = None
    CtC: Optional[np.ndarray] = None
    DtC: Optional[np.ndarray] = None
    CtG: Optional[np.ndarray] = None
    ion_tof_edges: Optional[np.ndarray] = None
    # TR mode (config 2).
    z_edges: Optional[np.ndarray] = None
    # XAS-scan mode (config 2).
    nominal_energies: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    @property
    def config(self) -> int:
        return int(self.metadata.get("config", 2))

    @property
    def mode(self) -> str:
        if self.nominal_energies is not None:
            return "xas_scan"
        if self.z_edges is not None:
            return "time_resolved"
        return "spectral"

    @property
    def n_gmd_bins(self) -> int:
        return int(self.gmd_edges.shape[0]) - 1

    @property
    def n_z_bins(self) -> Optional[int]:
        return None if self.z_edges is None else int(self.z_edges.shape[0]) - 1

    @property
    def n_energy_bins(self) -> Optional[int]:
        return None if self.nominal_energies is None else int(self.nominal_energies.shape[0])

    @property
    def n_pixels(self) -> Optional[int]:
        if self.vls_pixels is None:
            return None
        return int(self.vls_pixels.shape[0])

    @property
    def n_tof(self) -> Optional[int]:
        if self.tof_edges is None:
            return None
        return int(self.tof_edges.shape[0]) - 1

    @property
    def n_tof_i(self) -> Optional[int]:
        if self.ion_tof_edges is None:
            return None
        return int(self.ion_tof_edges.shape[0]) - 1

    @property
    def gmd_centres(self) -> np.ndarray:
        return 0.5 * (self.gmd_edges[:-1] + self.gmd_edges[1:])

    @property
    def z_centres(self) -> Optional[np.ndarray]:
        if self.z_edges is None:
            return None
        return 0.5 * (self.z_edges[:-1] + self.z_edges[1:])

    @property
    def tof_centres(self) -> Optional[np.ndarray]:
        if self.tof_edges is None:
            return None
        return 0.5 * (self.tof_edges[:-1] + self.tof_edges[1:])

    @property
    def ion_tof_centres(self) -> Optional[np.ndarray]:
        if self.ion_tof_edges is None:
            return None
        return 0.5 * (self.ion_tof_edges[:-1] + self.ion_tof_edges[1:])

    @property
    def var_G(self) -> np.ndarray:
        """Per-bin variance of G: ``<G²> − <G>²``. Useful for partial
        covariances against GMD."""
        return self.GtG - self.G ** 2


def load_aggregates(path: Union[str, Path]) -> AggregatesData:
    """
    Load aggregates produced by :func:`compute_aggregates`. The mode is
    inferred from whether ``z_edges`` is present in the file.

    Parameters
    ----------
    path : str or Path
        Path to the aggregates H5 file.

    Returns
    -------
    AggregatesData
        All aggregate arrays + metadata.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        meta = {}
        for k in f.attrs:
            v = f.attrs[k]
            meta[k] = v.tolist() if isinstance(v, np.ndarray) else v

        def _opt(k):
            return f[k][:] if k in f else None

        return AggregatesData(
            # always present
            G=f["G"][:],
            GtG=f["GtG"][:],
            n_per_bin=f["n_per_bin"][:],
            gmd_edges=f["gmd_edges"][:],
            # eTOF (spectral / TR modes)
            D=_opt("D"),
            DtD=_opt("DtD"),
            DtG=_opt("DtG"),
            tof_edges=_opt("tof_edges"),
            # config 2
            A=_opt("A"),
            AtA=_opt("AtA"),
            AtD=_opt("AtD"),
            AtG=_opt("AtG"),
            vls_pixels=_opt("vls_pixels"),
            background=_opt("background"),
            # config 1
            C=_opt("C"),
            CtC=_opt("CtC"),
            DtC=_opt("DtC"),
            CtG=_opt("CtG"),
            ion_tof_edges=_opt("ion_tof_edges"),
            # TR mode
            z_edges=_opt("z_edges"),
            # XAS-scan mode
            nominal_energies=_opt("nominal_energies"),
            metadata=meta,
        )


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _load_python_config(path: Path):
    spec = importlib.util.spec_from_file_location("aggregates_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load config module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Public alias for use by sibling scripts (compute_xas_aggregates.py).
load_python_config = _load_python_config


def _iter_chunks(n_total: int, chunk_size: int):
    for s in range(0, n_total, chunk_size):
        yield s, min(s + chunk_size, n_total)


def _build_keep_mask(
    f: h5py.File, trim_start: int, trim_end: int
) -> np.ndarray:
    """
    Boolean mask of length ``n_total``: True for trains that pass the
    bad-train filter (``between_tdc_files == False``) and are not in
    the first ``trim_start`` or last ``trim_end`` of the surviving set.
    """
    btf = f["between_tdc_files"][:].astype(bool)
    n_total = btf.shape[0]
    good_pos = np.where(~btf)[0]
    if trim_end > 0:
        good_pos = good_pos[trim_start:-trim_end]
    else:
        good_pos = good_pos[trim_start:]
    keep_mask = np.zeros(n_total, dtype=bool)
    keep_mask[good_pos] = True
    return keep_mask


def _histogram_tof_chunk(
    tofs: np.ndarray, tof_edges: np.ndarray
) -> np.ndarray:
    """
    Convert ``(n, m, max_hits)`` zero-padded TOF hits to ``(n, m, n_bins)``
    histograms via a single ``np.histogram2d`` call. Zero entries are
    treated as padding and excluded.
    """
    n, m, _ = tofs.shape
    n_bins = len(tof_edges) - 1
    valid = tofs != 0
    if not valid.any():
        return np.zeros((n, m, n_bins), dtype=np.float64)
    tof_h = tofs[valid]
    train_of_hit = np.broadcast_to(
        np.arange(n)[:, None, None], tofs.shape
    )[valid]
    bunch_of_hit = np.broadcast_to(
        np.arange(m)[None, :, None], tofs.shape
    )[valid]
    shot_id = train_of_hit * m + bunch_of_hit
    H, _, _ = np.histogram2d(
        shot_id, tof_h,
        bins=[np.arange(n * m + 1) - 0.5, tof_edges],
    )
    return H.reshape(n, m, n_bins).astype(np.float64)


def _compute_auto_background(
    f: h5py.File,
    keep_mask: np.ndarray,
    bg_start: int,
    bg_end: int,
    roi_min: int,
    roi_max: int,
    chunk_size: int,
) -> np.ndarray:
    """
    Stream-average VLS over (kept trains, bunches in
    ``[bg_start, bg_end)``) within the cropped pixel ROI. Returns a
    ``(roi_max - roi_min,)`` float64 spectrum.
    """
    accum = np.zeros(roi_max - roi_min, dtype=np.float64)
    count = 0
    for s, e in _iter_chunks(keep_mask.shape[0], chunk_size):
        sub_mask = keep_mask[s:e]
        if not sub_mask.any():
            continue
        block = f["vls"][s:e, bg_start:bg_end, roi_min:roi_max][sub_mask]
        accum += block.sum(axis=(0, 1), dtype=np.float64)
        count += block.shape[0] * block.shape[1]
    if count == 0:
        raise RuntimeError("no shots available for auto background")
    return accum / count


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def _verify_h5_for_config(f: h5py.File, config: int) -> None:
    """
    Sanity-check that the H5 file actually contains the per-shot
    datasets that match the ``config`` declared in the config module.
    """
    keys = set(f.keys())
    if config == 1:
        required = {"gmd", "tofs_e", "tofs_i", "between_tdc_files", "tID"}
        wrong   = {"vls", "liq_tofs_e"} & keys
    elif config == 2:
        required = {"gmd", "vls", "liq_tofs_e", "between_tdc_files", "tID"}
        wrong   = {"tofs_e", "tofs_i"} & keys
    else:
        raise ValueError(f"config must be 1 or 2, got {config}")
    missing = required - keys
    if missing:
        raise RuntimeError(
            f"CONFIG = {config} but the H5 file is missing the expected "
            f"datasets {sorted(missing)}. Check the config file."
        )
    if wrong:
        print(f"warning: CONFIG = {config} but the H5 file also contains "
              f"{sorted(wrong)}; ignoring those datasets.")


def compute_aggregates(
    input_h5: Union[str, Path],
    output_h5: Union[str, Path],
    *,
    gmd_edges: np.ndarray,
    config: int,
    mode: str = "spectral",
    tof_edges: Optional[np.ndarray] = None,
    ion_tof_edges: Optional[np.ndarray] = None,
    crop_roi: Optional[Tuple[int, int]] = None,
    background_spec: Optional[dict] = None,
    z_edges: Optional[np.ndarray] = None,
    tof_roi: Optional[Tuple[float, float]] = None,
    chunk_size: int = 200,
    trim_start: int = 0,
    trim_end: int = 0,
    config_path: Optional[Path] = None,
    verbose: bool = True,
) -> None:
    """
    Stream the input H5 in chunks and write binned aggregates to disk.

    Supports two H5 layouts:

    * ``config = 1`` — electron + ion TOF (``tofs_e``, ``tofs_i``); no
      VLS. Computes per-bin ``D`` (electron spectrum), ``C`` (ion
      spectrum) and the cross-product ``DtC``. ``crop_roi`` and
      ``background_spec`` are unused.

    * ``config = 2`` — VLS + liquid-jet eTOF (``vls``, ``liq_tofs_e``).
      Computes ``A`` (VLS), ``D`` (eTOF), and ``AtD``. ``crop_roi`` and
      ``background_spec`` are required.

    Both configs always compute the GMD aggregates ``G = <G_i>``,
    ``GtG = <G_i²>``, ``DtG = <D_i G_i>`` (and ``AtG`` / ``CtG`` where
    those modalities are present). These let downstream code form
    partial covariances against GMD.

    Modes
    -----
    ``mode = "spectral"`` (default)
        Bin by GMD only. Aggregate shapes have a leading ``N_GMD`` axis.
    ``mode = "time_resolved"`` (config 2 only)
        Bin by GMD × stage z, with the eTOF reduced per shot to a scalar
        count inside ``tof_roi``. Aggregate shapes have leading
        ``(N_GMD, N_Z)`` axes; ``n_tof`` collapses to 1.

    Parameters
    ----------
    input_h5, output_h5 : str or Path
    gmd_edges : array-like
        GMD bin edges. Shots outside the first / last edge are dropped.
    config : int  (1 or 2)
    mode : {"spectral", "time_resolved"}
    tof_edges : array-like
        eTOF histogram edges (100 ps) — required for spectral mode.
    ion_tof_edges : array-like
        Ion-TOF histogram edges (100 ps) — required for ``config = 1``.
    crop_roi : (int, int)
        Half-open VLS pixel ROI — required for ``config = 2``.
    background_spec : dict
        ``{"type": "auto",  "roi": (s, e)}`` |
        ``{"type": "array", "path": "..."}`` |
        ``{"type": "none"}`` — required for ``config = 2``.
    z_edges, tof_roi : array-like
        Required for ``mode = "time_resolved"`` (config 2 only).
    chunk_size, trim_start, trim_end : int
    config_path : Path, optional
        Recorded as a provenance attribute.
    verbose : bool
    """
    input_h5 = Path(input_h5)
    output_h5 = Path(output_h5)
    config = int(config)
    gmd_edges = np.asarray(gmd_edges, dtype=np.float64)
    n_gmd = len(gmd_edges) - 1

    # ----- per-config validation + key resolution ------------------------
    if config == 1:
        if mode != "spectral":
            raise NotImplementedError(
                "mode='time_resolved' is currently only supported for config=2"
            )
        if tof_edges is None or ion_tof_edges is None:
            raise ValueError("config=1 requires tof_edges and ion_tof_edges")
        tof_edges     = np.asarray(tof_edges, dtype=np.float64)
        ion_tof_edges = np.asarray(ion_tof_edges, dtype=np.float64)
        n_tof   = len(tof_edges) - 1
        n_tof_i = len(ion_tof_edges) - 1
        n_pixels = None
        e_tof_key = "tofs_e"
    elif config == 2:
        if tof_edges is None and mode == "spectral":
            raise ValueError("mode='spectral' (config=2) requires tof_edges")
        if crop_roi is None or background_spec is None:
            raise ValueError("config=2 requires crop_roi and background_spec")
        e_tof_key = "liq_tofs_e"
        n_tof_i = None
        ion_tof_edges = None
        roi_min, roi_max = int(crop_roi[0]), int(crop_roi[1])
        n_pixels = roi_max - roi_min
        if mode == "spectral":
            tof_edges = np.asarray(tof_edges, dtype=np.float64)
            n_tof = len(tof_edges) - 1
        elif mode == "time_resolved":
            if z_edges is None or tof_roi is None:
                raise ValueError(
                    "mode='time_resolved' requires z_edges and tof_roi"
                )
            tof_roi_arr = np.asarray(tof_roi, dtype=np.float64)
            if tof_roi_arr.shape != (2,):
                raise ValueError(f"tof_roi must be (2,), got {tof_roi_arr.shape}")
            tof_edges = tof_roi_arr.copy()
            n_tof = 1
        else:
            raise ValueError(f"unknown mode {mode!r}")
    else:
        raise ValueError(f"config must be 1 or 2, got {config}")

    if mode == "time_resolved":
        z_edges_arr = np.asarray(z_edges, dtype=np.float64)
        n_z = len(z_edges_arr) - 1
    else:
        z_edges_arr = None
        n_z = None

    log = print if verbose else (lambda *a, **k: None)

    with h5py.File(input_h5, "r") as f:
        _verify_h5_for_config(f, config)
        keep_mask = _build_keep_mask(f, trim_start, trim_end)
        n_total = keep_mask.shape[0]
        m = f["gmd"].shape[1]

        log(f"input              : {input_h5}")
        log(f"config             : {config}")
        log(f"mode               : {mode}")
        log(f"trains             : {n_total} total, "
            f"{int(keep_mask.sum())} kept after bad-train filter + trim")
        log(f"bunches / train    : {m}")
        log(f"GMD bins           : {n_gmd}  edges {gmd_edges}")
        if config == 2:
            log(f"VLS ROI            : [{roi_min}, {roi_max}) = {n_pixels} pixels")
        if mode == "spectral":
            log(f"eTOF bins          : {n_tof} ({tof_edges[0]:.1f} .. {tof_edges[-1]:.1f} 100ps)")
            if config == 1:
                log(f"ion TOF bins       : {n_tof_i} ({ion_tof_edges[0]:.1f} .. {ion_tof_edges[-1]:.1f} 100ps)")
        else:
            log(f"TOF ROI            : [{tof_edges[0]:.1f}, {tof_edges[-1]:.1f}) 100ps (scalar D)")
            log(f"Z bins             : {n_z}  edges {z_edges_arr}")

        # --- resolve background (config 2 only) ------------------------
        background = None
        background_meta = None
        if config == 2:
            bg_type = background_spec["type"]
            if bg_type == "auto":
                bg_start, bg_end = background_spec["roi"]
                log(f"computing auto background over bunches [{bg_start}, {bg_end})...")
                background = _compute_auto_background(
                    f, keep_mask, int(bg_start), int(bg_end),
                    roi_min, roi_max, chunk_size,
                )
                background_meta = {"type": "auto",
                                   "roi": [int(bg_start), int(bg_end)]}
            elif bg_type == "array":
                bg_arr = np.load(background_spec["path"])
                if bg_arr.shape != (n_pixels,):
                    raise ValueError(
                        f"background array shape {bg_arr.shape} != "
                        f"VLS ROI ({n_pixels},)"
                    )
                background = bg_arr.astype(np.float64)
                background_meta = {"type": "array",
                                   "path": str(background_spec["path"])}
            elif bg_type == "none":
                background = np.zeros(n_pixels, dtype=np.float64)
                background_meta = {"type": "none"}
            else:
                raise ValueError(f"unknown background type: {bg_type!r}")

        # --- allocate aggregate sums -----------------------------------
        if mode == "spectral":
            agg_prefix = (n_gmd,)
            n_bins_total = n_gmd
        else:
            agg_prefix = (n_gmd, n_z)
            n_bins_total = n_gmd * n_z

        # Always present (both configs).
        D_sum   = np.zeros(agg_prefix + (n_tof,),       dtype=np.float64)
        DtD_sum = np.zeros(agg_prefix + (n_tof, n_tof), dtype=np.float64)
        G_sum   = np.zeros(agg_prefix,                  dtype=np.float64)
        GtG_sum = np.zeros(agg_prefix,                  dtype=np.float64)
        DtG_sum = np.zeros(agg_prefix + (n_tof,),       dtype=np.float64)
        n_per_bin = np.zeros(agg_prefix, dtype=np.int64)

        # Config-specific.
        if config == 2:
            A_sum   = np.zeros(agg_prefix + (n_pixels,),          dtype=np.float64)
            AtA_sum = np.zeros(agg_prefix + (n_pixels, n_pixels), dtype=np.float64)
            AtD_sum = np.zeros(agg_prefix + (n_pixels, n_tof),    dtype=np.float64)
            AtG_sum = np.zeros(agg_prefix + (n_pixels,),          dtype=np.float64)
        else:
            C_sum   = np.zeros(agg_prefix + (n_tof_i,),           dtype=np.float64)
            CtC_sum = np.zeros(agg_prefix + (n_tof_i, n_tof_i),   dtype=np.float64)
            DtC_sum = np.zeros(agg_prefix + (n_tof, n_tof_i),     dtype=np.float64)
            CtG_sum = np.zeros(agg_prefix + (n_tof_i,),           dtype=np.float64)

        n_skipped_gmd_oor = 0
        n_skipped_gmd_nan = 0
        n_skipped_z_oor   = 0
        n_skipped_z_nan   = 0

        log("accumulating...")
        for s, e in _iter_chunks(n_total, chunk_size):
            sub_mask = keep_mask[s:e]
            if not sub_mask.any():
                continue

            gmd_block = f["gmd"][s:e][sub_mask]                            # (n_keep, m)
            tof_block = f[e_tof_key][s:e][sub_mask]                        # (n_keep, m, max_hits)
            if config == 2:
                vls_block = f["vls"][s:e, :, roi_min:roi_max][sub_mask]    # (n_keep, m, n_pixels)
                A_block = vls_block.astype(np.float64) - background
            else:
                ion_tof_block = f["tofs_i"][s:e][sub_mask]                 # (n_keep, m, max_hits_i)
                C_block = _histogram_tof_chunk(ion_tof_block, ion_tof_edges)
            if mode == "time_resolved":
                z_block = f["z"][s:e][sub_mask]

            n_keep = gmd_block.shape[0]

            if mode == "spectral":
                D_block = _histogram_tof_chunk(tof_block, tof_edges)
            else:
                roi_lo, roi_hi = float(tof_edges[0]), float(tof_edges[-1])
                in_roi = (tof_block >= roi_lo) & (tof_block < roi_hi) & (tof_block != 0)
                D_block = in_roi.sum(axis=-1).astype(np.float64)[:, :, None]

            # GMD bin per shot.
            gmd_flat = gmd_block.ravel()
            gmd_bin  = np.digitize(gmd_flat, gmd_edges) - 1
            gmd_nan  = ~np.isfinite(gmd_flat)
            gmd_oor  = ~gmd_nan & ((gmd_bin < 0) | (gmd_bin >= n_gmd))
            valid    = ~gmd_nan & ~gmd_oor
            n_skipped_gmd_nan += int(gmd_nan.sum())
            n_skipped_gmd_oor += int(gmd_oor.sum())

            if mode == "spectral":
                bin_id = gmd_bin
            else:
                z_flat = z_block.ravel()
                z_bin  = np.digitize(z_flat, z_edges_arr) - 1
                z_nan  = ~np.isfinite(z_flat)
                z_oor  = ~z_nan & ((z_bin < 0) | (z_bin >= n_z))
                n_skipped_z_nan += int(z_nan.sum())
                n_skipped_z_oor += int(z_oor.sum())
                valid = valid & ~z_nan & ~z_oor
                bin_id = gmd_bin * n_z + z_bin

            D_flat = D_block.reshape(-1, n_tof)
            if config == 2:
                A_flat = A_block.reshape(-1, n_pixels)
            else:
                C_flat = C_block.reshape(-1, n_tof_i)

            for b in np.unique(bin_id[valid]):
                mask = (bin_id == b) & valid
                if mode == "spectral":
                    idx = (int(b),)
                else:
                    idx = (int(b) // n_z, int(b) % n_z)

                G_b = gmd_flat[mask]                # (n_b,)  per-shot scalar
                D_b = D_flat[mask]                  # (n_b, n_tof)
                n_b = G_b.shape[0]

                n_per_bin[idx] += n_b
                G_sum[idx]   += G_b.sum()
                GtG_sum[idx] += float(G_b @ G_b)
                D_sum[idx]   += D_b.sum(axis=0)
                DtD_sum[idx] += D_b.T @ D_b
                DtG_sum[idx] += D_b.T @ G_b

                if config == 2:
                    A_b = A_flat[mask]
                    A_sum[idx]   += A_b.sum(axis=0)
                    AtA_sum[idx] += A_b.T @ A_b
                    AtD_sum[idx] += A_b.T @ D_b
                    AtG_sum[idx] += A_b.T @ G_b
                else:
                    C_b = C_flat[mask]
                    C_sum[idx]   += C_b.sum(axis=0)
                    CtC_sum[idx] += C_b.T @ C_b
                    DtC_sum[idx] += D_b.T @ C_b
                    CtG_sum[idx] += C_b.T @ G_b

            log(f"  [{s:>6d}..{e:>6d})  trains={n_keep:>4d}  "
                f"binned={int(valid.sum()):>5d}")

    # --- means ---------------------------------------------------------
    safe_n = np.where(n_per_bin > 0, n_per_bin, 1).astype(np.float64)
    empty = n_per_bin == 0

    def _norm_vec(s):
        m = s / safe_n[..., None]
        m[empty] = np.nan
        return m

    def _norm_mat(s):
        m = s / safe_n[..., None, None]
        m[empty] = np.nan
        return m

    def _norm_scalar(s):
        m = s / safe_n
        m[empty] = np.nan
        return m

    means = {
        "D":   _norm_vec(D_sum),
        "DtD": _norm_mat(DtD_sum),
        "G":   _norm_scalar(G_sum),
        "GtG": _norm_scalar(GtG_sum),
        "DtG": _norm_vec(DtG_sum),
    }
    if config == 2:
        means.update({
            "A":   _norm_vec(A_sum),
            "AtA": _norm_mat(AtA_sum),
            "AtD": _norm_mat(AtD_sum),
            "AtG": _norm_vec(AtG_sum),
        })
    else:
        means.update({
            "C":   _norm_vec(C_sum),
            "CtC": _norm_mat(CtC_sum),
            "DtC": _norm_mat(DtC_sum),
            "CtG": _norm_vec(CtG_sum),
        })

    # --- save ----------------------------------------------------------
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    log(f"writing {output_h5}")
    with h5py.File(output_h5, "w") as fout:
        for name, arr in means.items():
            fout.create_dataset(name, data=arr, compression="gzip")
        fout.create_dataset("n_per_bin", data=n_per_bin)
        fout.create_dataset("gmd_edges", data=gmd_edges)
        fout.create_dataset("tof_edges", data=tof_edges)
        if config == 2:
            fout.create_dataset("vls_pixels", data=np.arange(roi_min, roi_max, dtype=np.int64))
            fout.create_dataset("background", data=background)
        else:
            fout.create_dataset("ion_tof_edges", data=ion_tof_edges)
        if mode == "time_resolved":
            fout.create_dataset("z_edges", data=z_edges_arr)

        fout.attrs["input_h5"] = str(input_h5)
        if config_path is not None:
            fout.attrs["config_path"] = str(config_path)
        fout.attrs["mode"]            = mode
        fout.attrs["config"]          = int(config)
        fout.attrs["trim_start"]      = int(trim_start)
        fout.attrs["trim_end"]        = int(trim_end)
        fout.attrs["chunk_size"]      = int(chunk_size)
        fout.attrs["n_skipped_gmd_oor"] = int(n_skipped_gmd_oor)
        fout.attrs["n_skipped_gmd_nan"] = int(n_skipped_gmd_nan)
        if config == 2:
            fout.attrs["vls_crop_roi"]    = np.asarray([roi_min, roi_max], dtype=np.int64)
            fout.attrs["background_type"] = background_meta["type"]
            if "roi" in background_meta:
                fout.attrs["background_roi"] = np.asarray(
                    background_meta["roi"], dtype=np.int64
                )
            if "path" in background_meta:
                fout.attrs["background_path"] = background_meta["path"]
        if mode == "time_resolved":
            fout.attrs["n_skipped_z_oor"] = int(n_skipped_z_oor)
            fout.attrs["n_skipped_z_nan"] = int(n_skipped_z_nan)
            fout.attrs["tof_roi"]         = np.asarray(tof_edges, dtype=np.float64)

    summary = (f"done: {int(n_per_bin.sum())} shots aggregated "
               f"across {n_bins_total} bins (config={config}, mode={mode})")
    if mode == "time_resolved":
        summary += (f"; skipped {n_skipped_gmd_oor} GMD-oor, "
                    f"{n_skipped_z_oor} z-oor, "
                    f"{n_skipped_gmd_nan + n_skipped_z_nan} nan")
    else:
        summary += (f"; skipped {n_skipped_gmd_oor} GMD-oor, "
                    f"{n_skipped_gmd_nan} nan")
    log(summary)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config_path", type=Path,
        help="Python module exposing CROP_ROI, BACKGROUND, GMD_EDGES, "
             "CONFIG, plus MODE-specific extras (TOF_EDGES, or "
             "Z_EDGES + TOF_ROI)",
    )
    parser.add_argument(
        "input_h5", type=Path, help="combined H5 file to process",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output H5 (default: <input>_aggregates.h5 next to the input)",
    )
    args = parser.parse_args(argv)

    cfg = _load_python_config(args.config_path)

    cfg_n = int(cfg.CONFIG)
    mode = getattr(cfg, "MODE", "spectral")
    base = "_aggregates_tr" if mode == "time_resolved" else "_aggregates"
    user_suffix = getattr(cfg, "OUTPUT_SUFFIX", "") or ""
    out = args.output or args.input_h5.with_name(
        args.input_h5.stem + base + user_suffix + ".h5"
    )

    kwargs = dict(
        input_h5=args.input_h5,
        output_h5=out,
        gmd_edges=np.asarray(cfg.GMD_EDGES, dtype=float),
        config=cfg_n,
        mode=mode,
        chunk_size=int(getattr(cfg, "CHUNK_SIZE", 200)),
        trim_start=int(getattr(cfg, "TRIM_START", 0)),
        trim_end=int(getattr(cfg, "TRIM_END", 0)),
        config_path=args.config_path,
    )

    if cfg_n == 2:
        background = cfg.BACKGROUND
        # Shorthand: tuple/list of length 2 is treated as an auto bunch ROI.
        if isinstance(background, (tuple, list)) and len(background) == 2:
            background = {"type": "auto", "roi": list(background)}
        kwargs["crop_roi"] = cfg.CROP_ROI
        kwargs["background_spec"] = background

    if mode == "spectral":
        kwargs["tof_edges"] = np.asarray(cfg.TOF_EDGES, dtype=float)
        if cfg_n == 1:
            kwargs["ion_tof_edges"] = np.asarray(cfg.ION_TOF_EDGES, dtype=float)
    else:
        kwargs["z_edges"] = np.asarray(cfg.Z_EDGES, dtype=float)
        kwargs["tof_roi"] = tuple(cfg.TOF_ROI)

    compute_aggregates(**kwargs)


if __name__ == "__main__":
    main()
