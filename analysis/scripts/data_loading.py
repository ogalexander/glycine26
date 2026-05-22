"""
Load a combined-H5 dataset into an :class:`ExperimentData` container.

The ``ExperimentData`` class itself lives in ``experiment_data.py``; it
is re-exported here for backwards compatibility with code that imports
``from data_loading import ExperimentData``.
"""

import h5py
import numpy as np

from experiment_data import ExperimentData

__all__ = ["ExperimentData", "load_data", "save_vls_moments"]

_VLS_MOMENTS_GROUP = "vls_moments"


def load_data(
    filepath: str,
    config: int,
    trim_start: int = 2,
    trim_end: int = 2,
    downsample_N: int = 1,
) -> ExperimentData:
    """
    Load experiment data from an HDF5 file.

    Bad trains (``between_tdc_files == True``) are removed, as are the
    first ``trim_start`` and last ``trim_end`` trains.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    config : int
        Experiment configuration: 1 (electron + ion TOF) or 2 (VLS + liquid TOF).
    trim_start : int
        Number of trains to remove from the start after bad-train filtering.
    trim_end : int
        Number of trains to remove from the end after bad-train filtering.
    downsample_N : int
        Only load 1 in every downsample_N trains (1 to load every train), for testing purposes.

    Returns
    -------
    ExperimentData
        Loaded and filtered data container.
    """
    with h5py.File(filepath, "r") as f:

        between_tdc = f["between_tdc_files"][:].astype(bool)  # (n,)
        good_mask = ~between_tdc

        # Apply bad-train mask first, then trim edges
        indices = np.where(good_mask)[0]
        if trim_end > 0:
            indices = indices[trim_start:-trim_end:downsample_N]
        else:
            indices = indices[trim_start::downsample_N]

        def load(key):
            return f[key][indices]

        tID              = load("tID")               # (n,)
        gmd              = load("gmd")               # (n, m)
        mpe              = load("mpe")               # (n,)
        z                = load("z")                 # (n, m)
        btf              = load("between_tdc_files") # (n,)

        tofs_e     = None
        tofs_i     = None
        liq_tofs_e = None
        vls        = None

        if config == 1:
            tofs_e = load("tofs_e")  # (n, m, 50)
            tofs_i = load("tofs_i")  # (n, m, 120)
        elif config == 2:
            liq_tofs_e = load("liq_tofs_e")  # (n, m, ?)
            vls        = load("vls")          # (n, m, ?)
        else:
            raise ValueError(f"config must be 1 or 2, got {config}")

        vls_sums = vls_coms = vls_widths = None
        vls_crop_roi = vls_background_roi = None
        if _VLS_MOMENTS_GROUP in f:
            grp = f[_VLS_MOMENTS_GROUP]
            vls_sums   = grp["sums"][indices]
            vls_coms   = grp["coms"][indices]
            vls_widths = grp["widths"][indices]
            if "crop_roi" in grp.attrs:
                vls_crop_roi = tuple(int(x) for x in grp.attrs["crop_roi"])
            if "background_roi" in grp.attrs:
                vls_background_roi = tuple(int(x) for x in grp.attrs["background_roi"])
            print(f"vls_moments       : loaded {vls_sums.shape}")
            print(f"  crop_roi        : {vls_crop_roi}")
            print(f"  background_roi  : {vls_background_roi}")

    return ExperimentData(
        config=config,
        tID=tID,
        gmd=gmd,
        mpe=mpe,
        z=z,
        between_tdc_files=btf,
        tofs_e=tofs_e,
        tofs_i=tofs_i,
        liq_tofs_e=liq_tofs_e,
        vls=vls,
        vls_sums=vls_sums,
        vls_coms=vls_coms,
        vls_widths=vls_widths,
        vls_crop_roi=vls_crop_roi,
        vls_background_roi=vls_background_roi,
    )


def save_vls_moments(
    data: ExperimentData,
    filepath: str,
    overwrite: bool = False,
) -> None:
    """
    Persist per-shot VLS moments and their preprocessing metadata into a
    combined H5 file.

    The moments are scattered onto the file's full train axis (NaN where
    the train was filtered out or not loaded) by matching ``data.tID``
    against the file's ``tID`` dataset, so the file remains a valid
    combined-H5 file and ``load_data`` will pick the right rows back up
    on the next call.

    Parameters
    ----------
    data : ExperimentData
        Dataset with ``vls_sums``, ``vls_coms``, ``vls_widths``
        populated (call :meth:`ExperimentData.compute_vls_moments` first).
    filepath : str
        Path to the combined H5 file to append to. Should be the same
        file the data was loaded from.
    overwrite : bool
        If a ``vls_moments`` group already exists in the file, delete
        and replace it. Default ``False`` (raises ``FileExistsError``).

    Raises
    ------
    ValueError
        If moments are not populated, or if any train ID in ``data`` is
        not present in the file.
    FileExistsError
        If the moments group already exists and ``overwrite`` is False.

    Notes
    -----
    The ``vls_moments/`` group is written with three ``(n_trains_full,
    n_bunches)`` datasets (``sums``, ``coms``, ``widths``) plus two
    attributes documenting the preprocessing applied to the spectra
    before moment calculation: ``crop_roi`` (pixel ROI) and
    ``background_roi`` (bunch ROI used for auto background subtraction).
    Either attribute is omitted if the corresponding preprocessing was
    not applied.
    """
    if data.vls_sums is None or data.vls_coms is None or data.vls_widths is None:
        raise ValueError(
            "data has no VLS moments. Call data.compute_vls_moments() first."
        )

    with h5py.File(filepath, "a") as f:
        if _VLS_MOMENTS_GROUP in f:
            if not overwrite:
                raise FileExistsError(
                    f"Group '{_VLS_MOMENTS_GROUP}' already exists in {filepath}; "
                    "pass overwrite=True to replace."
                )
            del f[_VLS_MOMENTS_GROUP]

        file_tID = f["tID"][:]
        sorter = np.argsort(file_tID)
        pos = np.searchsorted(file_tID[sorter], data.tID)
        full_idx = sorter[pos]
        if not np.array_equal(file_tID[full_idx], data.tID):
            raise ValueError(
                "Some train IDs in `data` are not present in the file; "
                "cannot scatter moments back to the full train axis."
            )

        n_full = file_tID.shape[0]
        m = data.n_bunches
        sums_full   = np.full((n_full, m), np.nan, dtype=np.float64)
        coms_full   = np.full((n_full, m), np.nan, dtype=np.float64)
        widths_full = np.full((n_full, m), np.nan, dtype=np.float64)
        sums_full[full_idx]   = data.vls_sums
        coms_full[full_idx]   = data.vls_coms
        widths_full[full_idx] = data.vls_widths

        grp = f.create_group(_VLS_MOMENTS_GROUP)
        grp.create_dataset("sums",   data=sums_full,   compression="gzip")
        grp.create_dataset("coms",   data=coms_full,   compression="gzip")
        grp.create_dataset("widths", data=widths_full, compression="gzip")

        if data.vls_crop_roi is not None:
            grp.attrs["crop_roi"] = np.asarray(data.vls_crop_roi, dtype=np.int64)
        if data.vls_background_roi is not None:
            grp.attrs["background_roi"] = np.asarray(
                data.vls_background_roi, dtype=np.int64
            )
