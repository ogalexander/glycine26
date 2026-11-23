"""
Load a combined-H5 dataset into an :class:`ExperimentData` container.

The ``ExperimentData`` class itself lives in ``experiment_data.py``; it
is re-exported here for backwards compatibility with code that imports
``from data_loading import ExperimentData``.
"""

import glob
from pathlib import Path

import h5py
import numpy as np

from experiment_data import ExperimentData

__all__ = ["ExperimentData", "load_data", "load_raw_h5", "save_vls_moments"]

_VLS_MOMENTS_GROUP = "vls_moments"

# Raw FLASH H5 dataset paths (duplicated from write_h5.py so this module
# does not pull in the legacy TDC decoder dependency).
_GMD_INDEX = "/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall/index"
_GMD_VALUE = "/FL2/Photon Diagnostic/GMD/Pulse resolved energy/energy hall/value"
_MPE_INDEX = "/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/mean photon energy/index"
_MPE_VALUE = "/FL2/Photon Diagnostic/Wavelength/OPIS tunnel/Processed/mean photon energy/value"
_VLS_INDEX = "/FL2/Support Infrastructure/Gotthard/images/index"
_VLS_VALUE = "/FL2/Support Infrastructure/Gotthard/images/value"
_SHUTTER_INDEX = "/FL2/Beamlines/Fast Shutter/shutter/index"
_SHUTTER_VALUE = "/FL2/Beamlines/Fast Shutter/shutter/value"


def load_data(
    filepath: str,
    config: int,
    trim_start: int = 2,
    trim_end: int = 2,
    downsample_N: int = 1,
    load_vls: bool = True,
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
        Experiment configuration: 1 (gas arm: electron + ion TOF, plus VLS
        when the file has it) or 2 (liquid arm: VLS + liquid TOF).
    trim_start : int
        Number of trains to remove from the start after bad-train filtering.
    trim_end : int
        Number of trains to remove from the end after bad-train filtering.
    downsample_N : int
        Only load 1 in every downsample_N trains (1 to load every train), for testing purposes.
    load_vls : bool
        Load the ``vls`` dataset when present (default True). The VLS is
        by far the largest field (~0.5 MB per train as float32), so pass
        False when only TOF / GMD are needed.

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

        # Shutter was added to the combined-H5 schema after the initial
        # write_h5 release; treat as optional so older files still load.
        shutter = load("shutter") if "shutter" in f else None

        tofs_e     = None
        tofs_i     = None
        liq_tofs_e = None
        vls        = None

        # The TOF datasets are optional: write_h5.py omits them when the
        # run has no TDC .lst files. VLS is optional in both configs
        # (config 1 runs without the Gotthard, pathological cfg-2 files).
        # Anything missing stays None on the result.
        if config == 1:
            if "tofs_e" in f:
                tofs_e = load("tofs_e")  # (n, m, 50)
            if "tofs_i" in f:
                tofs_i = load("tofs_i")  # (n, m, 120)
        elif config == 2:
            if "liq_tofs_e" in f:
                liq_tofs_e = load("liq_tofs_e")  # (n, m, ?)
        else:
            raise ValueError(f"config must be 1 or 2, got {config}")
        if load_vls and "vls" in f:
            vls = load("vls")                    # (n, m, n_pixels)

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
        shutter=shutter,
        vls_sums=vls_sums,
        vls_coms=vls_coms,
        vls_widths=vls_widths,
        vls_crop_roi=vls_crop_roi,
        vls_background_roi=vls_background_roi,
    )


def _align_by_tID(src_idx, master_tID):
    """
    Return ``(matched, src_pos)`` such that ``src_idx[src_pos]`` are the
    entries of the source dataset whose train IDs match ``master_tID``.

    ``matched`` is a boolean mask of shape ``master_tID.shape`` — True at
    positions where a match was found. ``src_pos`` is the array of source
    positions for the matched master entries (length ``matched.sum()``).
    Both ``src_idx`` and ``master_tID`` must be sorted ascending.
    """
    pos = np.searchsorted(src_idx, master_tID)
    in_range = pos < len(src_idx)
    matched = np.zeros(master_tID.shape, dtype=bool)
    matched[in_range] = src_idx[pos[in_range]] == master_tID[in_range]
    return matched, pos[matched]


def load_raw_h5(
    run_no: int,
    config: int,
    raw_dir=None,
    train_length: int = None,
    n_vls_pixels: int = 1280,
    max_files: int = None,
    load_vls: bool = True,
    vls_offset: int = None,
    train_step: int = 1,
) -> ExperimentData:
    """
    Load experiment data directly from the raw FLASH H5 files for one run.

    Useful for inspecting GMD / VLS data during the experiment when no
    combined H5 has been built yet. The SDU (``z``, ``z_std``) and TDC
    (``tofs_e``, ``tofs_i``, ``liq_tofs_e``) streams are not read here:
    ``z`` is NaN-filled and the TOF arrays are left as ``None``.

    The VLS train-ID axis is taken as the master for config 2 (so every
    output train has a VLS spectrum); config 1 uses the GMD axis and
    attaches the VLS (when the files have it) by train ID. GMD, MPE and
    shutter are aligned to the master by train ID; trains missing from a
    given source are NaN.

    Read-only: the raw H5 directory is treated as a read-only input.
    Computed VLS moments cannot be persisted back to it — call
    :func:`save_vls_moments` only against a writable combined H5.

    Parameters
    ----------
    run_no : int
        FLASH run number used to glob raw H5 files.
    config : int
        1 (gas arm; VLS attached when present) or 2 (liquid arm; VLS is
        the master axis).
    raw_dir : str or pathlib.Path, optional
        Override the raw-H5 directory. Defaults to ``config.RAW_H5_DIR``.
    train_length : int, optional
        Bunch slots per train. Defaults to ``config.TRAIN_LENGTH`` (101)
        for config 1 and 100 for config 2. Per-train arrays narrower than
        this are NaN-padded; wider ones are truncated.
    n_vls_pixels : int
        Width of the VLS pixel axis.
    max_files : int, optional
        Read only the first ``max_files`` raw H5 files (smallest run
        numbers first). Default ``None`` reads all of them — be aware
        that each file can be several GB once the VLS stack is cast to
        float32.
    load_vls : bool
        Config 1 only: attach the VLS when the files have it (default
        True). Config 2 always loads it (it is the master axis).
    vls_offset : int, optional
        Gotthard line stored as bunch 0, as in ``write_h5.py
        --vls-offset`` (line ``b + vls_offset`` -> bunch ``b``). Defaults
        to ``config.VLS_OFFSET`` (10) for config 1 and 0 for config 2.
    train_step : int
        Keep every ``train_step``-th train (default 1 = all). The VLS is
        read only for the kept trains, in blocks, so memory scales with
        the trains kept (~0.5 MB per train as float32).

    Returns
    -------
    ExperimentData
        Container with ``tID``, ``gmd``, ``mpe``, ``shutter`` populated.
        ``vls`` is populated for config 2, and for config 1 when present;
        ``z`` is NaN-filled; TOF fields are ``None``.
        ``between_tdc_files`` is all-False (no TDC stream).
    """
    # Local import: rename so the int ``config`` argument doesn't shadow
    # the path-management module.
    import config as path_config

    if config not in (1, 2):
        raise ValueError(f"config must be 1 or 2, got {config!r}")
    if train_length is None:
        train_length = path_config.TRAIN_LENGTH if config == 1 else 100
    if vls_offset is None:
        vls_offset = path_config.VLS_OFFSET if config == 1 else 0
    train_step = max(1, int(train_step))

    def _file_order_key(path_str: str):
        """Sort by numeric index in ..._fileNN_... (fallback to name)."""
        name = Path(path_str).name
        if "_file" in name:
            tail = name.split("_file", 1)[1]
            num_str = tail.split("_", 1)[0]
            if num_str.isdigit():
                return (0, int(num_str), name)
        return (1, 0, name)

    raw_dir = Path(raw_dir) if raw_dir is not None else Path(path_config.RAW_H5_DIR)
    pattern = str(raw_dir / f"*run{run_no}*.h5")
    h5_paths = sorted(glob.glob(pattern), key=_file_order_key)
    if not h5_paths:
        raise FileNotFoundError(f"No raw H5 files matching {pattern!r}.")
    if max_files is not None:
        h5_paths = h5_paths[:max_files]
    print(f"Loading {len(h5_paths)} raw H5 file(s) for run {run_no} "
          f"(config {config}).")

    tID_chunks = []
    gmd_chunks = []
    mpe_chunks = []
    shutter_chunks = []
    # VLS: always for cfg 2 (master axis); for cfg 1 when requested and at
    # least one file carries it (NaN for trains / files without a frame).
    if config == 2:
        want_vls = True
    elif load_vls:
        want_vls = False
        for path in h5_paths:
            with h5py.File(path, "r") as f:
                if _VLS_VALUE in f:
                    want_vls = True
                    break
    else:
        want_vls = False
    vls_chunks = []

    for path in h5_paths:
        print(f"  reading {Path(path).name}")
        with h5py.File(path, "r") as f:
            # Master tID axis: VLS for cfg 2 (so every row has a spectrum),
            # GMD for cfg 1; thinned by train_step.
            if config == 2:
                master_tID = f[_VLS_INDEX][...][::train_step]
            else:
                master_tID = f[_GMD_INDEX][...][::train_step]
            n = master_tID.shape[0]
            tID_chunks.append(master_tID.astype(np.float64))

            # GMD: dataset is (n_trains, 8, m_raw); keep channel 0 (intensity).
            gmd_idx = f[_GMD_INDEX][...]
            m_raw = f[_GMD_VALUE].shape[2]
            m = min(m_raw, train_length)
            gmd_slice = f[_GMD_VALUE][:, 0, :m]      # (n_gmd, m)
            gmd_buf = np.full((n, train_length), np.nan, dtype=np.float32)
            matched, src_pos = _align_by_tID(gmd_idx, master_tID)
            if matched.any():
                gmd_buf[matched, :m] = gmd_slice[src_pos].astype(np.float32)
            gmd_chunks.append(gmd_buf)

            # MPE: scalar per train.
            mpe_idx = f[_MPE_INDEX][...]
            mpe_val = f[_MPE_VALUE][...]
            mpe_buf = np.full(n, np.nan, dtype=np.float32)
            matched, src_pos = _align_by_tID(mpe_idx, master_tID)
            if matched.any():
                mpe_buf[matched] = mpe_val[src_pos].astype(np.float32)
            mpe_chunks.append(mpe_buf)

            # Fast shutter: per-pulse samples -> one level per train.
            shutter_buf = np.full(n, np.nan, dtype=np.float32)
            if _SHUTTER_INDEX in f and _SHUTTER_VALUE in f:
                sh_val = f[_SHUTTER_VALUE][...].astype(np.float64)
                sh_val = np.nanmean(sh_val.reshape(sh_val.shape[0], -1), axis=1)
                matched, src_pos = _align_by_tID(f[_SHUTTER_INDEX][...], master_tID)
                shutter_buf[matched] = sh_val[src_pos]
            shutter_chunks.append(shutter_buf)

            # VLS: dataset is (n_trains, n_lines, n_px_raw); lined up with
            # the bunch axis via vls_offset and aligned to master by tID.
            # Only the matched rows are read, in blocks, to bound memory.
            if want_vls:
                vls_buf = np.full((n, train_length, n_vls_pixels),
                                  np.nan, dtype=np.float32)
                if _VLS_VALUE in f:
                    matched, src_pos = _align_by_tID(f[_VLS_INDEX][...], master_tID)
                    dst_rows = np.flatnonzero(matched)
                    dset = f[_VLS_VALUE]
                    for i in range(0, len(src_pos), 256):
                        rows = src_pos[i:i + 256]   # increasing -> valid h5py selection
                        vls_buf[dst_rows[i:i + 256]] = _crop_vls_block(
                            dset[rows], train_length, n_vls_pixels, vls_offset,
                        )
                vls_chunks.append(vls_buf)

    tID = np.concatenate(tID_chunks)
    gmd = np.concatenate(gmd_chunks)
    mpe = np.concatenate(mpe_chunks)
    shutter = np.concatenate(shutter_chunks)
    vls = np.concatenate(vls_chunks) if vls_chunks else None

    # Consecutive raw files can overlap by a few trains: keep the first
    # occurrence of each train ID.
    _, first = np.unique(tID, return_index=True)
    if len(first) < len(tID):
        keep = np.sort(first)
        print(f"  dropped {len(tID) - len(keep)} duplicate train(s) at file boundaries")
        tID, gmd, mpe, shutter = tID[keep], gmd[keep], mpe[keep], shutter[keep]
        if vls is not None:
            vls = vls[keep]

    n_total = tID.shape[0]
    z = np.full((n_total, train_length), np.nan, dtype=np.float32)
    between_tdc_files = np.zeros(n_total, dtype=bool)

    print(f"Loaded {n_total} trains x {train_length} bunches"
          f"{' (+ VLS, line b+%d -> bunch b)' % vls_offset if vls is not None else ''}.")

    return ExperimentData(
        config=config,
        tID=tID,
        gmd=gmd,
        mpe=mpe,
        z=z,
        between_tdc_files=between_tdc_files,
        vls=vls,
        shutter=shutter,
    )


def _crop_vls_block(raw_vls, train_length, n_vls_pixels, vls_offset=0):
    """
    Line a block of Gotthard frames ``(n, n_lines, n_px)`` up with the
    bunch axis: line ``b + vls_offset`` -> bunch ``b``, cropped / NaN-padded
    to ``(n, train_length, n_vls_pixels)`` float32. Same convention as
    ``write_h5._crop_vls_lines`` (which works on a single train).
    """
    n = raw_vls.shape[0]
    out = np.full((n, train_length, n_vls_pixels), np.nan, dtype=np.float32)
    src0 = max(vls_offset, 0)
    dst0 = max(-vls_offset, 0)
    m = min(raw_vls.shape[1] - src0, train_length - dst0)
    if m > 0:
        p = min(raw_vls.shape[2], n_vls_pixels)
        out[:, dst0:dst0 + m, :p] = raw_vls[:, src0:src0 + m, :p]
    return out


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
