"""
Container for a loaded combined-H5 dataset, with inspection / processing
methods.

Keep cheap, side-effect-free transformations on this class — they should
return a *new* ``ExperimentData`` so the original survives further calls
(e.g. ``data.crop_vls(380, 500)`` doesn't mutate ``data``). Anything
expensive or that produces a fundamentally different object (figures,
flat shot arrays, etc.) belongs in ``processing.py`` / ``plotting.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np


def _mask_float(arr: Optional[np.ndarray], mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Return a copy of ``arr`` with ``arr[~mask]`` set to NaN.

    Works for 2D ``(n_trains, n_bunches)`` arrays (gmd, z, moments) and
    3D ``(n_trains, n_bunches, ...)`` arrays (vls); the NaN broadcasts
    over the trailing axis. ``arr`` must have a float dtype.
    """
    if arr is None:
        return None
    out = arr.copy()
    out[~mask] = np.nan
    return out


def _mask_hits(arr: Optional[np.ndarray], mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Return a copy of a zero-padded TOF hits array with masked shots zeroed.

    ``arr`` is ``(n_trains, n_bunches, max_hits)``; cells where
    ``mask`` is False have every hit slot set to 0, which the existing
    zero-padding convention treats as "no hit".
    """
    if arr is None:
        return None
    out = arr.copy()
    out[~mask] = 0
    return out


@dataclass
class ExperimentData:
    """
    Container for experiment data loaded from a combined HDF5 file.

    Attributes
    ----------
    config : int
        Experiment configuration (1 or 2).
    tID : np.ndarray
        Train IDs, shape (n,).
    gmd : np.ndarray
        Pulse energy in µJ, shape (n, m).
    mpe : np.ndarray
        Photon energy from upstream detector (eV), shape (n,).
    z : np.ndarray
        Delay stage position, shape (n, m).
    between_tdc_files : np.ndarray
        Boolean flag for bad/incomplete trains, shape (n,).
    tofs_e : np.ndarray, optional
        Electron TOF hits (config 1), shape (n, m, 50). Units: ns.
    tofs_i : np.ndarray, optional
        Ion TOF hits (config 1), shape (n, m, 120). Units: ns.
    liq_tofs_e : np.ndarray, optional
        Electron TOF hits (config 2), shape (n, m, max_hits). Units: ns.
    vls : np.ndarray, optional
        X-ray spectrum (config 2), shape (n, m, n_pixels).
    shutter : np.ndarray, optional
        Per-train fast-shutter signal (multi-dim raw samples are reduced
        to a scalar via nanmean by ``write_h5.py``), shape (n,).
        ``None`` for combined H5 files written before shutter was added
        to the schema.
    vls_pixel_ax : np.ndarray, optional
        Source-pixel indices for the VLS pixel axis. ``None`` means no
        cropping has been applied (use ``np.arange(vls.shape[-1])``).
        After ``crop_vls`` this holds the indices of the kept pixels so
        plot ticks remain in the original pixel coordinates.
    vls_sums, vls_coms, vls_widths : np.ndarray, optional
        Per-shot spectral moments, shape ``(n_trains, n_bunches)``.
        ``coms`` and ``widths`` are in ``vls_pixels`` units (i.e. the
        source-pixel frame after any crop). Populated by
        :meth:`compute_vls_moments` or restored from H5 by ``load_data``.
        Any transform that mutates ``vls`` (``crop_vls``,
        ``subtract_background``, ``auto_subtract_background``) clears
        these to avoid stale values.
    vls_crop_roi : tuple of int, optional
        ``(roi_min, roi_max)`` half-open pixel ROI applied by the most
        recent :meth:`crop_vls` call. Persisted alongside the moments so
        the preprocessing used to compute them is recoverable.
    vls_background_roi : tuple of int, optional
        ``(bunch_start, bunch_end)`` half-open bunch range used by the
        most recent :meth:`auto_subtract_background`. Persisted with
        the moments as preprocessing metadata.
    shot_mask : np.ndarray, optional
        Cumulative boolean mask of shape ``(n_trains, n_bunches)`` —
        ``True`` where the shot is valid, ``False`` where filtered out
        by :meth:`filter_shots`. ``None`` means no filtering has been
        applied (all shots valid). Downstream code that averages over
        shots should use :attr:`valid_shots` / :attr:`n_valid_shots` so
        that the denominator reflects how many shots actually survive.
    """

    config: int
    tID: np.ndarray
    gmd: np.ndarray
    mpe: np.ndarray
    z: np.ndarray
    between_tdc_files: np.ndarray
    tofs_e: Optional[np.ndarray] = field(default=None)
    tofs_i: Optional[np.ndarray] = field(default=None)
    liq_tofs_e: Optional[np.ndarray] = field(default=None)
    vls: Optional[np.ndarray] = field(default=None)
    shutter: Optional[np.ndarray] = field(default=None)
    vls_pixel_ax: Optional[np.ndarray] = field(default=None)
    vls_sums: Optional[np.ndarray] = field(default=None)
    vls_coms: Optional[np.ndarray] = field(default=None)
    vls_widths: Optional[np.ndarray] = field(default=None)
    vls_crop_roi: Optional[tuple] = field(default=None)
    vls_background_roi: Optional[tuple] = field(default=None)
    shot_mask: Optional[np.ndarray] = field(default=None)

    # ------------------------------------------------------------------
    # Derived quantities (cheap; computed on access)
    # ------------------------------------------------------------------

    @property
    def n_trains(self) -> int:
        """Number of trains in the dataset."""
        return self.tID.shape[0]

    @property
    def n_bunches(self) -> int:
        """Number of bunches per train (taken from the GMD axis)."""
        return self.gmd.shape[1]

    @property
    def mpe_broadcast(self) -> np.ndarray:
        """
        Broadcast ``mpe`` from shape (n,) to (n, m) so each bunch in a
        train inherits the train's photon energy.
        """
        return np.broadcast_to(self.mpe[:, np.newaxis], self.gmd.shape).copy()

    @property
    def vls_pixels(self) -> np.ndarray:
        """
        Source-pixel index axis for the VLS spectra.

        Returns the explicit ``vls_pixel_ax`` if a crop has been applied,
        otherwise ``np.arange(n_vls_pixels)``.
        """
        if self.vls is None:
            raise AttributeError("This dataset has no VLS data (config != 2).")
        if self.vls_pixel_ax is not None:
            return self.vls_pixel_ax
        return np.arange(self.vls.shape[-1])

    @property
    def valid_shots(self) -> np.ndarray:
        """
        Boolean ``(n_trains, n_bunches)`` mask: ``True`` where the shot
        is valid (i.e. not filtered out).

        Returns ``shot_mask`` if any filtering has been applied, otherwise
        an all-True broadcast view (no extra allocation). Use this rather
        than ``shot_mask`` directly when normalising sums or averages so
        the no-filter case is handled transparently.
        """
        if self.shot_mask is None:
            return np.broadcast_to(True, (self.n_trains, self.n_bunches))
        return self.shot_mask

    @property
    def n_valid_shots(self) -> int:
        """
        Number of valid shots after any :meth:`filter_shots` calls.

        Equals ``n_trains * n_bunches`` when no filtering has been applied.
        """
        if self.shot_mask is None:
            return self.n_trains * self.n_bunches
        return int(self.shot_mask.sum())

    # ------------------------------------------------------------------
    # Transformations — each returns a NEW ExperimentData
    # ------------------------------------------------------------------

    def crop_vls(self, roi_min: int, roi_max: int) -> "ExperimentData":
        """
        Return a copy of this dataset with the VLS cropped to a pixel ROI.

        Parameters
        ----------
        roi_min : int
            First source-pixel index to keep (inclusive).
        roi_max : int
            Last source-pixel index to keep (exclusive — half-open like
            Python slicing).

        Returns
        -------
        ExperimentData
            New dataset with ``vls`` cropped along the last axis and
            ``vls_pixel_ax`` holding the source-pixel indices of the
            kept pixels.

        Notes
        -----
        Bounds checking and the actual slicing are delegated to
        ``processing.crop_to_roi`` so the same logic backs both the
        free function and this method.
        """
        if self.vls is None:
            raise AttributeError("This dataset has no VLS data (config != 2).")

        # Local import avoids an import cycle (processing.py imports
        # ExperimentData for type hints).
        from processing import crop_to_roi

        cropped, pixel_ax = crop_to_roi(self.vls, roi_min, roi_max)
        return replace(
            self,
            vls=cropped,
            vls_pixel_ax=pixel_ax,
            vls_crop_roi=(int(roi_min), int(roi_max)),
            vls_sums=None,
            vls_coms=None,
            vls_widths=None,
        )

    def subtract_background(self, background: np.ndarray) -> "ExperimentData":
        """
        Return a copy with a background subtracted from every VLS shot.

        The background can be a single 1D spectrum (broadcast over both
        the train and bunch axes) or a 2D per-bunch background of shape
        ``(n_bunches, n_pixels)`` (broadcast over the train axis only).
        The 2D form is useful when the background spectrum depends on
        bunch position within the train.

        Parameters
        ----------
        background : np.ndarray
            Either ``(n_pixels,)`` (one spectrum for every shot) or
            ``(n_bunches, n_pixels)`` (one spectrum per bunch). The
            pixel axis must match the current VLS pixel axis — i.e. the
            *cropped* axis if ``crop_vls`` has already been applied.

        Returns
        -------
        ExperimentData
            New dataset with ``vls`` equal to ``self.vls - background``
            broadcast over the leading axes.
        """
        if self.vls is None:
            raise AttributeError("This dataset has no VLS data (config != 2).")

        background = np.asarray(background)
        n_bunches = self.vls.shape[1]
        n_pixels  = self.vls.shape[2]
        valid_shapes = ((n_pixels,), (n_bunches, n_pixels))
        if background.shape not in valid_shapes:
            raise ValueError(
                f"background shape {background.shape} must be "
                f"({n_pixels},) for a per-pixel background or "
                f"({n_bunches}, {n_pixels}) for a per-bunch background."
            )

        return replace(
            self,
            vls=self.vls - background,
            vls_sums=None,
            vls_coms=None,
            vls_widths=None,
        )

    def auto_subtract_background(
        self, background_roi: tuple[int, int]
    ) -> "ExperimentData":
        """
        Subtract an automatically-computed background from the VLS spectra.

        The background is the mean spectrum over all trains and over the
        bunches in ``background_roi`` — typically a range of bunch indices
        outside the FEL bunch train where the signal is known to be absent.

        Parameters
        ----------
        background_roi : tuple of int
            ``(bunch_start, bunch_end)`` half-open bunch index range
            (``bunch_end`` exclusive, like Python slicing). The mean
            spectrum over ``vls[:, bunch_start:bunch_end, :]`` is used as
            the background.

        Returns
        -------
        ExperimentData
            New dataset with the auto-computed background subtracted.
        """
        if self.vls is None:
            raise AttributeError("This dataset has no VLS data (config != 2).")

        bunch_start, bunch_end = background_roi
        if not 0 <= bunch_start < bunch_end <= self.n_bunches:
            raise ValueError(
                f"Invalid background_roi {background_roi} for "
                f"n_bunches={self.n_bunches}: require "
                f"0 <= bunch_start < bunch_end <= n_bunches."
            )

        background = np.nanmean(
            self.vls[:, bunch_start:bunch_end, :], axis=(0, 1)
        )
        subbed = self.subtract_background(background)
        return replace(
            subbed,
            vls_background_roi=(int(bunch_start), int(bunch_end)),
        )

    def auto_subtract_background_trainwise(
        self, background_roi: tuple[int, int]
    ) -> "ExperimentData":
        """
        Per-train counterpart of :meth:`auto_subtract_background`.

        For each train, computes the mean spectrum over the bunches in
        ``background_roi`` and subtracts it from every bunch of that
        train. The background is allowed to drift train-to-train
        (unlike :meth:`auto_subtract_background`, which uses a single
        background averaged across all trains). Useful when the
        baseline / dark level changes shot-to-shot but stays constant
        within a train across non-signal bunches.

        Parameters
        ----------
        background_roi : tuple of int
            ``(bunch_start, bunch_end)`` half-open bunch index range
            used for the per-train mean.

        Returns
        -------
        ExperimentData
            New dataset with the per-train background subtracted.
        """
        if self.vls is None:
            raise AttributeError("This dataset has no VLS data (config != 2).")

        bunch_start, bunch_end = background_roi
        if not 0 <= bunch_start < bunch_end <= self.n_bunches:
            raise ValueError(
                f"Invalid background_roi {background_roi} for "
                f"n_bunches={self.n_bunches}: require "
                f"0 <= bunch_start < bunch_end <= n_bunches."
            )

        per_train_bg = np.nanmean(
            self.vls[:, bunch_start:bunch_end, :], axis=1,
        )
        return replace(
            self,
            vls=self.vls - per_train_bg[:, None, :],
            vls_background_roi=(int(bunch_start), int(bunch_end)),
            vls_sums=None,
            vls_coms=None,
            vls_widths=None,
        )

    def roll_vls_bunches(self, shift: int) -> "ExperimentData":
        """
        Cyclically shift the VLS along its bunch axis by ``shift`` bunches.

        Useful for aligning the VLS bunch axis with the GMD/TOF bunch
        axis when the Gotthard line index is offset from the FEL bunch
        train. Positive ``shift`` moves entries to higher bunch indices;
        negative ``shift`` moves them to lower indices. Bunches that
        roll off one end wrap around to the other.

        Per-shot VLS moments are cleared (they would no longer match the
        rolled spectra). The cumulative ``shot_mask`` is left untouched
        — shifting the spectra does not change which shots have been
        marked invalid.

        Parameters
        ----------
        shift : int
            Number of bunches to roll by. Equivalent to
            ``np.roll(vls, shift, axis=1)``.

        Returns
        -------
        ExperimentData
            New dataset with the VLS bunch axis rolled.
        """
        if self.vls is None:
            raise AttributeError("This dataset has no VLS data (config != 2).")

        shift = int(shift)
        return replace(
            self,
            vls=np.roll(self.vls, shift, axis=1),
            vls_sums=None,
            vls_coms=None,
            vls_widths=None,
        )

    def roll_tofs_trains(self, shift: int) -> "ExperimentData":
        """
        Roll the TDC hit arrays along the train axis and trim the ends.

        Corrects a small integer train-ID offset between the TDC stream
        (``tofs_e`` / ``tofs_i`` / ``liq_tofs_e``) and the
        raw-H5 / SDU streams (``gmd``, ``mpe``, ``z``, ...). Positive
        ``shift`` moves TDC entries to higher train indices (equivalent
        to ``np.roll(tofs, shift, axis=0)``); negative ``shift`` moves
        them to lower indices.

        To remove the wrap-around that ``np.roll`` introduces, every
        field is then trimmed by ``abs(shift)`` trains at *each* end.
        The output has ``n_trains - 2 * abs(shift)`` rows. Trimming both
        ends (rather than only the wrap-around end) keeps the surviving
        shot count independent of the sign of ``shift``, which makes
        run-vs-run correlation sweeps directly comparable.

        Per-shot VLS moments are preserved (the VLS is not rolled, so
        the moments are still valid on the trimmed train range). The
        cumulative ``shot_mask`` is trimmed along with everything else.

        Parameters
        ----------
        shift : int
            Number of trains to roll the TDC arrays by. ``0`` is a
            no-op copy. ``abs(shift)`` must be smaller than
            ``n_trains // 2`` so at least one train survives the trim.

        Returns
        -------
        ExperimentData
            New dataset with the TDC arrays rolled and every field
            trimmed by ``abs(shift)`` trains at each end.

        Raises
        ------
        ValueError
            If ``2 * abs(shift) >= n_trains`` (no trains would survive).
        """
        shift = int(shift)
        n_trim = abs(shift)
        if 2 * n_trim >= self.n_trains:
            raise ValueError(
                f"shift={shift} would trim {2 * n_trim} of {self.n_trains} "
                f"trains; need 2 * abs(shift) < n_trains."
            )

        def _roll3(arr):
            return None if arr is None else np.roll(arr, shift, axis=0)

        rolled_tofs_e     = _roll3(self.tofs_e)
        rolled_tofs_i     = _roll3(self.tofs_i)
        rolled_liq_tofs_e = _roll3(self.liq_tofs_e)

        if n_trim == 0:
            return replace(
                self,
                tofs_e=rolled_tofs_e,
                tofs_i=rolled_tofs_i,
                liq_tofs_e=rolled_liq_tofs_e,
            )

        sl = slice(n_trim, self.n_trains - n_trim)

        def _1d(arr):
            return None if arr is None else arr[sl]

        def _2d(arr):
            return None if arr is None else arr[sl, :]

        def _3d(arr):
            return None if arr is None else arr[sl, :, :]

        return replace(
            self,
            tID=self.tID[sl],
            mpe=self.mpe[sl],
            shutter=_1d(self.shutter),
            between_tdc_files=self.between_tdc_files[sl],
            gmd=self.gmd[sl, :],
            z=self.z[sl, :],
            tofs_e=_3d(rolled_tofs_e),
            tofs_i=_3d(rolled_tofs_i),
            liq_tofs_e=_3d(rolled_liq_tofs_e),
            vls=_3d(self.vls),
            vls_sums=_2d(self.vls_sums),
            vls_coms=_2d(self.vls_coms),
            vls_widths=_2d(self.vls_widths),
            shot_mask=_2d(self.shot_mask),
        )

    def compute_vls_moments(self) -> "ExperimentData":
        """
        Return a copy with per-shot VLS sum, COM, and width populated.

        Computes :func:`processing.spectral_moments` over the current
        ``vls`` array (i.e. after whatever crop / background subtraction
        has already been applied) using ``vls_pixels`` as the pixel
        coordinate axis, so the COM and width come out in source-pixel
        units.

        Returns
        -------
        ExperimentData
            New dataset with ``vls_sums``, ``vls_coms``, ``vls_widths``
            set, shape ``(n_trains, n_bunches)`` each.
        """
        if self.vls is None:
            raise AttributeError("This dataset has no VLS data (config != 2).")

        from processing import spectral_moments

        sums, coms, widths = spectral_moments(
            self.vls, self.vls_pixels.astype(float)
        )
        return replace(self, vls_sums=sums, vls_coms=coms, vls_widths=widths)

    def filter_shots(
        self,
        values: np.ndarray,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
        inside: bool = True,
    ) -> "ExperimentData":
        """
        Filter shots by a per-train or per-shot scalar property.

        Bad shots are masked in place — array shapes are unchanged.
        Per-shot float arrays (``gmd``, ``z``, ``vls``, moments) have
        the offending cells set to NaN; zero-padded integer TOF hit
        arrays (``tofs_e``, ``tofs_i``, ``liq_tofs_e``) have the
        offending shots zeroed (the existing hit convention treats
        zeros as "no hit", so they fall out of all counting and
        histogramming naturally).

        The cumulative boolean mask is stored on
        :attr:`shot_mask` (``True`` = valid) and is updated by
        bitwise-AND on each call so successive filters compose. Use
        :attr:`n_valid_shots` to get the correct denominator when
        averaging downstream.

        Parameters
        ----------
        values : np.ndarray
            Property to threshold. Shape ``(n_trains,)`` (broadcast
            across the bunch axis — useful for ``tID``, ``mpe``) or
            ``(n_trains, n_bunches)`` (per-shot — e.g. ``gmd``,
            ``vls_sums``).
        lo : float, optional
            Lower bound. ``None`` means no lower bound.
        hi : float, optional
            Upper bound. ``None`` means no upper bound.
        inside : bool
            If ``True`` (default), KEEP shots with
            ``lo <= values <= hi``. If ``False``, KEEP shots OUTSIDE
            that band. ``NaN`` values in ``values`` are always dropped.

        Returns
        -------
        ExperimentData
            New dataset with updated ``shot_mask`` and per-shot arrays
            masked.

        Raises
        ------
        ValueError
            If both ``lo`` and ``hi`` are ``None``, or if ``values``
            has the wrong shape.
        """
        if lo is None and hi is None:
            raise ValueError("filter_shots needs at least one of lo, hi.")

        values = np.asarray(values)
        if values.shape == (self.n_trains,):
            values_2d = np.broadcast_to(
                values[:, None], (self.n_trains, self.n_bunches)
            )
        elif values.shape == (self.n_trains, self.n_bunches):
            values_2d = values
        else:
            raise ValueError(
                f"values shape {values.shape} must be (n_trains,) "
                f"= ({self.n_trains},) or (n_trains, n_bunches) "
                f"= ({self.n_trains}, {self.n_bunches})."
            )

        in_band = np.ones(values_2d.shape, dtype=bool)
        if lo is not None:
            in_band &= values_2d >= lo
        if hi is not None:
            in_band &= values_2d <= hi
        keep = in_band if inside else ~in_band
        # NaN comparisons return False; ~False = True would let NaNs
        # survive an `inside=False` filter. Drop them explicitly.
        if values_2d.dtype.kind == "f":
            keep &= ~np.isnan(values_2d)

        new_mask = keep if self.shot_mask is None else self.shot_mask & keep

        return replace(
            self,
            shot_mask=new_mask,
            gmd=_mask_float(self.gmd, new_mask),
            z=_mask_float(self.z, new_mask),
            tofs_e=_mask_hits(self.tofs_e, new_mask),
            tofs_i=_mask_hits(self.tofs_i, new_mask),
            liq_tofs_e=_mask_hits(self.liq_tofs_e, new_mask),
            vls=_mask_float(self.vls, new_mask),
            vls_sums=_mask_float(self.vls_sums, new_mask),
            vls_coms=_mask_float(self.vls_coms, new_mask),
            vls_widths=_mask_float(self.vls_widths, new_mask),
        )

    def filter_trains(
        self,
        values: np.ndarray,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
        inside: bool = True,
    ) -> "ExperimentData":
        """
        Drop whole trains whose per-train ``values`` lie outside the band.

        Unlike :meth:`filter_shots` (which masks individual shot cells in
        place), this shrinks every field along the train axis so
        ``n_trains`` decreases. Use it when the criterion is genuinely
        per-train (e.g. ``tID`` window, photon energy ``mpe`` band) and
        the offending trains are not worth keeping at all.

        Parameters
        ----------
        values : np.ndarray
            Per-train property, shape ``(n_trains,)``.
        lo : float, optional
            Lower bound. ``None`` means no lower bound.
        hi : float, optional
            Upper bound. ``None`` means no upper bound.
        inside : bool
            If ``True`` (default), KEEP trains with
            ``lo <= values <= hi``. If ``False``, KEEP trains OUTSIDE
            that band. ``NaN`` values in ``values`` are always dropped.

        Returns
        -------
        ExperimentData
            New dataset with the rejected trains removed.

        Raises
        ------
        ValueError
            If both ``lo`` and ``hi`` are ``None``, or if ``values``
            does not have shape ``(n_trains,)``.
        """
        if lo is None and hi is None:
            raise ValueError("filter_trains needs at least one of lo, hi.")

        values = np.asarray(values)
        if values.shape != (self.n_trains,):
            raise ValueError(
                f"values shape {values.shape} must be (n_trains,) "
                f"= ({self.n_trains},)."
            )

        in_band = np.ones(self.n_trains, dtype=bool)
        if lo is not None:
            in_band &= values >= lo
        if hi is not None:
            in_band &= values <= hi
        keep = in_band if inside else ~in_band
        if values.dtype.kind == "f":
            keep &= ~np.isnan(values)

        def _1d(arr):
            return None if arr is None else arr[keep]

        def _2d(arr):
            return None if arr is None else arr[keep, :]

        def _3d(arr):
            return None if arr is None else arr[keep, :, :]

        return replace(
            self,
            tID=self.tID[keep],
            mpe=self.mpe[keep],
            shutter=_1d(self.shutter),
            between_tdc_files=self.between_tdc_files[keep],
            gmd=self.gmd[keep, :],
            z=self.z[keep, :],
            tofs_e=_3d(self.tofs_e),
            tofs_i=_3d(self.tofs_i),
            liq_tofs_e=_3d(self.liq_tofs_e),
            vls=_3d(self.vls),
            vls_sums=_2d(self.vls_sums),
            vls_coms=_2d(self.vls_coms),
            vls_widths=_2d(self.vls_widths),
            shot_mask=_2d(self.shot_mask),
        )

    def trim_bunches(self, bunch_start: int, bunch_end: int) -> "ExperimentData":
        """
        Return a copy trimmed to bunch indices ``[bunch_start, bunch_end)``.

        Slices the bunch axis of every per-shot array and the
        ``shot_mask``. Train-level fields (``tID``, ``mpe``,
        ``between_tdc_files``) are untouched.

        Parameters
        ----------
        bunch_start : int
            First bunch index to keep (inclusive).
        bunch_end : int
            First bunch index NOT kept (exclusive — half-open like
            Python slicing).

        Returns
        -------
        ExperimentData
            New dataset with ``n_bunches = bunch_end - bunch_start``.
        """
        if not 0 <= bunch_start < bunch_end <= self.n_bunches:
            raise ValueError(
                f"Invalid bunch range [{bunch_start}, {bunch_end}) for "
                f"n_bunches={self.n_bunches}; require "
                f"0 <= bunch_start < bunch_end <= n_bunches."
            )

        sl = slice(bunch_start, bunch_end)

        def _2d(arr):
            return None if arr is None else arr[:, sl]

        def _3d(arr):
            return None if arr is None else arr[:, sl, :]

        return replace(
            self,
            gmd=self.gmd[:, sl],
            z=self.z[:, sl],
            tofs_e=_3d(self.tofs_e),
            tofs_i=_3d(self.tofs_i),
            liq_tofs_e=_3d(self.liq_tofs_e),
            vls=_3d(self.vls),
            vls_sums=_2d(self.vls_sums),
            vls_coms=_2d(self.vls_coms),
            vls_widths=_2d(self.vls_widths),
            shot_mask=_2d(self.shot_mask),
        )
