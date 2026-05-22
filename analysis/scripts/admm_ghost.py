"""
ADMM solver for regularised spectral-domain ghost imaging.

Per GMD bin we have two pre-computed covariance matrices

    M = Cov(A)      shape (n_pixels, n_pixels)   symmetric PSD
    B = Cov(A, D)   shape (n_pixels, n_tof)

and we want the spectral kernel ``X`` minimising

    J(X) = ½||M X − B||²_F
         + ½ λ_p ||D_p X||²_F      (smoothness along VLS pixels)
         + ½ λ_t ||X D_tᵀ||²_F     (smoothness along TOF bins)
         +     λ_s ||X||_1         (element-wise sparsity)

ADMM splits the non-smooth ℓ₁ term via an auxiliary ``Z = X``:

    primal:   ½||M X − B||²_F + ½ λ_p ||D_p X||²_F + ½ λ_t ||X D_tᵀ||²_F
              + λ_s ||Z||_1
    s.t.:     X − Z = 0

The augmented Lagrangian gives

    X ←  argmin_X  L(X, Z, U)        (x-step)
    Z ←  soft_threshold(X + U, λ_s/ρ) (z-step, prox of ℓ₁)
    U ←  U + X − Z                    (u-step, scaled dual)

The x-step is a Sylvester equation

    (MᵀM + λ_p D_pᵀD_p + ρI) X + X (λ_t D_tᵀD_t) = MᵀB + ρ(Z − U).

Both side matrices are symmetric, so we diagonalise them once via
``scipy.linalg.eigh`` and reduce the inner loop to two ``n_p × n_t``
matrix products plus an element-wise divide. Cost per iteration:
``O(n_p n_t (n_p + n_t))`` — for the test problem (120, 125) that is
~4M ops, sub-millisecond.

Usage
-----

    from compute_aggregates import load_aggregates
    from admm_ghost import solve_admm

    agg = load_aggregates("...aggregates.h5")
    b   = 3   # GMD bin
    M = agg.AtA[b] - np.outer(agg.A[b], agg.A[b])
    B = agg.AtD[b] - np.outer(agg.A[b], agg.D[b])

    res = solve_admm(M, B,
                     lambda_smooth_pixel=1e4,
                     lambda_smooth_tof=1e4,
                     lambda_sparse=1e2,
                     rho=1e3)
    X_star = res.X
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.linalg


__all__ = ["ADMMResult", "solve_admm"]


@dataclass
class ADMMResult:
    """
    Output of :func:`solve_admm`.

    Attributes
    ----------
    X : np.ndarray
        Reconstructed kernel, shape ``(n_pixels, n_tof)``.
    n_iter : int
        Number of iterations actually run.
    converged : bool
        Whether both residuals dropped below tolerance.
    history : dict
        Per-iteration arrays:
            ``primal``    — primal residual ‖X−Z‖_F
            ``dual``      — dual residual ρ‖Z−Z_prev‖_F
            ``objective`` — value of J(X) above
    """
    X: np.ndarray
    n_iter: int
    converged: bool
    history: dict = field(default_factory=dict)


def _soft_threshold(X: np.ndarray, kappa: float) -> np.ndarray:
    """Element-wise prox of ``kappa · ||·||_1`` — the L1 shrinkage."""
    if kappa <= 0:
        return X
    return np.sign(X) * np.maximum(np.abs(X) - kappa, 0.0)


def _diff_operator(n: int, order: int) -> np.ndarray:
    """
    Finite-difference matrix of shape ``(n - order, n)``.

    ``order=1`` gives the standard first difference; ``order=2`` gives
    the second difference (discrete Laplacian, free boundary).
    """
    if order < 1:
        raise ValueError("diff_order must be >= 1")
    return np.diff(np.eye(n), n=order, axis=0).astype(np.float64)


def solve_admm(
    M: np.ndarray,
    B: np.ndarray,
    *,
    lambda_smooth_pixel: float,
    lambda_smooth_tof: float,
    lambda_sparse: float,
    rho: float = 1.0,
    diff_order: int = 2,
    max_iter: int = 500,
    tol_primal: float = 1e-4,
    tol_dual: float = 1e-4,
    X0: Optional[np.ndarray] = None,
    track_objective: bool = True,
    verbose: bool = False,
) -> ADMMResult:
    """
    Solve the regularised inversion ``M X ≈ B`` via ADMM. See the
    module docstring for the full objective.

    Parameters
    ----------
    M : np.ndarray
        ``(n_pixels, n_pixels)`` operator (typically ``Cov(A)``).
    B : np.ndarray
        ``(n_pixels, n_tof)`` right-hand side (typically ``Cov(A, D)``).
    lambda_smooth_pixel : float
        Weight on ``½ ||D_p X||²_F`` — smoothness along the VLS pixel
        (photon-energy) axis. Set to 0 to disable.
    lambda_smooth_tof : float
        Weight on ``½ ||X D_tᵀ||²_F`` — smoothness along the eTOF axis.
        Set to 0 to disable.
    lambda_sparse : float
        Weight on ``||X||_1``. Set to 0 to disable (the solver still
        runs but the z-step is a no-op).
    rho : float
        ADMM penalty parameter. Larger values force ``X ≈ Z`` faster
        but slow convergence on the data fit. As a rule of thumb,
        pick ``rho`` on the order of a typical eigenvalue of
        ``MᵀM + λ D_pᵀD_p``.
    diff_order : int
        Order of the finite-difference operator (1 or 2). Default 2
        (discrete Laplacian).
    max_iter : int
        Hard cap on iterations.
    tol_primal, tol_dual : float
        Relative tolerances for primal and dual residuals.
    X0 : np.ndarray, optional
        Warm-start kernel. If None, starts from zero.
    track_objective : bool
        If True (default), records ``J(X)`` at each iteration.
    verbose : bool
        If True, prints residuals periodically.

    Returns
    -------
    ADMMResult
        With the reconstructed kernel, iteration count, convergence
        flag, and per-iteration residual / objective history.
    """
    M = np.asarray(M, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n_p = M.shape[0]
    n_t = B.shape[1]
    if M.shape != (n_p, n_p):
        raise ValueError(f"M must be square; got {M.shape}")
    if B.shape != (n_p, n_t):
        raise ValueError(f"B shape {B.shape} != (n_pixels={n_p}, n_tof)")

    # Difference operators.
    D_p = _diff_operator(n_p, diff_order)
    D_t = _diff_operator(n_t, diff_order)
    DtD_p = D_p.T @ D_p
    DtD_t = D_t.T @ D_t

    # Step-1 system:  A X + X C = RHS
    A_mat = M.T @ M + lambda_smooth_pixel * DtD_p + rho * np.eye(n_p)
    C_mat = lambda_smooth_tof * DtD_t

    # Both A and C are symmetric — diagonalise once and reuse.
    eig_A, U_A = scipy.linalg.eigh(A_mat)
    eig_C, U_C = scipy.linalg.eigh(C_mat)
    denom = eig_A[:, None] + eig_C[None, :]
    if (denom <= 0).any():
        raise RuntimeError(
            "Sylvester denominator non-positive — check λ_smooth ≥ 0 "
            "and rho > 0."
        )

    MtB = M.T @ B

    X = np.zeros_like(B) if X0 is None else np.asarray(X0, dtype=np.float64).copy()
    Z = X.copy()
    U = np.zeros_like(B)

    primal_hist = np.empty(max_iter, dtype=np.float64)
    dual_hist   = np.empty(max_iter, dtype=np.float64)
    obj_hist    = np.empty(max_iter, dtype=np.float64) if track_objective else None

    converged = False
    it = 0
    for it in range(max_iter):
        # --- x-step: diagonalised Sylvester ------------------------------
        rhs = MtB + rho * (Z - U)
        F = U_A.T @ rhs @ U_C
        Y = F / denom
        X = U_A @ Y @ U_C.T

        # --- z-step: soft threshold --------------------------------------
        Z_old = Z
        Z = _soft_threshold(X + U, lambda_sparse / rho)

        # --- u-step ------------------------------------------------------
        U = U + X - Z

        # --- residuals + objective ---------------------------------------
        primal_r = float(np.linalg.norm(X - Z))
        dual_r   = float(rho * np.linalg.norm(Z - Z_old))
        primal_hist[it] = primal_r
        dual_hist[it]   = dual_r

        if track_objective:
            residual = M @ X - B
            sm_p = D_p @ X
            sm_t = X @ D_t.T
            obj = (
                0.5 * float(np.einsum("ij,ij->", residual, residual))
                + 0.5 * lambda_smooth_pixel * float(np.einsum("ij,ij->", sm_p, sm_p))
                + 0.5 * lambda_smooth_tof  * float(np.einsum("ij,ij->", sm_t, sm_t))
                +       lambda_sparse      * float(np.abs(X).sum())
            )
            obj_hist[it] = obj

        if verbose and (it < 10 or it % 50 == 0 or it == max_iter - 1):
            obj_str = f"  obj={obj_hist[it]:.3e}" if track_objective else ""
            print(f"  iter {it:4d}  pri={primal_r:.3e}  dual={dual_r:.3e}{obj_str}")

        # --- convergence test --------------------------------------------
        eps_pri  = tol_primal * max(np.linalg.norm(X), np.linalg.norm(Z), 1e-12)
        eps_dual = tol_dual   * max(rho * np.linalg.norm(U), 1e-12)
        if primal_r < eps_pri and dual_r < eps_dual:
            converged = True
            break

    history = {
        "primal":    primal_hist[: it + 1],
        "dual":      dual_hist[: it + 1],
    }
    if track_objective:
        history["objective"] = obj_hist[: it + 1]

    return ADMMResult(
        X=X, n_iter=it + 1, converged=converged, history=history,
    )
