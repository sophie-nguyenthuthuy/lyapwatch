"""Lyapunov-function estimation from a scalar energy series.

A retraining loop is treated as a discrete dynamical system

    x_{k+1} = f(x_k)

whose state ``x_k`` we never observe directly. Instead we observe a vector of
metrics (loss, eval score, embedding drift, ...) which the :mod:`monitor`
collapses into a single non-negative *energy* ``V_k >= 0`` — a candidate
Lyapunov function with ``V = 0`` at the desired fixed point.

This module reasons purely about the scalar series ``[V_0, V_1, ...]``:

* :func:`fit_contraction` estimates the contraction factor ``rho`` from
  ``V_k ~= V_0 * rho^k`` (the discrete analogue of a Lyapunov exponent).
* :func:`drift_stats` measures the one-step Lyapunov drift ``Delta V_k``.
* :func:`steps_to_threshold` projects how many steps until a *diverging* loop
  crosses an instability threshold — the "bounded delegation" horizon.

Only :mod:`numpy` is required.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fit_contraction", "drift_stats", "steps_to_threshold"]

_EPS = 1e-12


def fit_contraction(v_series) -> tuple[float, float, float]:
    """Estimate the geometric contraction factor of an energy series.

    Fits ``log V_k = a + s * k`` by least squares and returns
    ``rho = exp(s)``. ``rho < 1`` is a contraction (the loop is settling onto
    its fixed point), ``rho > 1`` is expansion (divergence).

    Non-positive samples carry no log information and are dropped, but the
    surviving samples keep their original step index ``k`` so the slope stays
    in true per-step units.

    Parameters
    ----------
    v_series:
        Iterable of energies ``V_k >= 0``, in step order.

    Returns
    -------
    rho:
        Per-step contraction factor ``exp(slope)``.
    r2:
        Coefficient of determination of the log-linear fit, clamped to
        ``[0, 1]``. Used as a confidence weight — a clean geometric trend
        scores near 1, noise scores near 0.
    slope:
        The raw fitted slope ``s`` (``log rho``).
    """
    v = np.asarray(list(v_series), dtype=float)
    k = np.arange(v.size, dtype=float)
    mask = v > _EPS
    if mask.sum() < 2:
        # Not enough signal to claim contraction or expansion.
        return 1.0, 0.0, 0.0

    k = k[mask]
    y = np.log(v[mask])
    slope, intercept = np.polyfit(k, y, 1)

    y_hat = intercept + slope * k
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > _EPS else 1.0

    rho = float(np.exp(slope))
    return rho, float(np.clip(r2, 0.0, 1.0)), float(slope)


def drift_stats(v_series) -> tuple[float, float]:
    """One-step Lyapunov drift statistics.

    Returns the mean drift ``mean(Delta V_k)`` where
    ``Delta V_k = V_{k+1} - V_k``, and the fraction of steps with positive
    drift. A stable loop has negative mean drift and few positive steps;
    ``frac_pos`` guards against a single lucky slope read on a noisy series.
    """
    v = np.asarray(list(v_series), dtype=float)
    dv = np.diff(v)
    if dv.size == 0:
        return 0.0, 0.0
    return float(dv.mean()), float(np.mean(dv > 0.0))


def steps_to_threshold(v_now: float, rho: float, v_threshold: float):
    """Project steps until a diverging energy crosses ``v_threshold``.

    Solves ``v_now * rho^k = v_threshold`` for ``k``. Only meaningful while the
    loop is expanding (``rho > 1``) and still below the threshold; otherwise
    there is no finite crossing and ``None`` is returned.
    """
    if rho <= 1.0 or v_now <= _EPS or v_threshold <= v_now:
        return None
    return float(np.log(v_threshold / v_now) / np.log(rho))
