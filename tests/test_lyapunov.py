import numpy as np

from lyapwatch.lyapunov import drift_stats, fit_contraction, steps_to_threshold


def test_contraction_of_geometric_decay():
    v = [2.0 * 0.8**k for k in range(30)]
    rho, r2, slope = fit_contraction(v)
    assert abs(rho - 0.8) < 1e-6
    assert r2 > 0.999
    assert slope < 0


def test_expansion_of_geometric_growth():
    v = [0.1 * 1.25**k for k in range(20)]
    rho, r2, _ = fit_contraction(v)
    assert rho > 1.2
    assert r2 > 0.999


def test_flat_series_is_unit_rho():
    rho, _, slope = fit_contraction([1.0] * 10)
    assert abs(rho - 1.0) < 1e-9
    assert abs(slope) < 1e-9


def test_too_few_positive_samples_returns_neutral():
    assert fit_contraction([0.0, 0.0, 0.0]) == (1.0, 0.0, 0.0)
    assert fit_contraction([1.0]) == (1.0, 0.0, 0.0)


def test_nonpositive_samples_do_not_shift_slope():
    # Zeros are dropped but surviving indices keep true step spacing.
    clean = [0.8**k for k in range(10)]
    holed = list(clean)
    holed[3] = 0.0
    rho_clean, _, _ = fit_contraction(clean)
    rho_holed, _, _ = fit_contraction(holed)
    assert abs(rho_clean - rho_holed) < 1e-6


def test_drift_stats_signs():
    mean_dv, frac_pos = drift_stats([4.0, 3.0, 2.0, 1.0])
    assert mean_dv < 0
    assert frac_pos == 0.0
    mean_dv, frac_pos = drift_stats([1.0, 2.0, 3.0])
    assert mean_dv > 0
    assert frac_pos == 1.0


def test_drift_stats_single_point():
    assert drift_stats([1.0]) == (0.0, 0.0)


def test_steps_to_threshold_matches_closed_form():
    # v_now * rho^k = thresh -> k = log(thresh/v_now)/log(rho)
    k = steps_to_threshold(1.0, 2.0, 8.0)
    assert abs(k - 3.0) < 1e-9


def test_steps_to_threshold_none_when_not_diverging():
    assert steps_to_threshold(1.0, 0.9, 8.0) is None   # contracting
    assert steps_to_threshold(8.0, 2.0, 4.0) is None   # already past threshold
    assert steps_to_threshold(0.0, 2.0, 4.0) is None   # zero energy
