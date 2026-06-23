import pytest

from lyapwatch import Regime, StabilityMonitor


def _run(monitor, series):
    report = None
    for obs in series:
        report = monitor.update(obs)
    return report


def test_warming_up_before_min_samples():
    mon = StabilityMonitor({"loss": "min"}, min_samples=5)
    r = mon.update({"loss": 1.0})
    assert r.regime is Regime.WARMING_UP
    assert r.step == 0


def test_converging_loss_detected():
    mon = StabilityMonitor({"loss": "min"}, min_samples=5)
    series = [{"loss": 1.0 * 0.85**k} for k in range(25)]
    r = _run(mon, series)
    assert r.regime is Regime.CONVERGING
    assert r.contraction_rate < 1.0
    assert r.drift < 0
    assert r.horizon is None
    assert r.confidence > 0.9


def test_diverging_loss_detected_with_horizon():
    # Slow growth + a high threshold leaves a window where the loop is already
    # flagged diverging but has not yet crossed into instability -> a horizon.
    mon = StabilityMonitor({"loss": "min"}, min_samples=5, v_threshold=20.0)
    series = [{"loss": 0.2 * 1.1**k} for k in range(20)]
    reports = []
    for obs in series:
        reports.append(mon.update(obs))

    assert reports[-1].regime is Regime.DIVERGING
    assert reports[-1].contraction_rate > 1.0
    assert reports[-1].regime.is_actionable

    # Some step fired the alarm with a positive, finite steps-to-instability.
    horizons = [r.horizon for r in reports if r.regime is Regime.DIVERGING and r.horizon]
    assert horizons and all(h > 0 for h in horizons)


def test_stalled_when_flat():
    mon = StabilityMonitor({"loss": "min"}, min_samples=5)
    series = [{"loss": 0.5} for _ in range(15)]
    r = _run(mon, series)
    assert r.regime is Regime.STALLED
    assert r.regime.is_actionable


def test_eval_channel_uses_max_direction():
    # eval climbing toward 1.0 means shrinking error -> converging
    mon = StabilityMonitor({"eval": "max"}, min_samples=5)
    series = [{"eval": 1.0 - 0.5 * 0.8**k} for k in range(25)]
    r = _run(mon, series)
    assert r.regime is Regime.CONVERGING


def test_multichannel_energy_combines():
    mon = StabilityMonitor(
        {"loss": "min", "eval": "max", "drift": "min"}, min_samples=5
    )
    series = [
        {"loss": 0.9 * 0.85**k, "eval": 1.0 - 0.4 * 0.85**k, "drift": 0.1 * 0.85**k}
        for k in range(25)
    ]
    r = _run(mon, series)
    assert r.regime is Regime.CONVERGING
    assert r.energy >= 0.0


def test_custom_targets_shift_equilibrium():
    # Target loss floor of 0.3: a loss settling at 0.3 has ~zero error.
    mon = StabilityMonitor({"loss": "min"}, targets={"loss": 0.3}, min_samples=5)
    series = [{"loss": 0.3 + 0.7 * 0.8**k} for k in range(25)]
    r = _run(mon, series)
    assert r.regime is Regime.CONVERGING


def test_missing_channel_raises():
    mon = StabilityMonitor({"loss": "min", "eval": "max"})
    with pytest.raises(KeyError):
        mon.update({"loss": 1.0})


def test_reset_clears_history():
    mon = StabilityMonitor({"loss": "min"}, min_samples=5)
    _run(mon, [{"loss": 0.5} for _ in range(10)])
    mon.reset()
    r = mon.update({"loss": 1.0})
    assert r.step == 0
    assert r.regime is Regime.WARMING_UP


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        StabilityMonitor({})
    with pytest.raises(ValueError):
        StabilityMonitor({"x": "lower"})
    with pytest.raises(ValueError):
        StabilityMonitor({"x": "min"}, window=1)


def test_report_str_is_readable():
    mon = StabilityMonitor({"loss": "min"})
    s = str(mon.update({"loss": 1.0}))
    assert "step 0" in s
    assert "WARMING_UP" in s
