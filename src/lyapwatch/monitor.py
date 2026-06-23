"""Online stability monitor for retraining loops.

:class:`StabilityMonitor` ingests one dict of metrics per retraining step and,
on every update, returns a :class:`StabilityReport` classifying the loop as
converging, stalled, or diverging — with a projected horizon to instability.

The narrative is *bounded delegation*: a retraining loop is delegated authority
to update a model autonomously, and the monitor is the bound — it certifies the
delegation is still contracting toward a fixed point and pulls the andon cord
before the loop loses stability.

Energy construction
-------------------
Each channel is declared with a direction:

* ``"min"`` — lower is better (loss, embedding drift); equilibrium target
  defaults to ``0.0``.
* ``"max"`` — higher is better (eval score, accuracy); target defaults to
  ``1.0``.

For an observation ``m`` the per-channel error is the one-sided distance to its
target, normalized by the error seen on the first step so channels of different
units contribute comparably. The Lyapunov energy is the weighted sum of squared
normalized errors::

    V_k = sum_i  w_i * ( error_i(m) / error_i(m_0) )^2

``V_k >= 0`` with ``V_k = 0`` exactly at the targets, so it is a valid candidate
Lyapunov function. Stability is then a statement about the series ``V_k``,
analyzed by :mod:`lyapwatch.lyapunov`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping, Optional

from .lyapunov import drift_stats, fit_contraction, steps_to_threshold
from .regime import Regime

__all__ = ["StabilityMonitor", "StabilityReport"]

_EPS = 1e-9


@dataclass(frozen=True)
class StabilityReport:
    """Verdict for a single retraining step."""

    step: int
    energy: float                  #: current Lyapunov energy V_k
    drift: float                   #: mean one-step drift over the window
    contraction_rate: float        #: estimated rho (V_k ~ V_0 * rho^k)
    regime: Regime
    confidence: float              #: R^2 of the contraction fit, in [0, 1]
    horizon: Optional[float]       #: projected steps to instability, if diverging
    message: str                   #: human-readable one-liner

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[step {self.step}] {self.regime.value.upper()}: {self.message}"


class StabilityMonitor:
    """Streaming Lyapunov stability monitor.

    Parameters
    ----------
    channels:
        Mapping of channel name to direction, ``"min"`` or ``"max"``.
    targets:
        Optional per-channel equilibrium values overriding the defaults
        (``0.0`` for ``"min"`` channels, ``1.0`` for ``"max"``).
    weights:
        Optional per-channel weights in the energy sum (default ``1.0`` each).
    window:
        Number of most-recent steps used for the contraction fit.
    min_samples:
        Steps required before a verdict other than ``WARMING_UP`` is issued.
    conv_margin, div_margin:
        Dead-band around ``rho = 1``. ``rho < 1 - conv_margin`` reads as
        converging; ``rho > 1 + div_margin`` as diverging; in between, stalled.
    v_threshold:
        Energy at which the loop is considered to have lost stability, used for
        the divergence horizon. Energy starts near ``sum(weights)`` at step 0,
        so the default ``4.0`` is a few× the initial energy.

    Examples
    --------
    >>> mon = StabilityMonitor({"loss": "min", "eval": "max"})
    >>> r = mon.update({"loss": 1.0, "eval": 0.5})
    >>> r.regime
    <Regime.WARMING_UP: 'warming_up'>
    """

    def __init__(
        self,
        channels: Mapping[str, str],
        *,
        targets: Optional[Mapping[str, float]] = None,
        weights: Optional[Mapping[str, float]] = None,
        window: int = 20,
        min_samples: int = 5,
        conv_margin: float = 0.02,
        div_margin: float = 0.02,
        v_threshold: float = 4.0,
    ) -> None:
        if not channels:
            raise ValueError("at least one channel is required")
        for name, direction in channels.items():
            if direction not in ("min", "max"):
                raise ValueError(
                    f"channel {name!r} direction must be 'min' or 'max', got {direction!r}"
                )
        if window < 2:
            raise ValueError("window must be >= 2")
        if min_samples < 2:
            raise ValueError("min_samples must be >= 2")

        self.channels = dict(channels)
        self.targets = dict(targets or {})
        self.weights = {name: float((weights or {}).get(name, 1.0)) for name in channels}
        self.window = window
        self.min_samples = min_samples
        self.conv_margin = conv_margin
        self.div_margin = div_margin
        self.v_threshold = v_threshold

        self._scale: dict[str, float] = {}          # initial error per channel
        self._energy: deque[float] = deque(maxlen=window)
        self._step = -1

    # -- public API ---------------------------------------------------------

    def update(self, observation: Mapping[str, float]) -> StabilityReport:
        """Feed one step of metrics and get the current stability verdict."""
        missing = set(self.channels) - set(observation)
        if missing:
            raise KeyError(f"observation missing channels: {sorted(missing)}")

        self._step += 1
        energy = self._energy_of(observation)
        self._energy.append(energy)

        window = list(self._energy)
        rho, r2, _ = fit_contraction(window)
        mean_dv, frac_pos = drift_stats(window)
        regime, message = self._classify(len(window), rho, frac_pos)
        horizon = (
            steps_to_threshold(energy, rho, self.v_threshold)
            if regime is Regime.DIVERGING
            else None
        )
        if horizon is not None:
            message += f" (~{horizon:.0f} steps to V>{self.v_threshold:g})"

        return StabilityReport(
            step=self._step,
            energy=energy,
            drift=mean_dv,
            contraction_rate=rho,
            regime=regime,
            confidence=r2,
            horizon=horizon,
            message=message,
        )

    def reset(self) -> None:
        """Forget all history (scales, energy window, step counter)."""
        self._scale.clear()
        self._energy.clear()
        self._step = -1

    # -- internals ----------------------------------------------------------

    def _channel_error(self, name: str, value: float) -> float:
        direction = self.channels[name]
        default = 0.0 if direction == "min" else 1.0
        target = self.targets.get(name, default)
        if direction == "min":
            return max(0.0, value - target)
        return max(0.0, target - value)

    def _energy_of(self, observation: Mapping[str, float]) -> float:
        energy = 0.0
        for name in self.channels:
            err = self._channel_error(name, float(observation[name]))
            if name not in self._scale:
                # Lock the normalizing scale on first sight; a zero initial
                # error (already at target) falls back to unit scale.
                self._scale[name] = err if err > _EPS else 1.0
            norm = err / self._scale[name]
            energy += self.weights[name] * norm * norm
        return energy

    def _classify(self, n: int, rho: float, frac_pos: float) -> tuple[Regime, str]:
        if n < self.min_samples:
            return Regime.WARMING_UP, f"collecting samples ({n}/{self.min_samples})"

        contracting = rho < 1.0 - self.conv_margin and frac_pos < 0.5
        expanding = rho > 1.0 + self.div_margin and frac_pos > 0.5

        if contracting:
            return Regime.CONVERGING, f"energy contracting (rho={rho:.3f})"
        if expanding:
            return Regime.DIVERGING, f"energy expanding (rho={rho:.3f})"
        return Regime.STALLED, f"energy flat (rho={rho:.3f}) — not converging"
