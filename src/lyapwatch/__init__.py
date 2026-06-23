"""lyapwatch — Lyapunov stability monitoring for retraining loops.

Model a retraining pipeline as a discrete dynamical system, estimate a
Lyapunov function from its metric stream, and warn on divergence *before* the
loop loses stability. Bounded delegation for autonomous retraining.

>>> from lyapwatch import StabilityMonitor
>>> mon = StabilityMonitor({"loss": "min", "eval": "max", "drift": "min"})
>>> report = mon.update({"loss": 0.9, "eval": 0.60, "drift": 0.05})
>>> report.regime
<Regime.WARMING_UP: 'warming_up'>
"""

from .lyapunov import drift_stats, fit_contraction, steps_to_threshold
from .metrics import EmbeddingDriftTracker, embedding_drift
from .monitor import StabilityMonitor, StabilityReport
from .regime import Regime

__version__ = "0.1.0"

__all__ = [
    "StabilityMonitor",
    "StabilityReport",
    "Regime",
    "embedding_drift",
    "EmbeddingDriftTracker",
    "fit_contraction",
    "drift_stats",
    "steps_to_threshold",
    "__version__",
]
