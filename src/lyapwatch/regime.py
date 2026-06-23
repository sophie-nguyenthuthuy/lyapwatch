"""Stability regimes for a retraining loop."""

from __future__ import annotations

from enum import Enum

__all__ = ["Regime"]


class Regime(str, Enum):
    """Coarse stability verdict emitted on every monitor update.

    Ordered by escalating concern. The string values are stable and safe to log
    or compare against (``Regime.DIVERGING == "diverging"``).
    """

    WARMING_UP = "warming_up"   #: too few samples to judge yet
    CONVERGING = "converging"   #: energy contracting toward the fixed point
    STALLED = "stalled"         #: energy flat — neither settling nor blowing up
    DIVERGING = "diverging"     #: energy expanding — loss of stability ahead

    @property
    def is_actionable(self) -> bool:
        """True when the verdict warrants halting or rolling back delegation."""
        return self in (Regime.STALLED, Regime.DIVERGING)
