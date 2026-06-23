"""Helpers for turning raw training signals into monitor channels.

The :class:`~lyapwatch.monitor.StabilityMonitor` consumes a flat dict of
scalar observations per step. These helpers produce those scalars from the two
signals that are awkward to compute inline: embedding drift between successive
model checkpoints, and a robust running scale for noisy channels.
"""

from __future__ import annotations

import numpy as np

__all__ = ["embedding_drift", "EmbeddingDriftTracker"]

_EPS = 1e-12


def embedding_drift(prev, curr, metric: str = "cosine") -> float:
    """Distance between two embedding vectors (e.g. successive checkpoints).

    A standard signal for representational instability in a retraining loop:
    feed the drift of a fixed probe set's mean embedding as a monitor channel.

    Parameters
    ----------
    prev, curr:
        1-D array-likes of equal length.
    metric:
        ``"cosine"`` (default, range ``[0, 2]``) or ``"l2"``.
    """
    a = np.asarray(prev, dtype=float).ravel()
    b = np.asarray(curr, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"embedding shapes differ: {a.shape} vs {b.shape}")

    if metric == "cosine":
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < _EPS or nb < _EPS:
            return 1.0
        return float(1.0 - np.dot(a, b) / (na * nb))
    if metric == "l2":
        return float(np.linalg.norm(a - b))
    raise ValueError(f"unknown metric {metric!r}; use 'cosine' or 'l2'")


class EmbeddingDriftTracker:
    """Stateful convenience wrapper: feed checkpoints, get drift since the last.

    Returns ``0.0`` on the first checkpoint (no predecessor to compare).

    >>> t = EmbeddingDriftTracker()
    >>> t.update([1.0, 0.0])
    0.0
    >>> round(t.update([0.0, 1.0]), 3)
    1.0
    """

    def __init__(self, metric: str = "cosine") -> None:
        self.metric = metric
        self._prev = None

    def update(self, embedding) -> float:
        emb = np.asarray(embedding, dtype=float).ravel()
        if self._prev is None:
            self._prev = emb
            return 0.0
        d = embedding_drift(self._prev, emb, self.metric)
        self._prev = emb
        return d
