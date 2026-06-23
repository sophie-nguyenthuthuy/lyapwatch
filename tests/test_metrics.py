import numpy as np
import pytest

from lyapwatch.metrics import EmbeddingDriftTracker, embedding_drift


def test_cosine_drift_identical_is_zero():
    v = [0.3, 0.7, 1.1]
    assert embedding_drift(v, v) == pytest.approx(0.0, abs=1e-9)


def test_cosine_drift_orthogonal_is_one():
    assert embedding_drift([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)


def test_cosine_drift_opposite_is_two():
    assert embedding_drift([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0)


def test_zero_vector_returns_max_drift():
    assert embedding_drift([0.0, 0.0], [1.0, 1.0]) == 1.0


def test_l2_metric():
    assert embedding_drift([0.0, 0.0], [3.0, 4.0], metric="l2") == pytest.approx(5.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        embedding_drift([1.0, 2.0], [1.0, 2.0, 3.0])


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        embedding_drift([1.0], [1.0], metric="manhattan")


def test_tracker_first_update_is_zero_then_drifts():
    t = EmbeddingDriftTracker()
    assert t.update([1.0, 0.0]) == 0.0
    assert t.update([0.0, 1.0]) == pytest.approx(1.0)
    assert t.update([0.0, 1.0]) == pytest.approx(0.0)
