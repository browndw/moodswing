"""Tests for visualization helpers."""
from __future__ import annotations

import numpy as np

from moodswing.transforms.core import rolling_mean
from moodswing.viz.trajectory import prepare_trajectory


class _DummyTransform:
    def __init__(self):
        self.scale_range = False
        self.scale_values = False
    
    def transform(self, values):
        data = np.asarray(values, dtype=float)
        return np.array([data[0], data[-1]], dtype=float)


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    std = arr.std()
    if std == 0:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def test_prepare_trajectory_defaults_to_range_normalization() -> None:
    values = [0.0, 5.0, 10.0]
    trajectory = prepare_trajectory(values)
    np.testing.assert_allclose(trajectory.raw, np.array([-1.0, 0.0, 1.0]))
    assert trajectory.rolling is None
    assert trajectory.dct is None


def test_prepare_trajectory_can_skip_normalization() -> None:
    values = np.array([-1.0, 2.0, 5.0], dtype=float)
    trajectory = prepare_trajectory(values, normalize=None)
    np.testing.assert_allclose(trajectory.raw, values)


def test_prepare_trajectory_normalizes_all_components() -> None:
    values = np.array([-2.0, 0.0, 3.0, 6.0], dtype=float)
    trajectory = prepare_trajectory(
        values,
        rolling_window=2,
        dct_transform=_DummyTransform(),
        normalize="zscore",
    )
    np.testing.assert_allclose(trajectory.raw, _zscore(values))

    rolling = np.asarray(rolling_mean(values, window=2))
    np.testing.assert_allclose(trajectory.rolling, _zscore(rolling))

    expected_dct = np.array([values[0], values[-1]], dtype=float)
    np.testing.assert_allclose(trajectory.dct, _zscore(expected_dct))
