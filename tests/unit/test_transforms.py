"""Tests for transform utilities."""
from __future__ import annotations

import numpy as np

from moodswing.transforms.core import DCTTransform, rolling_mean


def test_dct_transform_basic():
    data = np.linspace(-1, 1, num=10)
    transformer = DCTTransform(
        low_pass_size=3, output_length=10, scale_range=True
        )
    result = transformer.transform(data)
    assert len(result) == 10
    assert max(result) <= 1.0
    assert min(result) >= -1.0


def test_rolling_mean_window_adjustment():
    data = [1, 2, 3, 4, 5]
    result = rolling_mean(data, window=4)
    assert len(result) == 5
    assert abs(result[2] - 2.5) < 1e-6
