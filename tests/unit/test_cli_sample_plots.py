"""Tests for the sample plot CLI helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from moodswing.cli import sample_plots
from moodswing.viz import TrajectoryComponents


def make_stub_trajectory() -> TrajectoryComponents:
    values = np.array([0.0, 0.5, -0.25, 0.75], dtype=float)
    return TrajectoryComponents(raw=values, rolling=None, dct=None)


def test_make_plot_returns_figure_without_output_dir() -> None:
    trajectory = make_stub_trajectory()
    figure = sample_plots.make_plot(
        "Doc Test",
        trajectory,
        method="demo",
        output_dir=None,
    )
    assert hasattr(figure, "savefig")
    plt.close(figure)


def test_make_plot_writes_file(tmp_path: Path) -> None:
    trajectory = make_stub_trajectory()
    target = sample_plots.make_plot(
        "Doc Test",
        trajectory,
        method="demo",
        output_dir=tmp_path,
    )
    assert isinstance(target, Path)
    assert target.is_file()
