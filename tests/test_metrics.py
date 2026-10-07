import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optonet.metrics import (count_metrics, coverage, detection_metrics,
                             interval_width, sequence_metrics)


def test_count_metrics_perfect_fit():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    m = count_metrics(y, y)
    assert m["r2"] == pytest.approx(1.0)
    assert m["mae"] == 0.0
    assert m["spearman"] == pytest.approx(1.0)


def test_count_metrics_worse_than_mean():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    m = count_metrics(y, np.full_like(y, 10.0))
    assert m["r2"] < 0


def test_detection_metrics_separable():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = detection_metrics(y, p)
    assert m["auroc"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)


def test_sequence_metrics_shapes_and_ranges():
    rng = np.random.default_rng(0)
    y = (rng.random((20, 1500)) < 0.01).astype(float)
    p = np.clip(y * 0.95, 0, 1)
    m = sequence_metrics(y, p, window_start=200)
    assert set(["bin_auroc", "bin_auprc", "evoked_r2", "psth_r",
                "total_r2"]).issubset(m)
    assert 0.0 <= m["bin_auroc"] <= 1.0
    assert m["evoked_r2"] > 0.5


def test_coverage_and_width():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo = np.array([0.0, 1.5, 2.5, 3.5])
    hi = np.array([2.0, 2.5, 3.5, 4.5])
    assert coverage(y, lo, hi) == pytest.approx(1.0)
    assert interval_width(lo, hi) == pytest.approx(np.mean(hi - lo))
