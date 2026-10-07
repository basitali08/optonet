import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optonet.config import load_config
from optonet.data import SPLIT_NAMES, load, sample_batch


@pytest.fixture(scope="module")
def bundle():
    return load()


def test_shapes(bundle):
    n = len(bundle.trials)
    assert bundle.light.shape == (n, 1500)
    assert bundle.raster.shape == (n, 1500)
    assert bundle.light.dtype == np.uint8
    assert bundle.indptr.shape == (n + 1,)
    assert bundle.indptr[0] == 0
    assert bundle.indptr[-1] == len(bundle.times)


def test_splits_partition_trials(bundle):
    split = bundle.trials["split"].to_numpy()
    assert set(np.unique(split)) == set(SPLIT_NAMES)
    counts = [int(np.sum(split == s)) for s in SPLIT_NAMES]
    assert sum(counts) == len(bundle.trials)
    assert min(counts) > 1000


def test_cells_partition_and_types(bundle):
    cells = bundle.cells
    assert set(np.unique(cells["cell_type"])) == {"pc", "pv", "som"}
    for s in ("train", "val", "test"):
        assert (cells["cell_split"] == s).sum() > 0


def test_raster_matches_csr(bundle):
    idx = np.linspace(0, len(bundle.trials) - 1, 30).astype(int)
    for i in idx:
        s, e = bundle.indptr[i], bundle.indptr[i + 1]
        assert bundle.raster[i].sum() == e - s
        assert bundle.trials["n_spikes_total"].iloc[i] == e - s


def test_light_contains_stimulus(bundle):
    idx = bundle.trial_index("train")[:50]
    assert (bundle.light[idx].max(axis=1) > 0).all()
    pre = bundle.light[idx][:, :200]
    assert pre.max() == 0


def test_cell_normalization(bundle):
    X = bundle.norm_cell_matrix()
    train_cells = bundle.cells["cell_split"].to_numpy() == "train"
    assert np.allclose(X[train_cells].mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(X[train_cells].std(axis=0), 1.0, atol=1e-4)


def test_sample_batch(bundle):
    idx = bundle.trial_index("val")[:16]
    light, raster, feats = sample_batch(bundle, idx)
    assert light.shape == (16, 1500)
    assert raster.shape == (16, 1500)
    assert feats.shape == (16, 14)
    assert light.max() <= 2.0 + 1e-6
    assert np.allclose(feats[:, :3].sum(axis=1), 1.0)


def test_config_roundtrip():
    cfg = load_config()
    assert cfg["data"]["trials_per_split"]["train"] > 0
    assert set(cfg["training"]["sequence_models"]) == {"glm", "cnn", "lstm",
                                                       "transformer"}
