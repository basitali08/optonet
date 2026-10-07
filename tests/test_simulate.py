import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optonet import simulate as S
from optonet.protocols import (MAX_PULSES, TRAIN_FAMILIES, OOD_FAMILIES,
                               sample_protocol, _chirp_times)


def test_rasterize_pulses_boundaries():
    pulses = np.array([0.0, 50.0])
    widths = np.array([5.0, 10.0])
    light = S.rasterize_pulses([pulses], [widths], np.array([1.0]),
                               n_steps=2000, dt=0.1, t_onset=10.0, l_max=0.6)
    assert light.shape == (1, 2000)
    assert light[0, 99] == 0.0
    assert light[0, 100] == pytest.approx(0.6, abs=1e-5)
    assert light[0, 149] == pytest.approx(0.6, abs=1e-5)
    assert light[0, 150] == 0.0
    assert light[0, 599] == 0.0
    assert light[0, 600] == pytest.approx(0.6, abs=1e-5)
    assert light[0, 699] == pytest.approx(0.6, abs=1e-5)
    assert light[0, 700] == 0.0


def test_rasterize_clips_past_end():
    light = S.rasterize_pulses([np.array([95.0])], [np.array([50.0])],
                               np.array([1.0]), n_steps=100, dt=0.1, t_onset=10.0)
    assert light[0, -1] == pytest.approx(0.6, abs=1e-5)
    assert np.all(light[0, :99] == 0.0)


def test_sample_cells_shapes_and_ranges():
    rng = np.random.default_rng(0)
    cells = S.sample_cells(50, "pc", rng)
    assert cells["C"].shape == (50,)
    assert np.all(cells["C"] > 0) and np.all(cells["gL"] > 0)
    assert np.all(cells["EL"] < 0) and np.all(cells["tau_w"] > 0)
    assert set(np.unique(cells["cell_type"])) == {"pc"}


def test_simulation_is_deterministic():
    rng = np.random.default_rng(1)
    cells = S.sample_cells(8, "pv", rng)
    light = np.zeros((8, S.N_STEPS), dtype=np.float32)
    light[:, 2000:2050] = 0.6
    a = S.simulate_chunk(cells, light, seed=3)
    b = S.simulate_chunk(cells, light, seed=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    c = S.simulate_chunk(cells, light, seed=4)
    assert not (np.array_equal(a[0], c[0]) and np.array_equal(a[1], c[1]))


def test_light_increases_firing():
    rng = np.random.default_rng(2)
    cells = S.sample_cells(48, "pv", rng, drive_scale=1.178)
    pulses = np.arange(0, 800, 25.0)
    widths = np.full(len(pulses), 5.0)
    counts = {}
    for inten in (0.0, 0.4, 1.0):
        light = S.rasterize_pulses([pulses], [widths], np.array([inten]))
        light = np.repeat(light, 48, axis=0)
        ind, times, _ = S.simulate_chunk(cells, light, seed=5)
        ev = np.array([(times[ind[i]:ind[i + 1]] * S.DT >= S.T_ONSET).sum()
                       for i in range(48)])
        counts[inten] = ev.mean()
    assert counts[1.0] > counts[0.4] > counts[0.0]


def test_spike_times_within_trial():
    rng = np.random.default_rng(3)
    cells = S.sample_cells(16, "som", rng)
    light = np.zeros((16, S.N_STEPS), dtype=np.float32)
    light[:, 2000:4000] = 0.5
    ind, times, vm = S.simulate_chunk(cells, light, record_vm=4, seed=6)
    assert ind[0] == 0 and ind[-1] == len(times)
    assert times.min() >= 0 and times.max() < S.N_STEPS
    assert vm.shape == (4, S.N_STEPS // 10)
    assert np.isfinite(vm).all()
    assert vm.max() > -70.0


@pytest.mark.parametrize("family", TRAIN_FAMILIES + OOD_FAMILIES)
def test_protocol_families_are_valid(family):
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = sample_protocol(family, rng)
        assert p.n_pulses <= MAX_PULSES
        if p.n_pulses:
            assert np.all(p.pulses >= 0)
            assert np.all(p.widths > 0)
            assert p.duration <= 1100.0 + 1e-6
            assert np.all(np.diff(p.pulses) >= 0)
        assert 0.05 <= p.intensity <= 2.0
        assert p.family == family


def test_ood_kwargs_shift_distributions():
    rng = np.random.default_rng(0)
    base = [sample_protocol("regular", rng, freq_lo=1.0, freq_hi=40.0).meta["freq"]
            for _ in range(50)]
    hi = [sample_protocol("regular", rng, freq_lo=50.0, freq_hi=120.0).meta["freq"]
          for _ in range(50)]
    assert max(base) <= 40.0 and min(hi) >= 50.0
    ints = [sample_protocol("regular", rng, intensity_lo=1.2, intensity_hi=2.0).intensity
            for _ in range(50)]
    assert min(ints) >= 1.2


def test_chirp_times_monotone():
    t = _chirp_times(2.0, 60.0, 800.0)
    assert len(t) > 10
    assert np.all(np.diff(t) > 0)
    assert t.max() <= 800.0
