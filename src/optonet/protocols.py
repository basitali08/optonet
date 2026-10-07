"""Optogenetic stimulation protocol library and dataset splits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

TRAIN_FAMILIES = ("regular", "step", "rand", "doublet", "burst")
OOD_FAMILIES = ("chirp", "sweep", "sparse", "chirp_hi")

MAX_PULSES = 220
T_WINDOW = 1100.0   # max light-evoked window (ms after onset)


@dataclass
class Protocol:
    family: str
    pulses: np.ndarray      # onset times relative to light onset (ms)
    widths: np.ndarray      # pulse widths (ms)
    intensity: float        # 0..~2, model units
    meta: Dict[str, float] = field(default_factory=dict)

    @property
    def n_pulses(self) -> int:
        return int(len(self.pulses))

    @property
    def duration(self) -> float:
        if self.n_pulses == 0:
            return 0.0
        return float(self.pulses[-1] + self.widths[-1])


def _clip_pulses(p: np.ndarray, w: np.ndarray, t_end: float = T_WINDOW
                 ) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(p, kind="stable")
    p, w = p[order], w[order]
    keep = (p < t_end) & (p + w <= t_end + 1e-6)
    p, w = p[keep], w[keep]
    if len(p) > MAX_PULSES:
        p, w = p[:MAX_PULSES], w[:MAX_PULSES]
    return np.round(p, 2), np.round(w, 2)


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_protocol(family: str, rng: np.random.Generator,
                    intensity_lo: float = 0.15, intensity_hi: float = 1.0,
                    freq_lo: float = 1.0, freq_hi: float = 40.0) -> Protocol:
    """Draw one stimulation protocol from a named family."""
    intensity = _log_uniform(rng, intensity_lo, intensity_hi)

    if family == "regular":
        freq = _log_uniform(rng, freq_lo, freq_hi)
        width = float(rng.uniform(2.0, 10.0))
        n = int(np.clip(rng.integers(3, 121), 3, np.floor(T_WINDOW / (1000.0 / freq))))
        p = np.arange(n) * (1000.0 / freq)
        w = np.full(n, width)
        meta = {"freq": freq, "width": width, "n_pulses": n}

    elif family == "step":
        width = _log_uniform(rng, 10.0, 800.0)
        p, w = np.zeros(1), np.array([width])
        meta = {"freq": 0.0, "width": width, "n_pulses": 1}

    elif family == "rand":
        rate = _log_uniform(rng, max(freq_lo, 1.0), freq_hi)
        cv = float(rng.uniform(0.1, 1.6))
        isi = _gamma_isi(rng, rate, cv, size=MAX_PULSES + 20)
        p = np.concatenate([[0.0], np.cumsum(isi)])[:MAX_PULSES]
        w = np.full(len(p), float(rng.uniform(2.0, 8.0)))
        p, w = _clip_pulses(p, w)
        meta = {"freq": rate, "cv": cv, "n_pulses": len(p)}

    elif family == "doublet":
        isi = _log_uniform(rng, 5.0, 400.0)
        width = float(rng.uniform(2.0, 8.0))
        p = np.array([0.0, isi])
        w = np.full(2, width)
        meta = {"freq": 1000.0 / isi, "isi": isi, "width": width, "n_pulses": 2}

    elif family == "burst":
        ibf = _log_uniform(rng, max(freq_lo, 2.0), 15.0)     # inter-burst freq
        bf = _log_uniform(rng, 20.0, max(freq_hi, 40.0))     # within-burst freq
        nb = int(rng.integers(1, 9))
        npb = int(rng.integers(2, 9))
        width = float(rng.uniform(2.0, 6.0))
        burst_dur = (npb - 1) * (1000.0 / bf)
        period = max(1000.0 / ibf, burst_dur + 5.0)
        p = np.concatenate([np.arange(npb) * (1000.0 / bf) + i * period
                            for i in range(nb)])
        w = np.full(len(p), width)
        p, w = _clip_pulses(p, w)
        meta = {"freq": bf, "inter_burst_freq": 1000.0 / period, "width": width,
                "n_pulses": len(p), "n_bursts": nb}

    elif family == "chirp":
        width = float(rng.uniform(2.0, 5.0))
        f0, f1 = 2.0, float(rng.uniform(40.0, 60.0))
        dur = float(rng.uniform(600.0, 1000.0))
        p = _chirp_times(f0, f1, dur, kind="exp")
        w = np.full(len(p), width)
        p, w = _clip_pulses(p, w)
        meta = {"freq": f1, "f0": f0, "f1": f1, "width": width, "n_pulses": len(p)}

    elif family == "sweep":
        width = float(rng.uniform(2.0, 5.0))
        lo, hi = float(rng.uniform(5.0, 15.0)), float(rng.uniform(35.0, 60.0))
        dur = float(rng.uniform(700.0, 1000.0))
        up = _chirp_times(lo, hi, dur / 2, kind="exp")
        down = dur / 2 + _chirp_times(hi, lo, dur / 2, kind="exp")
        p = np.concatenate([up, down])
        w = np.full(len(p), width)
        p, w = _clip_pulses(p, w)
        meta = {"freq": hi, "width": width, "n_pulses": len(p)}

    elif family == "sparse":
        rate = _log_uniform(rng, 0.5, 8.0)
        cv = float(rng.uniform(1.8, 3.5))
        isi = _gamma_isi(rng, rate, cv, size=MAX_PULSES + 20)
        p = np.concatenate([[0.0], np.cumsum(isi)])[:MAX_PULSES]
        w = np.full(len(p), float(rng.uniform(2.0, 10.0)))
        p, w = _clip_pulses(p, w)
        meta = {"freq": rate, "cv": cv, "n_pulses": len(p)}

    elif family == "chirp_hi":
        width = float(rng.uniform(2.0, 8.0))
        f0, f1 = 5.0, float(rng.uniform(70.0, 120.0))
        dur = float(rng.uniform(500.0, 900.0))
        p = _chirp_times(f0, f1, dur, kind="exp")
        w = np.full(len(p), width)
        p, w = _clip_pulses(p, w)
        meta = {"freq": f1, "f0": f0, "f1": f1, "width": width, "n_pulses": len(p)}

    else:
        raise ValueError(f"unknown family: {family}")

    return Protocol(family=family, pulses=p.astype(np.float64),
                    widths=w.astype(np.float64), intensity=intensity, meta=meta)


def _gamma_isi(rng: np.random.Generator, rate_hz: float, cv: float, size: int) -> np.ndarray:
    mean_isi = 1000.0 / max(rate_hz, 1e-3)
    k = max(1.0 / (cv ** 2), 0.05)
    theta = mean_isi / k
    return rng.gamma(shape=k, scale=theta, size=size)


def _chirp_times(f0: float, f1: float, dur: float, kind: str = "exp") -> np.ndarray:
    grid = np.arange(0.0, dur, 0.1)
    if grid[-1] < dur:
        grid = np.append(grid, dur)
    if kind == "exp":
        f = f0 * (f1 / f0) ** (grid / max(dur, 1e-6))
    else:
        f = f0 + (f1 - f0) * grid / max(dur, 1e-6)
    phase = np.concatenate([[0.0], np.cumsum(np.diff(grid) * f[:-1])])
    n_pulses = int(np.floor(phase[-1]))
    if n_pulses < 1:
        return np.zeros(0)
    return np.interp(np.arange(1, n_pulses + 1), phase, grid)


def families_for_split(split: str) -> Tuple[str, ...]:
    if split in ("train", "val", "test_id"):
        return TRAIN_FAMILIES
    if split == "test_ood_proto":
        return OOD_FAMILIES
    if split == "test_ood_freq":
        return ("regular",)
    if split == "test_ood_int":
        return ("regular", "step")
    raise ValueError(split)


def split_kwargs(split: str) -> Dict[str, float]:
    if split == "test_ood_freq":
        return {"freq_lo": 50.0, "freq_hi": 120.0}
    if split == "test_ood_int":
        return {"intensity_lo": 1.2, "intensity_hi": 2.0}
    return {}
