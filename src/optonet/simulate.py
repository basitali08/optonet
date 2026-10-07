"""Biophysical forward model for optogenetically driven spiking.

Neuron: Adaptive Exponential Integrate-and-Fire (Brette & Gerstner, 2005).
Opsin:  two-state ChR2 photocurrent with light-dependent activation and
        slow desensitization (Nagel et al., 2003; Foutz et al., 2012;
        Herman et al., 2019), including voltage-dependent deactivation.

Units: ms, mV, pF, nS, pA.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DT = 0.1                 # integration step (ms)
T_TOTAL = 1500.0         # trial duration (ms)
T_ONSET = 200.0          # light onset (ms)
N_STEPS = int(round(T_TOTAL / DT))
N_BINS = int(T_TOTAL)    # 1 kHz bins
K_D = 0.35               # light intensity for half activation (model units)
HILL_N = 2.0             # Hill coefficient of ChR2 activation
TAU_RISE = 3.0           # ms, channel opening time constant at low light
TAU_H_ON = 700.0         # ms, desensitization time constant during light
TAU_H_OFF = 2500.0       # ms, recovery time constant in darkness
H_DESens = 0.65          # fractional steady-state desensitization at saturating light
E_OPTO = 0.0             # mV, ChR2 reversal potential
V_DEACT_HALF = -40.0     # mV, voltage half-point of channel deactivation
V_DEACT_K = 6.0          # mV, slope of voltage-dependent deactivation
DEACT_FRAC = 0.75        # maximal fractional shortening of tau_rise when depolarized
V_SPIKE = 0.0            # mV, spike detection threshold
L_MAX = 0.6              # model units, intensity == 1.0 maps here

CELL_TYPES = ("pc", "pv", "som")


@dataclass(frozen=True)
class CellTypeSpec:
    name: str
    C: float
    gL: float
    EL: float
    VT: float
    dVT: float
    a: float
    b: float
    tau_w: float
    VR: float
    g_opto: float
    I0: float
    std_ou: float
    tau_ou: float
    target_rate: float
    jitter: Dict[str, float]


CELL_TYPE_SPECS: Dict[str, CellTypeSpec] = {
    "pc": CellTypeSpec(
        name="pc", C=200.0, gL=20.0, EL=-70.0, VT=-50.0, dVT=2.5,
        a=3.0, b=45.0, tau_w=100.0, VR=-70.0,
        g_opto=12.0, I0=280.0, std_ou=55.0, tau_ou=5.0,
        target_rate=3.0,
        jitter={"C": 0.18, "gL": 0.12, "EL": 3.0, "VT": 2.0, "dVT": 0.25,
                "a": 0.35, "b": 0.35, "tau_w": 0.30, "g_opto": 0.40,
                "I0": 0.10, "std_ou": 0.25},
    ),
    "pv": CellTypeSpec(
        name="pv", C=120.0, gL=12.0, EL=-70.0, VT=-50.0, dVT=2.0,
        a=0.5, b=1.0, tau_w=15.0, VR=-70.0,
        g_opto=7.0, I0=165.0, std_ou=45.0, tau_ou=5.0,
        target_rate=8.0,
        jitter={"C": 0.15, "gL": 0.12, "EL": 2.5, "VT": 1.8, "dVT": 0.20,
                "a": 0.50, "b": 0.50, "tau_w": 0.35, "g_opto": 0.40,
                "I0": 0.10, "std_ou": 0.25},
    ),
    "som": CellTypeSpec(
        name="som", C=140.0, gL=10.0, EL=-70.5, VT=-52.0, dVT=3.0,
        a=3.0, b=15.0, tau_w=45.0, VR=-70.5,
        g_opto=5.5, I0=125.0, std_ou=50.0, tau_ou=6.0,
        target_rate=4.0,
        jitter={"C": 0.18, "gL": 0.15, "EL": 3.0, "VT": 2.2, "dVT": 0.30,
                "a": 0.40, "b": 0.40, "tau_w": 0.35, "g_opto": 0.40,
                "I0": 0.12, "std_ou": 0.25},
    ),
}

CELL_PARAM_NAMES = ("C", "gL", "EL", "VT", "dVT", "a", "b", "tau_w",
                    "g_opto", "I0", "std_ou")


def sample_cells(n: int, cell_type: str, rng: np.random.Generator,
                 drive_scale: float = 1.0,
                 opto_scale: float = 1.0) -> Dict[str, np.ndarray]:
    """Sample a heterogeneous population of one cell type."""
    spec = CELL_TYPE_SPECS[cell_type]
    out: Dict[str, np.ndarray] = {"cell_type": np.array([cell_type] * n)}
    base = asdict(spec)
    for p in CELL_PARAM_NAMES:
        val = float(base[p])
        j = spec.jitter.get(p, 0.0)
        if j == 0.0:
            arr = np.full(n, val)
        elif p in ("EL", "VT"):
            arr = val + rng.normal(0.0, j, size=n)
        else:
            arr = val * np.exp(rng.normal(0.0, np.sqrt(np.log(1.0 + j ** 2)), size=n))
        out[p] = arr.astype(np.float64)
    out["I0"] = out["I0"] * drive_scale
    out["g_opto"] = out["g_opto"] * opto_scale
    out["VR"] = np.full(n, spec.VR)
    out["tau_ou"] = np.full(n, spec.tau_ou)
    return out


def rasterize_pulses(pulses: Sequence[np.ndarray], widths: Sequence[np.ndarray],
                     intensities: np.ndarray, n_steps: int = N_STEPS,
                     dt: float = DT, t_onset: float = T_ONSET,
                     l_max: float = L_MAX) -> np.ndarray:
    """Build a dense light trace (model units) for a batch of trials."""
    n = len(pulses)
    qmax = 255.0
    diff = np.zeros((n, n_steps), dtype=np.int16)
    for j in range(max((len(p) for p in pulses), default=0)):
        rows = np.array([i for i, p in enumerate(pulses) if len(p) > j], dtype=np.int64)
        if rows.size == 0:
            continue
        on = np.array([p[j] for p in pulses if len(p) > j], dtype=np.float64)
        wd = np.array([w[j] for w in widths if len(w) > j], dtype=np.float64)
        lvl = np.clip(np.rint(intensities[rows] * qmax), 1, int(qmax)).astype(np.int16)
        i0 = np.clip(np.rint((t_onset + on) / dt).astype(np.int64), 0, n_steps - 1)
        i1 = np.maximum(np.rint((t_onset + on + wd) / dt).astype(np.int64), i0 + 1)
        np.add.at(diff, (rows, i0), lvl)
        open_end = i1 < n_steps
        if open_end.any():
            np.add.at(diff, (rows[open_end], i1[open_end]), -lvl[open_end])
    np.cumsum(diff, axis=1, out=diff)
    return diff.astype(np.float32) * (l_max / qmax)


def simulate_chunk(cells: Dict[str, np.ndarray],
                   light: np.ndarray,
                   record_vm: int = 0,
                   seed: int = 0,
                   dt: float = DT) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate a batch of AdEx+ChR2 neurons driven by `light`.

    Returns (indptr, spike_step_indices, vm) in CSR form; vm rows are the
    first `record_vm` trials sampled at 1 kHz.
    """
    n = light.shape[0]
    n_steps = light.shape[1]
    rng = np.random.default_rng(seed)

    V = cells["EL"].astype(np.float64).copy()
    w = np.zeros(n)
    m = np.zeros(n)
    h = np.ones(n)
    I_ou = np.zeros(n)
    spk_row: List[np.ndarray] = []
    spk_t: List[np.ndarray] = []
    vm = np.zeros((record_vm, int(np.ceil(n_steps / 10))), dtype=np.float32) \
        if record_vm > 0 else np.zeros((0, 0), dtype=np.float32)
    vm_stride = 10

    C = cells["C"]
    gL = cells["gL"]
    EL = cells["EL"]
    VT = cells["VT"]
    dVT = cells["dVT"]
    a = cells["a"]
    b = cells["b"]
    tau_w = cells["tau_w"]
    VR = cells["VR"]
    g_opto = cells["g_opto"]
    I0 = cells["I0"]
    std_ou = cells["std_ou"]
    tau_ou = cells["tau_ou"]

    k2 = K_D ** HILL_N
    step_m = dt / TAU_RISE

    for t in range(n_steps):
        L = light[:, t]

        m_inf = L * L / (L * L + k2)
        tau_m = TAU_RISE - 1.2 * m_inf
        m += (m_inf - m) * (1.0 - np.exp(-dt / tau_m))

        h_ss = 1.0 - H_DESens * m_inf
        tau_h = np.where(L > 1e-6, TAU_H_ON, TAU_H_OFF)
        h += (h_ss - h) * (1.0 - np.exp(-dt / tau_h))

        deact = 1.0 - DEACT_FRAC / (1.0 + np.exp(-(V - V_DEACT_HALF) / V_DEACT_K))
        g_open = g_opto * m * h * deact
        I_opto = g_open * (E_OPTO - V)

        I_ou += (-I_ou / tau_ou) * dt + std_ou * np.sqrt(2.0 * dt / tau_ou) * rng.standard_normal(n)
        I_tot = I0 + I_ou + I_opto

        dv = (-(V - EL) * gL + gL * dVT * np.exp(np.clip((V - VT) / dVT, -80, 80))
              - w + I_tot) / C
        dw = (a * (V - EL) - w) / tau_w
        V = V + dv * dt
        w = w + dw * dt

        spk = V >= V_SPIKE
        if spk.any():
            rows = np.flatnonzero(spk)
            spk_row.append(rows)
            spk_t.append(np.full(rows.size, t, dtype=np.int32))
            V[spk] = VR[spk]
            w[spk] += b[spk]

        if record_vm > 0 and t % vm_stride == 0:
            vm[:, t // vm_stride] = V[:record_vm].astype(np.float32)

    if spk_row:
        rows = np.concatenate(spk_row)
        times = np.concatenate(spk_t)
        order = np.argsort(rows, kind="stable")
        rows = rows[order]
        times = times[order]
    else:
        rows = np.zeros(0, dtype=np.int64)
        times = np.zeros(0, dtype=np.int64)
    counts = np.bincount(rows, minlength=n)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return indptr, times.astype(np.int64), vm


def spikes_to_bincount(indptr: np.ndarray, times: np.ndarray, n_trials: int,
                       bin_ms: float = 1.0, n_bins: int = N_BINS,
                       dt: float = DT) -> np.ndarray:
    """Collapse CSR spike trains into per-trial binned spike counts."""
    counts = np.bincount(indptr[:n_trials], minlength=n_trials)
    return counts.reshape(n_trials, 1).astype(np.int32)


def spikes_to_raster(indptr: np.ndarray, times: np.ndarray, trial_ids: np.ndarray,
                     n_bins: int = N_BINS, dt: float = DT) -> np.ndarray:
    """Materialise a dense 0/1 raster (trials x bins at `bin_ms`=1 ms)."""
    out = np.zeros((len(trial_ids), n_bins), dtype=np.float32)
    for i, tr in enumerate(trial_ids):
        s, e = indptr[tr], indptr[tr + 1]
        if e > s:
            bins = (times[s:e] * dt).astype(np.int64)
            np.add.at(out[i], np.clip(bins, 0, n_bins - 1), 1.0)
    return out


def baseline_rates(n_per_type: int = 40, duration_ms: float = 4000.0,
                   drive_scale: Dict[str, float] | None = None,
                   seed: int = 0) -> Dict[str, float]:
    """Median spontaneous firing rate (Hz) per cell type with light off."""
    drive_scale = drive_scale or {t: 1.0 for t in CELL_TYPES}
    rates = {}
    rng = np.random.default_rng(seed)
    n_steps = int(duration_ms / DT)
    for k, ct in enumerate(CELL_TYPES):
        cells = sample_cells(n_per_type, ct, rng, drive_scale=drive_scale.get(ct, 1.0))
        light = np.zeros((n_per_type, n_steps), dtype=np.float32)
        indptr, times, _ = simulate_chunk(cells, light, seed=seed + 1000 + k)
        counts = np.diff(indptr)
        rates[ct] = float(np.mean(counts) / (duration_ms / 1000.0))
    return rates


def calibrate_drive(targets: Dict[str, float] = {"pc": 3.0, "pv": 8.0, "som": 4.0},
                    seed: int = 0, iters: int = 9) -> Dict[str, float]:
    """Bisect per-type background-drive scalars to hit target baseline rates."""
    lo = {ct: 0.5 for ct in targets}
    hi = {ct: 1.8 for ct in targets}
    scales = {ct: 1.0 for ct in targets}
    best = {ct: (np.inf, 1.0) for ct in targets}
    for _ in range(iters):
        rates = baseline_rates(duration_ms=3000.0, drive_scale=scales, seed=seed)
        for ct, target in targets.items():
            r = max(rates.get(ct, 1e-3), 1e-3)
            err = abs(np.log(r / target))
            if err < best[ct][0]:
                best[ct] = (err, scales[ct])
            if r < target:
                lo[ct] = scales[ct]
            else:
                hi[ct] = scales[ct]
            scales[ct] = 0.5 * (lo[ct] + hi[ct])
    return {ct: s for ct, (_err, s) in best.items()}
