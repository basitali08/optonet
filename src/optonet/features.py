"""Handcrafted tabular features used by the classical baselines."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .data import Bundle, CELL_FEATURES

FAMILIES = ("regular", "step", "rand", "doublet", "burst", "chirp", "sweep",
            "sparse", "chirp_hi")


def tabular_features(bundle: Bundle, trial_ids: np.ndarray
                     ) -> Tuple[np.ndarray, List[str]]:
    t = bundle.trials.iloc[trial_ids]
    names: List[str] = []
    cols: List[np.ndarray] = []

    onehot = bundle.onehot()[t["cell_id"].to_numpy()]
    for i, ct in enumerate(("pc", "pv", "som")):
        cols.append(onehot[:, i])
        names.append(f"type_{ct}")

    norm = bundle.norm_cell_matrix()[t["cell_id"].to_numpy()]
    for i, f in enumerate(CELL_FEATURES):
        cols.append(norm[:, i])
        names.append(f)

    fam = pd.Categorical(t["family"], categories=FAMILIES)
    fam_oh = np.eye(len(FAMILIES), dtype=np.float64)[fam.codes]
    for i, f in enumerate(FAMILIES):
        cols.append(fam_oh[:, i])
        names.append(f"fam_{f}")

    intensity = t["intensity"].to_numpy(dtype=np.float64)
    n_pulses = t["n_pulses"].to_numpy(dtype=np.float64)
    duration = t["duration"].to_numpy(dtype=np.float64)
    width = t["width"].to_numpy(dtype=np.float64)
    freq = t["freq"].to_numpy(dtype=np.float64)
    isi = t["isi"].to_numpy(dtype=np.float64)
    cv = t["cv"].to_numpy(dtype=np.float64)
    ibf = t["inter_burst_freq"].to_numpy(dtype=np.float64)
    nb = t["n_bursts"].to_numpy(dtype=np.float64)

    derived = {
        "intensity": intensity,
        "log_intensity": np.log(np.clip(intensity, 1e-3, None)),
        "n_pulses": n_pulses,
        "log_n_pulses": np.log1p(n_pulses),
        "duration": duration,
        "log_duration": np.log1p(duration),
        "dose": intensity * width * n_pulses,
        "log_dose": np.log1p(intensity * width * n_pulses),
        "duty_cycle": np.clip(width * n_pulses / np.maximum(duration, 1.0), 0, 4),
        "freq": freq,
        "log_freq": np.log(np.clip(freq, 1e-3, None)),
        "width": np.nan_to_num(width),
        "isi": np.nan_to_num(isi),
        "log_isi": np.log(np.clip(np.nan_to_num(isi), 1e-3, None)),
        "cv": np.nan_to_num(cv),
        "inter_burst_freq": np.nan_to_num(ibf),
        "n_bursts": np.nan_to_num(nb),
        "f0": np.nan_to_num(t["f0"].to_numpy(dtype=np.float64)),
        "f1": np.nan_to_num(t["f1"].to_numpy(dtype=np.float64)),
    }
    for k, v in derived.items():
        cols.append(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
        names.append(k)

    X = np.column_stack(cols).astype(np.float32)
    return X, names
