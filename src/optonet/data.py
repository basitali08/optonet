"""Dataset construction, storage and loading.

Generated data layout (data/):
  cells.csv     one row per simulated neuron (parameters + type)
  trials.csv    one row per trial (cell, protocol, split, response summary)
  dataset.npz   dense 1 kHz light traces and spike rasters + CSR spike times
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import simulate as S
from .config import load_config, repo_path
from .protocols import sample_protocol, families_for_split, split_kwargs

SPLIT_NAMES = ("train", "val", "test_id", "test_ood_proto", "test_ood_freq",
               "test_ood_int")
CELL_SPLITS = ("train", "val", "test")
CELL_FEATURES = ("C", "gL", "EL", "VT", "dVT", "a", "b", "tau_w", "g_opto",
                 "I0", "std_ou")
CELL_ARRAY_KEYS = CELL_FEATURES + ("VR", "tau_ou")
META_COLS = ("freq", "width", "isi", "cv", "inter_burst_freq", "n_bursts", "f0", "f1")


def build_cells(cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"] + 1)
    scales = cfg["data"]["drive_scales"]
    rows = []
    cell_id = 0
    for ct in S.CELL_TYPES:
        n = cfg["data"]["cells_per_type"]
        cells = S.sample_cells(n, ct, rng, drive_scale=scales[ct])
        for i in range(n):
            split = CELL_SPLITS[0] if i < cfg["data"]["cell_split"]["train"] else (
                CELL_SPLITS[1] if i < cfg["data"]["cell_split"]["train"] + cfg["data"]["cell_split"]["val"]
                else CELL_SPLITS[2])
            rec = {"cell_id": cell_id, "cell_type": ct, "cell_split": split}
            for k in CELL_FEATURES:
                rec[k] = float(cells[k][i])
            rows.append(rec)
            cell_id += 1
    return pd.DataFrame(rows)


def build_trials(cfg: dict, cells: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"] + 2)
    pcfg = cfg["protocols"]
    rows: List[dict] = []
    trial_id = 0
    for split, n_trials in cfg["data"]["trials_per_split"].items():
        families = families_for_split(split)
        kwargs = split_kwargs(split)
        if split == "train":
            pool = cells[cells["cell_split"] == "train"]
        elif split == "val":
            pool = cells[cells["cell_split"] == "val"]
        else:
            pool = cells[cells["cell_split"] == "test"]
        pool_ids = pool["cell_id"].to_numpy()
        pool_types = pool["cell_type"].to_numpy()
        n_done = 0
        while n_done < n_trials:
            k = int(rng.integers(0, len(pool_ids)))
            fam = families[int(rng.integers(0, len(families)))]
            proto = sample_protocol(
                fam, rng,
                intensity_lo=kwargs.get("intensity_lo", pcfg["intensity_lo"]),
                intensity_hi=kwargs.get("intensity_hi", pcfg["intensity_hi"]),
                freq_lo=kwargs.get("freq_lo", pcfg["freq_lo"]),
                freq_hi=kwargs.get("freq_hi", pcfg["freq_hi"]))
            rec = {
                "trial_id": trial_id,
                "cell_id": int(pool_ids[k]),
                "cell_type": str(pool_types[k]),
                "split": split,
                "family": proto.family,
                "intensity": proto.intensity,
                "n_pulses": proto.n_pulses,
                "duration": proto.duration,
                "pulses": json.dumps(np.round(proto.pulses, 2).tolist()),
                "widths": json.dumps(np.round(proto.widths, 2).tolist()),
            }
            for c in META_COLS:
                rec[c] = proto.meta.get(c, np.nan)
            rows.append(rec)
            trial_id += 1
            n_done += 1
    return pd.DataFrame(rows)


def light_trace_from_trial(row: pd.Series, n_steps: Optional[int] = None,
                           dt: float = S.DT, t_onset: float = S.T_ONSET,
                           l_max: float = S.L_MAX) -> np.ndarray:
    pulses = np.asarray(json.loads(row["pulses"]), dtype=np.float64)
    widths = np.asarray(json.loads(row["widths"]), dtype=np.float64)
    n_steps = n_steps or S.N_STEPS
    return S.rasterize_pulses([pulses], [widths],
                              np.array([float(row["intensity"])]),
                              n_steps=n_steps, dt=dt, t_onset=t_onset,
                              l_max=l_max)[0]


def generate(cfg: Optional[dict] = None, out_dir: Optional[Path] = None) -> Path:
    cfg = cfg or load_config()
    out_dir = Path(out_dir or repo_path(cfg["data"]["root"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cells = build_cells(cfg)
    trials = build_trials(cfg, cells)
    n = len(trials)
    print(f"[data] {len(cells)} cells, {n} trials", flush=True)

    light_u8 = np.zeros((n, S.N_BINS), dtype=np.uint8)
    chunk = int(cfg["data"]["chunk_size"])
    time_parts: List[np.ndarray] = []
    vm_store: Dict[int, np.ndarray] = {}
    all_counts: List[np.ndarray] = []

    cell_arrs = {k: cells[k].to_numpy(dtype=np.float64) for k in CELL_FEATURES}
    cell_arrs["VR"] = cells["cell_type"].map(
        {ct: S.CELL_TYPE_SPECS[ct].VR for ct in S.CELL_TYPES}).to_numpy(dtype=np.float64)
    cell_arrs["tau_ou"] = cells["cell_type"].map(
        {ct: S.CELL_TYPE_SPECS[ct].tau_ou for ct in S.CELL_TYPES}).to_numpy(dtype=np.float64)
    seed = int(cfg["seed"])

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sub = trials.iloc[start:end]
        idx = sub["cell_id"].to_numpy()
        batch_cells = {k: v[idx] for k, v in cell_arrs.items()}
        pulses = [json.loads(p) for p in sub["pulses"]]
        widths = [json.loads(w) for w in sub["widths"]]
        intens = sub["intensity"].to_numpy(dtype=np.float64)
        light01 = S.rasterize_pulses(pulses, widths, intens)
        rec_rows = np.flatnonzero((sub["trial_id"].to_numpy() % 400) == 0)
        ind, times, vm = S.simulate_chunk(batch_cells, light01,
                                           record_vm=len(rec_rows),
                                           seed=seed + start)
        for j, r in enumerate(rec_rows):
            vm_store[int(sub["trial_id"].iloc[r])] = vm[j]
        light_ms = light01.reshape(end - start, S.N_BINS, 10).mean(axis=2)
        light_u8[start:end] = np.clip(np.rint(light_ms / (2.0 * S.L_MAX) * 255.0), 0, 255).astype(np.uint8)
        all_counts.append(np.diff(ind))
        time_parts.append(times)
        print(f"[data] {end}/{n}  ({time.time() - t0:.0f}s)", flush=True)

    counts = np.concatenate(all_counts)
    times_all = np.concatenate(time_parts)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    raster = np.zeros((n, S.N_BINS), dtype=np.uint8)
    rows_idx = np.repeat(np.arange(n, dtype=np.int64), counts)
    cols_idx = np.clip((times_all * S.DT).astype(np.int64), 0, S.N_BINS - 1)
    np.add.at(raster, (rows_idx, cols_idx), 1)

    pre = np.zeros(n, dtype=np.int64)
    onset_step = int(round(S.T_ONSET / S.DT))
    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        pre[i] = int(np.sum(times_all[s:e] < onset_step))
    trials["n_spikes_pre"] = pre
    trials["n_spikes_evoked"] = counts - pre
    trials["n_spikes_total"] = counts

    vm_ids = np.array(sorted(vm_store), dtype=np.int64)
    vm_arr = np.stack([vm_store[i] for i in vm_ids]) if len(vm_ids) else np.zeros((0, 0), np.float32)

    np.savez_compressed(out_dir / "dataset.npz",
                        light=light_u8, raster=raster, indptr=indptr,
                        times=times_all, vm=vm_arr, vm_trial_ids=vm_ids)
    cells.to_csv(out_dir / "cells.csv", index=False)
    trials.to_csv(out_dir / "trials.csv", index=False)
    meta = {
        "n_cells": int(len(cells)), "n_trials": int(n),
        "n_spikes": int(len(times_all)),
        "drive_scales": cfg["data"]["drive_scales"],
        "t_total_ms": S.T_TOTAL, "t_onset_ms": S.T_ONSET, "dt_ms": S.DT,
        "l_max": S.L_MAX, "generated_in_sec": round(time.time() - t0, 1),
        "splits": {s: int((trials["split"] == s).sum()) for s in SPLIT_NAMES},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[data] wrote {out_dir} in {meta['generated_in_sec']}s", flush=True)
    return out_dir


@dataclass
class Bundle:
    cells: pd.DataFrame
    trials: pd.DataFrame
    light: np.ndarray
    raster: np.ndarray
    indptr: np.ndarray
    times: np.ndarray
    vm: np.ndarray
    vm_trial_ids: np.ndarray
    cell_mean: np.ndarray = field(default=None)
    cell_std: np.ndarray = field(default=None)

    @property
    def cell_feature_names(self) -> List[str]:
        return list(CELL_FEATURES)

    def cell_matrix(self) -> np.ndarray:
        return self.cells[list(CELL_FEATURES)].to_numpy(dtype=np.float64)

    def fit_normalizer(self, mask_cells: np.ndarray) -> None:
        X = self.cell_matrix()[mask_cells]
        self.cell_mean = X.mean(axis=0)
        self.cell_std = X.std(axis=0) + 1e-8

    def norm_cell_matrix(self) -> np.ndarray:
        if self.cell_mean is None:
            self.fit_normalizer(np.arange(len(self.cells)))
        return (self.cell_matrix() - self.cell_mean) / self.cell_std

    def onehot(self) -> np.ndarray:
        types = list(S.CELL_TYPES)
        idx = self.cells["cell_type"].map({t: i for i, t in enumerate(types)}).to_numpy()
        return np.eye(len(types), dtype=np.float32)[idx]

    def trial_index(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.trials["split"].to_numpy() == split)

    def counts(self) -> np.ndarray:
        return np.diff(self.indptr)


LEAN_COLUMNS = ["trial_id", "cell_id", "cell_type", "split", "family", "intensity",
                "n_pulses", "width", "freq", "n_spikes_pre", "n_spikes_evoked",
                "n_spikes_total"]


def load(data_dir: Optional[Path] = None, lean: bool = False) -> Bundle:
    """Load the dataset.

    `lean=True` skips the (large) per-trial JSON pulse arrays, which keeps the
    resident footprint low enough to train on small machines.
    """
    data_dir = Path(data_dir or repo_path("data"))
    z = np.load(data_dir / "dataset.npz")
    cells = pd.read_csv(data_dir / "cells.csv")
    trials = pd.read_csv(data_dir / "trials.csv",
                         usecols=LEAN_COLUMNS if lean else None)
    bundle = Bundle(cells=cells, trials=trials, light=z["light"], raster=z["raster"],
                    indptr=z["indptr"], times=z["times"], vm=z["vm"],
                    vm_trial_ids=z["vm_trial_ids"])
    train_cells = cells["cell_split"].to_numpy() == "train"
    bundle.fit_normalizer(train_cells)
    return bundle


def sample_batch(bundle: Bundle, trial_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (light intensity, raster, cell features) for the given trials."""
    light = bundle.light[trial_ids].astype(np.float32) * (2.0 / 255.0)
    raster = bundle.raster[trial_ids].astype(np.float32)
    cell_ids = bundle.trials["cell_id"].to_numpy()[trial_ids]
    feats = np.concatenate([bundle.onehot()[cell_ids],
                            bundle.norm_cell_matrix()[cell_ids].astype(np.float32)], axis=1)
    return light, raster, feats
