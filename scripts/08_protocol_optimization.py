"""Closed-loop stimulation design: use the trained surrogate to search for
pulse patterns that drive a target spike output, then validate them in the
ground-truth simulator.

Methods compared (equal simulation budget is NOT used - only the surrogate
search differs; every candidate is finally scored by the simulator):
  ours        differential evolution over a 10-segment frequency schedule
  constant    differential evolution over a single constant frequency
  random      uniform random search over the same schedule space
  oracle      constant frequency scanned directly in the simulator

Writes results/tables/protocol_opt.csv and results/protocol_opt.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optonet import simulate as S
from optonet.config import load_config, repo_path
from optonet.data import load
from optonet.models import build_model
from optonet.train import make_store

SEGMENTS = 10
SEG_MS = 100
TOTAL_MS = 1000
F_MIN, F_MAX = 1.0, 60.0
I_MIN, I_MAX = 0.15, 1.0
WIDTH = 5.0
N_EVALS = 4000


def schedule_to_pulses(freqs: np.ndarray) -> np.ndarray:
    """Integrate a piecewise-constant frequency schedule into pulse times (ms)."""
    pulses = []
    phase = 0.0
    f_prev = 0.0
    for t in range(TOTAL_MS):
        seg = min(int(t // SEG_MS), SEGMENTS - 1)
        f = float(np.clip(freqs[seg], F_MIN, F_MAX))
        if t > 0:
            phase += (f + f_prev) * 0.5 / 1000.0
        f_prev = f
        if np.floor(phase) > len(pulses):
            pulses.append(float(t))
    return np.asarray(pulses, dtype=np.float64)


def pulses_to_light_ms(pulses: np.ndarray, intensity: float,
                       n_bins: int = 1500, onset: int = 200) -> np.ndarray:
    light = np.zeros(n_bins, dtype=np.float32)
    for p in pulses:
        a = onset + int(round(p))
        b = min(a + max(int(round(WIDTH)), 1), n_bins)
        if a < n_bins:
            light[a:b] = intensity
    return light


def dose(pulses: np.ndarray, intensity: float) -> float:
    return float(intensity * WIDTH * len(pulses))


class Surrogate:
    def __init__(self, cfg, name: str, cell_ids: np.ndarray):
        self.cfg = cfg
        self.name = name
        bundle = load()
        self.store = make_store(bundle, cond="full")
        self.cells = self.store.cell[cell_ids]
        ckpts = repo_path("results", "checkpoints")
        seeds = [s for s in cfg["training"]["seeds"]
                 if (ckpts / f"{name}_seed{s}.pt").exists()]
        self.models = []
        for s in seeds:
            ck = torch.load(ckpts / f"{name}_seed{s}.pt", map_location="cpu",
                            weights_only=False)
            m = build_model(name, self.store.cell.shape[1], cfg, cond=ck["cond"])
            m.load_state_dict(ck["state_dict"])
            m.eval()
            self.models.append(m)

    @torch.no_grad()
    def predict_counts(self, lights: np.ndarray) -> np.ndarray:
        """lights: [B, 1500] intensity units -> predicted evoked spike counts."""
        out = np.zeros(len(lights), dtype=np.float64)
        for m in self.models:
            for s in range(0, len(lights), 128):
                lb = torch.from_numpy(lights[s:s + 128])
                cs = self.cells[(np.arange(s, min(s + 128, len(lights)))
                                 % len(self.cells))]
                p = torch.sigmoid(m(lb, cs)).numpy()
                out[s:s + lb.shape[0]] += p[:, 200:].sum(axis=1)
        return out / len(self.models)


def evaluate_ground_truth(cfg, pulses: np.ndarray, intensity: float,
                          cells_arrays: dict, n_cells: int, seed: int) -> np.ndarray:
    if len(pulses) == 0:
        return np.zeros(n_cells)
    sub = {k: v[:n_cells] for k, v in cells_arrays.items()}
    light = S.rasterize_pulses([pulses], [np.full(len(pulses), WIDTH)],
                               np.array([intensity]))
    light = np.repeat(light, n_cells, axis=0)
    ind, times, _ = S.simulate_chunk(sub, light, seed=seed)
    ev = np.array([(times[ind[i]:ind[i + 1]] * S.DT >= S.T_ONSET).sum()
                   for i in range(n_cells)])
    return ev.astype(float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default="transformer")
    ap.add_argument("--target", type=int, default=20)
    ap.add_argument("--n-evals", type=int, default=N_EVALS)
    args = ap.parse_args()
    cfg = load_config(args.config)
    target = args.target

    bundle = load()
    cells = bundle.cells
    from optonet.data import CELL_ARRAY_KEYS, CELL_FEATURES
    cell_arrs = {k: cells[k].to_numpy(dtype=np.float64) for k in CELL_FEATURES}
    cell_arrs["VR"] = cells["cell_type"].map(
        {ct: S.CELL_TYPE_SPECS[ct].VR for ct in S.CELL_TYPES}).to_numpy()
    cell_arrs["tau_ou"] = cells["cell_type"].map(
        {ct: S.CELL_TYPE_SPECS[ct].tau_ou for ct in S.CELL_TYPES}).to_numpy()

    rows = []
    traces = {}
    t0 = time.time()

    for ct in S.CELL_TYPES:
        test_cells = cells[(cells["cell_type"] == ct) &
                           (cells["cell_split"] == "test")]["cell_id"].to_numpy()
        ref_cells = test_cells[:8]
        val_cells = test_cells[8:40]
        sub_val = {k: v[val_cells] for k, v in cell_arrs.items()}
        surrogate = Surrogate(cfg, args.model, ref_cells)

        def obj(freqs: np.ndarray, intensity: float) -> float:
            pulses = schedule_to_pulses(freqs)
            light = pulses_to_light_ms(pulses, intensity)[None, :]
            pred = surrogate.predict_counts(np.repeat(light, 1, axis=0))[0]
            return (pred - target) ** 2

        # --- ours: differential evolution over the schedule + intensity -----
        from scipy.optimize import differential_evolution
        bounds = [(F_MIN, F_MAX)] * SEGMENTS + [(I_MIN, I_MAX)]
        def obj_full(x):
            return obj(np.asarray(x[:SEGMENTS]), float(x[SEGMENTS]))
        res = differential_evolution(obj_full, bounds, maxiter=args.n_evals // 200,
                                     popsize=12, seed=0, polish=True,
                                     updating="immediate", workers=1)
        f_ours, i_ours = np.asarray(res.x[:SEGMENTS]), float(res.x[SEGMENTS])

        # --- constant frequency, surrogate-optimised -----------------------
        res_c = differential_evolution(lambda x: obj(np.full(SEGMENTS, x[0]), x[1]),
                                       [(F_MIN, F_MAX), (I_MIN, I_MAX)],
                                       maxiter=60, popsize=12, seed=0, polish=True)
        f_const = np.full(SEGMENTS, res_c.x[0])
        i_const = float(res_c.x[1])

        # --- random search --------------------------------------------------
        rng = np.random.default_rng(0)
        best, best_v = None, np.inf
        for _ in range(args.n_evals // 4):
            f_r = rng.uniform(F_MIN, F_MAX, SEGMENTS)
            i_r = float(rng.uniform(I_MIN, I_MAX))
            v = obj(f_r, i_r)
            if v < best_v:
                best, best_v = (f_r, i_r), v
        f_rand, i_rand = best

        # --- oracle: constant frequency scanned in the simulator -------------
        best_err, best_f, best_i = np.inf, 20.0, 0.5
        for f in np.linspace(2, 60, 15):
            for i in np.linspace(0.2, 1.0, 5):
                p = schedule_to_pulses(np.full(SEGMENTS, f))
                ev = evaluate_ground_truth(cfg, p, i, sub_val, len(val_cells), seed=7)
                err = abs(ev.mean() - target)
                if err < best_err:
                    best_err, best_f, best_i = err, f, i
        f_oracle = np.full(SEGMENTS, best_f)
        i_oracle = best_i

        for method, f_sched, inten in (("ours", f_ours, i_ours),
                                       ("constant", f_const, i_const),
                                       ("random", f_rand, i_rand),
                                       ("oracle", f_oracle, i_oracle)):
            pulses = schedule_to_pulses(f_sched)
            pred = surrogate.predict_counts(
                pulses_to_light_ms(pulses, inten)[None, :])[0]
            gt = evaluate_ground_truth(cfg, pulses, inten, sub_val, len(val_cells),
                                       seed=11)
            rows.append({
                "cell_type": ct, "method": method, "target": target,
                "pred_count": float(pred), "gt_mean": float(gt.mean()),
                "gt_std": float(gt.std()), "abs_error": float(abs(gt.mean() - target)),
                "n_pulses": int(len(pulses)), "intensity": float(inten),
                "dose": dose(pulses, inten),
                "mean_freq": float(len(pulses) / (TOTAL_MS / 1000.0)),
                "freq_std": float(f_sched.std()),
                "within3": float(np.mean(np.abs(gt - target) <= 3)),
            })
            traces[f"{ct}_{method}_freq"] = f_sched.astype(np.float32)
            traces[f"{ct}_{method}_pulses"] = pulses.astype(np.float32)
            traces[f"{ct}_{method}_intensity"] = np.float32(inten)
            traces[f"{ct}_{method}_gt"] = gt.astype(np.float32)
            print(f"{ct} {method:9s} pred={pred:5.1f} gt={gt.mean():5.1f}"
                  f"+-{gt.std():4.1f} err={abs(gt.mean()-target):4.1f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(repo_path("results", "tables", "protocol_opt.csv"), index=False)
    np.savez_compressed(repo_path("results", "protocol_opt.npz"), **traces)
    print("\n" + df.groupby("method")[["abs_error", "within3", "dose"]]
          .mean().round(2).to_string())
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
