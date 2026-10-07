"""Uncertainty quantification: deep ensembles + split-conformal intervals.

Writes:
  results/tables/uncertainty.csv
  results/tables/calibration.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optonet.config import load_config, repo_path

SPLITS = ("test_id", "test_ood_proto", "test_ood_freq", "test_ood_int")


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    return float(np.quantile(scores, np.clip(level, 0, 1)))


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -1e-9, 1 + 1e-9
    idx = np.digitize(p, edges[1:-1])
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": b, "n": int(m.sum()), "p_mean": float(p[m].mean()),
                     "y_mean": float(y[m].mean())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    alpha = float(cfg["uncertainty"]["conformal_alpha"])

    z = np.load(repo_path("results", "pred_counts.npz"))
    tables = repo_path("results", "tables")
    tables.mkdir(parents=True, exist_ok=True)
    models = sorted({k.split("__")[0] for k in z.files})

    probs_path = repo_path("results", "probs_test_id.npz")
    probs_z = np.load(probs_path) if probs_path.exists() else None
    raster_z = None
    test_idx = None

    rows, cal_rows = [], []
    for model in models:
        def get(split, key):
            return z[f"{model}__{split}__{key}"].astype(np.float64)

        if f"{model}__val__y" not in z.files:
            continue
        cal_y = get("val", "y")
        cal_p = get("val", "p").mean(axis=0)
        scores = np.abs(cal_y - cal_p)
        q = conformal_quantile(scores, alpha)

        for split in SPLITS:
            y = get(split, "y")
            p_seeds = get(split, "p")
            mu = p_seeds.mean(axis=0)
            sd = p_seeds.std(axis=0)
            lo, hi = mu - q, mu + q
            cover = float(np.mean((y >= lo) & (y <= hi)))
            rows.append({
                "model": model, "split": split, "alpha": alpha,
                "q": q, "coverage": cover,
                "mean_width": float(np.mean(hi - lo)),
                "ensemble_std": float(np.mean(sd)),
                "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
                "n": len(y), "n_seeds": int(p_seeds.shape[0]),
                "std_ratio_vs_id": np.nan,
            })
        id_row = rows[-len(SPLITS)]
        id_std = id_row["ensemble_std"]
        for r in rows[-len(SPLITS):]:
            r["std_ratio_vs_id"] = r["ensemble_std"] / max(id_std, 1e-9)

        # bin-level calibration on the in-distribution test split
        if probs_z is not None and model in probs_z.files:
            if raster_z is None:
                from optonet.data import load
                bundle = load()
                raster_z = np.load(repo_path("data", "dataset.npz"))["raster"]
                test_idx = bundle.trial_index("test_id")
            probs = probs_z[model].astype(np.float64)
            rng = np.random.default_rng(0)
            mask = rng.choice(len(test_idx), size=min(4000, len(test_idx)), replace=False)
            yb = (raster_z[test_idx[mask]] > 0).ravel().astype(int)
            pb = probs[mask].ravel()
            cb = calibration_bins(yb, pb)
            cb["model"] = model
            cal_rows.append(cb)

    df = pd.DataFrame(rows)
    df.to_csv(tables / "uncertainty.csv", index=False)
    if cal_rows:
        pd.concat(cal_rows, ignore_index=True).to_csv(tables / "calibration.csv",
                                                       index=False)
    print(df[["model", "split", "coverage", "mean_width", "ensemble_std",
              "std_ratio_vs_id"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
