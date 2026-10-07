"""Evaluate trained sequence models on every benchmark split.

Writes:
  results/tables/sequence_metrics.csv      per model / seed / split
  results/tables/ensemble_metrics.csv      per model / split (seed ensemble)
  results/pred_counts.npz                  per-trial predicted counts (for UQ/plots)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optonet.config import load_config, repo_path
from optonet.data import load
from optonet.metrics import sequence_metrics
from optonet.models import build_model
from optonet.train import make_store, predict

SPLITS = ("val", "test_id", "test_ood_proto", "test_ood_freq", "test_ood_int")


def load_checkpoint(path: Path, cfg: dict, store_cell_feats: int) -> torch.nn.Module:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["model"], store_cell_feats, cfg, cond=ck.get("cond", "full"))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    tables = repo_path("results", "tables")
    ckpts = repo_path("results", "checkpoints")
    tables.mkdir(parents=True, exist_ok=True)

    bundle = load()
    store = make_store(bundle, cond="full")
    seeds = list(cfg["training"]["seeds"])
    models = list(cfg["training"]["sequence_models"])

    rows = []
    ensemble_rows = []
    counts_out = {}
    probs_test = {}

    for name in models:
        available = [s for s in seeds if (ckpts / f"{name}_seed{s}.pt").exists()]
        if not available:
            print(f"[skip] {name}: no checkpoints", flush=True)
            continue
        print(f"[eval] {name} seeds={available}", flush=True)
        for split in SPLITS:
            idx = bundle.trial_index(split)
            y = store.raster[idx].numpy().astype(np.float64)
            y_ev = y[:, 200:].sum(axis=1)
            prob_sum = np.zeros((len(idx), y.shape[1]), dtype=np.float32)
            per_seed_counts = []
            for seed in available:
                model = load_checkpoint(ckpts / f"{name}_seed{seed}.pt", cfg,
                                        store.cell.shape[1])
                probs = predict(model, store, idx)
                prob_sum += probs
                per_seed_counts.append(probs[:, 200:].sum(axis=1))
                m = sequence_metrics(y, probs, window_start=200)
                m.update({"model": name, "seed": seed, "split": split,
                          "n_trials": len(idx)})
                rows.append(m)
            ens = sequence_metrics(y, prob_sum / len(available), window_start=200)
            ens.update({"model": name, "seed": "ensemble", "split": split,
                        "n_trials": len(idx)})
            ensemble_rows.append(ens)
            if split == "test_id":
                probs_test[name] = (prob_sum / len(available)).astype(np.float16)
            counts_out[f"{name}__{split}__y"] = y_ev.astype(np.float16)
            counts_out[f"{name}__{split}__p"] = np.stack(per_seed_counts).astype(np.float16)
            print(f"  {split}: auprc={ens['bin_auprc']:.4f} "
                  f"evoked_r2={ens['evoked_r2']:.3f} psth_r={ens['psth_r']:.3f}",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(tables / "sequence_metrics.csv", index=False)
    ens_df = pd.DataFrame(ensemble_rows)
    ens_df.to_csv(tables / "ensemble_metrics.csv", index=False)
    np.savez_compressed(repo_path("results", "pred_counts.npz"), **counts_out)
    np.savez_compressed(repo_path("results", "probs_test_id.npz"), **probs_test)
    print("\n=== ensemble summary (evoked_r2 / bin_auprc / psth_r) ===")
    print(ens_df.pivot(index="model", columns="split",
                       values="evoked_r2").round(3).to_string())
    print(ens_df.pivot(index="model", columns="split",
                       values="bin_auprc").round(4).to_string())


if __name__ == "__main__":
    main()
