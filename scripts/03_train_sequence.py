"""Train the 1 kHz sequence models (GLM, CNN, LSTM, OptoFormer).

Resumable: a model/seed pair already present in results/tables/train_log.csv
is skipped.  Writes checkpoints to results/checkpoints/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch

from optonet.config import load_config, repo_path
from optonet.data import load
from optonet.models import build_model
from optonet.train import evaluate, make_store, set_seed, train_sequence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", default=None, help="comma separated subset")
    ap.add_argument("--seeds", default=None, help="comma separated subset")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tables = repo_path("results", "tables")
    ckpts = repo_path("results", "checkpoints")
    tables.mkdir(parents=True, exist_ok=True)
    ckpts.mkdir(parents=True, exist_ok=True)
    log_path = tables / "train_log.csv"

    models = (args.models.split(",") if args.models
              else list(cfg["training"]["sequence_models"]))
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else list(cfg["training"]["seeds"]))

    bundle = load(lean=True)
    store = make_store(bundle, cond="full")
    val_idx = bundle.trial_index("val")

    done = set()
    if log_path.exists():
        prev = pd.read_csv(log_path)
        done = set(zip(prev["model"], prev["seed"]))

    rows = []
    for name in models:
        mcfg = cfg["models"][name]
        for seed in seeds:
            if (name, seed) in done:
                print(f"[skip] {name} seed={seed}", flush=True)
                continue
            print(f"[train] {name} seed={seed}", flush=True)
            set_seed(seed)
            model = build_model(name, store.cell.shape[1], cfg, cond="full")
            n_params = sum(p.numel() for p in model.parameters())
            t0 = time.time()
            model, best, history = train_sequence(
                model, store, bundle.trial_index("train"), val_idx,
                epochs=int(mcfg["epochs"]), lr=float(mcfg["lr"]),
                weight_decay=float(mcfg.get("weight_decay", 1e-4)),
                batch_size=int(mcfg.get("batch_size", 512)),
                patience=int(cfg["training"]["early_stopping_patience"]),
                seed=seed)
            train_sec = time.time() - t0
            val_metrics = evaluate(model, store, val_idx)
            torch.save({"model": name, "seed": seed, "cond": "full",
                        "state_dict": model.state_dict(),
                        "n_params": int(n_params),
                        "best_epoch": best["epoch"], "val_loss": best["loss"]},
                       ckpts / f"{name}_seed{seed}.pt")
            (ckpts / f"{name}_seed{seed}_history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8")
            row = {"model": name, "seed": seed, "n_params": int(n_params),
                   "best_epoch": best["epoch"], "val_loss": best["loss"],
                   "train_sec": round(train_sec, 1),
                   "epochs_run": len(history)}
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            rows.append(row)
            df_new = pd.DataFrame(rows)
            if log_path.exists():
                df_new = pd.concat([pd.read_csv(log_path), df_new], ignore_index=True)
            df_new.to_csv(log_path, index=False)
            rows = []
            print(f"  done in {train_sec:.0f}s  val auprc={val_metrics['bin_auprc']:.4f} "
                  f"evoked_r2={val_metrics['evoked_r2']:.3f} params={n_params}", flush=True)

    print("[done] sequence training complete", flush=True)


if __name__ == "__main__":
    main()
