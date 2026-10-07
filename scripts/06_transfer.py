"""Cross-cell-type transfer: pretrain on two cell types, adapt to the third.

Conditions evaluated on the held-out cell type (in-distribution protocols):
  oracle      model trained with all three types present
  zero_shot   pretrained on the other two types, no target-type data
  few_shot    pretrained, then fine-tuned on k labelled target-type cells
  scratch     trained from random init on the same k cells
  finetune_oracle  oracle fine-tuned on the same k cells

Writes results/tables/transfer.csv
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

from optonet.config import load_config, repo_path
from optonet.data import load
from optonet.metrics import sequence_metrics
from optonet.models import build_model
from optonet.train import evaluate, make_store, predict, set_seed, train_sequence


def trial_types(bundle) -> np.ndarray:
    return bundle.trials["cell_type"].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default="cnn")
    args = ap.parse_args()
    cfg = load_config(args.config)
    holdout = cfg["transfer"]["holdout_type"]
    name = args.model
    mcfg = dict(cfg["models"][name])
    seed = list(cfg["training"]["seeds"])[0]

    bundle = load()
    store = make_store(bundle, cond="full")
    types = trial_types(bundle)
    split = bundle.trials["split"].to_numpy()

    src_idx = np.flatnonzero((split == "train") & (types != holdout))
    tgt_train_idx = np.flatnonzero((split == "train") & (types == holdout))
    tgt_eval_idx = np.flatnonzero((split == "test_id") & (types == holdout))
    tgt_val_idx = np.flatnonzero((split == "val") & (types == holdout))
    all_train_idx = np.flatnonzero(split == "train")
    oracle_val_idx = np.flatnonzero(split == "val")

    print(f"[transfer] holdout={holdout} src={len(src_idx)} tgt_train={len(tgt_train_idx)} "
          f"eval={len(tgt_eval_idx)}", flush=True)
    rows = []

    def new_model():
        set_seed(seed)
        return build_model(name, store.cell.shape[1], cfg, cond="full")

    def run(tag, model, epochs, train_idx, val_idx, lr=None):
        t0 = time.time()
        model, best, _ = train_sequence(
            model, store, train_idx, val_idx, epochs=epochs,
            lr=lr or float(mcfg["lr"]),
            weight_decay=float(mcfg.get("weight_decay", 1e-4)),
            batch_size=int(mcfg.get("batch_size", 512)),
            patience=max(int(cfg["training"]["early_stopping_patience"]), 2), seed=seed)
        m = evaluate(model, store, tgt_eval_idx)
        m.update({"condition": tag, "model": name, "seed": seed,
                  "n_train": len(train_idx), "train_sec": round(time.time() - t0, 1),
                  "holdout": holdout})
        rows.append(m)
        print(f"  {tag:20s} n={len(train_idx):5d} evoked_r2={m['evoked_r2']:.3f} "
              f"auprc={m['bin_auprc']:.4f}", flush=True)
        return model

    # oracle: trained with every cell type present
    oracle = run("oracle", new_model(), int(mcfg["epochs"]), all_train_idx, oracle_val_idx)

    # pretrain on the two source types only
    pretrained = run("zero_shot", new_model(), int(mcfg["epochs"]), src_idx,
                     oracle_val_idx)

    # k-shot adaptation on held-out-type cells
    rng = np.random.default_rng(seed)
    cells = np.unique(bundle.trials["cell_id"].to_numpy()[tgt_train_idx])
    for k in cfg["transfer"]["shot_sizes"]:
        k = int(k)
        chosen = rng.choice(cells, size=min(k, len(cells)), replace=False)
        sub = tgt_train_idx[np.isin(bundle.trials["cell_id"].to_numpy()[tgt_train_idx],
                                    chosen)]
        set_seed(seed)
        ft = build_model(name, store.cell.shape[1], cfg, cond="full")
        ft.load_state_dict(pretrained.state_dict())
        run(f"few_shot_{k}", ft, int(cfg["transfer"]["epochs"]), sub, tgt_val_idx,
            lr=float(mcfg["lr"]) * 0.5)

        set_seed(seed)
        sc = build_model(name, store.cell.shape[1], cfg, cond="full")
        run(f"scratch_{k}", sc, int(cfg["transfer"]["epochs"]), sub, tgt_val_idx)

        set_seed(seed)
        fo = build_model(name, store.cell.shape[1], cfg, cond="full")
        fo.load_state_dict(oracle.state_dict())
        run(f"finetune_oracle_{k}", fo, int(cfg["transfer"]["epochs"]), sub,
            tgt_val_idx, lr=float(mcfg["lr"]) * 0.5)

    df = pd.DataFrame(rows)
    df.to_csv(repo_path("results", "tables", "transfer.csv"), index=False)
    print(df[["condition", "n_train", "evoked_r2", "bin_auprc", "evoked_mae"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
