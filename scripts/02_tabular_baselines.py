"""Tabular baselines: gradient-boosted trees on engineered features.

Writes:
  results/tables/tabular_metrics.csv
  results/checkpoints/hgb_*.joblib
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

from optonet.config import load_config, repo_path
from optonet.data import load
from optonet.features import tabular_features
from optonet.metrics import count_metrics, detection_metrics
from optonet.models import GBMClassifier, GBMRegressor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    bundle = load()
    tables = repo_path("results", "tables")
    ckpts = repo_path("results", "checkpoints")
    tables.mkdir(parents=True, exist_ok=True)
    ckpts.mkdir(parents=True, exist_ok=True)

    splits = [s for s in ("train", "val", "test_id", "test_ood_proto",
                          "test_ood_freq", "test_ood_int")]
    idx = {s: bundle.trial_index(s) for s in splits}
    X = {}
    for s in splits:
        X[s], feat_names = tabular_features(bundle, idx[s])
    y_evoked = bundle.trials["n_spikes_evoked"].to_numpy().astype(float)
    y_total = bundle.trials["n_spikes_total"].to_numpy().astype(float)

    hgb_cfg = cfg["models"]["hgb"]
    rows = []
    for target_name, y in (("evoked", y_evoked), ("total", y_total)):
        for seed in cfg["training"]["seeds"]:
            model = GBMRegressor(
                max_iter=hgb_cfg["max_iter"], learning_rate=hgb_cfg["learning_rate"],
                max_leaf_nodes=hgb_cfg["max_leaf_nodes"],
                min_samples_leaf=hgb_cfg["min_samples_leaf"], random_state=seed)
            t0 = time.time()
            model.fit(X["train"], y[idx["train"]])
            train_sec = time.time() - t0
            for s in splits:
                pred = model.predict(X[s])
                m = count_metrics(y[idx[s]], pred)
                m.update({"task": "count_" + target_name, "model": "hgb",
                          "seed": seed, "split": s, "train_sec": round(train_sec, 1)})
                rows.append(m)
            if seed == cfg["training"]["seeds"][0]:
                import joblib
                joblib.dump(model, ckpts / f"hgb_{target_name}.joblib")

    det_y = (y_evoked > 0).astype(int)
    for seed in cfg["training"]["seeds"]:
        clf = GBMClassifier(max_iter=hgb_cfg["detection_max_iter"],
                            learning_rate=hgb_cfg["learning_rate"],
                            max_leaf_nodes=hgb_cfg["max_leaf_nodes"],
                            min_samples_leaf=hgb_cfg["min_samples_leaf"],
                            random_state=seed)
        clf.fit(X["train"], det_y[idx["train"]])
        thr = None
        for s in splits:
            prob = clf.predict_proba(X[s])
            m = detection_metrics(det_y[idx[s]], prob, threshold=thr)
            if s == "val":
                thr = m["threshold"]
            m.update({"task": "detection", "model": "hgb", "seed": seed,
                      "split": s, "train_sec": 0.0})
            rows.append(m)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(tables / "tabular_metrics.csv", index=False)
    summary = (df.groupby(["task", "split"])[
        ["r2", "mae", "auroc", "auprc", "f1", "spearman"]].mean().round(4))
    print(summary.to_string())
    df.to_csv(tables / "tabular_metrics.csv", index=False)


if __name__ == "__main__":
    main()
