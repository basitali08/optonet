"""Statistics: bootstrap confidence intervals and paired model comparisons.

Writes:
  results/tables/statistics.csv
  paper/tables/main_table.tex
  paper/tables/ood_table.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optonet.config import load_config, repo_path

SPLITS = ["test_id", "test_ood_proto", "test_ood_freq", "test_ood_int"]
MODELS = ["glm", "cnn", "lstm", "transformer"]
N_BOOT = 1000


def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def mae(y, p):
    return np.mean(np.abs(y - p))


def spearmanr_safe(y, p):
    from scipy.stats import spearmanr
    if np.std(p) < 1e-12:
        return 0.0
    return float(spearmanr(y, p).statistic)


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(max((m - i) * p, running), 1.0)
        running = adj
        out[k] = adj
    return out


def inject_into_paper(name: str, block: str) -> None:
    """Replace the content between AUTO <name> markers in paper/main.tex."""
    main = repo_path("paper", "main.tex")
    if not main.exists():
        return
    begin, end = f"% BEGIN AUTO {name}", f"% END AUTO {name}"
    tex = main.read_text(encoding="utf-8")
    if begin in tex and end in tex:
        head, rest = tex.split(begin, 1)
        _, tail = rest.split(end, 1)
        tex = head + begin + "\n" + block + "\n" + end + tail
        main.write_text(tex, encoding="utf-8")
        print(f"  injected {name} into {main}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    cfg = load_config(args.config)

    z = np.load(repo_path("results", "pred_counts.npz"))
    rng = np.random.default_rng(0)
    rows, per_trial_err = [], {}

    for split in SPLITS:
        for model in MODELS:
            key_y = f"{model}__{split}__y"
            key_p = f"{model}__{split}__p"
            if key_y not in z.files:
                continue
            y = z[key_y].astype(np.float64)
            p = z[key_p].astype(np.float64)
            mu = p.mean(axis=0)
            n = len(y)
            boots = np.empty((args.n_boot, 3))
            for b in range(args.n_boot):
                idx = rng.integers(0, n, n)
                boots[b] = [r2(y[idx], mu[idx]), mae(y[idx], mu[idx]),
                            spearmanr_safe(y[idx], mu[idx])]
            point = [r2(y, mu), mae(y, mu), spearmanr_safe(y, mu)]
            seed_r2 = [r2(y, p[s]) for s in range(p.shape[0])]
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            rows.append({
                "split": split, "model": model, "n_trials": n,
                "r2": point[0], "r2_lo": lo[0], "r2_hi": hi[0],
                "r2_seed_mean": float(np.mean(seed_r2)),
                "r2_seed_sd": float(np.std(seed_r2)),
                "mae": point[1], "mae_lo": lo[1], "mae_hi": hi[1],
                "spearman": point[2], "spearman_lo": lo[2], "spearman_hi": hi[2],
            })
            per_trial_err[(split, model)] = np.abs(y - mu)

    df = pd.DataFrame(rows)
    df.to_csv(repo_path("results", "tables", "statistics.csv"), index=False)

    # paired comparisons of per-trial absolute error (Holm-corrected)
    comp_rows = []
    for split in SPLITS:
        pairs = {}
        for a in MODELS:
            for b in MODELS:
                if a >= b:
                    continue
                ka, kb = (split, a), (split, b)
                if ka not in per_trial_err or kb not in per_trial_err:
                    continue
                ea, eb = per_trial_err[ka], per_trial_err[kb]
                try:
                    stat, p = wilcoxon(ea, eb)
                except ValueError:
                    p = np.nan
                pairs[f"{a} vs {b}"] = p
        adj = holm({k: (v if np.isfinite(v) else 1.0) for k, v in pairs.items()})
        for k, v in adj.items():
            a, b = k.split(" vs ")
            comp_rows.append({"split": split, "comparison": k, "p_holm": v,
                              "mae_a": float(per_trial_err[(split, a)].mean()),
                              "mae_b": float(per_trial_err[(split, b)].mean())})
    pd.DataFrame(comp_rows).to_csv(
        repo_path("results", "tables", "significance.csv"), index=False)

    # LaTeX tables (inlined into the manuscript between markers)
    tabdir = repo_path("paper", "tables")
    tabdir.mkdir(parents=True, exist_ok=True)

    def fmt(v, d=3):
        return "--" if not np.isfinite(v) else f"{v:.{d}f}"

    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Model & Params & $R^2$ (95\% CI) & MAE & $\rho$ & Bin AUPRC \\",
             r"\midrule"]
    tab = repo_path("results", "tables", "ensemble_metrics.csv")
    if tab.exists():
        ens = pd.read_csv(tab)
        train_log = repo_path("results", "tables", "train_log.csv")
        nparams = {}
        if train_log.exists():
            tl = pd.read_csv(train_log)
            nparams = tl.groupby("model")["n_params"].mean().to_dict()
        for m in MODELS:
            row = df[(df["split"] == "test_id") & (df["model"] == m)]
            e = ens[(ens["split"] == "test_id") & (ens["model"] == m)]
            if not len(row):
                continue
            r = row.iloc[0]
            au = e["bin_auprc"].mean() if len(e) else np.nan
            np_ = nparams.get(m, np.nan)
            nps = f"{np_/1000:.0f}k" if np.isfinite(np_) else "--"
            lines.append(f"{m.upper()} & {nps} & {fmt(r['r2'])} "
                         f"({fmt(r['r2_lo'])}, {fmt(r['r2_hi'])}) & "
                         f"{fmt(r['mae'],2)} & {fmt(r['spearman'])} & {fmt(au,4)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tabdir / "main_table.tex").write_text("\n".join(lines), encoding="utf-8")

    ood_lines = [r"\begin{tabular}{lcccc}", r"\toprule",
                 r"Model & ID & OOD protocol & OOD freq. & OOD intensity \\",
                 r"\midrule"]
    for m in MODELS:
        cells = []
        for s in SPLITS:
            r = df[(df["split"] == s) & (df["model"] == m)]
            cells.append(fmt(r.iloc[0]["r2"]) if len(r) else "--")
        ood_lines.append(f"{m.upper()} & " + " & ".join(cells) + r" \\")
    ood_lines += [r"\bottomrule", r"\end{tabular}"]

    print(df.round(4).to_string(index=False))
    if comp_rows:
        print("\nPaired Wilcoxon (per-trial |error|), Holm-corrected:")
        print(pd.DataFrame(comp_rows).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
