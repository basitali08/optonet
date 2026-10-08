"""Publication figures for the OptoNet paper.

Figures written to results/figures (PNG + PDF).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optonet import simulate as S
from optonet.config import load_config, repo_path

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 8.5,
    "axes.titlesize": 9.5, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.constrained_layout.use": True,
})

PALETTE = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1", "#b83280"]
MODELS = ["glm", "cnn", "lstm", "transformer"]
SPLIT_ORDER = ["val", "test_id", "test_ood_proto", "test_ood_freq", "test_ood_int"]
SPLIT_LABEL = {"val": "val", "test_id": "ID test", "test_ood_proto": "OOD protocol",
               "test_ood_freq": "OOD frequency", "test_ood_int": "OOD intensity"}


def save(fig, name):
    figdir = repo_path("results", "figures")
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}", flush=True)


def fig_protocols():
    from optonet.protocols import sample_protocol, TRAIN_FAMILIES, OOD_FAMILIES
    rng = np.random.default_rng(3)
    families = list(TRAIN_FAMILIES) + list(OOD_FAMILIES)
    fig, axes = plt.subplots(2, 5, figsize=(11.5, 3.4), sharex=True, sharey=True)
    for ax, fam in zip(axes.ravel(), families):
        p = sample_protocol(fam, rng)
        for t, w in zip(p.pulses, p.widths):
            ax.add_patch(plt.Rectangle((200 + t, 0), max(w, 1.0), 1,
                                       color=PALETTE[0], alpha=0.85, lw=0))
        ax.set_xlim(0, 1500)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(f"{fam}  (n={p.n_pulses})", fontsize=8)
        ax.set_xlabel("time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("light")
    save(fig, "fig1_protocol_examples")


def fig_simulator(cfg):
    """Dose-response, frequency-following and PSTH sanity checks."""
    rng = np.random.default_rng(5)
    scales = cfg["data"]["drive_scales"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))

    ints = np.array([0.0, 0.15, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    freqs = np.array([5, 10, 20, 40, 80])
    for ci, ct in enumerate(S.CELL_TYPES):
        cells = S.sample_cells(64, ct, rng, drive_scale=scales[ct])
        curve = []
        for inten in ints:
            p = np.arange(0, 1000, 50.0)
            light = S.rasterize_pulses([p], [np.full(len(p), 5.0)],
                                       np.array([inten]))
            light = np.repeat(light, 64, axis=0)
            ind, times, _ = S.simulate_chunk(cells, light, seed=3)
            ev = np.array([(times[ind[i]:ind[i + 1]] * S.DT >= S.T_ONSET).sum()
                           for i in range(64)])
            curve.append(ev.mean())
        axes[0].plot(ints, curve, "o-", color=PALETTE[ci], label=ct.upper(), ms=3)
        fid = []
        for f in freqs:
            p = np.arange(0, 1000, 1000.0 / f)
            light = S.rasterize_pulses([p], [np.full(len(p), 5.0)], np.array([1.0]))
            light = np.repeat(light, 64, axis=0)
            ind, times, _ = S.simulate_chunk(cells, light, seed=4)
            ev = np.array([(times[ind[i]:ind[i + 1]] * S.DT >= S.T_ONSET).sum()
                           for i in range(64)])
            fid.append(ev.mean() / len(p))
        axes[1].plot(freqs, fid, "o-", color=PALETTE[ci], label=ct.upper(), ms=3)
    axes[0].set_xlabel("light intensity (model units)")
    axes[0].set_ylabel("evoked spikes per trial")
    axes[0].set_title("Dose-response")
    axes[0].legend(frameon=False)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("pulse frequency (Hz)")
    axes[1].set_ylabel("spikes per pulse (fidelity)")
    axes[1].set_ylim(0, 2)
    axes[1].axhline(1.0, color="0.6", lw=0.7, ls="--")
    axes[1].set_title("Following vs. frequency")
    axes[1].legend(frameon=False)

    # example PSTH raster
    cells = S.sample_cells(1, "pv", rng, drive_scale=scales["pv"])
    p = np.arange(0, 800, 25.0)
    light = S.rasterize_pulses([p], [np.full(len(p), 5.0)], np.array([0.8]))
    ind, times, _ = S.simulate_chunk(cells, light, seed=9)
    spikes = times * S.DT
    for i, sp in enumerate(spikes):
        axes[2].vlines(sp, i, i + 0.8, color=PALETTE[1], lw=0.5)
    axes[2].set_xlim(0, 1200)
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([])
    axes[2].set_xlabel("time (ms)")
    axes[2].set_title("Example PV+ response (20 ms pulse width, 40 Hz)")
    save(fig, "fig2_simulator_validation")


def fig_dataset():
    b = pd.read_csv(repo_path("data", "trials.csv"))
    fams = ["regular", "step", "rand", "doublet", "burst", "chirp", "sweep",
            "sparse", "chirp_hi"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0))
    splits = SPLIT_ORDER
    counts = pd.crosstab(b["split"], b["family"]).reindex(splits).fillna(0)
    im = axes[0].imshow(counts.values, cmap="Blues", aspect="auto")
    axes[0].set_xticks(range(len(fams)))
    axes[0].set_xticklabels(fams, rotation=45, ha="right", fontsize=7)
    axes[0].set_yticks(range(len(splits)))
    axes[0].set_yticklabels([SPLIT_LABEL[s] for s in splits], fontsize=7.5)
    axes[0].set_title("Protocol x split composition")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    for i, s in enumerate(SPLIT_ORDER):
        m = b["split"] == s
        axes[1].hist(np.log10(b.loc[m, "n_spikes_evoked"] + 1), bins=40, histtype="step",
                     lw=1.2, label=SPLIT_LABEL[s], color=PALETTE[i])
    axes[1].set_xlabel("log10(1 + evoked spikes)")
    axes[1].set_ylabel("trials")
    axes[1].legend(frameon=False, fontsize=6.5)

    for i, ct in enumerate(S.CELL_TYPES):
        m = (b["split"] == "train") & (b["cell_type"] == ct)
        axes[2].hist(b.loc[m, "intensity"], bins=40, histtype="step", lw=1.2,
                     color=PALETTE[i], label=f"{ct.upper()}  n={m.sum()}")
    axes[2].set_xlabel("light intensity (model units)")
    axes[2].legend(frameon=False)
    axes[2].set_title("Training stimuli by cell type")
    save(fig, "fig3_dataset_overview")


def fig_main_results():
    path = repo_path("results", "tables", "ensemble_metrics.csv")
    if not path.exists():
        print("  [skip] fig4: run 04_evaluate.py first")
        return
    e = pd.read_csv(path)
    e = e[e["model"].isin(MODELS)]
    tab = pd.read_csv(repo_path("results", "tables", "tabular_metrics.csv"))
    tab = tab[(tab["task"] == "count_evoked")]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.1))
    splits = [s for s in SPLIT_ORDER if s != "val"]
    w = 0.2
    x = np.arange(len(splits))
    for i, m in enumerate(MODELS):
        vals = [e[(e["model"] == m) & (e["split"] == s)]["evoked_r2"].mean()
                for s in splits]
        errs = [e[(e["model"] == m) & (e["split"] == s)]["evoked_r2"].std()
                for s in splits]
        axes[0].bar(x + (i - 1.5) * w, vals, w, yerr=errs, capsize=2,
                    color=PALETTE[i], label=m.upper())
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([SPLIT_LABEL[s] for s in splits], rotation=15, fontsize=7.5)
    axes[0].set_ylabel("$R^2$ of evoked spike count")
    axes[0].set_title("Count prediction")
    axes[0].axhline(0, color="0.4", lw=0.8)
    axes[0].legend(frameon=False, ncol=2, fontsize=7)

    for i, m in enumerate(MODELS):
        vals = [e[(e["model"] == m) & (e["split"] == s)]["bin_auprc"].mean()
                for s in splits]
        axes[1].bar(x + (i - 1.5) * w, vals, w, color=PALETTE[i], label=m.upper())
    hgb = [tab[tab["split"] == s]["auprc"].mean() for s in splits]
    axes[1].plot(x + 0.4, hgb, "D--", color="k", ms=4, label="GBM (features)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([SPLIT_LABEL[s] for s in splits], rotation=15, fontsize=7.5)
    axes[1].set_ylabel("AUPRC (1 kHz bins)")
    axes[1].set_title("Spike detection")
    axes[1].legend(frameon=False, ncol=2, fontsize=7)

    for i, m in enumerate(MODELS):
        vals = [e[(e["model"] == m) & (e["split"] == s)]["psth_r"].mean()
                for s in splits]
        axes[2].bar(x + (i - 1.5) * w, vals, w, color=PALETTE[i], label=m.upper())
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([SPLIT_LABEL[s] for s in splits], rotation=15, fontsize=7.5)
    axes[2].set_ylabel("PSTH correlation")
    axes[2].set_title("Temporal fidelity")
    axes[2].legend(frameon=False, ncol=2, fontsize=7)
    save(fig, "fig4_main_results")


def fig_psth_examples():
    p = repo_path("results", "probs_test_id.npz")
    if not p.exists():
        print("  [skip] fig5: run 04_evaluate.py first")
        return
    from optonet.data import load
    b = load()
    z = np.load(p)
    if "transformer" not in z.files:
        print("  [skip] fig5: no transformer probs")
        return
    probs = z["transformer"].astype(np.float32)
    idx = b.trial_index("test_id")
    t = b.trials.iloc[idx]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 2.7))
    for k, ct in enumerate(S.CELL_TYPES):
        sel = np.flatnonzero((t["cell_type"].to_numpy() == ct) &
                             (t["n_pulses"].to_numpy() >= 8))
        if len(sel) < 5:
            continue
        obs = b.raster[idx[sel]][:, 200:].mean(axis=0)
        pred = probs[sel][:, 200:].mean(axis=0)
        axes[k].plot(np.arange(200, 1500), obs, color="k", lw=1.1, label="observed")
        axes[k].plot(np.arange(200, 1500), pred, color=PALETTE[1], lw=1.1, ls="--",
                     label="OptoFormer")
        axes[k].set_title(f"{ct.upper()} mean PSTH (n={len(sel)})")
        axes[k].set_xlabel("time (ms)")
        axes[k].set_ylabel("spikes / ms / trial")
        if k == 0:
            axes[k].legend(frameon=False)
    save(fig, "fig5_psth_examples")


def fig_uncertainty():
    p = repo_path("results", "tables", "uncertainty.csv")
    if not p.exists():
        print("  [skip] fig6: run 05_uncertainty.py first")
        return
    u = pd.read_csv(p)
    u = u[u["split"] != "val"]
    splits = [s for s in SPLIT_ORDER if s != "val"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9))
    for i, m in enumerate(sorted(u["model"].unique())):
        vals = [u[(u["model"] == m) & (u["split"] == s)]["coverage"].mean()
                for s in splits]
        axes[0].plot(range(len(splits)), vals, "o-", color=PALETTE[i % 5],
                     label=m.upper(), ms=4)
        sd = [u[(u["model"] == m) & (u["split"] == s)]["ensemble_std"].mean()
              for s in splits]
        axes[1].plot(range(len(splits)), sd, "o-", color=PALETTE[i % 5],
                     label=m.upper(), ms=4)
    for ax, ylab, title in ((axes[0], "empirical coverage", "Conformal coverage (90% nominal)"),
                            (axes[1], "mean ensemble SD (spikes)", "Epistemic uncertainty")):
        ax.set_xticks(range(len(splits)))
        ax.set_xticklabels([SPLIT_LABEL[s] for s in splits], rotation=15, fontsize=7.5)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(frameon=False)
    axes[0].axhline(0.9, color="0.5", ls="--", lw=0.8)
    axes[0].set_ylim(0, 1.05)
    save(fig, "fig6_uncertainty")


def fig_interpretability():
    p = repo_path("results", "interp.npz")
    if not p.exists():
        print("  [skip] fig7: run 07_interpretability.py first")
        return
    z = np.load(p)
    keys = [f"sal_{m}" for m in MODELS if f"sal_{m}" in z.files]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 2.8))
    for i, k in enumerate(keys):
        sal = z[k]
        axes[0].plot(sal[:, 200:].mean(axis=0), color=PALETTE[i], lw=0.9,
                     label=k.split("_", 1)[1].upper())
    axes[0].set_xlim(0, 1000)
    axes[0].set_xlabel("time after stimulus onset (ms)")
    axes[0].set_ylabel("mean |d logit / d light|")
    axes[0].set_title("Saliency profile")
    axes[0].legend(frameon=False)

    for i, ct in enumerate(("pc", "pv", "som")):
        k = f"glm_kernel_{ct}"
        if k in z.files:
            axes[1].plot(z[k], color=PALETTE[i], lw=1.0, label=ct.upper())
    axes[1].set_xlabel("lag (ms)")
    axes[1].set_ylabel("kernel weight")
    axes[1].set_title("GLM impulse response by cell type")
    axes[1].legend(frameon=False)

    if "attn_rollout" in z.files:
        a = z["attn_rollout"]
        dist = np.abs(np.subtract.outer(np.arange(a.shape[0]), np.arange(a.shape[1])))
        num = (a * (dist <= 1)).sum() / a.sum()
        axes[2].imshow(a, cmap="magma", aspect="auto")
        axes[2].set_xlabel("key patch")
        axes[2].set_ylabel("query patch")
        base = np.mean([min(3, i + 1) / (i + 1) for i in range(a.shape[0])])
        axes[2].set_title(f"Attention (adjacent={num:.2f}; uniform={base:.2f})")
    save(fig, "fig7_interpretability")


def fig_transfer():
    p = repo_path("results", "tables", "transfer.csv")
    if not p.exists():
        print("  [skip] fig8: run 06_transfer.py first")
        return
    t = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    groups = [("oracle", "Oracle\n(all types)"), ("zero_shot", "Zero-shot"),
              ("few_shot_50", "Few-shot 50"), ("scratch_50", "Scratch 50"),
              ("finetune_oracle_50", "Oracle + FT 50")]
    labels, vals, colors = [], [], []
    for cond, lab in groups:
        m = t[t["condition"] == cond]
        if len(m) == 0:
            continue
        labels.append(lab)
        vals.append(m["evoked_r2"].mean())
        colors.append(PALETTE[1] if cond.startswith("few") else PALETTE[0])
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7.5)
    ax.set_ylabel("$R^2$ (evoked count)")
    ax.set_title(f"Few-shot transfer to held-out {t['holdout'].iloc[0].upper()} cells")
    save(fig, "fig8_transfer")


def fig_protocol_optimization():
    p = repo_path("results", "protocol_opt.npz")
    if not p.exists():
        print("  [skip] fig9: run 08_protocol_optimization.py first")
        return
    z = np.load(p)
    cts = [c for c in S.CELL_TYPES if f"{c}_ours_freq" in z.files]
    fig, axes = plt.subplots(len(cts), 3, figsize=(11.0, 2.3 * len(cts)),
                             squeeze=False)
    methods = [("ours", PALETTE[0]), ("constant", PALETTE[1]),
               ("random", PALETTE[3])]
    for r, ct in enumerate(cts):
        for m, col in methods:
            f = z[f"{ct}_{m}_freq"]
            axes[r][0].plot(np.arange(len(f)) * 100, f, color=col, lw=1.2, label=m)
        axes[r][0].set_ylabel(f"{ct.upper()} (Hz)")
        axes[r][0].set_xlabel("time (ms)")
        if r == 0:
            axes[r][0].set_title("Designed frequency schedule")
            axes[r][0].legend(frameon=False)
        for m, col in methods:
            gt = z[f"{ct}_{m}_gt"]
            axes[r][1].hist(gt, bins=np.arange(0, max(41, int(gt.max()) + 2)), alpha=0.55,
                            color=col, label=m)
        axes[r][1].set_ylabel("cells")
        if r == 0:
            axes[r][1].set_title("Simulator response counts")
            axes[r][1].legend(frameon=False)
        for m, col in methods:
            pulses = z[f"{ct}_{m}_pulses"]
            inten = float(z[f"{ct}_{m}_intensity"])
            for t_, w in zip(pulses, [5.0] * len(pulses)):
                axes[r][2].add_patch(plt.Rectangle((200 + t_, 0), max(w, 1), 1,
                                                   color=col, alpha=0.8, lw=0))
        axes[r][2].set_xlim(0, 1500)
        axes[r][2].set_ylim(0, 1)
        axes[r][2].set_yticks([])
        axes[r][2].set_xlabel("time (ms)")
        if r == 0:
            axes[r][2].set_title("Designed light protocol")
    save(fig, "fig9_protocol_optimization")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    only = set(args.only.split(",")) if args.only else None

    all_figs = {
        "protocols": lambda: fig_protocols(),
        "simulator": lambda: fig_simulator(cfg),
        "dataset": lambda: fig_dataset(),
        "main": fig_main_results,
        "psth": fig_psth_examples,
        "uncertainty": fig_uncertainty,
        "interpret": fig_interpretability,
        "transfer": fig_transfer,
        "optimization": fig_protocol_optimization,
    }
    print("[figures] generating")
    for name, fn in all_figs.items():
        if only and name not in only:
            continue
        try:
            fn()
        except Exception as exc:  # keep going, figures are best-effort
            print(f"  [warn] {name}: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
