"""Interpretability analyses.

1. input-gradient saliency: where in the stimulus does each model look?
2. pulse-alignment enrichment and adaptation profile (saliency vs pulse index)
3. GLM impulse-response kernels per cell type
4. transformer attention rollout (effective receptive field)
5. negative control: light-trace shuffling

Writes results/interp.npz and results/tables/interpretability.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optonet.config import load_config, repo_path
from optonet.data import load
from optonet.models import build_model
from optonet.train import make_store, predict

N_TRIALS = 600
T_ONSET = 200


def saliency_batch(model, light_t, cell_t, batch=64):
    grads = []
    for s in range(0, light_t.shape[0], batch):
        l = light_t[s:s + batch].clone().requires_grad_(True)
        c = cell_t[s:s + batch]
        out = model(l, c)
        score = out[:, T_ONSET:].sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        grads.append(l.grad.detach().abs().cpu())
    return torch.cat(grads).numpy()


def pulse_mask(trials_sub: pd.DataFrame, n_bins: int) -> np.ndarray:
    m = np.zeros((len(trials_sub), n_bins), dtype=bool)
    for i, row in enumerate(trials_sub.itertuples()):
        pulses = json.loads(row.pulses)
        widths = json.loads(row.widths)
        for p, w in zip(pulses, widths):
            a = int(T_ONSET + p)
            b = int(T_ONSET + p + w) + 1
            if a < n_bins:
                m[i, a:min(b, n_bins)] = True
    return m


def attention_rollout(model, light_t, cell_t, patch: int, batch: int = 32):
    """Mean attention weight from late patches to each earlier patch."""
    rolls = []
    b0, t = light_t.shape
    pad = (-t) % patch
    for s in range(0, b0, batch):
        l = light_t[s:s + batch]
        c = cell_t[s:s + batch]
        x = torch.nn.functional.pad(l.unsqueeze(1), (0, pad))
        n = x.shape[-1] // patch
        v = x.reshape(l.shape[0], n, patch) + model.sub_pos
        tok = model.proj(v) + model.pos[:, :n]
        gamma, beta = model.film(c).chunk(2, dim=-1)
        tok = tok * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        att = []
        for layer in model.encoder.layers:
            _, a = layer.self_attn(tok, tok, tok, need_weights=True,
                                   average_attn_weights=True)
            att.append(a.detach().mean(dim=0))   # [n, n]
        a_bar = torch.stack(att).mean(dim=0)     # [n, n]
        rolls.append(a_bar.cpu().numpy())
    return np.mean(rolls, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    bundle = load()
    store = make_store(bundle, cond="full")
    ckpts = repo_path("results", "checkpoints")
    seeds = list(cfg["training"]["seeds"])
    models = list(cfg["training"]["sequence_models"])

    rng = np.random.default_rng(0)
    idx = bundle.trial_index("test_id")
    idx = rng.choice(idx, size=min(N_TRIALS, len(idx)), replace=False)
    idx = np.sort(idx)
    trials_sub = bundle.trials.iloc[idx].reset_index(drop=True)

    light = torch.from_numpy(bundle.light[idx].astype(np.float32) * (2.0 / 255.0))
    cell = store.cell[idx]
    raster = bundle.raster[idx].astype(np.float64)
    pmask = pulse_mask(trials_sub, bundle.light.shape[1])
    pulse_idx = trials_sub["n_pulses"].to_numpy().astype(int)
    ctype = trials_sub["cell_type"].to_numpy()

    out = {}
    rows = []

    for name in models:
        avail = [s for s in seeds if (ckpts / f"{name}_seed{s}.pt").exists()]
        if not avail:
            continue
        grads, probs_list = [], []
        for seed in avail:
            ck = torch.load(ckpts / f"{name}_seed{seed}.pt", map_location="cpu",
                            weights_only=False)
            model = build_model(name, store.cell.shape[1], cfg, cond=ck["cond"])
            model.load_state_dict(ck["state_dict"])
            model.eval()
            grads.append(saliency_batch(model, light, cell))
            probs_list.append(predict(model, store, idx))
        sal = np.mean(grads, axis=0)
        probs = np.mean(probs_list, axis=0)
        out[f"sal_{name}"] = sal.astype(np.float32)

        in_p = float(sal[:, pmask].mean()) if pmask.any() else 0.0
        out_p = float(sal[~pmask].mean())
        enrichment = in_p / max(out_p, 1e-12)

        # negative control: shuffle the light trace across trials
        shuf = np.random.default_rng(1).permutation(len(idx))
        from optonet.metrics import sequence_metrics
        ctrl = sequence_metrics(raster, probs[shuf], window_start=T_ONSET)

        # saliency as a function of pulse ordinal position
        prof = []
        for row_i, row in enumerate(trials_sub.itertuples()):
            pulses = json.loads(row.pulses)
            widths = json.loads(row.widths)
            for k, (p, w) in enumerate(zip(pulses, widths)):
                a = int(T_ONSET + p)
                b = min(int(T_ONSET + p + w) + 3, sal.shape[1])
                if b > a:
                    prof.append((row_i, k, sal[row_i, a:b].sum()))
        prof = pd.DataFrame(prof, columns=["trial", "pulse", "sal"])
        prof["cell_type"] = ctype[prof["trial"].to_numpy()]
        prof["rel"] = prof.groupby("trial")["pulse"].transform(lambda s: s / max(s.max(), 1))
        out[f"prof_{name}"] = prof.to_numpy(dtype=np.float32)

        rows.append({
            "model": name, "n_trials": len(idx),
            "sal_enrichment": enrichment,
            "sal_pulse_mean": in_p, "sal_offpulse_mean": out_p,
            "shuffle_bin_auprc": ctrl["bin_auprc"],
            "shuffle_evoked_r2": ctrl["evoked_r2"],
            "bin_auprc": sequence_metrics(raster, probs, window_start=T_ONSET)["bin_auprc"],
            "sal_first_half": float(sal[:, T_ONSET:T_ONSET + 500].mean()),
            "sal_second_half": float(sal[:, T_ONSET + 500:].mean()),
        })

        if name == "glm":
            for ci, ct in enumerate(("pc", "pv", "som")):
                k = model.kernel[ci].detach().numpy().ravel()
                out[f"glm_kernel_{ct}"] = k.astype(np.float32)
                peak = int(np.argmax(np.abs(k)))
                rows.append({"model": "glm", "n_trials": len(idx),
                             "sal_enrichment": np.nan, "sal_pulse_mean": np.nan,
                             "sal_offpulse_mean": np.nan, "shuffle_bin_auprc": np.nan,
                             "shuffle_evoked_r2": np.nan, "bin_auprc": np.nan,
                             "sal_first_half": np.nan, "sal_second_half": np.nan,
                             "kernel_type": ct, "kernel_peak_lag_ms": peak,
                             "kernel_peak_val": float(k[peak]),
                             "kernel_area": float(k.sum())})
        if name == "transformer":
            roll = attention_rollout(model, light, cell,
                                     cfg["models"]["transformer"]["patch_ms"])
            out["attn_rollout"] = roll.astype(np.float32)
            full = sequence_metrics(raster, probs, window_start=T_ONSET)
            rows.append({"model": name, "n_trials": len(idx),
                         "sal_enrichment": enrichment, "sal_pulse_mean": in_p,
                         "sal_offpulse_mean": out_p,
                         "shuffle_bin_auprc": ctrl["bin_auprc"],
                         "shuffle_evoked_r2": ctrl["evoked_r2"],
                         "bin_auprc": full["bin_auprc"],
                         "sal_first_half": float(sal[:, T_ONSET:T_ONSET + 500].mean()),
                         "sal_second_half": float(sal[:, T_ONSET + 500:].mean()),
                         "attn_distant_ratio": float(
                             roll[-1, :3].sum() / max(roll[-1, 3:].sum(), 1e-12))})
        print(f"[interp] {name}: enrichment={enrichment:.2f} "
              f"shuffle_auprc={ctrl['bin_auprc']:.4f}", flush=True)

    np.savez_compressed(repo_path("results", "interp.npz"), **out)
    pd.DataFrame(rows).to_csv(repo_path("results", "tables", "interpretability.csv"),
                              index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
