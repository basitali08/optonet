"""OptoNet interactive demo.

Draw a light protocol, predict the spike train with the trained OptoFormer,
and compare against the ground-truth biophysical simulator.

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / "src"))

from optonet import simulate as S
from optonet.config import load_config, repo_path
from optonet.protocols import sample_protocol

st.set_page_config(page_title="OptoNet demo", layout="wide")

@st.cache_resource
def load_bundle():
    from optonet.data import load
    return load()


@st.cache_resource
def load_model(name: str):
    import torch
    from optonet.models import build_model
    cfg = load_config()
    ck = repo_path("results", "checkpoints", f"{name}_seed11.pt")
    if not ck.exists():
        return None, cfg
    ckd = torch.load(ck, map_location="cpu", weights_only=False)
    per_cell = np.concatenate(
        [load_bundle().onehot().astype(np.float32),
         load_bundle().norm_cell_matrix().astype(np.float32)], axis=1)
    model = build_model(ckd["model"], per_cell.shape[1], cfg, cond=ckd["cond"])
    model.load_state_dict(ckd["state_dict"])
    model.eval()
    return model, cfg


def protocol_light(pulses, widths, intensity, n_steps=S.N_STEPS, onset=200):
    return S.rasterize_pulses([pulses], [widths], np.array([intensity]),
                              n_steps=n_steps, t_onset=onset)[0]


def protocol_light_1ms(pulses, widths, intensity, n_bins=1500, onset=200):
    light = np.zeros(n_bins, dtype=np.float32)
    for p, w in zip(pulses, widths):
        a = onset + int(round(float(p)))
        b = min(a + max(int(round(float(w))), 1), n_bins)
        if a < n_bins:
            light[a:b] = intensity
    return light


def predict_spikes(model, light1ms, cell_feats):
    import torch
    with torch.no_grad():
        lt = torch.from_numpy(light1ms[None, :])
        cf = torch.from_numpy(cell_feats[None, :])
        p = torch.sigmoid(model(lt, cf)).numpy()[0]
    return p


def simulate_ground_truth(bundle, cell_id, pulses, widths, intensity, seed=0):
    cell = bundle.cells.iloc[cell_id]
    arrays = {}
    for k in ("C", "gL", "EL", "VT", "dVT", "a", "b", "tau_w", "g_opto", "I0",
              "std_ou"):
        arrays[k] = np.array([float(cell[k])])
    ct = cell["cell_type"]
    arrays["VR"] = np.array([S.CELL_TYPE_SPECS[ct].VR])
    arrays["tau_ou"] = np.array([S.CELL_TYPE_SPECS[ct].tau_ou])
    light = protocol_light(pulses, widths, intensity)
    ind, times, _ = S.simulate_chunk(arrays, light, seed=seed)
    return times * S.DT


st.title("🔬 OptoNet — design & predict optogenetic responses")

bundle = load_bundle()
cfg = load_config()
st.caption("Deep learning predicts and designs neuron responses to optogenetic "
           "stimulation. Prediction = trained model; ground truth = AdEx + ChR2 simulator.")

model, _ = load_model("transformer")
if model is None:
    st.warning("No trained checkpoint found. Run `python scripts/03_train_sequence.py` "
               "first — the demo will still show the ground-truth simulation.")
else:
    st.success("Loaded OptoFormer checkpoint (seed 11).")

# ---- protocol controls ----
c1, c2, c3 = st.columns(3)
with c1:
    cell_type = st.selectbox("Cell type", ["pc", "pv", "som"])
    pool = bundle.cells[(bundle.cells["cell_type"] == cell_type) &
                        (bundle.cells["cell_split"] == "test")]
    cell_row = int(pool.iloc[st.slider("cell index", 0, len(pool) - 1, 0)].name)
    cell_feats = np.concatenate([
        bundle.onehot()[cell_row:cell_row + 1],
        bundle.norm_cell_matrix()[cell_row:cell_row + 1]], axis=1).astype(np.float32)

with c2:
    family = st.selectbox("Protocol", ["regular", "step", "rand", "doublet",
                                       "burst", "chirp", "sweep", "sparse",
                                       "chirp_hi"])
    intensity = st.slider("light intensity (model units)", 0.0, 2.0, 0.8, 0.05)

with c3:
    n = st.slider("draw length (pulses)", 1, 120, 20)
    freq = st.slider("frequency (Hz)", 1.0, 120.0, 20.0, 1.0)
    width = st.slider("pulse width (ms)", 1.0, 20.0, 5.0, 1.0)
    chirp_to = st.slider("sweep target freq (Hz)", 5.0, 120.0, 60.0, 1.0)
    seed = st.slider("random seed", 0, 999, 0)

if family == "step":
    pulses, widths = np.zeros(1), np.array([max(n * 1000.0 / max(freq, 1), 10.0)])
elif family in ("chirp", "chirp_hi", "sweep"):
    from optonet.protocols import _chirp_times
    dur = min(n * 1000.0 / max(freq, 1), 1100.0)
    if family == "sweep":
        up = _chirp_times(max(freq, 5.0), chirp_to, dur / 2)
        down = dur / 2 + _chirp_times(chirp_to, max(freq, 5.0), dur / 2)
        pulses = np.concatenate([up, down])
    else:
        pulses = _chirp_times(max(freq, 2.0), chirp_to, dur)
    widths = np.full(len(pulses), width)
else:
    if family == "doublet":
        pulses = np.array([0.0, max(n * 1000.0 / max(freq, 1), 5.0)])
    elif family == "burst":
        ibf = 5.0
        period = max(1000.0 / ibf, n * 1000.0 / max(freq, 1) + 5)
        pulses = np.concatenate([np.arange(n) * (1000.0 / freq) + i * period
                                 for i in range(3)])
    else:
        pulses = np.arange(0, min(n * (1000.0 / freq), 1100.0), 1000.0 / freq)
    widths = np.full(len(pulses), width)

# ---- simulate ----
light01 = protocol_light(pulses, widths, intensity)
light1ms = protocol_light_1ms(pulses, widths, intensity)[None, :]
gt_times = simulate_ground_truth(bundle, cell_row, pulses, widths, intensity, seed=seed)
pred_prob = predict_spikes(model, light1ms, cell_feats) if model is not None else None

# ---- plot ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1.4, 1.2]})
t = np.arange(1500)
axes[0].step(t, light1ms[0], where="post", color="#2b6cb0", lw=1.0)
axes[0].set_ylabel("light")
axes[0].set_title(f"{cell_type.upper()} cell #{cell_row} — {family}, "
                  f"{len(pulses)} pulses, I={intensity}")

r = st.empty()
if pred_prob is not None:
    axes[1].plot(t, pred_prob * 1000, color="#c05621", lw=1.0)
    axes[1].set_ylabel("P(spike)\n(x1000)")
    axes[1].set_ylim(0, max(pred_prob.max() * 1400, 1e-3))
    pred_count = float(pred_prob[:, 200:].sum())
else:
    axes[1].set_ylabel("prediction")
    pred_count = float("nan")

for sp in gt_times:
    axes[2].vlines(sp, 0, 1, color="#2f855a", lw=0.7)
axes[2].set_ylabel("ground truth")
axes[2].set_ylim(0, 1)
axes[2].set_xlabel("time (ms)")
for ax in axes:
    ax.axvline(200, color="0.6", ls="--", lw=0.7)
st.pyplot(fig)

gt_count = float((gt_times >= 200).sum())
m1, m2, m3 = st.columns(3)
m1.metric("Ground-truth spikes", f"{gt_count:.0f}")
m2.metric("Predicted spikes", "n/a" if np.isnan(pred_count) else f"{pred_count:.1f}")
m3.metric("Light dose", f"{intensity * width * len(pulses):.0f}")

st.markdown("---")
st.caption("Reference: Brette & Gerstner (2005) AdEx; Nagel et al. (2003) / "
           "Foutz et al. (2012) ChR2 kinetics. Full benchmark: see scripts/ and paper/main.tex.")
