# Methods

This document specifies the forward model, the benchmark construction and the
training protocol in enough detail to reimplement them independently.

## 1. Forward model

Units: milliseconds (ms), millivolts (mV), picofarads (pF), nanosiemens (nS),
picoamperes (pA). Integration: forward Euler, Δt = 0.1 ms, trial length
1500 ms, stimulus onset at 200 ms.

### 1.1 Neuron (AdEx)

```
C dV/dt = -g_L (V - E_L) + g_L Δ_T exp((V - V_T)/Δ_T) - w + I_bg + I_op
τ_w dw/dt = a (V - E_L) - w
if V ≥ 0:  V ← V_r ;  w ← w + b
```

All updates within a step use the pre-update state, so the reset cannot
contaminate the adaptation variable.

### 1.2 Opsin (two-state ChR2)

```
m∞(L)   = L² / (L² + K²),  K = 0.35          (Hill n = 2)
τ_m(L)  = 3.0 − 1.2 m∞  ms
dm/dt   = (m∞ − m)/τ_m
h∞(L)   = 1 − 0.65 m∞
τ_h     = 700 ms if L > 0 else 2500 ms
dh/dt   = (h∞ − h)/τ_h
σ(V)    = 1 − 0.75 / (1 + exp(−(V + 40)/6))
I_op    = g_opsin · m · h · σ(V) · (0 − V)
```

`m` is channel availability, `h` is the slow desensitisation/recovery
variable (the mechanism behind light-induced depression in real ChR2), and
`σ` implements voltage-dependent deactivation that terminates the photocurrent
during spikes. Light intensity `L` in model units maps to the experimental
intensity parameter as `L = intensity × 0.6`, so intensity 1.0 gives
m∞ ≈ 0.75.

### 1.3 Background drive

Ornstein–Uhlenbeck noise: `dI = −I/τ dt + σ√(2/τ) dW`, giving a stationary
standard deviation equal to `σ`. The mean drive `I0` of each cell type is
calibrated by bisection so that the median spontaneous rate matches a target
(pc ≈ 3 Hz, pv ≈ 8 Hz, som ≈ 4 Hz).

### 1.4 Cell types and heterogeneity

| type | C (pF) | g_L (nS) | E_L | V_T | Δ_T | a (nS) | b (pA) | τ_w (ms) | g_opsin (nS) |
|------|--------|----------|-----|-----|-----|--------|--------|----------|--------------|
| pc   | 200    | 20       | −70  | −50 | 2.5 | 3.0    | 45     | 100      | 12.0         |
| pv   | 120    | 12       | −70  | −50 | 2.0 | 0.5    | 1      | 15       | 7.0          |
| som  | 140    | 10       | −70.5| −52 | 3.0 | 3.0    | 15     | 45       | 5.5         |

Per cell, every parameter is multiplied by an independent log-normal factor
(CV 10–40 %), `E_L` and `V_T` receive additive Gaussian jitter, and `I0` is
scaled by the calibrated per-type factor (pc 1.191, pv 1.178, som 1.387). A
model therefore cannot ignore cell identity.

## 2. Stimulus families

| family | description | split |
|--------|-------------|-------|
| regular | trains, frequency log-uniform 1–40 Hz, width 2–10 ms | train |
| step | single pulse, width 10–800 ms | train |
| rand | gamma-distributed ISIs, rate 1–40 Hz, CV 0.1–1.6 | train |
| doublet | two pulses, ISI 5–400 ms | train |
| burst | 1–8 bursts × 2–8 pulses, inter-burst period ≥ burst duration | train |
| chirp | exponential sweep 2→(40–60) Hz | OOD |
| sweep | triangle sweep 5–60 Hz | OOD |
| sparse | very low rate (0.5–8 Hz), CV 1.8–3.5 | OOD |
| chirp_hi | exponential sweep to 70–120 Hz | OOD |

Intensity is log-uniform in 0.15–1.0 for training protocols; the intensity OOD
split uses 1.2–2.0. The frequency OOD split uses 50–120 Hz. Pulses are clipped
to the first 1100 ms after onset and to 220 pulses per trial; pulse times are
sorted and non-overlapping by construction.

## 3. Splits

900 cells (300 per type): 200/40/60 per type assigned to train/val/test.
Trials: 36,000 train, 7,000 val, 7,000 ID test (test cells, training families),
5,000 OOD-protocol, 3,000 OOD-frequency, 3,000 OOD-intensity. Cells are disjoint
across splits, so no cell appears in two splits.

## 4. Training

Loss: binary cross-entropy over all 1500 ms at 1 kHz (no positive
re-weighting, so predicted probabilities are calibrated and their sums are the
predicted spike counts). Optimiser AdamW, cosine schedule, gradient-norm
clipping at 1.0, early stopping on validation loss with patience 3. Three seeds
(11, 22, 33) per model. Models are selected by best validation loss.

Hyper-parameters live in `configs/default.yaml`: GLM 512 ms filter, lr 2e-2,
15 epochs; CNN 8 ms stem, 32 channels, 7 dilated blocks (dilations 1…64),
kernel 5, dropout 0.1, lr 1e-3, 10 epochs; LSTM 8 ms stem, 96 hidden units,
1 layer, lr 1e-3, 10 epochs; OptoFormer 32 ms patches, d_model 64, 4 heads,
3 layers, FFN 128, dropout 0.1, lr 5e-4, 10 epochs. Batch size 256.

## 5. Metrics

* **Count**: R², MAE, RMSE and Spearman correlation of predicted vs observed
  evoked spike counts (spikes at t ≥ 200 ms).
* **Spike detection**: AUROC and AUPRC over all 1 kHz bins (base rate 0.008),
  plus Brier score.
* **Temporal fidelity**: per-trial Pearson correlation between the 41 ms
  Hanning-smoothed observed raster and the smoothed predicted rate, averaged
  over trials with non-degenerate variance.
* **Uncertainty**: deep-ensemble standard deviation of predicted counts, and
  split-conformal 90 % intervals calibrated on the validation split.
* **Statistics**: 1000-sample bootstrap percentile intervals over trials;
  paired Wilcoxon signed-rank tests on per-trial absolute error with Holm
  correction.
