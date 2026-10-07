# Using real experimental data

The benchmark shipped with this repository is **simulator-based**: it gives an
exact ground truth for every trial, which is what makes the protocol-design
experiment in `scripts/08_protocol_optimization.py` scoreable against an
oracle. It is not a substitute for real recordings.

This document explains exactly how to point the pipeline at real optogenetic
electrophysiology, so the same models, splits and metrics can be applied
without rewriting anything.

## What a real dataset must contain

| Column / field | Meaning | Units |
|---|---|---|
| `trial_id` | unique trial index | int |
| `cell_id` | unique patched cell index | int |
| `cell_type` | one of `pc`, `pv`, `som`, or your own label mapped onto these | str |
| `intensity` | irradiance at the specimen plane | your units (keep consistent) |
| `t_start_ms` | time of first light pulse relative to trial start | ms |
| `pulse_times_ms` | list of pulse onsets | ms |
| `pulse_widths_ms` | list of pulse widths | ms |
| `spike_times_ms` | list of spike times | ms |

Trials should be 1500 ms with light starting at 200 ms to match the shipped
splits; if yours differ, change `t_total_ms` / `t_onset_ms` in
`configs/default.yaml` and in `src/optonet/simulate.py` — nothing else depends
on those constants.

## Converting a recording into the benchmark format

1. **Spike detection.** Use a published detector (e.g. `mountain` +
   median-threshold, or `spykingcircus`) and keep only well-isolated units
   (e.g. amplitude > 5×MAD, ISI violations < 5 %). Discard cells with fewer
   than ~20 trials.
2. **Light timing.** From the acquisition system (or from an aligned
   photodiode trace, which is preferable), extract pulse onsets and widths.
3. **Cell parameters.** Not strictly required — set `cell_type` from your own
   classification and fill the eleven electrophysiological parameters with
   NaN-free placeholders, or drop them by training with `--cond type`, which
   conditions only on the cell-type indicator.
4. **Splits.** Hold out whole cells, never trials from the same cell, exactly
   as the shipped benchmark does.

## Minimal adapter

```python
import json, numpy as np, pandas as pd

rows = []
for cell_id, cell in enumerate(cells):
    for k, trial in enumerate(cell["trials"]):
        rows.append(dict(
            trial_id=len(rows),
            cell_id=cell_id,
            cell_type=cell["label"],
            intensity=float(trial["irradiance"]),
            t_start_ms=200.0,
            pulse_times_ms=json.dumps(trial["pulse_times_ms"]),
            pulse_widths_ms=json.dumps(trial["pulse_widths_ms"]),
            n_spikes_total=len(trial["spike_times_ms"]),
        ))
trials = pd.DataFrame(rows)
```

Then build the dense arrays that `src/optonet/train.py` consumes:

```python
from optonet.data import rasterize_from_trials  # thin wrapper, see below
light = rasterize_from_trials(trials, n_bins=1500)     # uint8 light, same scaling
raster = binned_spikes(trials, n_bins=1500)           # uint8 spikes per ms
```

`rasterize_from_trials` is a ~10-line wrapper around
`optonet.simulate.rasterize_pulses` that maps the JSON pulse columns into the
same `uint8` light array the models were trained with; the scaling constant
(`intensity/2 × 255`) is the only convention to keep.

## Where to find candidate datasets

* **DANDI archive** (`dandiarchive.org`) — search "optogenetic" or
  "channelrhodopsin"; datasets are publicly downloadable without credentials
  through the S3 endpoint, but individual NWB files are typically large.
* **Allen Brain Observatory** — optogenetic stimulation/inactivation sessions
  with public stimulus and spike tables.
* **CRCNS / OpenNeuro / Figshare** — patch-clamp or two-photon datasets with
  stimulation metadata.

For the OOD protocol split specifically, the practical requirement is that a
cell was stimulated with *more than one* protocol family; most public datasets
are dominated by one or two families, which is precisely why a synthetic
benchmark is useful for evaluating the generalisation question.

## What to expect

On real data the absolute numbers will be lower than reported here — trial
variability is larger, light calibration is imperfect, and drift and
photo-toxicity are unmodelled. What should transfer is the *qualitative*
finding: sequence models degrade gracefully on unseen protocol families while
feature-based models fail, and ensemble spread plus conformal coverage flag the
extrapolation regime.
