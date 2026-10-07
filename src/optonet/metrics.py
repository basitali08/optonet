"""Evaluation metrics for spike-train prediction."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)


def count_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    out = {
        "r2": float(r2_score(y_true, y_pred)) if np.std(y_true) > 0 else 0.0,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": float(spearmanr(y_true, y_pred).statistic)
        if np.std(y_pred) > 0 else 0.0,
    }
    return out


def detection_metrics(y_true: np.ndarray, probs: np.ndarray,
                      threshold: Optional[float] = None) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    out = {
        "auroc": float(roc_auc_score(y_true, probs)),
        "auprc": float(average_precision_score(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
    }
    if threshold is None:
        grid = np.unique(np.quantile(probs, np.linspace(0.5, 0.999, 60)))
        f1s = [f1_score(y_true, probs >= g, zero_division=0) for g in grid]
        threshold = float(grid[int(np.argmax(f1s))])
    out["f1"] = float(f1_score(y_true, probs >= threshold, zero_division=0))
    out["threshold"] = float(threshold)
    return out


def sequence_metrics(y_bin: np.ndarray, probs: np.ndarray,
                     window_start: int = 200) -> Dict[str, float]:
    """Bin-level and trial-level metrics for 1 kHz spike rasters."""
    y_bin = y_bin.astype(np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    flat_y = (y_bin.ravel() > 0).astype(int)
    flat_p = probs.ravel()
    out = {
        "bin_auroc": float(roc_auc_score(flat_y, flat_p)),
        "bin_auprc": float(average_precision_score(flat_y, flat_p)),
        "bin_brier": float(brier_score_loss(flat_y, flat_p)),
    }
    y_count = y_bin[:, window_start:].sum(axis=1)
    p_count = probs[:, window_start:].sum(axis=1)
    out.update({f"evoked_{k}": v for k, v in count_metrics(y_count, p_count).items()})
    y_count_all = y_bin.sum(axis=1)
    p_count_all = probs.sum(axis=1)
    out.update({f"total_{k}": v for k, v in count_metrics(y_count_all, p_count_all).items()})

    ker = np.hanning(41)
    ker /= ker.sum()
    rs = []
    for i in range(y_bin.shape[0]):
        yy = np.convolve(y_bin[i], ker, mode="same")
        pp = np.convolve(probs[i], ker, mode="same")
        if yy.std() > 1e-9 and pp.std() > 1e-9:
            rs.append(np.corrcoef(yy, pp)[0, 1])
    out["psth_r"] = float(np.mean(rs)) if rs else 0.0
    return out


def coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def interval_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(np.asarray(hi) - np.asarray(lo)))
