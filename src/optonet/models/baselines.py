"""Classical baselines: gradient-boosted trees and a Poisson-style GLM."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikeGLM(nn.Module):
    """Cell-type specific causal linear-nonlinear encoding model.

    rate_t = softplus( (K_type * light)_{<=t} + bias_type + w^T cell )
    with a learned causal kernel K over the last `n_lags` ms.
    """

    def __init__(self, n_cell_feats: int, n_types: int = 3, n_lags: int = 512):
        super().__init__()
        self.n_lags = n_lags
        self.n_types = n_types
        self.kernel = nn.Parameter(0.1 * torch.randn(n_types, 1, n_lags))
        self.bias = nn.Parameter(torch.zeros(n_types))
        self.global_bias = nn.Parameter(torch.zeros(1))
        n_extra = max(n_cell_feats - n_types, 0)
        self.feat_w = nn.Linear(n_extra, 1) if n_extra else None
        if self.feat_w is not None:
            nn.init.zeros_(self.feat_w.bias)

    def forward(self, light: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        b, t = light.shape
        x = F.pad(light.unsqueeze(1), (self.n_lags - 1, 0))
        conv = F.conv1d(x, self.kernel)                       # [b, n_types, t]
        oh = cell[:, :self.n_types]
        y = (conv * oh.unsqueeze(-1)).sum(dim=1)
        y = y + (self.bias * oh).sum(dim=1, keepdim=True) + self.global_bias
        if self.feat_w is not None and cell.shape[1] > self.n_types:
            y = y + self.feat_w(cell[:, self.n_types:])
        return y.squeeze(1)


class GBMRegressor:
    """Thin wrapper around sklearn histogram gradient boosting."""

    def __init__(self, **kwargs):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.model = HistGradientBoostingRegressor(**kwargs)

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.model.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)


class GBMClassifier:
    def __init__(self, **kwargs):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.model = HistGradientBoostingClassifier(**kwargs)

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.model.fit(x, y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]
