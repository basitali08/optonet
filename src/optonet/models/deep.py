"""Deep sequence models: dilated CNN, Conv-LSTM and the OptoFormer."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import SpikeGLM


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor,
                  bias: Optional[torch.Tensor], dilation: int) -> torch.Tensor:
    k = weight.shape[-1]
    return F.conv1d(F.pad(x, ((k - 1) * dilation, 0)), weight, bias,
                    dilation=dilation)


class CausalBlock(nn.Module):
    """Residual causal conv block with position-wise LayerNorm (no cross-time
    normalization statistics, so no future leakage)."""

    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, channels, kernel))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dilation: int) -> torch.Tensor:
        h = causal_conv1d(x, self.weight, self.bias, dilation)
        h = h.transpose(1, 2)
        h = self.norm(h).transpose(1, 2)
        h = self.drop(F.gelu(h))
        return x + h


class SpikeCNN(nn.Module):
    """Causal dilated CNN over 1 kHz light traces with cell conditioning.

    A strided stem reduces the sequence to `stem_stride` ms frames (stimulus
    lookahead <= stem_stride - 1 ms); dilated blocks then operate on frames
    and the logits are upsampled back to 1 kHz.
    """

    def __init__(self, n_cell_feats: int, channels: int = 32, kernel: int = 5,
                 dropout: float = 0.1, n_blocks: int = 7, cond: str = "full",
                 stem_stride: int = 8, stem_channels: int = 48):
        super().__init__()
        self.cond = cond
        self.stride = stem_stride
        self.n_cell_feats = n_cell_feats
        in_ch = 1 + (n_cell_feats if cond == "full" else (3 if cond == "type" else 0))
        self.stem = nn.Conv1d(in_ch, stem_channels, kernel_size=stem_stride,
                              stride=stem_stride)
        self.blocks = nn.ModuleList([
            CausalBlock(channels, kernel, 2 ** i, dropout) for i in range(n_blocks)])
        self.up = nn.Conv1d(stem_channels, channels, 1)
        self.head = nn.Conv1d(channels, 1, 1)

    def forward(self, light: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        b, t = light.shape
        if self.cond == "full":
            feats = cell
        elif self.cond == "type":
            feats = cell[:, :3]
        else:
            feats = None
        x = light.unsqueeze(1)
        if feats is not None:
            x = torch.cat([x, feats.unsqueeze(2).expand(b, feats.shape[1], t)], dim=1)
        pad = (-t) % self.stride
        if pad:
            x = F.pad(x, (0, pad))
        h = self.up(self.stem(x))
        for i, blk in enumerate(self.blocks):
            h = blk(h, 2 ** i)
        return self.head(h).squeeze(1).repeat_interleave(self.stride, dim=-1)[:, :t]


class SpikeLSTM(nn.Module):
    """Strided causal convolutional front-end + LSTM, logits upsampled to 1 kHz."""

    def __init__(self, n_cell_feats: int, hidden: int = 64, layers: int = 1,
                 stem_stride: int = 8, dropout: float = 0.1, cond: str = "full",
                 stem_channels: int = 48):
        super().__init__()
        self.cond = cond
        self.stride = stem_stride
        self.hidden = hidden
        self.stem = nn.Conv1d(1, stem_channels, kernel_size=stem_stride,
                              stride=stem_stride)
        self.lstm = nn.LSTM(stem_channels, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        n_cond = n_cell_feats if cond == "full" else (3 if cond == "type" else 0)
        self.cell_emb = nn.Linear(n_cond, stem_channels) if n_cond else None
        self.head = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.n_cell_feats = n_cell_feats

    def forward(self, light: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        b, t = light.shape
        s = self.stride
        if self.cond == "full":
            feats = cell
        elif self.cond == "type":
            feats = cell[:, :3]
        else:
            feats = None
        pad = (-t) % s
        x = F.pad(light.unsqueeze(1), (0, pad))
        h = self.stem(x).transpose(1, 2)                 # [b, T/s, stem_channels]
        if feats is not None and self.cell_emb is not None:
            h = h + self.cell_emb(feats).unsqueeze(1)
        h, _ = self.lstm(h)
        logits = self.head(self.drop(h)).squeeze(-1)     # [b, T/s]
        return logits.repeat_interleave(s, dim=-1)[:, :t]


class OptoFormer(nn.Module):
    """Transformer encoder over light patches with FiLM cell conditioning.

    Each patch of `patch_ms` milliseconds is embedded, modulated by a
    cell-type / cell-parameter embedding (FiLM), processed by a transformer
    encoder and decoded back to per-millisecond spike logits.
    """

    def __init__(self, n_cell_feats: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 256,
                 patch_ms: int = 16, dropout: float = 0.1, cond: str = "full"):
        super().__init__()
        self.cond = cond
        self.patch = patch_ms
        self.n_cell_feats = n_cell_feats
        self.sub_pos = nn.Parameter(torch.zeros(1, 1, patch_ms))
        self.proj = nn.Linear(patch_ms, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 400, d_model))
        n_cond = n_cell_feats if cond == "full" else 3
        self.film = nn.Sequential(nn.Linear(n_cond, d_model * 2), nn.GELU(),
                                  nn.Linear(d_model * 2, d_model * 2))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, patch_ms)
        nn.init.zeros_(self.pos)
        nn.init.zeros_(self.sub_pos)

    def forward(self, light: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        b, t = light.shape
        s = self.patch
        pad = (s - t % s) % s
        x = F.pad(light.unsqueeze(1), (0, pad)) if pad else light.unsqueeze(1)
        n_patches = x.shape[-1] // s
        v = x.reshape(b, n_patches, s) + self.sub_pos
        tok = self.proj(v) + self.pos[:, :n_patches]
        feats = cell if self.cond == "full" else cell[:, :3]
        gamma, beta = self.film(feats).chunk(2, dim=-1)
        tok = tok * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        causal = torch.triu(torch.full((n_patches, n_patches), float("-inf"),
                                       dtype=tok.dtype), diagonal=1)
        tok = self.encoder(tok, mask=causal)
        return self.head(tok).reshape(b, -1)[:, :t]


MODEL_REGISTRY = {
    "glm": SpikeGLM,
    "cnn": SpikeCNN,
    "lstm": SpikeLSTM,
    "transformer": OptoFormer,
}


def build_model(name: str, n_cell_feats: int, cfg: dict, cond: str = "full"):
    if name == "glm":
        return SpikeGLM(n_cell_feats=n_cell_feats, n_lags=512)
    mcfg = cfg["models"][name]
    kwargs = {k: v for k, v in mcfg.items()
              if k in ("channels", "kernel", "dropout", "hidden", "layers",
                       "d_model", "nhead", "num_layers", "dim_feedforward",
                       "patch_ms", "stem_stride", "stem_channels", "n_blocks")}
    kwargs.setdefault("dropout", 0.1)
    return MODEL_REGISTRY[name](n_cell_feats=n_cell_feats, cond=cond, **kwargs)
