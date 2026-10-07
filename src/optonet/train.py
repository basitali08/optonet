"""Training and evaluation loops for the sequence (1 kHz) models."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .data import Bundle
from .metrics import sequence_metrics


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


@dataclass
class TensorStore:
    light: torch.Tensor      # uint8 [N, T]
    raster: torch.Tensor     # uint8 [N, T]
    cell: torch.Tensor       # float32 [N, F]

    @property
    def n(self) -> int:
        return int(self.light.shape[0])


def make_store(bundle: Bundle, cond: str = "full") -> TensorStore:
    per_cell = np.concatenate(
        [bundle.onehot().astype(np.float32),
         bundle.norm_cell_matrix().astype(np.float32)], axis=1)
    cell = per_cell[bundle.trials["cell_id"].to_numpy()]
    if cond == "type":
        cell = cell[:, :3]
    elif cond == "none":
        cell = np.zeros((len(cell), 3), dtype=np.float32)
    return TensorStore(light=torch.from_numpy(bundle.light),
                       raster=torch.from_numpy(bundle.raster),
                       cell=torch.from_numpy(np.ascontiguousarray(cell)))


def _batch_loss(model, store, idx, criterion) -> float:
    light = store.light[idx].float().mul_(2.0 / 255.0)
    target = store.raster[idx].float()
    cell = store.cell[idx]
    logits = model(light, cell)
    return float(criterion(logits, target))


@torch.no_grad()
def predict(model, store, idx, batch_size: int = 128) -> np.ndarray:
    model.eval()
    out = np.empty((len(idx), store.light.shape[1]), dtype=np.float32)
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        light = store.light[b].float().mul_(2.0 / 255.0)
        logits = model(light, store.cell[b])
        out[s:s + len(b)] = torch.sigmoid(logits).cpu().numpy()
    return out


def train_sequence(model: nn.Module, store: TensorStore,
                   train_idx: np.ndarray, val_idx: np.ndarray,
                   epochs: int = 20, lr: float = 1e-3, weight_decay: float = 1e-4,
                   batch_size: int = 512, patience: int = 5, seed: int = 0,
                   verbose: bool = False) -> Tuple[nn.Module, Dict, List[dict]]:
    set_seed(seed)
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(epochs, 1))
    rng = np.random.default_rng(seed)
    history: List[dict] = []
    best = {"loss": np.inf, "epoch": -1, "state": None}
    wait = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_idx)
        losses = []
        for s in range(0, len(order), batch_size):
            b = torch.from_numpy(np.ascontiguousarray(order[s:s + batch_size]))
            light = store.light[b].float().mul_(2.0 / 255.0)
            target = store.raster[b].float()
            logits = model(light, store.cell[b])
            loss = criterion(logits, target)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(loss.item())
        sched.step()

        val_loss = float(np.mean([float(_batch_loss(model, store, val_idx[s:s + 4096], criterion))
                                  for s in range(0, len(val_idx), 4096)]))
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "val_loss": val_loss, "lr": optim.param_groups[0]["lr"],
               "time_sec": time.time() - t0}
        history.append(rec)
        if val_loss < best["loss"] - 1e-6:
            best = {"loss": val_loss, "epoch": epoch,
                    "state": copy.deepcopy({k: v.clone() for k, v in model.state_dict().items()})}
            wait = 0
        else:
            wait += 1
        if verbose:
            print(f"  epoch {epoch}: train {rec['train_loss']:.5f} val {val_loss:.5f}",
                  flush=True)
        if wait >= patience:
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, best, history


def evaluate(model, store, trial_ids: np.ndarray, batch_size: int = 512,
             window_start: int = 200) -> Dict[str, float]:
    probs = predict(model, store, trial_ids, batch_size=batch_size)
    y = store.raster[trial_ids].numpy().astype(np.float64)
    return sequence_metrics(y, probs, window_start=window_start)
