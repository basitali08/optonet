import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optonet.config import load_config
from optonet.models import build_model
from optonet.models.baselines import SpikeGLM

CFG = load_config()
N_FEATS = 14
B, T = 4, 1500

CASES = [("glm", 1), ("cnn", 8), ("lstm", 8), ("transformer", 32)]


@pytest.mark.parametrize("name,group", CASES)
def test_output_shape(name, group):
    model = build_model(name, N_FEATS, CFG, cond="full")
    light = torch.rand(B, T) * 2
    cell = torch.randn(B, N_FEATS)
    out = model(light, cell)
    assert out.shape == (B, T)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name,group", CASES)
def test_conditioning_ablation_shapes(name, group):
    for cond, n in (("type", 3), ("none", 3)):
        model = build_model(name, N_FEATS, CFG, cond=cond)
        out = model(torch.rand(B, T) * 2, torch.randn(B, n))
        assert out.shape == (B, T)


@pytest.mark.parametrize("name,group", CASES)
def test_no_future_leakage_beyond_group(name, group):
    """logits up to time t must ignore light strictly after the frame of t."""
    torch.manual_seed(0)
    model = build_model(name, N_FEATS, CFG, cond="full")
    model.eval()
    light = (torch.rand(B, T) * 2)
    cell = torch.randn(B, N_FEATS)
    t = 700
    cutoff = ((t // group) + 1) * group
    with torch.no_grad():
        ref = model(light, cell)[:, :t + 1]
        light2 = light.clone()
        light2[:, cutoff:] = torch.rand_like(light2[:, cutoff:]) * 2
        alt = model(light2, cell)[:, :t + 1]
    assert torch.allclose(ref, alt, atol=1e-4), f"{name}: future leakage"


@pytest.mark.parametrize("name", ["glm", "cnn", "transformer"])
def test_gradients_reach_input(name):
    model = build_model(name, N_FEATS, CFG, cond="full")
    light = torch.rand(2, T, requires_grad=True)
    cell = torch.randn(2, N_FEATS)
    out = model(light, cell)
    out.sum().backward()
    assert light.grad is not None
    assert light.grad.abs().sum() > 0


def test_glm_kernel_is_causal_and_typed():
    m = SpikeGLM(n_cell_feats=14, n_lags=64)
    assert m.kernel.shape == (3, 1, 64)
    light = torch.rand(3, 200) * 2
    cell = torch.zeros(3, 14)
    cell[0, 0] = cell[1, 1] = cell[2, 2] = 1.0
    out = m(light, cell)
    assert out.shape == (3, 200)
    with torch.no_grad():
        r1 = m(light, cell)
        light2 = light.clone()
        light2[:, 100:] = torch.rand_like(light2[:, 100:]) * 2
        r2 = m(light2, cell)
    assert torch.allclose(r1[:, :100], r2[:, :100], atol=1e-5)
    assert not torch.allclose(r1[:, 100:], r2[:, 100:], atol=1e-3)


def test_registry_covers_config():
    cfg = load_config()
    for name in cfg["training"]["sequence_models"]:
        assert name in ("glm", "cnn", "lstm", "transformer")
        build_model(name, N_FEATS, cfg)
