"""Milestone 3b: spatial VAE, Dice/combined loss, raster road-network metrics, expansion invariants."""

import numpy as np
import pandas as pd
import pytest
import torch

from urbanmorph.metrics.road_raster import prune_spurs, road_network_metrics, zhang_suen_thin
from urbanmorph.models.losses import combined_reconstruction, kl_divergence, reconstruction_loss, soft_dice
from urbanmorph.models.train import build_model, load_checkpoint, save_checkpoint
from urbanmorph.models.vae import SpatialVAE, count_parameters


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(0)
    return (torch.rand(2, 3, 128, 128, generator=g) > 0.7).float()


def test_spatial_vae_shapes_params_and_backward(batch):
    m = SpatialVAE(16)
    logits, mu, logvar = m(batch)
    assert logits.shape == batch.shape and mu.shape == (2, 1024) and logvar.shape == (2, 1024)
    assert m.encode(batch)[0].shape == (2, 16, 8, 8)
    assert count_parameters(m) == 2_122_435
    loss = reconstruction_loss(logits, batch)[0] + kl_divergence(mu, logvar)
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())
    assert m.decoder(torch.randn(3, 1024)).shape == (3, 3, 128, 128)
    assert m.sample(2, device="cpu").shape == (2, 3, 128, 128)


def test_build_model_and_checkpoint_for_spatial(tmp_path, batch):
    cfg = {"model": {"type": "spatial", "latent_channels": 8, "latent_dim": 32, "variational": True}}
    m = build_model(cfg).eval()
    assert isinstance(m, SpatialVAE) and m.latent_shape == (8, 8, 8)
    save_checkpoint(tmp_path / "s.pt", m, cfg, 1, 0.0)
    m2, _ = load_checkpoint(tmp_path / "s.pt")
    with torch.no_grad():
        assert torch.allclose(m(batch)[0], m2(batch)[0])


def test_soft_dice_and_combined_loss(batch):
    big = 20.0
    perfect = torch.where(batch > 0, big, -big)  # logits matching the target
    assert torch.allclose(soft_dice(perfect, batch), torch.ones(2, 3), atol=1e-3)
    wrong = -perfect
    assert (soft_dice(wrong, batch) < 0.01).all()
    logits = torch.randn_like(batch)
    plain, _ = combined_reconstruction(logits, batch)
    assert plain.item() == pytest.approx(reconstruction_loss(logits, batch)[0].item())
    _, per_w = combined_reconstruction(logits, batch, channel_weights=[3.0, 1.0, 1.0])
    _, per_p = reconstruction_loss(logits, batch)
    assert per_w[0].item() == pytest.approx(3 * per_p[0].item(), rel=1e-5) and per_w[1].item() == pytest.approx(per_p[1].item())
    with_dice, _ = combined_reconstruction(logits, batch, dice_weight=1.0)
    assert with_dice > plain


def test_thinning_gives_one_pixel_lines():
    m = np.zeros((40, 40), bool)
    m[18:24, 2:38] = True  # 6-px-wide horizontal bar
    s = zhang_suen_thin(m)
    assert s.sum() > 20 and (s.sum(axis=0) <= 1).all()  # at most one pixel per column
    assert prune_spurs(s, 3).sum() == s.sum() - 6         # both ends shortened by 3 px


def test_road_network_metrics_plus_vs_blob():
    plus = np.zeros((128, 128), bool)
    plus[60:68, :] = True
    plus[:, 60:68] = True
    r = road_network_metrics(plus)
    assert r["skeleton_components"] == 1
    assert r["intersections_per_km2"] == pytest.approx(1 / 0.262144)
    assert 25 < r["mean_road_width_m"] < 45  # the bar is 32 m wide
    yy, xx = np.mgrid[:128, :128]
    blob = (yy - 64) ** 2 + (xx - 64) ** 2 < 40 ** 2
    assert road_network_metrics(blob)["mean_road_width_m"] > 1000  # a blob is not a street
    assert road_network_metrics(np.zeros((128, 128), bool))["road_px"] == 0


def test_expansion_is_train_only_and_respects_splits():
    from urbanmorph.data.m2_dataset import EXPANSION_INDEX_PATH, load_index
    from urbanmorph.splits.geographic import block_ids, edge_distance
    from urbanmorph.sampling.study_area import load_config
    if not EXPANSION_INDEX_PATH.exists():
        pytest.skip("expansion not built")
    cfg = load_config()
    base = load_index()
    exp = pd.read_csv(EXPANSION_INDEX_PATH)
    ds = base[base.in_dataset]
    held = ds[ds.split.isin(["val", "test"])]
    assert (exp.split == "train").all() and exp.usable.all()
    assert set(exp.sample_id).isdisjoint(ds.sample_id)
    assert not set(block_ids(exp, cfg["splits"]["block_size_m"])) & set(block_ids(held, cfg["splits"]["block_size_m"]))
    for code, g in exp.groupby("city_code"):
        h = held[held.city_code == code]
        d = min(edge_distance(a, b) for _, a in g.iterrows() for _, b in h.iterrows())
        assert d >= cfg["splits"]["buffer_m"]
    # the original split assignment is unchanged: 97 / 19 / 19
    assert ds.split.value_counts().to_dict() == {"train": 97, "val": 19, "test": 19}
