"""Milestone 3 model tests (CPU, synthetic tensors, fast)."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from urbanmorph.models import evaluate as ev
from urbanmorph.models.data import dihedral
from urbanmorph.models.losses import beta_at, kl_divergence, pos_weight_from_frequencies, reconstruction_loss
from urbanmorph.models.train import load_checkpoint, run_epoch, save_checkpoint
from urbanmorph.models.vae import ConvVAE, Decoder, Encoder, count_parameters, reparameterize


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(0)
    return (torch.rand(4, 3, 128, 128, generator=g) > 0.7).float()


def test_encoder_output_dimensions(batch):
    mu, logvar = Encoder(16)(batch)
    assert mu.shape == (4, 16) and logvar.shape == (4, 16)
    mu, logvar = Encoder(16, variational=False)(batch)
    assert mu.shape == (4, 16) and logvar is None


def test_decoder_output_dimensions():
    out = Decoder(32)(torch.randn(5, 32))
    assert out.shape == (5, 3, 128, 128)


def test_vae_forward_and_parameter_count(batch):
    m = ConvVAE(32)
    logits, mu, logvar = m(batch)
    assert logits.shape == batch.shape and mu.shape == (4, 32)
    assert count_parameters(m) == 3_665_891  # documented in docs/MILESTONE_3.md
    assert m.sample(3, device="cpu").shape == (3, 3, 128, 128)
    p = m.sample(2, device="cpu")
    assert (p >= 0).all() and (p <= 1).all()


def test_reparameterize_statistics_and_gradients():
    torch.manual_seed(0)
    mu = torch.full((20000, 2), 3.0, requires_grad=True)
    logvar = torch.full((20000, 2), math.log(4.0), requires_grad=True)  # sigma = 2
    z = reparameterize(mu, logvar)
    assert z.mean().item() == pytest.approx(3.0, abs=0.05)
    assert z.std().item() == pytest.approx(2.0, abs=0.05)
    z.sum().backward()
    assert mu.grad is not None and logvar.grad is not None  # gradients flow through the sample


def test_kl_divergence_known_values():
    zero = torch.zeros(3, 8)
    assert kl_divergence(zero, zero).item() == pytest.approx(0.0)  # posterior == prior
    mu = torch.ones(1, 2)
    # KL(N(1,1)||N(0,1)) = 0.5 per dim
    assert kl_divergence(mu, torch.zeros(1, 2)).item() == pytest.approx(1.0)
    # KL(N(0, s^2)||N(0,1)) = 0.5 (s^2 - 1 - ln s^2)
    lv = torch.full((1, 1), math.log(4.0))
    assert kl_divergence(torch.zeros(1, 1), lv).item() == pytest.approx(0.5 * (4 - 1 - math.log(4)))


def test_reconstruction_loss_matches_manual_bce(batch):
    logits = torch.randn_like(batch)
    total, per_ch = reconstruction_loss(logits, batch)
    manual = F.binary_cross_entropy(torch.sigmoid(logits), batch, reduction="sum") / batch.shape[0]
    assert total.item() == pytest.approx(manual.item(), rel=1e-4)
    assert per_ch.shape == (3,) and per_ch.sum().item() == pytest.approx(total.item(), rel=1e-5)
    # positive weighting increases the loss on positives only
    w = pos_weight_from_frequencies([0.1, 0.25, 0.1], "sqrt")
    assert w.flatten().tolist() == pytest.approx([3.0, math.sqrt(3), 3.0], rel=1e-5)
    weighted, _ = reconstruction_loss(logits, batch, w)
    assert weighted > total
    all_zero = torch.zeros_like(batch)
    assert reconstruction_loss(logits, all_zero, w)[0].item() == pytest.approx(reconstruction_loss(logits, all_zero)[0].item())


def test_beta_warmup():
    assert beta_at(0, 1.0, 20) == 0.0 and beta_at(10, 1.0, 20) == 0.5 and beta_at(50, 0.5, 20) == 0.5
    assert beta_at(3, 1.0, 0) == 1.0


def test_forward_backward_updates_all_parameters(batch):
    torch.manual_seed(0)
    m = ConvVAE(16)
    logits, mu, logvar = m(batch)
    loss = reconstruction_loss(logits, batch)[0] + kl_divergence(mu, logvar)
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert missing == []


def test_training_epochs_reduce_loss_and_are_deterministic_on_cpu(batch):
    def fit():
        torch.manual_seed(1)
        m = ConvVAE(8)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        dl = DataLoader(TensorDataset(batch, torch.arange(4)), batch_size=4)
        first = run_epoch(m, dl, "cpu", None, 0.1, opt)["loss"]
        for _ in range(5):
            last = run_epoch(m, dl, "cpu", None, 0.1, opt)["loss"]
        return first, last
    a, b = fit(), fit()
    assert a[1] < a[0]
    assert a == pytest.approx(b)


def test_checkpoint_roundtrip(tmp_path, batch):
    cfg = {"model": {"latent_dim": 16, "variational": True}, "experiment": {"name": "t", "seed": 0}}
    m = ConvVAE(16).eval()
    save_checkpoint(tmp_path / "best.pt", m, cfg, epoch=7, val_loss=1.5, extra={"note": "x"})
    m2, ck = load_checkpoint(tmp_path / "best.pt", "cpu")
    assert ck["epoch"] == 7 and ck["extra"]["note"] == "x" and ck["config"] == cfg
    with torch.no_grad():
        assert torch.allclose(m(batch)[0], m2(batch)[0])


def test_dihedral_group():
    x = torch.arange(16.0).view(1, 4, 4)
    imgs = [dihedral(x, k) for k in range(8)]
    assert len({tuple(i.flatten().tolist()) for i in imgs}) == 8  # all distinct
    assert torch.equal(dihedral(dihedral(x, 1), 3), x)            # rotations invert
    assert torch.equal(dihedral(dihedral(x, 4), 4), x)            # mirror is an involution


def test_iou_coverage_and_components():
    t = np.zeros((1, 3, 8, 8), bool)
    p = np.zeros((1, 3, 8, 8), bool)
    t[0, 0, :, :4] = True
    p[0, 0, :, :2] = True
    iou = ev.iou_per_tile(p, t)
    assert iou[0, 0] == pytest.approx(0.5) and iou[0, 1] == 1.0  # empty vs empty counts as 1
    cov = ev.coverage_errors(p, t)
    assert cov["bias"][0] == pytest.approx(-0.25)
    roads = np.zeros((1, 20, 20), bool)
    roads[0, 2, :] = True          # one long road
    roads[0, 10:12, 5:15] = True   # a second piece
    roads[0, 18, 18] = True        # a 1-pixel fragment
    lab = ev.label_components(roads[0])
    assert lab.max() == 3
    stats = ev.road_connectivity(roads)
    assert stats["components_median"] == 2 and stats["tiny_fragments_median"] == 1


def test_trivial_baseline_and_latent_probes():
    rng = np.random.default_rng(0)
    train = (rng.random((10, 3, 16, 16)) < 0.2).astype(np.float32)
    tgt = (rng.random((5, 3, 16, 16)) < 0.2).astype(np.float32)
    tb = ev.trivial_baseline(train, tgt, 0.5)
    assert tb["roads"]["iou"] == 0.0  # constant p < 0.5 never predicts a positive
    z_tr = np.concatenate([rng.normal(0, 0.1, (20, 2)), rng.normal(5, 0.1, (20, 2))])
    y_tr = ["a"] * 20 + ["b"] * 20
    assert ev.knn_accuracy(z_tr, y_tr, np.array([[0, 0], [5, 5]]), ["a", "b"]) == 1.0
    t = z_tr[:, 0] * 2 + 1
    assert ev.ridge_r2(z_tr, t, z_tr, t, lam=1e-6) == pytest.approx(1.0, abs=1e-6)


def test_layout_dataset_uses_existing_splits():
    from urbanmorph.data.m2_dataset import INDEX_PATH
    if not INDEX_PATH.exists():
        pytest.skip("M2 dataset not built")
    from urbanmorph.models.data import LayoutDataset
    tr, va = LayoutDataset("train"), LayoutDataset("val")
    assert tr.x.shape[1:] == (3, 128, 128) and tr.x.dtype == torch.float32
    assert set(torch.unique(tr.x).tolist()) <= {0.0, 1.0}
    assert set(tr.rows.sample_id).isdisjoint(va.rows.sample_id)
    assert (tr.rows.split == "train").all() and (va.rows.split == "val").all()
    aug = LayoutDataset("train", augment=True, seed=0)
    x0, i0 = aug[0]
    assert x0.shape == (3, 128, 128) and x0.sum() == tr.x[i0].sum()  # symmetry preserves pixel counts
