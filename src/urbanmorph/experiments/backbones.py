"""Trained backbones (Milestone 12 TerrainGraphTurtle checkpoints) and the terrain each one is shown.

Backbones: ``TN_s{0,1,2}`` (raw DEM), ``TNlp_s*`` (DEM low-passed at 16 m), ``TNse_s*`` (DEM with street beds erased).
The low-pass / street-erased variants are leakage controls: urban bare-earth DEMs keep the imprint of real streets.
Only the MODEL input changes; feasibility rules and metrics always use the original terrain.
"""
from __future__ import annotations

import numpy as np
import torch

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.terrain_sequence import channel_list, lowpass
from urbanmorph.models.terrain_graph_turtle import TerrainGraphTurtle

CK = PROJECT_ROOT / "checkpoints" / "m12"
SE = PROJECT_ROOT / "data" / "processed" / "m13" / "terrain_se"


def load_backbone(run: str, device):
    """(model, terrain channel names, feature normaliser, training config) of checkpoints/m12/<run>/best.pt."""
    ck = torch.load(CK / run / "best.pt", map_location=device, weights_only=False)
    c = ck["config"]
    m = TerrainGraphTurtle(c["model"]["hidden"], c["model"]["n_mix"], c["model"]["n_classes"],
                           n_terrain=len(channel_list(c["terrain"]))).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, c["terrain"], ck["norm"], c


GLO30 = (PROJECT_ROOT / "data" / "processed" / "m18" / "glo30_eval", PROJECT_ROOT / "data" / "processed" / "m18" / "terrain_glo30")


def glo30_elev(sid):
    """GLO-30 elevation of a window (scripts/m18_glo30_eval.py for evaluation windows, m18_build_glo30_terrain.py for
    the training tiles); a window without one is an error, never a silent fallback to LiDAR."""
    for d in GLO30:
        f = d / f"{sid}.npz"
        if f.exists():
            return np.load(f)["elev"]
    raise FileNotFoundError(f"no GLO-30 terrain for {sid}")


def model_inputs(cfg, sids, ds):
    """What the MODEL sees: low-passed or street-erased terrain; rules and metrics keep ``ds``. None = ``ds`` itself."""
    if cfg.get("dem_variant") == "glo30":                       # M18 TNg: Copernicus GLO-30 on the same grid
        return [{**d, "elev": glo30_elev(s)} for s, d in zip(sids, ds)]
    if cfg.get("dem_variant"):
        f = cfg["dem_variant"].split("_")[1]
        return [{**d, "elev": np.load(SE / f / f"{s}.npz")["elev"]} for s, d in zip(sids, ds)]
    if cfg.get("lowpass_m"):
        return [lowpass(d, float(cfg["lowpass_m"])) for d in ds]
    return None
