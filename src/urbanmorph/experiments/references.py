"""Shared reference quantities of the Milestone 12 evaluation, as a library (previously imported from
scripts/m12_evaluate.py by later runners; the values are identical).

* ``TH``: the street feasibility thresholds (configs/m12_feasibility.json), used by the procedural baseline P and by
  post-hoc repair;
* ``median_density()``: median intersections per km2 of the TRAINING street graphs, P's density target.
"""
from __future__ import annotations

import pickle
from functools import lru_cache

import numpy as np

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs.road_graph import graph_metrics
from urbanmorph.terrain.road_feasibility import Thresholds

TH = Thresholds.load(PROJECT_ROOT / "configs" / "m12_feasibility.json")
GRAPHS3D = PROJECT_ROOT / "data" / "processed" / "m12" / "graphs3d.pkl"


@lru_cache(maxsize=1)
def median_density() -> float:
    dev = pickle.loads(GRAPHS3D.read_bytes())                   # always the TRAINING distribution
    return float(np.median([graph_metrics(g)["intersections_per_km2"] for _, _, g in dev["train"]]))
