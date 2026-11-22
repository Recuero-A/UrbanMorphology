"""Milestone 15 sites: synthetic terrain interventions (docs/M15_PROTOCOL.md §2) and the real-city pilot windows (§5).

Every site is a terrain dict (graphs/fixtures.py format) plus a stable ``site`` key. Development and evaluation sites
are different instances of the same conditions (other azimuths and parameters), so tuning never sees an evaluation
case. Conditions: flat, incline (8 % and 15 %), ridge, valley, water barrier.

Negative control (``mismatch``): the sampler is given a comparable site with the terrain rotated by 90 degrees about
the tile centre (a different incline / ridge / valley / river direction, same statistics); metrics are always computed
on the TRUE terrain. Flat sites have no mismatch partner and are excluded from that control.
"""
from __future__ import annotations

import pickle

import numpy as np

from urbanmorph.config import PROJECT_ROOT
from urbanmorph.graphs import fixtures as fx

REAL = PROJECT_ROOT / "data" / "processed" / "m15" / "real"
GRID_B_MIN_R4 = 0.5          # Task B windows: real orthogonal order >= this (protocol §5)

CONDITIONS = ("flat", "incline8", "incline15", "ridge", "valley", "water")

_DEV = {"flat": dict(kind="flat"),
        "incline8": dict(kind="incline", grade=0.08, az=110.0),
        "incline15": dict(kind="incline", grade=0.15, az=200.0),
        "ridge": dict(kind="ridge", h=45.0, w=160.0, az=75.0),
        "valley": dict(kind="valley", h=45.0, w=160.0, az=165.0),
        "water": dict(kind="water", w=36.0, az=120.0)}
_EVAL = {"flat": dict(kind="flat", base=80.0),
         "incline8": dict(kind="incline", grade=0.08, az=25.0),
         "incline15": dict(kind="incline", grade=0.15, az=300.0),
         "ridge": dict(kind="ridge", h=40.0, w=150.0, az=15.0),
         "valley": dict(kind="valley", h=40.0, w=150.0, az=130.0),
         "water": dict(kind="water", w=40.0, az=40.0)}


# Milestone 16 confirmation sites: fresh instances of the same six conditions (declared before any M16 run; never
# used for development or inspected before the confirmatory protocol is frozen, docs/M16_PROTOCOL.md)
_CONF = {"flat": dict(kind="flat", base=120.0),
         "incline8": dict(kind="incline", grade=0.08, az=160.0),
         "incline15": dict(kind="incline", grade=0.15, az=65.0),
         "ridge": dict(kind="ridge", h=50.0, w=170.0, az=105.0),
         "valley": dict(kind="valley", h=50.0, w=170.0, az=250.0),
         "water": dict(kind="water", w=32.0, az=85.0)}

# Milestone 16 replays: the procedural landscapes of the M12 / M14 demo cities (terrain/synthesis.py; central 512 m tile
# of a 1024 m window, as scripts/m14_demo3d.py builds them)
DEMO_SCENARIOS = (("rolling_hills", 4), ("high_hills", 4), ("steep_hillside", 0), ("mountain_ridges", 0),
                  ("deep_valley", 0), ("mountain_pass", 0))


def _demo(name: str, seed: int) -> dict:
    from urbanmorph.terrain.synthesis import generate, scenario
    t = generate(scenario(name, seed), -256, -256, 768, 768)
    return {"elev": t.elev, "valid": np.ones(t.elev.shape, bool), "water": t.water_kind,
            "water_surface": t.water_surface, "site": f"demo_{name}{seed}"}


def _site(name: str, p: dict, rotate: float = 0.0) -> dict:
    p = dict(p)
    kind = p.pop("kind")
    if rotate and "az" in p:
        p["az"] = (p["az"] + rotate) % 360.0
    return {**fx.terrain(kind, **p), "site": name}


def real_tiles(dem: str = "glo30", countries=None) -> list[dict]:
    """Usable development windows of the real-city pilot (manifest status ok), experiment-driver format. Held-out
    metros are not in the manifest's window list and cannot be returned."""
    import pandas as pd
    m = pd.read_csv(REAL / "manifest.csv")
    graphs = {sid: (g, b) for sid, _, g, b in pickle.loads((REAL / "graphs.pkl").read_bytes())}
    out = []
    for r in m.itertuples():
        if r.status != "ok" or r.role != "development" or (countries and r.country not in countries):
            continue
        f = REAL / f"terrain_{dem}" / f"{r.sample_id}.npz"
        if not f.exists():
            continue
        g, b = graphs[r.sample_id]
        out.append({"sid": r.sample_id, "metro": r.metro, "country": r.country, "stratum": r.stratum, "block": b,
                    "graph": g, "d": {**dict(np.load(f)), "site": r.sample_id}})
    return out


def grid_b(tiles: list[dict]) -> list[dict]:
    """Task B: windows whose REAL network has orthogonal order R4 >= 0.5, with the declared grid orientation taken from
    the real network (the R4 phasor's angle / 4). R4 is then a conditioned quantity, excluded as evidence."""
    from urbanmorph.graphs.style_metrics import _lengths_angles, orth_order
    out = []
    for t in tiles:
        g = t["graph"]
        if not (orth_order(g) >= GRID_B_MIN_R4):
            continue
        L, th, _ = _lengths_angles(g)
        theta0 = float(np.degrees(np.angle((L * np.exp(4j * th)).sum()) / 4))
        out.append({**t, "d": {**t["d"], "style_decl": {"style_theta0": theta0}}, "theta0": theta0})
    return out


def site_set(name: str) -> list[dict]:
    """``syn_dev`` / ``syn_eval``: one site per condition; ``syn_eval_mismatch``: the eval sites the SAMPLER is shown
    under the negative control (flat excluded); ``syn_eval_nonflat``: their true counterparts, same order."""
    if name == "syn_dev":
        return [_site(f"dev_{c}", _DEV[c]) for c in CONDITIONS]
    if name == "syn_eval":
        return [_site(f"eval_{c}", _EVAL[c]) for c in CONDITIONS]
    if name == "syn_conf":
        return [_site(f"conf_{c}", _CONF[c]) for c in CONDITIONS]
    if name == "syn_conf_mismatch":
        return [{**_site(f"conf_{c}", _CONF[c], rotate=90.0), "site": f"conf_{c}"} for c in CONDITIONS if c != "flat"]
    if name == "syn_conf_nonflat":
        return [_site(f"conf_{c}", _CONF[c]) for c in CONDITIONS if c != "flat"]
    if name == "demo_scen":
        return [_demo(n, k) for n, k in DEMO_SCENARIOS]
    if name == "holdout_m17":
        return holdout_m17()
    if name in ("named_glo30", "named_mdt05"):
        return named(name.split("_")[1])
    if name == "syn_eval_nonflat":
        return [_site(f"eval_{c}", _EVAL[c]) for c in CONDITIONS if c != "flat"]
    if name == "real_dev":
        return [t["d"] for t in real_tiles("glo30")]
    if name == "real_dev_spain_mdt05":
        return [t["d"] for t in real_tiles("mdt05", countries=("spain",))]
    if name == "real_dev_spain_glo30":
        return [t["d"] for t in real_tiles("glo30", countries=("spain",))]
    if name == "real_dev_gridB":
        return [t["d"] for t in grid_b(real_tiles("glo30"))]
    if name == "syn_eval_mismatch":
        return [{**_site(f"eval_{c}", _EVAL[c], rotate=90.0), "site": f"eval_{c}"} for c in CONDITIONS if c != "flat"]
    raise KeyError(name)


PACK = PROJECT_ROOT / "data" / "processed" / "m16" / "pack"
NAMED = ("KAIST", "SEL_HBC", "MAD_LAV", "GUA_ESC")


def named(dem: str) -> list[dict]:
    """M16 named real sites (recurring comparisons and demos; configs/m16_pack): terrain only, never street data."""
    out = []
    for w in NAMED:
        f = PACK / "terrain" / f"{w}__{dem}.npz"
        if f.exists():
            out.append({**dict(np.load(f)), "site": f"named_{w}_{dem}"})
    return out


HOLDOUT = PROJECT_ROOT / "data" / "processed" / "m17" / "holdout"


def holdout_m17() -> list[dict]:
    """M17 fresh geographic holdout (configs/m17_holdout.json; terrain only, never street data)."""
    man = __import__("json").loads((PROJECT_ROOT / "configs" / "m17_holdout_windows.json").read_text())["windows"]
    return [{**dict(np.load(HOLDOUT / "terrain" / f"{w['window']}.npz")), "site": f"hold_{w['window']}"} for w in man]


def condition(site: str) -> str:
    return site.split("_", 1)[1]
