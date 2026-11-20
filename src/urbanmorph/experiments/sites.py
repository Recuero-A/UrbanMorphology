"""Milestone 15 sites: synthetic terrain interventions (docs/M15_PROTOCOL.md §2).

Every site is a terrain dict (graphs/fixtures.py format) plus a stable ``site`` key. Development and evaluation sites
are different instances of the same conditions (other azimuths and parameters), so tuning never sees an evaluation
case. Conditions: flat, incline (8 % and 15 %), ridge, valley, water barrier.

Negative control (``mismatch``): the sampler is given a comparable site with the terrain rotated by 90 degrees about
the tile centre (a different incline / ridge / valley / river direction, same statistics); metrics are always computed
on the TRUE terrain. Flat sites have no mismatch partner and are excluded from that control.
"""
from __future__ import annotations

from urbanmorph.graphs import fixtures as fx

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


def _site(name: str, p: dict, rotate: float = 0.0) -> dict:
    p = dict(p)
    kind = p.pop("kind")
    if rotate and "az" in p:
        p["az"] = (p["az"] + rotate) % 360.0
    return {**fx.terrain(kind, **p), "site": name}


def site_set(name: str) -> list[dict]:
    """``syn_dev`` / ``syn_eval``: one site per condition; ``syn_eval_mismatch``: the eval sites the SAMPLER is shown
    under the negative control (flat excluded); ``syn_eval_nonflat``: their true counterparts, same order."""
    if name == "syn_dev":
        return [_site(f"dev_{c}", _DEV[c]) for c in CONDITIONS]
    if name == "syn_eval":
        return [_site(f"eval_{c}", _EVAL[c]) for c in CONDITIONS]
    if name == "syn_eval_nonflat":
        return [_site(f"eval_{c}", _EVAL[c]) for c in CONDITIONS if c != "flat"]
    if name == "syn_eval_mismatch":
        return [{**_site(f"eval_{c}", _EVAL[c], rotate=90.0), "site": f"eval_{c}"} for c in CONDITIONS if c != "flat"]
    raise KeyError(name)


def condition(site: str) -> str:
    return site.split("_", 1)[1]
