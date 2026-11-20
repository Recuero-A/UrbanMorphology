"""The single registry of named sampler configurations used in the papers' comparisons.

Every method is the SAME trained backbone; only inference differs. Names, what they are, and what they control for:

| Name        | Preset (graphs/*)      | Settings                         | Role                                                 |
|-------------|------------------------|----------------------------------|------------------------------------------------------|
| A           | tc_sampler standard    | K = 1, no rules                  | unconstrained ancestral sampling                     |
| M*          | tc_sampler energy      | K = 8, hard rules, terrain pots  | M13 terrain-aware baseline                           |
| M*+repair   | as M* + graphs/repair  | post-hoc A*-style connection     | the simple alternative to network-aware inference    |
| P           | procedural_terrain     | density-matched to training      | rule-based procedural baseline                       |
| M13-SMC     | tc_sampler smc         | M* proposal, N = 4 particles     | the effect of SMC alone (no network potentials)      |
| NA-myopic   | m14_sampler na_myopic  | M* + network potentials          | network potentials without look-ahead                |
| NA-fixed    | m14_sampler na_fixed   | + 2-step look-ahead everywhere   | network potentials + look-ahead (M15 main sampler)   |
| NA-adaptive | m14_sampler na_adaptive| look-ahead where trigger fires   | adaptive allocation (M14 H3: not better than random) |
| NA-random   | m14_sampler na_adaptive| random trigger, same firing rate | control for NA-adaptive                              |
| NA-SMC      | m14_sampler na_smc     | network potentials, N = 4        | network potentials + SMC                             |

Milestone 15 adds optional style guidance on top of any of these (``with_style``); with style "off" the configuration
is byte-identical to the base method.

The network-potential weights (lam_dead, lam_island) are the moment-matched values in configs/m14_lambda.json; the
NA-adaptive trigger and NA-random firing rate are frozen in configs/m14_frozen.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from urbanmorph.config import PROJECT_ROOT

CFG = PROJECT_ROOT / "configs"
LAM = CFG / "m14_lambda.json"
FROZEN = CFG / "m14_frozen.json"


@dataclass(frozen=True)
class Method:
    label: str
    preset: str                       # tc_sampler.PRESETS or m14_sampler.NET_PRESETS key, or "P"
    overrides: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)   # run-level options, e.g. {"repair": True}

    def as_tuple(self):
        return self.label, self.preset, dict(self.overrides), dict(self.extra)


def lam(mult=1.0, dead=True, island=True) -> dict:
    """Network-potential weights (moment-matched on training tiles), optionally scaled or switched off."""
    L = json.loads(LAM.read_text())
    return {"lam_dead": round(L["lam_dead"] * mult, 4) if dead else 0.0,
            "lam_island": round(L["lam_island"] * mult, 4) if island else 0.0}


def frozen() -> dict:
    return json.loads(FROZEN.read_text())


def confirmatory_methods(fz: dict | None = None, include_slow: bool = True) -> list[Method]:
    """The M14 confirmatory method set, in the M14 order (run IDs hash these exact settings)."""
    fz = frozen() if fz is None else fz
    m = [Method("A", "standard"), Method("M*", "energy", {"K": 8, "lam_ratio": 0.5}),
         Method("M*+repair", "energy", {"K": 8, "lam_ratio": 0.5}, {"repair": True}),
         Method("P", "P"),
         Method("NA-myopic", "na_myopic", lam()),
         Method("NA-adaptive", "na_adaptive", {**lam(), "trigger": fz["trigger"]}),
         Method("NA-random", "na_adaptive", {**lam(), "trigger": f"random:{fz['fire_share']:.4f}"})]
    if include_slow:
        m += [Method("NA-fixed", "na_fixed", lam()), Method("NA-SMC", "na_smc", lam()),
              Method("M13-SMC", "smc", {"K": 8, "lam_ratio": 0.5, "particles": 4})]
    return m


def replay_methods() -> list[Method]:
    """Milestone 16 replays of the M12 generation loop (graphs/generate.sample_graphs, the learned start head): without
    rules, and with the M12 hard rules + replanning (TerrainPolicy "HR": per-class grade limits, redraw, halve, contour,
    far bank), the configuration of the M12 TN+HR evaluation and of every M12 demo city."""
    return [Method("M12-loop", "legacy"), Method("M12-HR", "legacy", {"legacy_policy": "HR"})]


def method(label: str, fz: dict | None = None) -> Method:
    for m in confirmatory_methods(fz) + replay_methods():
        if m.label == label:
            return m
    raise KeyError(label)
