"""Milestone 15: optional, composable street-style guidance for the K-candidate samplers (docs/M15_METHOD.md).

Style guidance is one more multiplicative factor on each candidate's weight, applied on top of any existing sampler
(M*, M13-SMC, NA-*) through ``TCSampler.candidates``; with style "off" the sampler is unchanged bit for bit.

Heading fields (a cross field: a street aligned with the field direction OR perpendicular to it is preferred):
* ``grid``     a constant direction theta0 (the declared grid orientation);
* ``organic``  a spatially smooth, variable direction: theta0 + amplitude x a sum of 6 low-frequency random Fourier
               modes (wavelengths in [scale, 2 scale]), keyed by (site, inference seed) so it is stable and replayable;
* ``radial``   the direction from a declared centre (default: the tile centre); with the cross field this prefers
               radial AND tangential (ring) streets, as in spoke-and-ring layouts.
Mixtures (``mix:grid=0.5+organic=0.5``) weight the per-field log-factors.

For a candidate edge a -> q with axial direction theta at the edge midpoint m:
    log w_style = gain[class] * kappa * sum_j w_j * (cos(4 (theta - theta_j(m))) - 1) / 2      in [-kappa, 0]
so an aligned (or perpendicular) street keeps its weight and a 45-degree street loses kappa nats. ``gain`` guides
local streets fully and the connecting backbone less (classes major, mid, local, minor).

Selection normalisation (``style_norm``, default on): within each decision the style log-factors of the feasible
candidates are shifted by their log-mean, so E_k[w_style] = 1. Style then changes WHICH continuation is chosen but not
the arm's stop probability (an un-normalised factor <= 1 would shrink the stop normaliser Z and make styled networks
shorter, i.e. "succeed" by accepting fewer roads). Termination is controlled separately (``stop_bias``). With the
normalisation the target is decision-local, so no sampler, SMC included, is exact for a fixed tilt once style is on.

Cul-de-sac controls, independent of the heading field and combinable with any of them:
* local termination: ``stop_bias`` (SamplerConfig) is added to every arm's stop logit;
* loop closure: a candidate that joins the component it starts from (a merge, T-junction or crossing that closes a
  cycle) loses gain[class] * ``style_loop`` nats.

Grade ceiling (optional, ``grade_cap``): a ground edge whose maximum 30 m sub-span grade exceeds the cap is
infeasible (reason ``grade_cap``). "real_p99" uses the class-specific p99 of real streets (configs/m12_feasibility.json);
a number applies to every class. Off by default: M13 kept steep grades soft because real streets are built on steep land.
"""
from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass

import numpy as np

from urbanmorph.config import PROJECT_ROOT

FIELDS = ("grid", "organic", "radial")
CLOSING = ("merge_node", "t_junction", "crossing_split", "crossing_endpoint")
TILE_CENTRE = (256.0, 256.0)


def parse_mix(style: str) -> dict[str, float]:
    """'off' -> {}; 'grid' -> {'grid': 1}; 'mix:grid=0.5+organic=0.5' -> normalised weights."""
    if style in ("off", "", "culdesac"):
        return {}
    if style in FIELDS:
        return {style: 1.0}
    if not style.startswith("mix:"):
        raise ValueError(f"unknown style {style!r}")
    w = {}
    for part in style[4:].split("+"):
        k, v = part.split("=")
        if k not in FIELDS:
            raise ValueError(f"unknown style field {k!r}")
        w[k] = float(v)
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}


def stable_key(*parts) -> int:
    """A 32-bit seed from (site, seed, ...): stable across runs, machines and Python hash randomisation."""
    return zlib.crc32("|".join(map(str, parts)).encode())


def grade_caps(spec) -> dict[int, float] | None:
    if spec in (None, "", "off"):
        return None
    if spec == "real_p99":
        g = json.loads((PROJECT_ROOT / "configs" / "m12_feasibility.json").read_text())["grade_max"]
        return {int(k): float(v) for k, v in g.items()}
    return {c: float(spec) for c in range(4)}


@dataclass
class StyleField:
    """The style factor of one tile (built once per tile and inference seed)."""
    weights: dict
    kappa: float
    theta0: float                  # radians
    centre: tuple
    gain: tuple
    loop: float
    modes: np.ndarray | None = None   # organic: (6, 4) = (kx, ky, phase, amplitude)
    amp: float = 0.0

    @classmethod
    def build(cls, cfg, site: str, seed: int) -> "StyleField":
        w = parse_mix(cfg.style)
        modes = None
        if "organic" in w:
            rng = np.random.default_rng(stable_key("organic", site, seed, cfg.style_key))
            lam = rng.uniform(cfg.style_scale_m, 2 * cfg.style_scale_m, 6)
            ang = rng.uniform(0, 2 * math.pi, 6)
            amp = rng.normal(0, 1, 6)
            amp /= max(np.sqrt((amp ** 2).sum() / 2), 1e-9)                 # unit RMS of the field
            modes = np.c_[np.cos(ang) / lam, np.sin(ang) / lam, rng.uniform(0, 2 * math.pi, 6), amp]
        cx = TILE_CENTRE[0] if not np.isfinite(cfg.style_cx) else cfg.style_cx
        cy = TILE_CENTRE[1] if not np.isfinite(cfg.style_cy) else cfg.style_cy
        gain = tuple(float(x) for x in cfg.style_class_gain.split(","))
        return cls(w, cfg.style_kappa, math.radians(cfg.style_theta0), (cx, cy), gain, cfg.style_loop, modes,
                   math.radians(cfg.style_amp_deg))

    @property
    def active(self) -> bool:
        return bool(self.weights and self.kappa) or bool(self.loop)

    def direction(self, kind: str, m: np.ndarray) -> float:
        """Field direction (radians) of one heading field at point m (tile metres)."""
        if kind == "grid":
            return self.theta0
        if kind == "radial":
            return math.atan2(m[1] - self.centre[1], m[0] - self.centre[0])
        x = self.modes
        return self.theta0 + self.amp * float((x[:, 3] * np.sin(2 * math.pi * (x[:, 0] * m[0] + x[:, 1] * m[1]) + x[:, 2])).sum())

    def heading_lw(self, a: np.ndarray, q: np.ndarray, cls: int) -> float:
        d = q - a
        if not self.weights or not self.kappa or d @ d < 1.0:
            return 0.0
        th = math.atan2(d[1], d[0])
        m = (a + q) / 2
        s = sum(w * (math.cos(4 * (th - self.direction(k, m))) - 1) / 2 for k, w in self.weights.items())
        return self.gain[min(max(cls, 0), 3)] * self.kappa * s

    def loop_lw(self, closes_loop: bool, cls: int) -> float:
        return -self.gain[min(max(cls, 0), 3)] * self.loop if (closes_loop and self.loop) else 0.0


def closes_loop(s, u: int, o: dict, comp: np.ndarray) -> bool:
    """Does committing ``o`` from node u join u's own component (i.e. close a cycle)?"""
    if o["kind"] not in CLOSING:
        return False
    if "v" in o and o["v"] >= 0:
        return bool(comp[o["v"]] == comp[u])
    if "edge" in o:
        return bool(comp[o["edge"][0]] == comp[u])
    return False


# frozen M15 style presets (docs/M15_PROTOCOL.md; strengths frozen on development terrain before confirmation)
STYLE_PRESETS = {
    "off": dict(style="off"),
    "grid": dict(style="grid"),
    "organic": dict(style="organic"),
    "radial": dict(style="radial"),
    "culdesac": dict(style="off"),
}
