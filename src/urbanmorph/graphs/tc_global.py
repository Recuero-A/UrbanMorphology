"""Milestone 13: tile-free (global) generation over an L x L domain (docs/M13_METHOD.md §6).

One GenState over the whole domain (one node numbering, one queue, one terrain field); the turtle's 192 m crops are
taken from the global canvas; streets are clipped only at the outer boundary. The start head, trained on 512 m tiles,
is applied to 512 m WINDOWS (a 3 x 3 grid for L = 1536 m) in a seeded random order: when the queue is empty, the next
window that has not ended proposes a start or ends itself. The model's position features use the node's position
inside the 512 m cell that contains it (an approximation: training positions are tile-relative).
"""
from __future__ import annotations

import numpy as np
import torch

from urbanmorph.graphs.generate import MAX_COMPONENTS, MAX_NODES, GenState
from urbanmorph.graphs.sequence import CELL, GRID, PAD
from urbanmorph.graphs.tc_potentials import TerrainField
from urbanmorph.graphs.tc_sampler import TCSampler
from urbanmorph.graphs.terrain_generate import TerrainPolicy
from urbanmorph.graphs.terrain_sequence import tile_canvas
from urbanmorph.terrain.heightfield import Terrain

W = 512.0


class GlobalTCSampler(TCSampler):
    def __init__(self, model, big: dict, L: float, names, norm, pots, cfg, device="cpu", crop_big: dict | None = None):
        """``big``: terrain dict on the 4 m grid of [-256, L + 256]^2 (row 0 = north), keys elev / valid / water."""
        self.L = float(L)
        super().__init__(model, [big], names, norm, pots, cfg, device, crop_tiles=[crop_big] if crop_big else None)
        t = Terrain(np.where(big["valid"], big["elev"], np.nan).astype(np.float32), -256.0, self.L + 256.0, 4.0)
        t.water_kind = big["water"].astype(np.int8)
        self.tfs = [TerrainField(t)]
        n = int(self.L // W)
        self.windows = [(c * W, r * W) for r in range(n) for c in range(n)]       # (x0, y0) world, south-west corner
        lay = self.pol.layers[0]
        self.wcanv = [tile_canvas(lay, self.pol.names, norm, window=(int((self.L - y0 - W) / 4), int(x0 / 4))) if self.pol.names
                      else np.zeros((0, 128, 128), np.float16) for x0, y0 in self.windows]

    def local_pos(self, p):
        q = np.asarray(p, float) - np.floor(np.asarray(p, float) / W) * W
        return np.clip(q, 0, W - 1e-6)

    def new_state(self):
        s = GenState(size=self.L, max_nodes=MAX_NODES * len(self.windows))
        s.win_done = np.zeros(len(self.windows), bool)
        s.win_ncomp = np.zeros(len(self.windows), int)
        s.win_order = list(self._rng.permutation(len(self.windows)))
        return s

    def run(self, seed: int = 0, max_rounds: int = 6000):
        self._rng = np.random.default_rng(seed + 17)
        return super().run(seed, max_rounds)

    def do_starts(self, need, gen, rng):
        model = self.model
        for i, p, s in need:
            todo = [w for w in s.win_order if not s.win_done[w]]
            if not todo or s.n >= s.max_nodes:
                s.done = True
                continue
            w = todo[0]
            x0, y0 = self.windows[w]
            r0, c0 = int((self.L - y0 - W) / 4), int(x0 / 4)
            canvas = torch.from_numpy((s.edge_map[PAD + r0:PAD + r0 + GRID, PAD + c0:PAD + c0 + GRID] < s.t + 0.5)[None, None]).to(self.device)
            ncomp = torch.tensor([float(s.win_ncomp[w])], device=self.device)
            tcan = torch.from_numpy(self.wcanv[w][None]).to(self.device)
            stop_logit, loc = model.start_outputs(canvas, ncomp, tcanvas=tcan)
            self.n_fwd += 1
            stp = bool(torch.rand(1, generator=gen).item() < torch.sigmoid(stop_logit).cpu().item())
            cell = int(torch.multinomial(torch.softmax(loc, -1).cpu(), 1, generator=gen).item())
            s.win_order.remove(w)
            s.win_order.append(w)                                           # round robin over open windows
            if stp or s.win_ncomp[w] >= MAX_COMPONENTS:
                s.win_done[w] = True
                continue
            row, col = divmod(cell, GRID // CELL)
            px = x0 + (col * CELL + rng.uniform(0, CELL)) * 4.0
            py = y0 + W - (row * CELL + rng.uniform(0, CELL)) * 4.0
            s.win_ncomp[w] += 1
            s.ncomp += 1
            self.place_start(i, s, px, py)


def big_terrain(scenario_params, L: float) -> dict:
    """A procedural terrain dict over [-256, L + 256]^2 for global generation (synthetic stress tests)."""
    from urbanmorph.terrain.synthesis import generate
    t = generate(scenario_params, -256.0, -256.0, L + 256.0, L + 256.0)
    return {"elev": t.elev.astype(np.float32), "valid": np.ones(t.elev.shape, bool), "water": t.water_kind.astype(np.int8)}


def window_tiles(big: dict, L: float) -> list[dict]:
    """The 512 m windows of a big terrain dict as ordinary tile dicts (each with its 256 m margin)."""
    n = int(L // W)
    out = []
    H = big["elev"].shape[0]
    for r in range(n):
        for c in range(n):
            x0, y0 = c * W, r * W
            r0 = int((L - y0 - W) / 4)
            c0 = int(x0 / 4)
            sl = (slice(r0, r0 + 256), slice(c0, c0 + 256))
            out.append({k: v[sl] for k, v in big.items()})
    assert H == int(L / 4) + 128
    return out
