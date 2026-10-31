"""Milestone 10: buildings for every block of a street graph with BlockGen (docs/M10_BLOCK_GENERATION.md §4).

Blocks come from ``extract_blocks`` (road surface from the class widths). Each block is drawn in its canonical
window(s); window k of every block is sampled in one batch, then window k + 1 with the buildings placed so far
(and the area already covered) as the known region: outpainting. Generated pixels are vectorised only outside
the known region, clipped to the block, and the pieces of one block are merged at the end.
"""

from __future__ import annotations

import numpy as np
import shapely
import torch

from urbanmorph.layout.block_canvas import from_canvas, to_canvas, window_region, windows, block_frame
from urbanmorph.layout.blocks import extract_blocks
from urbanmorph.layout.objects import MIN_BUILDING_M2

BATCH = 96


def generate_buildings(model, graphs, cities, class_widths, device="cpu", seed=0, steps=50) -> list[list]:
    """One list of building polygons per (graph, city index) pair."""
    gen = torch.Generator().manual_seed(seed)
    items = []                                            # (tile, block polygon, frame, windows)
    for ti, g in enumerate(graphs):
        for b in extract_blocks(g, class_widths):
            fr = block_frame(b["polygon"])
            items.append({"tile": ti, "poly": b["polygon"], "frame": fr, "wins": windows(fr), "pieces": [], "covered": []})
    k = 0
    while True:
        todo = [it for it in items if len(it["wins"]) > k]
        if not todo:
            break
        for s in range(0, len(todo), BATCH):
            part = todo[s:s + BATCH]
            conds = []
            for it in part:
                fr, win = it["frame"], it["wins"][k]
                blk = to_canvas([it["poly"]], fr, win)
                known_m = to_canvas([shapely.union_all(it["covered"]).intersection(it["poly"])], fr, win) if it["covered"] else np.zeros_like(blk)
                known_b = to_canvas(it["pieces"], fr, win) if it["pieces"] else np.zeros_like(blk)
                conds.append(np.stack([blk, known_b * known_m, known_m]).astype(np.float32))
            cond = torch.from_numpy(np.stack(conds)).to(device)
            city = torch.as_tensor([cities[it["tile"]] for it in part], device=device)
            out = model.sample(cond, city, steps=steps, generator=gen).cpu().numpy()[:, 0]
            for it, o, c in zip(part, out, conds):
                fr, win = it["frame"], it["wins"][k]
                new = (o > 0.5) & (c[2] < 0.5)
                it["pieces"] += from_canvas(new, fr, win, clip=it["poly"])
                it["covered"].append(window_region(fr, win))
        k += 1
    out = [[] for _ in graphs]
    for it in items:
        if not it["pieces"]:
            continue
        u = shapely.union_all([p.buffer(0.05) for p in it["pieces"]]).buffer(-0.05).intersection(it["poly"])
        out[it["tile"]] += [q for q in getattr(u, "geoms", [u]) if q.geom_type == "Polygon" and q.area >= MIN_BUILDING_M2]
    return out
