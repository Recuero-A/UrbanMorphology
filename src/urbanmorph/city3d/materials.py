"""Procedural PBR material library (no downloaded textures): base colour (linear RGB), roughness, metallic,
and an optional procedural-texture recipe the Blender stage builds from nodes (docs/M11_RENDERING_PIPELINE.md).
Palettes are small and muted so neighbouring buildings vary in a controlled way, never at random."""

from __future__ import annotations


def srgb(h: str) -> list[float]:
    """'#rrggbb' -> linear RGB (Blender / glTF base colours are linear)."""
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return [round(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4, 4) for x in c]


FACADE_FAMILIES = {
    "plaster": ["#e8dcc6", "#efe6d6", "#e2cfb0", "#d9c3a5", "#f1ece2", "#e6d3c3"],
    "brick": ["#9b5a43", "#a8664d", "#8a4f3c", "#b07a5f", "#7f4a39"],
    "stone": ["#cbbfa8", "#bfb39c", "#d4cab6", "#b8ad98"],
    "concrete": ["#b9b7b2", "#a9a7a2", "#c6c4bf", "#9d9b97"],
    "wood_siding": ["#d8d2c4", "#c9d1d3", "#e3dccb", "#b9c4c0", "#d6c7b0"],
}
FACADE_ROUGHNESS = {"plaster": 0.85, "brick": 0.9, "stone": 0.8, "concrete": 0.9, "wood_siding": 0.75}
TEXTURE = {"plaster": "noise", "brick": "brick", "stone": "noise", "concrete": "noise", "wood_siding": "stripes"}

ROOF = {
    "flat": [("#8d8a85", 0.9), ("#a19e98", 0.9), ("#7a7874", 0.9)],          # concrete / bitumen
    "tile": [("#a4553a", 0.8), ("#8f4b35", 0.8), ("#6e6a66", 0.8), ("#4f5257", 0.75)],   # terracotta, slate-grey
}

GROUND = {
    "asphalt": ("#3b3c3e", 0.95, "noise"),
    "ground": ("#b5ad9c", 0.95, "noise"),           # bare / paved open land
    "pavement": ("#c7c3ba", 0.9, "tiles"),
    "grass_public": ("#5f8a3c", 0.95, "noise"),
    "grass_private": ("#6d8f4a", 0.95, "noise"),
    "glass": ("#3d4a55", 0.08, None),
    "window_frame": ("#e8e6e1", 0.6, None),
    "door": ("#5a3e2b", 0.7, None),
    "marking": ("#e9e7df", 0.7, None),
    "tree_trunk": ("#5b4636", 0.9, None),
    "tree_leaves": ("#4c7a34", 0.85, None),
    "metal": ("#4a4d50", 0.45, None),
}


class MaterialTable:
    """Name -> index, with the parameters the Blender stage needs."""

    def __init__(self):
        self.names, self.params = [], []

    def get(self, name: str) -> int:
        if name in self.names:
            return self.names.index(name)
        kind, *rest = name.split(":")
        if kind == "facade":
            fam, v = rest[0], int(rest[1])
            p = {"base_color": srgb(FACADE_FAMILIES[fam][v]), "roughness": FACADE_ROUGHNESS[fam], "texture": TEXTURE[fam]}
        elif kind == "roof":
            fam, v = rest[0], int(rest[1])
            h, r = ROOF[fam][v]
            p = {"base_color": srgb(h), "roughness": r, "texture": "tiles" if fam == "tile" else "noise"}
        else:
            h, r, t = GROUND[kind]
            p = {"base_color": srgb(h), "roughness": r, "texture": t}
        p.update(name=name, metallic=0.0)
        self.names.append(name)
        self.params.append(p)
        return len(self.names) - 1
