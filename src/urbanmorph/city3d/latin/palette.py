"""Latin Caribbean palette and materials (themes/latin_caribbean/DESIGN.md §2).

Colours were sampled from the three reference images (architectural features/IMG_3376-3378) and corrected for the
sunset light they were painted in, then kept within a renderable albedo range (no channel above ~0.9 in sRGB for
walls). Wall colours are grouped by how often a street should show them; trims are mostly white.
Every material is (hex, roughness, image or None, metres per texture repeat, alpha cut-out). Images are tinted by
the hex colour (glTF baseColorFactor x baseColorTexture), so one grey plaster image serves every wall colour.
"""

from __future__ import annotations

from urbanmorph.city3d.materials import srgb

# walls: (name, hex, weight). Yellow / ochre / coral / orange dominate the references; pink, turquoise and blue
# are the Cartagena accents; cream and white rest the eye.
WALLS = [
    ("amarillo", "#f2c14e", 1.4), ("ocre", "#e9a93f", 1.2), ("naranja", "#e58a3f", 1.0), ("coral", "#e5694a", 1.1),
    ("salmon", "#ef9877", 0.9), ("terracota", "#c9573a", 0.6), ("rosa", "#e0577f", 0.7), ("rosa_palo", "#f0a3a8", 0.5),
    ("turquesa", "#45b3ad", 0.6), ("azul", "#5f97b4", 0.6), ("celeste", "#8ccbe0", 0.4), ("menta", "#9fd1a2", 0.35),
    ("lila", "#b19ad0", 0.25), ("crema", "#f3dfb8", 0.9), ("blanco", "#f3eee4", 0.7),
]
CHURCH_WALLS = [("coral", "#e5694a"), ("amarillo", "#f2c14e"), ("blanco", "#f3eee4"), ("salmon", "#ef9877"), ("ocre", "#e9a93f")]
TRIMS = [("blanco", "#f6f2e9", 0.75), ("crema", "#f1e3c6", 0.2), ("gris", "#c9c2b5", 0.05)]
# plinth (zocalo): contrasting band at the foot of the wall
PLINTHS = [("gris_piedra", "#8f877c"), ("terracota", "#a94d33"), ("azul_oscuro", "#2f5d7c"), ("verde_oscuro", "#2f5e48"),
           ("blanco", "#ece6da"), ("ocre_oscuro", "#b5792d")]
# carpentry: doors, shutters, window grilles and wooden balconies
WOODS = [("madera", "#8a4a24"), ("madera_clara", "#c46a32"), ("madera_oscura", "#5a2f18"), ("verde", "#2f7a5a"),
         ("turquesa", "#2e9c9a"), ("azul", "#2f6f9e"), ("blanco", "#efeadf")]
ROOF_TILES = [("teja", "roof_tiles_a.png"), ("teja_vieja", "roof_tiles_b.png")]
DOMES = [("cupula_teja", "#ffffff", "dome_tiles.png"), ("cupula_azul", "#5b8fa6", "plaster.png"),
         ("cupula_amarilla", "#efc04b", "plaster.png"), ("cupula_blanca", "#f3eee4", "plaster.png")]

# name -> (hex, roughness, image, metres per repeat, alpha cut-out)
BASE = {
    "trim": ("#f6f2e9", 0.7, "plaster.png", 3.0, False),
    "glass": ("#26323a", 0.12, None, 1.0, False),
    "void": ("#1d1a18", 0.95, None, 1.0, False),              # open arches of belfries, deep shadow
    "iron": ("#1f1f1f", 0.45, None, 1.0, False),
    "iron_railing": ("#262626", 0.5, "iron_railing.png", 1.0, True),
    "stone": ("#d9cfbe", 0.85, "plaster.png", 3.0, False),
    "roof_flat": ("#c9b49a", 0.9, "pavers.png", 3.0, False),  # azotea (roof terrace) in clay pavers
    "cobble": ("#ffffff", 0.95, "cobble.png", 3.0, False),
    "asphalt": ("#4a4643", 0.95, "asphalt.png", 6.0, False),
    "pavement": ("#ffffff", 0.9, "pavers.png", 2.4, False),
    "plaza": ("#ffffff", 0.9, "brick_plaza.png", 2.4, False),
    "marking": ("#eeeae0", 0.7, None, 1.0, False),
    "grass_public": ("#ffffff", 0.95, "grass.png", 5.0, False),
    "grass_private": ("#e8f0d8", 0.95, "grass.png", 5.0, False),
    "ground": ("#ffffff", 0.95, "grass.png", 8.0, False),              # tropical open land: grass, earth patches (soil.png)
    "jungle_floor": ("#ffffff", 0.95, "jungle_floor.png", 10.0, False),
    "sand": ("#ffffff", 0.95, "sand.png", 6.0, False),
    "rock": ("#8b8173", 0.95, "plaster.png", 6.0, False),
    "water": ("#2a8c9a", 0.04, None, 1.0, False),
    "bridge_concrete": ("#c8bfb0", 0.85, "plaster.png", 4.0, False),
    "lamp": ("#ffd9a0", 0.3, None, 1.0, False),
    "gold": ("#c9a443", 0.35, None, 1.0, False),
    "trunk_coconut": ("#ffffff", 0.9, "palm_trunk.png", 1.0, False),
    "trunk_royal": ("#d8d4cc", 0.8, "plaster.png", 2.0, False),
    "crownshaft": ("#6f9a3c", 0.6, None, 1.0, False),
    "bark": ("#6b5443", 0.9, "plaster.png", 2.0, False),
    "coconut": ("#6e7a2c", 0.6, None, 1.0, False),
    "frond": ("#ffffff", 0.7, "palm_frond.png", 1.0, True),
    "frond_dry": ("#c8a466", 0.8, "palm_frond.png", 1.0, True),
    "fan_leaf": ("#ffffff", 0.7, "fan_leaf.png", 1.0, True),
    "banana_leaf": ("#ffffff", 0.6, "banana_leaf.png", 1.0, True),
    "leaves": ("#ffffff", 0.85, "leaves.png", 4.0, False),
    "leaves_dark": ("#b8c8b0", 0.85, "leaves.png", 4.0, False),
    "bougainvillea": ("#ffffff", 0.8, "bougainvillea.png", 2.0, False),
    "flamboyant": ("#ffffff", 0.8, "flamboyant.png", 3.0, False),
    "flower_card": ("#ffffff", 0.7, "flowers.png", 1.0, True),
}


def material(name: str) -> dict:
    """Params for a 'latin:<name>' material (the city3d MaterialTable format plus image fields)."""
    kind, *rest = name.split(":", 1)
    if kind == "wall":
        h, r, img, tm, a = dict((n, c) for n, c, _ in WALLS)[rest[0]], 0.88, "plaster.png", 3.0, False
    elif kind == "church":
        h, r, img, tm, a = dict(CHURCH_WALLS)[rest[0]], 0.88, "plaster.png", 3.0, False
    elif kind == "trimc":
        h, r, img, tm, a = dict((n, c) for n, c, _ in TRIMS)[rest[0]], 0.7, "plaster.png", 3.0, False
    elif kind == "plinth":
        h, r, img, tm, a = dict(PLINTHS)[rest[0]], 0.85, "plaster.png", 3.0, False
    elif kind in ("door", "shutter", "wood", "baluster"):
        h = dict(WOODS)[rest[0]]
        img, tm, a = {"door": ("door.png", 1.0, False), "shutter": ("shutter.png", 1.0, False),
                      "wood": ("wood.png", 1.5, False), "baluster": ("balusters.png", 1.0, True)}[kind]
        r = 0.7
    elif kind == "roof":
        h, r, img, tm, a = "#ffffff", 0.8, dict(ROOF_TILES)[rest[0]], 2.4, False
    elif kind == "dome":
        d = {n: (c, i) for n, c, i in DOMES}[rest[0]]
        h, r, img, tm, a = d[0], 0.7, d[1], 2.0, False
    else:
        h, r, img, tm, a = BASE[name]
    return {"base_color": srgb(h), "roughness": r, "texture": None, "image": img, "tile_m": tm, "alpha": a}
