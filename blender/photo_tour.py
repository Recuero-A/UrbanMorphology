"""Photo tour: sunset photographs of a finished city (runs INSIDE Blender, on the <tile>/scene.blend of m11_scene.py).

    /Applications/Blender.app/Contents/MacOS/Blender -b <tile>/scene.blend --python-exit-code 1 --python blender/photo_tour.py -- \
        --tile <tile> [--photos <tile>/photos.json] [--only a,b] [--res 2400] [--samples 128] [--out photos]

photos.json: [{"name", "location", "look_at", "lens", optional "sun_el" (deg, default 8), "sun_az" (deg, 140),
"exposure", "haze_m" (aerial perspective scale, m), "glow" (lit windows / lanterns, emission W/m2), "clouds" (cover 0-1),
"sun_disc", "dof": {"focus": [x, y, z], "fstop"}}]; optional "sun_tint_el" (the sun lamp coloured as at this elevation:
dusty air reddens a higher sun), "sun_k" (sun lamp strength, x), "disc_k" (sun disc brightness, x), "bloom" (glare strength, 0.18), "dust": {"r",
"density", "g"} (a box of dusty air about the camera, Cycles: light shafts), "house_lights": {"share", "strength",
"cell", "rgb", "tint"} (a random share of the houses' windows lit warm: 3D cells of about ``cell`` m, a floor high, in
"rgb" (amber); "tint": each lit window's colour drawn between "rgb" and this RGB), "night":
{"lamp_w", "crown_w", "foot_w", "dome"} (dusk lights as the Godot viewer's --light dusk: a warm point light under every
street lamp's head, narrow warm uplights on the towers of <tile>/viewer.json "crowns", domes and gilt glowing). Per photo:
* sky: the physical sky (multiple scattering) with a procedural cloud layer lit by the low sun (warm towards the sun,
  lavender away from it); the sun lamp's colour and strength follow its elevation;
* aerial perspective: every pixel is mixed towards the horizon sky of its own direction (a sky-only pre-render with the
  view rays flattened onto the horizon) by 1 - exp(-depth / haze_m); the sky itself is left alone;
* warm windows of the skyscrapers and the lanterns glow; a light bloom; optional depth of field.
"on_surface": the camera's z is its height above the first surface straight below (a roof terrace; ray cast).
Writes <tile>/<out>/<name>.jpg (quality 95) and <tile>/<out>/photo_timing.json (seconds and the camera actually used, per photo).
Deterministic (no random numbers).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

# sun lamp by elevation (deg): W/m2 and colour, interpolated (the low sun is weaker and redder)
SUN_EL = [0.0, 2.0, 5.0, 8.0, 14.0]
SUN_W = [1.2, 3.2, 4.8, 6.0, 6.5]
SUN_RGB = [(1.0, 0.28, 0.12), (1.0, 0.36, 0.16), (1.0, 0.47, 0.23), (1.0, 0.56, 0.30), (1.0, 0.68, 0.43)]
SKY_K = 0.30                                   # sky strength of the m11 sunset preset
GLOW = {"latin:glass_warm": (1.0, 0.38, 0.10), "latin:lamp": (1.0, 0.55, 0.22), "latin:glass_lit": (1.0, 0.5, 0.18)}


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", required=True)
    ap.add_argument("--photos", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--res", type=int, default=2400)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--out", default="photos")
    ap.add_argument("--engine", default="EEVEE", choices=["EEVEE", "CYCLES"], help="CYCLES: path traced (bounce light)")
    return ap.parse_args(argv)


def node(nt, kind, **inputs):
    n = nt.nodes.new(kind)
    for k, v in inputs.items():
        if k.startswith("_"):
            setattr(n, k[1:], v)
        else:
            n.inputs[k].default_value = v
    return n


def sock(coll, ident):
    """Socket by identifier (the Mix node repeats its names: A_Float, A_Vector, A_Color, ...)."""
    return next(s for s in coll if s.identifier == ident)


def mix(nt, fac, A, B, blend="MIX"):
    """Colour mix A -> B by fac (number or socket); A, B: RGBA tuples or sockets."""
    n = node(nt, "ShaderNodeMix", _data_type="RGBA", _blend_type=blend)
    for ident, x in (("Factor_Float", fac), ("A_Color", A), ("B_Color", B)):
        if isinstance(x, (int, float, tuple)):
            sock(n.inputs, ident).default_value = x
        else:
            nt.links.new(x, sock(n.inputs, ident))
    return sock(n.outputs, "Result_Color")


def math(nt, op, a, b=None):
    n = node(nt, "ShaderNodeMath", _operation=op)
    for i, x in enumerate((a, b)):
        if x is None:
            continue
        if isinstance(x, (int, float)):
            n.inputs[i].default_value = x
        else:
            nt.links.new(x, n.inputs[i])
    return n.outputs[0]


def build_world(world, el, az, clouds, flat=False, disc=False, aerosol=1.0, disc_k=1.0):
    """Physical sky + sunset clouds; ``flat``: no clouds (the haze colour: every ray below 0.6 deg sees the sky there)."""
    nt = world.node_tree
    nt.nodes.clear()
    L = nt.links.new
    out = node(nt, "ShaderNodeOutputWorld")
    bg = node(nt, "ShaderNodeBackground", Strength=SKY_K)
    L(bg.outputs[0], out.inputs["Surface"])
    tc = node(nt, "ShaderNodeTexCoord")
    D = node(nt, "ShaderNodeVectorMath", _operation="NORMALIZE")
    L(tc.outputs["Generated"], D.inputs[0])
    sep = node(nt, "ShaderNodeSeparateXYZ")
    L(D.outputs[0], sep.inputs[0])
    # az: the sun's direction counter-clockwise from +x (east), as the sun lamp below; the sky texture's sun_rotation is
    # a compass bearing (clockwise from north; measured 2026-11-23: rotation 0 -> brightest towards +y, 140 -> 310 deg)
    sky = node(nt, "ShaderNodeTexSky", _sky_type="MULTIPLE_SCATTERING", _sun_elevation=np.radians(el),
               _sun_rotation=np.radians(90.0 - az), _sun_disc=False, _aerosol_density=aerosol)
    # rays below the horizon see the horizon (the physical sky is black there; no dark seam past the land's edge)
    comb = node(nt, "ShaderNodeCombineXYZ")
    L(sep.outputs[0], comb.inputs[0])
    L(sep.outputs[1], comb.inputs[1])
    L(math(nt, "MAXIMUM", sep.outputs[2], 0.01), comb.inputs[2])
    Dn = node(nt, "ShaderNodeVectorMath", _operation="NORMALIZE")
    L(comb.outputs[0], Dn.inputs[0])
    L(Dn.outputs[0], sky.inputs["Vector"])
    if flat or clouds <= 0:
        L(sky.outputs[0], bg.inputs["Color"])
        return
    # cloud layer: the view ray hits a plane (u, v) = (x, y) / z; streaks along the sun's azimuth
    zc = math(nt, "MAXIMUM", sep.outputs[2], 0.012)
    a = np.radians(az)
    u = math(nt, "DIVIDE", math(nt, "ADD", math(nt, "MULTIPLY", sep.outputs[0], np.cos(a)), math(nt, "MULTIPLY", sep.outputs[1], np.sin(a))), zc)
    v = math(nt, "DIVIDE", math(nt, "SUBTRACT", math(nt, "MULTIPLY", sep.outputs[1], np.cos(a)), math(nt, "MULTIPLY", sep.outputs[0], np.sin(a))), zc)
    P = node(nt, "ShaderNodeCombineXYZ")
    L(math(nt, "MULTIPLY", u, 0.35), P.inputs[0])
    L(math(nt, "MULTIPLY", v, 1.0), P.inputs[1])
    P.inputs[2].default_value = 3.7
    nz = node(nt, "ShaderNodeTexNoise", Scale=0.55, Detail=8.0, Roughness=0.62, Lacunarity=2.1, Distortion=0.35)
    L(P.outputs[0], nz.inputs["Vector"])
    thr = 0.62 - 0.22 * clouds
    cover = node(nt, "ShaderNodeMapRange", _interpolation_type="SMOOTHSTEP", **{"From Min": thr, "From Max": thr + 0.16})
    L(nz.outputs["Fac"], cover.inputs["Value"])
    # gone by the clamp (z 0.012): below it (u, v) depend on the azimuth only, and a high camera saw them as vertical
    # streaks over the horizon (aerials, 2026-11-27)
    fade = node(nt, "ShaderNodeMapRange", _interpolation_type="SMOOTHSTEP", **{"From Min": 0.015, "From Max": 0.075})
    L(sep.outputs[2], fade.inputs["Value"])
    mask = math(nt, "MULTIPLY", math(nt, "MULTIPLY", cover.outputs[0], fade.outputs[0]), 0.92)
    # light on the clouds: half the sky's light, plus the reddened low sun (everywhere, and a strong forward lobe
    # around the sun); thick parts self-shadow. Raw sky units (x SKY_K): zenith ~(0.5, 0.9, 1.8) and the horizon
    # ~(5.8, 6.2, 5.5) away from an 8 deg sun, ~(33, 23, 12) next to it (measured 2026-11-23)
    S = (np.cos(np.radians(el)) * np.cos(a), np.cos(np.radians(el)) * np.sin(a), np.sin(np.radians(el)))
    dot = node(nt, "ShaderNodeVectorMath", _operation="DOT_PRODUCT")
    L(D.outputs[0], dot.inputs[0])
    dot.inputs[1].default_value = S
    A = float(np.interp(el, [1.0, 4.0, 8.0], [1.8, 2.6, 3.0]))
    lobe = math(nt, "POWER", math(nt, "ADD", math(nt, "MULTIPLY", dot.outputs["Value"], 0.5), 0.5), 8.0)
    w = math(nt, "ADD", math(nt, "MULTIPLY", lobe, 8.0 * A), A)
    tint = tuple(float(np.interp(el, [1.0, 4.0, 8.0], [c0, c1, c2])) for c0, c1, c2 in
                 ((1.0, 1.0, 1.0), (0.30, 0.42, 0.55), (0.14, 0.22, 0.32)))
    sun_term = node(nt, "ShaderNodeVectorMath", _operation="SCALE")
    sun_term.inputs[0].default_value = tint
    L(w, sun_term.inputs["Scale"])
    amb = node(nt, "ShaderNodeVectorMath", _operation="SCALE")
    L(sky.outputs[0], amb.inputs[0])
    amb.inputs["Scale"].default_value = 0.5
    lit = node(nt, "ShaderNodeVectorMath", _operation="ADD")
    L(amb.outputs[0], lit.inputs[0])
    L(sun_term.outputs[0], lit.inputs[1])
    thick = node(nt, "ShaderNodeMapRange", **{"From Min": thr, "From Max": thr + 0.4, "To Min": 1.0, "To Max": 0.5})
    L(nz.outputs["Fac"], thick.inputs["Value"])
    shade = node(nt, "ShaderNodeVectorMath", _operation="SCALE")
    L(lit.outputs[0], shade.inputs[0])
    L(thick.outputs[0], shade.inputs["Scale"])
    shade = shade.outputs[0]
    col = mix(nt, mask, sky.outputs[0], shade)
    if disc:                                    # the sun's disc (0.55 deg; the sky node hides its Vector input with its own)
        sd = node(nt, "ShaderNodeMapRange", _interpolation_type="SMOOTHSTEP",
                  **{"From Min": np.cos(np.radians(0.32)), "From Max": np.cos(np.radians(0.26))})
        L(dot.outputs["Value"], sd.inputs["Value"])
        col = mix(nt, 1.0, col, mix(nt, sd.outputs[0], (0, 0, 0, 1), (900.0 * disc_k, 420.0 * disc_k, 160.0 * disc_k, 1)), "ADD")
    L(col, bg.inputs["Color"])


def compositor(sc, haze_img, haze_m, far, beyond=(1800.0, 2500.0), bloom=0.18):
    """Aerial perspective and a light bloom: the render is mixed towards the haze image by 1 - T, T = exp(-depth / haze_m
    - max(depth - beyond[0], 0) / beyond[1]) (the second term dissolves the flat land past the map's edge into the
    horizon); the sky is kept."""
    tree = bpy.data.node_groups.get("photo") or bpy.data.node_groups.new("photo", "CompositorNodeTree")
    tree.nodes.clear()
    if not tree.interface.items_tree:
        tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    L = tree.links.new
    rl = tree.nodes.new("CompositorNodeRLayers")
    out = tree.nodes.new("NodeGroupOutput")
    img = rl.outputs["Image"]
    if haze_img is not None:
        hz = tree.nodes.new("CompositorNodeImage")
        hz.image = haze_img
        # depth eroded by one pixel (normalised to [0, 1] for the mask node): an anti-aliased edge pixel between the
        # land and the sky takes the land's depth (with the sky's it went unhazed: a dark dashed line on the horizon)
        dn = node(tree, "CompositorNodeDilateErode", Size=-1)
        L(math(tree, "DIVIDE", math(tree, "MINIMUM", rl.outputs["Depth"], far), far), dn.inputs["Mask"])
        d = math(tree, "MULTIPLY", dn.outputs[0], far)
        far_d = math(tree, "MAXIMUM", math(tree, "SUBTRACT", d, beyond[0]), 0.0)
        T = math(tree, "EXPONENT", math(tree, "SUBTRACT", math(tree, "MULTIPLY", d, -1.0 / haze_m),
                                        math(tree, "MULTIPLY", far_d, 1.0 / beyond[1])))
        T = math(tree, "MAXIMUM", T, math(tree, "GREATER_THAN", d, 0.99 * far))
        img = mix(tree, T, hz.outputs["Image"], img)
    gl = tree.nodes.new("CompositorNodeGlare")
    for k, v in (("Type", "Bloom"), ("Threshold", 1.2), ("Strength", bloom), ("Size", 0.6)):
        try:
            gl.inputs[k].default_value = v
        except (KeyError, TypeError):
            pass
    L(img, gl.inputs["Image"])
    L(gl.outputs["Image"], out.inputs[0])
    sc.compositing_node_group = tree


def set_sun(sun, el, az, tint_el=None, k=1.0):
    sun.data.energy = float(np.interp(el, SUN_EL, SUN_W)) * k
    te = el if tint_el is None else tint_el
    sun.data.color = [float(np.interp(te, SUN_EL, [c[i] for c in SUN_RGB])) for i in range(3)]
    elev, rot = np.radians(el), np.radians(az)
    sun.rotation_euler = (np.pi / 2 - elev, 0.0, rot + np.pi / 2)


def set_glow(strength):
    for name, col in GLOW.items():
        m = bpy.data.materials.get(name)
        if m is None or not m.use_nodes:
            continue
        b = m.node_tree.nodes.get("Principled BSDF")
        b.inputs["Emission Color"].default_value = (*col, 1.0)
        b.inputs["Emission Strength"].default_value = float(strength) * (2.5 if name == "latin:lamp" else 1.0)


def set_house_lights(spec):
    """Warm light in a random share of the house windows (latin:glass): a cell noise in object space (cells about a
    window bay wide and a floor high), each cell on or off, its brightness varied; ``spec`` None: all off."""
    m = bpy.data.materials.get("latin:glass")
    if m is None or not m.use_nodes:
        return
    nt = m.node_tree
    for n in [n for n in nt.nodes if n.name.startswith("hl_")]:
        nt.nodes.remove(n)
    b = nt.nodes.get("Principled BSDF")
    b.inputs["Emission Strength"].default_value = 0.0
    if not spec:
        return
    new = len(nt.nodes)
    cell = float(spec.get("cell", 3.0))
    tc = node(nt, "ShaderNodeTexCoord")
    sc = node(nt, "ShaderNodeVectorMath", _operation="MULTIPLY")
    sc.inputs[1].default_value = (1.0 / cell, 1.0 / cell, 1.0 / 3.2)
    nt.links.new(tc.outputs["Object"], sc.inputs[0])
    vo = node(nt, "ShaderNodeTexVoronoi", Scale=1.0, Randomness=0.0)
    nt.links.new(sc.outputs[0], vo.inputs["Vector"])
    sep = node(nt, "ShaderNodeSeparateColor")
    nt.links.new(vo.outputs["Color"], sep.inputs[0])
    on = math(nt, "GREATER_THAN", sep.outputs[0], 1.0 - float(spec.get("share", 0.35)))
    k = math(nt, "MULTIPLY", on, math(nt, "ADD", math(nt, "MULTIPLY", sep.outputs[1], 0.8), 0.2))
    nt.links.new(math(nt, "MULTIPLY", k, float(spec.get("strength", 6.0))), b.inputs["Emission Strength"])
    rgb = (*spec.get("rgb", (1.0, 0.5, 0.2)), 1.0)
    b.inputs["Emission Color"].default_value = rgb
    if spec.get("tint"):                            # each lit window its own lamp: "rgb" to the "tint" colour
        nt.links.new(mix(nt, sep.outputs[2], rgb, (*spec["tint"], 1.0)), b.inputs["Emission Color"])
    for n in list(nt.nodes)[new:]:
        n.name = "hl_" + n.name


LAMP_RGB = (1.0, 0.62, 0.33)                   # the viewer's #ffb468
UP_RGB = (1.0, 0.70, 0.42)                     # the viewer's floodlight #ffc07a
DOME_RGB = (1.0, 0.64, 0.38)                   # #ffb46e


def night_lights(sc, tile):
    """The dusk lights of the Godot viewer (main.gd add_night_lights / add_crown_lights), made once and switched per
    photo: a point light under the head of every street lamp (lamp_* objects; the lamps themselves cast no shadow, so
    that the light leaves the lantern) and the tower floodlights: for each tall tower of viewer.json "crowns" (Godot
    coordinates: ground point g, axis ax in world x, y, height top, crown base y0, crown and base half widths cr, br),
    three narrow uplights per face on the ledge under the crown and two per face at the foot, leaning into the wall.
    -> (lamp lights, crown lights, foot lights)."""
    col = bpy.data.collections.new("night_lights")
    sc.collection.children.link(col)

    def light(kind, name, p, rgb):
        lt = bpy.data.lights.new(name, kind)
        lt.color = rgb
        lt.energy = 0.0
        ob = bpy.data.objects.new(name, lt)
        ob.location = p
        col.objects.link(ob)
        return ob

    lamps = []
    for o in [o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith("lamp")]:
        bb = [o.matrix_world @ Vector(c) for c in o.bound_box]
        p = Vector((sum(v.x for v in bb) / 8, sum(v.y for v in bb) / 8, max(v.z for v in bb) - 0.35))
        ob = light("POINT", "nl_" + o.name, p, LAMP_RGB)
        ob.data.shadow_soft_size = 0.12
        o.visible_shadow = False
        lamps.append(ob)
    vj = tile / "viewer.json"
    crowns = json.loads(vj.read_text()).get("crowns", []) if vj.exists() else []
    up = Vector((0.0, 0.0, 1.0))
    crown, foot = [], []

    def spot(p, nrm, angle, out):
        ob = light("SPOT", f"nu_{len(crown) + len(foot)}", p, UP_RGB)
        ob.data.spot_size = np.radians(2.0 * angle)
        ob.data.spot_blend = 0.6
        ob.data.shadow_soft_size = 0.05
        ob.rotation_euler = (up * 10.0 - nrm * 1.1).normalized().to_track_quat("-Z", "Y").to_euler()
        out.append(ob)

    for c in crowns:
        g = Vector((c["g"][0], -c["g"][2], c["g"][1]))
        ax = Vector((c["ax"][0], c["ax"][1], 0.0)).normalized()
        ay = up.cross(ax).normalized()
        for nrm in (ax, -ax, ay, -ay):
            tan = up.cross(nrm)
            for s in (-0.55, 0.0, 0.55):
                spot(g + up * (c["y0"] + 0.35) + nrm * (c["cr"] + 0.45) + tan * (s * c["cr"]), nrm, 15.0, crown)
            for s in (-0.45, 0.45):
                spot(g + up * 0.4 + nrm * (c["br"] + 0.7) + tan * (s * c["br"]), nrm, 12.0, foot)
    print("night lights:", len(lamps), "lamps,", len(crown) + len(foot), "uplights", flush=True)
    return lamps, crown, foot


def set_night(nl, spec):
    """Switch the dusk lights for one photo (``spec`` None: all off, domes unlit)."""
    spec = spec or {}
    for obs, key in zip(nl, ("lamp_w", "crown_w", "foot_w")):
        for ob in obs:
            ob.data.energy = float(spec.get(key, 0.0))
            ob.hide_render = not spec.get(key)
    for m in bpy.data.materials:
        if (m.name.startswith("latin:dome:") or m.name == "latin:gold") and m.use_nodes:
            b = m.node_tree.nodes.get("Principled BSDF")
            if b is not None:
                b.inputs["Emission Color"].default_value = (*DOME_RGB, 1.0)
                b.inputs["Emission Strength"].default_value = float(spec.get("dome", 0.0)) * (3.0 if m.name == "latin:gold" else 1.0)


def dust_box(sc, loc, dust):
    """A box of dusty air about the camera (homogeneous volume, forward scattering): shafts of the low sun."""
    r = float(dust.get("r", 250.0))
    me = bpy.data.meshes.new("dust")
    V = [(x, y, z) for x in (-r, r) for y in (-r, r) for z in (-r, r)]
    me.from_pydata(V, [], [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
    ob = bpy.data.objects.new("dust", me)
    ob.location = loc
    m = bpy.data.materials.new("dust")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    vs = node(nt, "ShaderNodeVolumeScatter", Color=(1.0, 0.93, 0.82, 1.0), Density=float(dust.get("density", 0.002)),
              Anisotropy=float(dust.get("g", 0.7)))
    o = node(nt, "ShaderNodeOutputMaterial")
    nt.links.new(vs.outputs[0], o.inputs["Volume"])
    me.materials.append(m)
    sc.collection.objects.link(ob)
    return ob


def main():
    a = args()
    tile = Path(a.tile).resolve()
    photos = json.loads(Path(a.photos or tile / "photos.json").read_text())
    if a.only:
        keep = a.only.split(",")
        photos = [p for p in photos if p["name"] in keep]
    out = tile / a.out
    out.mkdir(exist_ok=True)
    sc = bpy.context.scene
    cyc = a.engine == "CYCLES"
    if cyc:                                         # path traced: bounce light, soft sky shadows, real glow of windows
        sc.render.engine = "CYCLES"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for dv in prefs.devices:
            dv.use = dv.type == "METAL"
        sc.cycles.device = "GPU"
        sc.cycles.use_adaptive_sampling = True
        sc.cycles.adaptive_threshold = 0.02
        sc.cycles.use_denoising = True
        sc.cycles.max_bounces, sc.cycles.diffuse_bounces, sc.cycles.glossy_bounces = 8, 4, 3
        sc.cycles.transparent_max_bounces = 48          # cut-out fronds, leaves and balusters stack deep
        sc.cycles.caustics_reflective = sc.cycles.caustics_refractive = False
        sc.cycles.blur_glossy = 1.0
        sc.render.use_persistent_data = True            # the BVH is built once for every photo
    else:
        sc.render.engine = "BLENDER_EEVEE"
        sc.eevee.use_shadows = True
        sc.eevee.use_raytracing = True
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = "AgX - Punchy"
    except TypeError:
        pass
    sc.view_layers[0].use_pass_z = True
    sc.render.resolution_x, sc.render.resolution_y = a.res, int(a.res * 9 / 16)
    sun = bpy.data.objects["sun"]
    hide = [c for c in (bpy.data.collections.get("city"), bpy.data.collections.get("props")) if c is not None]
    hide_ob = [o for o in (bpy.data.objects.get("surround"), bpy.data.objects.get("sea_plane")) if o is not None]
    world = sc.world
    sur = bpy.data.objects.get("surround")                  # the land around the map: jungle-dark, hazed away
    if sur is not None:
        for v in sur.data.vertices[:4]:
            v.co.x, v.co.y = v.co.x * 1.6, v.co.y * 1.6          # ~19 km: inside the camera's far clip
        bsdf = sur.data.materials[0].node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.035, 0.06, 0.03, 1.0)
        bsdf.inputs["Roughness"].default_value = 1.0         # matte far land: no sheen of the bright sky at grazing
        bsdf.inputs["Specular IOR Level"].default_value = 0.0  # angles from high cameras
    cam = bpy.data.objects.new("photo_cam", bpy.data.cameras.new("photo_cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    far = 30000.0
    cam.data.clip_end = far
    timing = {"res": [sc.render.resolution_x, sc.render.resolution_y], "samples": a.samples, "photos": {}, "cameras": {}}
    nl = night_lights(sc, tile) if any(p.get("night") for p in photos) else None
    if nl is not None:
        hide.append(bpy.data.collections["night_lights"])    # not in the sky-only haze pass
    for p in photos:
        t0 = time.time()
        el, az = float(p.get("sun_el", 8.0)), float(p.get("sun_az", 140.0))
        aerosol = float(p.get("aerosol", 2.0))
        loc = Vector(p["location"])
        if p.get("on_surface"):                         # z: height above the first surface below (a roof terrace)
            hit, at, *_ = sc.ray_cast(bpy.context.evaluated_depsgraph_get(), Vector((loc.x, loc.y, 5000.0)), Vector((0, 0, -1)))
            loc.z += at.z if hit else 0.0
        cam.location = loc
        cam.rotation_euler = (Vector(p["look_at"]) - loc).to_track_quat("-Z", "Y").to_euler()
        if p.get("roll"):
            cam.rotation_euler.rotate_axis("Z", np.radians(p["roll"]))
        cam.data.lens = float(p["lens"])
        cam.data.shift_x, cam.data.shift_y = float(p.get("shift_x", 0.0)), float(p.get("shift_y", 0.0))   # shift lens: level
        cam.data.sensor_fit = "HORIZONTAL"                                                                    # camera, straight verticals
        cam.data.clip_start = float(p.get("clip_start", 0.3))
        dof = p.get("dof")
        cam.data.dof.use_dof = bool(dof)
        if dof:
            cam.data.dof.focus_distance = float((Vector(dof["focus"]) - loc).length)
            cam.data.dof.aperture_fstop = float(dof.get("fstop", 2.8))
        set_sun(sun, el, az, p.get("sun_tint_el"), float(p.get("sun_k", 1.0)))
        dust = dust_box(sc, loc, p["dust"]) if p.get("dust") and cyc else None
        if dust is not None:
            dust.hide_render = True
        haze_m = float(p.get("haze_m", 0))
        haze_img = None
        if haze_m > 0:                                  # pass 1: the horizon sky of every view ray (linear EXR)
            build_world(world, el, az, 0.0, flat=True, aerosol=aerosol)
            for c in hide:
                c.hide_render = True
            for o in hide_ob:
                o.hide_render = True
            sc.render.use_compositing = False
            if cyc:
                sc.cycles.samples, sc.cycles.use_denoising = 1, False
            else:
                sc.eevee.taa_render_samples = 1
            sc.render.image_settings.file_format = "OPEN_EXR"
            hp = out / f"_haze_{p['name']}.exr"
            sc.render.filepath = str(hp)
            bpy.ops.render.render(write_still=True)
            for c in hide:
                c.hide_render = False
            for o in hide_ob:
                o.hide_render = False
            haze_img = bpy.data.images.load(str(hp), check_existing=False)
        build_world(world, el, az, float(p.get("clouds", 0.5)), disc=bool(p.get("sun_disc")), aerosol=aerosol,
                    disc_k=float(p.get("disc_k", 1.0)))
        set_glow(p.get("glow", 0.0))
        set_house_lights(p.get("house_lights"))
        if nl is not None:
            set_night(nl, p.get("night"))
        compositor(sc, haze_img, haze_m, far, bloom=float(p.get("bloom", 0.18)))
        if dust is not None:
            dust.hide_render = False
        sc.render.use_compositing = True
        sc.view_settings.exposure = float(p.get("exposure", -0.25))
        try:
            sc.view_settings.look = p.get("look", "AgX - Punchy")
        except TypeError:
            pass
        if cyc:
            sc.cycles.samples, sc.cycles.use_denoising = a.samples, True
        else:
            sc.eevee.taa_render_samples = a.samples
        sc.render.image_settings.file_format = "JPEG"
        sc.render.image_settings.quality = 95
        sc.render.filepath = str(out / f"{p['name']}.jpg")
        bpy.ops.render.render(write_still=True)
        if haze_img is not None:
            bpy.data.images.remove(haze_img)
            hp.unlink()
        if dust is not None:
            bpy.data.objects.remove(dust)
        timing["photos"][p["name"]] = round(time.time() - t0, 2)
        timing["cameras"][p["name"]] = [round(v, 3) for v in loc]
        print("PHOTO", p["name"], timing["photos"][p["name"]], flush=True)
    tf = out / "photo_timing.json"
    if a.only and tf.exists():                          # a partial re-render keeps the other photos' cameras
        old = json.loads(tf.read_text())
        for k in ("photos", "cameras"):
            old[k].update(timing[k])
        timing = old
    tf.write_text(json.dumps(timing, indent=1))


main()
