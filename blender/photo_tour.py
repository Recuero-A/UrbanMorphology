"""Photo tour: sunset photographs of a finished city (runs INSIDE Blender, on the <tile>/scene.blend of m11_scene.py).

    /Applications/Blender.app/Contents/MacOS/Blender -b <tile>/scene.blend --python-exit-code 1 --python blender/photo_tour.py -- \
        --tile <tile> [--photos <tile>/photos.json] [--only a,b] [--res 2400] [--samples 128] [--out photos]

photos.json: [{"name", "location", "look_at", "lens", optional "sun_el" (deg, default 8), "sun_az" (deg, 140),
"exposure", "haze_m" (aerial perspective scale, m), "glow" (lit windows / lanterns, emission W/m2), "clouds" (cover 0-1),
"sun_disc", "dof": {"focus": [x, y, z], "fstop"}}]. Per photo:
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
GLOW = {"latin:glass_warm": (1.0, 0.38, 0.10), "latin:lamp": (1.0, 0.55, 0.22)}


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", required=True)
    ap.add_argument("--photos", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--res", type=int, default=2400)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--out", default="photos")
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


def build_world(world, el, az, clouds, flat=False, disc=False, aerosol=1.0):
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
    fade = node(nt, "ShaderNodeMapRange", _interpolation_type="SMOOTHSTEP", **{"From Min": 0.0, "From Max": 0.06})
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
        col = mix(nt, 1.0, col, mix(nt, sd.outputs[0], (0, 0, 0, 1), (900.0, 420.0, 160.0, 1)), "ADD")
    L(col, bg.inputs["Color"])


def compositor(sc, haze_img, haze_m, far, beyond=(1800.0, 2500.0)):
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
    for k, v in (("Type", "Bloom"), ("Threshold", 1.2), ("Strength", 0.18), ("Size", 0.6)):
        try:
            gl.inputs[k].default_value = v
        except (KeyError, TypeError):
            pass
    L(img, gl.inputs["Image"])
    L(gl.outputs["Image"], out.inputs[0])
    sc.compositing_node_group = tree


def set_sun(sun, el, az):
    sun.data.energy = float(np.interp(el, SUN_EL, SUN_W))
    sun.data.color = [float(np.interp(el, SUN_EL, [c[i] for c in SUN_RGB])) for i in range(3)]
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
    cam = bpy.data.objects.new("photo_cam", bpy.data.cameras.new("photo_cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    far = 30000.0
    cam.data.clip_end = far
    timing = {"res": [sc.render.resolution_x, sc.render.resolution_y], "samples": a.samples, "photos": {}, "cameras": {}}
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
        cam.data.clip_start = float(p.get("clip_start", 0.3))
        dof = p.get("dof")
        cam.data.dof.use_dof = bool(dof)
        if dof:
            cam.data.dof.focus_distance = float((Vector(dof["focus"]) - loc).length)
            cam.data.dof.aperture_fstop = float(dof.get("fstop", 2.8))
        set_sun(sun, el, az)
        haze_m = float(p.get("haze_m", 0))
        haze_img = None
        if haze_m > 0:                                  # pass 1: the horizon sky of every view ray (linear EXR)
            build_world(world, el, az, 0.0, flat=True, aerosol=aerosol)
            for c in hide:
                c.hide_render = True
            for o in hide_ob:
                o.hide_render = True
            sc.render.use_compositing = False
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
        build_world(world, el, az, float(p.get("clouds", 0.5)), disc=bool(p.get("sun_disc")), aerosol=aerosol)
        set_glow(p.get("glow", 0.0))
        compositor(sc, haze_img, haze_m, far)
        sc.render.use_compositing = True
        sc.view_settings.exposure = float(p.get("exposure", -0.25))
        sc.eevee.taa_render_samples = a.samples
        sc.render.image_settings.file_format = "JPEG"
        sc.render.image_settings.quality = 95
        sc.render.filepath = str(out / f"{p['name']}.jpg")
        bpy.ops.render.render(write_still=True)
        if haze_img is not None:
            bpy.data.images.remove(haze_img)
            hp.unlink()
        timing["photos"][p["name"]] = round(time.time() - t0, 2)
        timing["cameras"][p["name"]] = [round(v, 3) for v in loc]
        print("PHOTO", p["name"], timing["photos"][p["name"]], flush=True)
    (out / "photo_timing.json").write_text(json.dumps(timing, indent=1))


main()
