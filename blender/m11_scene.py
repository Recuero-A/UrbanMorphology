"""Milestone 11: build, light, render and export one tile in Blender (runs INSIDE Blender, not the project venv).

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python blender/m11_scene.py -- \
        --tile outputs/m11/<tile_id> [--views aerial,oblique,street,closeup] [--engine EEVEE|CYCLES] [--res 1600]

Reads <tile>/meshes.npz (scripts/m11_build_city.py) and <tile>/city.json (camera placement only). Writes
<tile>/renders/<view>.png, <tile>/scene.glb, <tile>/scene.blend and <tile>/blender_timing.json.
Deterministic: no random numbers are drawn here (all variation is decided upstream from the seed).
Frame: metres, Z up (identical to city.json); the glTF exporter converts to Y up.
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import bpy
import numpy as np

# Daylight calibration (2026-11-01, aerial test renders): the physical sky alone was as bright as the sun alone
# at strength 0.35 and washed colours out; 0.08 keeps sky-blue fill light in the shadows. Exposure -1 stops
# mid-grey roofs from clipping to white under a 4 W/m2 sun with AgX.
SUN_W_M2, SKY_STRENGTH, EXPOSURE = 4.0, 0.08, -1.0


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", required=True)
    ap.add_argument("--views", default="aerial,oblique,street,closeup")
    ap.add_argument("--engine", default="EEVEE", choices=["EEVEE", "CYCLES"])
    ap.add_argument("--res", type=int, default=1600)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--no-export", action="store_true")
    return ap.parse_args(argv)


# ----------------------------------------------------------------------------------------------- materials
def make_material(p: dict) -> bpy.types.Material:
    m = bpy.data.materials.new(p["name"])
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    base = list(p["base_color"]) + [1.0]
    bsdf.inputs["Base Color"].default_value = base
    bsdf.inputs["Roughness"].default_value = p["roughness"]
    bsdf.inputs["Metallic"].default_value = p.get("metallic", 0.0)
    m["base_color"] = base                       # kept for the plain-material glTF export
    tex = p.get("texture")
    if tex is None:
        return m
    uv = nt.nodes.new("ShaderNodeUVMap")
    col = nt.nodes.new("ShaderNodeMix")
    col.data_type = "RGBA"
    dark = [c * 0.82 for c in base[:3]] + [1.0]
    if tex == "noise":                           # subtle large-scale mottling (plaster, concrete, asphalt, grass)
        t = nt.nodes.new("ShaderNodeTexNoise")
        t.inputs["Scale"].default_value = 0.6
        t.inputs["Detail"].default_value = 6.0
        nt.links.new(uv.outputs["UV"], t.inputs["Vector"])
        fac = t.outputs["Fac"]
        col.inputs["A"].default_value = dark
        col.inputs["B"].default_value = [min(1.0, c * 1.08) for c in base[:3]] + [1.0]
    elif tex in ("brick", "tiles"):              # running-bond courses in metres
        t = nt.nodes.new("ShaderNodeTexBrick")
        t.inputs["Scale"].default_value = 1.0
        t.inputs["Brick Width"].default_value = 0.25 if tex == "brick" else 0.4
        t.inputs["Row Height"].default_value = 0.075 if tex == "brick" else 0.25
        t.inputs["Mortar Size"].default_value = 0.01 if tex == "brick" else 0.02
        t.inputs["Color1"].default_value = base
        t.inputs["Color2"].default_value = dark
        t.inputs["Mortar"].default_value = [0.55, 0.53, 0.5, 1.0]
        nt.links.new(uv.outputs["UV"], t.inputs["Vector"])
        nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
        return m
    else:                                        # stripes: horizontal siding boards
        t = nt.nodes.new("ShaderNodeTexWave")
        t.inputs["Scale"].default_value = 0.5
        t.wave_type = "BANDS"
        t.bands_direction = "Y"
        nt.links.new(uv.outputs["UV"], t.inputs["Vector"])
        fac = t.outputs["Fac"]
        col.inputs["A"].default_value = dark
        col.inputs["B"].default_value = base
    nt.links.new(fac, col.inputs["Factor"])
    nt.links.new(col.outputs["Result"], bsdf.inputs["Base Color"])
    return m


def plain_materials():
    """glTF keeps only Principled factors: drop procedural links so base colours survive the export."""
    for m in bpy.data.materials:
        if not m.use_nodes or "base_color" not in m:
            continue
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        for link in list(bsdf.inputs["Base Color"].links):
            m.node_tree.links.remove(link)
        bsdf.inputs["Base Color"].default_value = list(m["base_color"])


# ----------------------------------------------------------------------------------------------- geometry
def make_object(name, V, F, mat, UV, materials, coll):
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], F.tolist())
    used = sorted(set(mat.tolist()))
    for k in used:
        me.materials.append(materials[k])
    slot = {k: i for i, k in enumerate(used)}
    me.polygons.foreach_set("material_index", [slot[int(k)] for k in mat])
    uvl = me.uv_layers.new(name="UVMap")
    uvl.data.foreach_set("uv", UV.reshape(-1).astype(np.float32))      # from_pydata keeps face-corner order
    me.validate(clean_customdata=False)
    me.update()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def look_at(ob, target):
    d = np.asarray(target, float) - np.asarray(ob.location, float)
    from mathutils import Vector
    ob.rotation_euler = Vector(d).to_track_quat("-Z", "Y").to_euler()


def street_camera(city):
    """1.7 m eye height on the longest street near the tile centre, looking along it."""
    best, score = None, -1
    for s in city["streets"]:
        xy = np.asarray(s["centerline"])
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        L = seg.sum()
        mid = xy[len(xy) // 2]
        sc = L - 0.5 * np.linalg.norm(mid - 256)
        if L > 60 and sc > score:
            best, score = xy, sc
    if best is None:
        return (256, 30, 1.7), (256, 256, 1.7)
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(best, axis=0), axis=1))]
    p = lambda t: np.array([np.interp(t, cum, best[:, 0]), np.interp(t, cum, best[:, 1])])
    a, b = p(cum[-1] * 0.15), p(min(cum[-1], cum[-1] * 0.15 + 80))
    return (*a, 1.7), (*b, 2.5)


def closeup_camera(city):
    """A taller building near the centre seen from the nearest street at 8 m height."""
    blds = [b for b in city["buildings"] if b["floors"] >= 3] or city["buildings"]
    c = lambda b: np.asarray(b["footprint"][0]).mean(0)
    b = min(blds, key=lambda b: np.linalg.norm(c(b) - 256) - 2 * b["height_m"])
    cb = c(b)
    pts = np.vstack([np.asarray(s["centerline"]) for s in city["streets"]]) if city["streets"] else np.array([[256, 0]])
    near = pts[np.argmin(np.linalg.norm(pts - cb, axis=1))]
    d = near - cb
    d = d / (np.linalg.norm(d) + 1e-9)
    dist = max(25.0, 1.2 * b["height_m"])
    cam = cb + d * dist
    return (*cam, 8.0), (*cb, b["eave_m"] * 0.45), b["id"]


def main():
    a = args()
    tile = Path(a.tile).resolve()
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    data = np.load(tile / "meshes.npz")
    city = json.loads((tile / "city.json").read_text())
    mats_p = json.loads(str(data["materials"]))
    materials = [make_material(p) for p in mats_p]
    coll = bpy.data.collections.new("city")
    sc.collection.children.link(coll)
    for g in ("ground", "roads", "green", "walls", "roofs"):
        if len(data[f"{g}__F"]):
            make_object(g, data[f"{g}__V"], data[f"{g}__F"], data[f"{g}__mat"], data[f"{g}__UV"], materials, coll)
    # render-only surround (the physical sky is black below the horizon); excluded from the GLB export
    sur = bpy.data.meshes.new("surround")
    R = 4000.0
    z = -0.02                                   # a frame AROUND the tile, never under it (no coplanar depth fight)
    sur.from_pydata([(-R, -R, z), (R, -R, z), (R, R, z), (-R, R, z), (0, 0, z), (512, 0, z), (512, 512, z), (0, 512, z)], [],
                    [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    surround_mat = make_material({"name": "surround", "base_color": [0.32, 0.31, 0.29], "roughness": 0.95, "texture": None})
    sur.materials.append(surround_mat)
    sur_ob = bpy.data.objects.new("surround", sur)
    sc.collection.objects.link(sur_ob)
    t_scene = time.time() - t0
    # ---- light: physical sky + sun (same direction)
    world = bpy.data.worlds.new("sky")
    sc.world = world
    world.use_nodes = True
    nt = world.node_tree
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    elev, rot = np.radians(42.0), np.radians(140.0)
    sky.sun_elevation, sky.sun_rotation = elev, rot
    sky.sun_disc = False
    nt.links.new(sky.outputs["Color"], nt.nodes["Background"].inputs["Color"])
    nt.nodes["Background"].inputs["Strength"].default_value = SKY_STRENGTH
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = SUN_W_M2
    sun.data.angle = np.radians(0.6)
    sun.rotation_euler = (np.pi / 2 - elev, 0.0, rot + np.pi / 2)
    sc.collection.objects.link(sun)
    # ---- render settings
    if a.engine == "EEVEE":
        sc.render.engine = "BLENDER_EEVEE"
        sc.eevee.taa_render_samples = a.samples
        sc.eevee.use_shadows = True
        sc.eevee.use_raytracing = True
    else:
        sc.render.engine = "CYCLES"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == "METAL"
        sc.cycles.device = "GPU"
        sc.cycles.samples = a.samples
        sc.cycles.use_denoising = True
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.exposure = EXPOSURE
    sc.render.image_settings.file_format = "PNG"
    out = tile / "renders"
    out.mkdir(exist_ok=True)
    timing = {"engine": a.engine, "samples": a.samples, "scene_build_s": round(t_scene, 2), "renders": {}}
    target = bpy.data.objects.new("target", None)
    sc.collection.objects.link(target)
    for view in a.views.split(","):
        cam = bpy.data.objects.new(f"cam_{view}", bpy.data.cameras.new(f"cam_{view}"))
        sc.collection.objects.link(cam)
        sc.camera = cam
        cam.data.clip_end = 6000
        cam.data.clip_start = 0.1 if view in ("street", "closeup") else 1.0     # depth precision at distance
        if view == "aerial":
            cam.data.type = "ORTHO"
            cam.data.ortho_scale = 512
            cam.location = (256, 256, 800)
            cam.rotation_euler = (0, 0, 0)
            sc.render.resolution_x = sc.render.resolution_y = a.res
        else:
            sc.render.resolution_x, sc.render.resolution_y = a.res, int(a.res * 9 / 16)
            if view == "oblique":
                cam.data.lens = 30
                cam.location = (-120, -160, 330)
                look_at(cam, (256, 256, 0))
            elif view == "street":
                cam.data.lens = 22
                loc, tgt = street_camera(city)
                cam.location = loc
                look_at(cam, tgt)
            else:
                cam.data.lens = 28
                loc, tgt, bid = closeup_camera(city)
                cam.location = loc
                look_at(cam, tgt)
                timing["closeup_building"] = bid
        sc.render.filepath = str(out / f"{view}.png")
        t = time.time()
        bpy.ops.render.render(write_still=True)
        timing["renders"][view] = {"seconds": round(time.time() - t, 2), "res": [sc.render.resolution_x, sc.render.resolution_y],
                                   "camera": [round(x, 2) for x in cam.location]}
        print("rendered", view, timing["renders"][view], flush=True)
    sc.render.filepath = str(out / "last.png")
    if not a.no_export:
        t = time.time()
        bpy.ops.wm.save_as_mainfile(filepath=str(tile / "scene.blend"))
        timing["blend_save_s"] = round(time.time() - t, 2)
        plain_materials()
        t = time.time()
        for ob in list(sc.objects):
            ob.select_set(ob.type == "MESH" and ob.name != "surround")
        bpy.ops.export_scene.gltf(filepath=str(tile / "scene.glb"), export_format="GLB", use_selection=True,
                                  export_apply=True, export_yup=True, export_cameras=False, export_lights=False,
                                  export_texcoords=True, export_normals=True, export_materials="EXPORT")
        timing["glb_export_s"] = round(time.time() - t, 2)
        timing["glb_bytes"] = (tile / "scene.glb").stat().st_size
        timing["blend_bytes"] = (tile / "scene.blend").stat().st_size
    timing["total_s"] = round(time.time() - t0, 2)
    timing["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 1)   # bytes on macOS
    timing["triangles"] = int(sum(len(data[f"{g}__F"]) for g in ("ground", "roads", "green", "walls", "roofs")))
    (tile / "blender_timing.json").write_text(json.dumps(timing, indent=1))
    print("TIMING", json.dumps(timing), flush=True)


main()
