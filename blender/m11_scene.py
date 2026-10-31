"""Milestone 11: build, light, render and export one tile in Blender (runs INSIDE Blender, not the project venv).

    /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python-exit-code 1 --python blender/m11_scene.py -- \
        --tile outputs/m11/<tile_id> [--views aerial,oblique,street,closeup] [--engine EEVEE|CYCLES] [--res 1600]

Reads <tile>/meshes.npz and <tile>/cameras.json (both from scripts/m11_build_city.py). Writes
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


def main():
    a = args()
    tile = Path(a.tile).resolve()
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    data = np.load(tile / "meshes.npz")
    cams = json.loads((tile / "cameras.json").read_text())        # from city3d.cameras (python stage)
    mats_p = json.loads(str(data["materials"]))
    materials = [make_material(p) for p in mats_p]
    coll = bpy.data.collections.new("city")
    sc.collection.children.link(coll)
    groups = sorted({k.split("__")[0] for k in data.files if k.endswith("__F")})
    for g in groups:
        if len(data[f"{g}__F"]) and not g.startswith("proto_"):
            make_object(g, data[f"{g}__V"], data[f"{g}__F"], data[f"{g}__mat"], data[f"{g}__UV"], materials, coll)
    # instanced props: one mesh per prototype, linked duplicates per placement (glTF keeps the sharing)
    n_inst = 0
    props = bpy.data.collections.new("props")
    sc.collection.children.link(props)
    for g in groups:
        if not g.startswith("proto_") or f"inst_{g[6:]}__T" not in data.files:
            continue
        proto = make_object(g, data[f"{g}__V"], data[f"{g}__F"], data[f"{g}__mat"], data[f"{g}__UV"], materials, props)
        mesh = proto.data
        bpy.data.objects.remove(proto)
        for k, t in enumerate(data[f"inst_{g[6:]}__T"]):
            ob = bpy.data.objects.new(f"{g[6:]}_{k}", mesh)
            ob.location = (float(t[0]), float(t[1]), float(t[2]))
            if len(t) == 5:                                   # tree: x, y, z, scale, yaw
                ob.scale = (float(t[3]),) * 3
                ob.rotation_euler = (0.0, 0.0, float(t[4]))
            else:                                             # streetlight: x, y, z, yaw
                ob.rotation_euler = (0.0, 0.0, float(t[3]))
            props.objects.link(ob)
            n_inst += 1
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
        spec = cams.get(view)
        if spec is None:
            print("no camera for", view, flush=True)
            continue
        cam.location = spec["location"]
        if spec["type"] == "ORTHO":
            cam.data.type = "ORTHO"
            cam.data.ortho_scale = spec["ortho_scale"]
            cam.rotation_euler = (0, 0, 0)
            sc.render.resolution_x = sc.render.resolution_y = a.res
        else:
            cam.data.lens = spec["lens"]
            look_at(cam, spec["look_at"])
            sc.render.resolution_x, sc.render.resolution_y = a.res, int(a.res * 9 / 16)
        if "building" in spec:
            timing["closeup_building"] = spec["building"]
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
    timing["triangles"] = int(sum(len(data[f"{g}__F"]) for g in groups if not g.startswith("proto_")))
    timing["instances"] = n_inst
    (tile / "blender_timing.json").write_text(json.dumps(timing, indent=1))
    print("TIMING", json.dumps(timing), flush=True)


main()
