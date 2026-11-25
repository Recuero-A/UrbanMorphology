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
    ap.add_argument("--suffix", default="", help="render / timing file suffix (e.g. _cycles); implies --no-export")
    ap.add_argument("--light", default="default", choices=["default", "sunset", "midday", "studio"],
                    help="themes: golden hour (sunset) or high sun (midday); default = the Milestone 11 calibration")
    ap.add_argument("--sun-az", type=float, default=None, help="override the preset sun azimuth (deg)")
    ap.add_argument("--view", default=None, help="view transform (default AgX; Standard for the studio light)")
    ap.add_argument("--exposure", type=float, default=None, help="override the preset exposure")
    return ap.parse_args(argv)


# light presets: sun elevation and azimuth (deg), sun W/m2 and colour, sky strength, exposure
LIGHTS = {"default": (42.0, 140.0, SUN_W_M2, (1.0, 1.0, 1.0), SKY_STRENGTH, EXPOSURE),
          "sunset": (8.0, 140.0, 6.0, (1.0, 0.56, 0.30), 0.30, -0.25),
          "midday": (62.0, 140.0, 4.5, (1.0, 0.97, 0.92), 0.09, -1.0),
          "studio": (36.0, 140.0, 4.2, (1.0, 0.95, 0.86), 0.55, -0.2)}     # landmark sheets: bright, warm, soft fill


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
    if p.get("image"):                           # themes: image x colour (glTF baseColorTexture x baseColorFactor)
        img = nt.nodes.new("ShaderNodeTexImage")
        img.image = bpy.data.images.load(p["image"], check_existing=True)
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        nt.links.new(img.outputs["Color"], mix.inputs["A"])
        mix.inputs["B"].default_value = base
        nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
        if p.get("alpha"):                       # cut-out: a ROUND node exports as alphaMode MASK
            r = nt.nodes.new("ShaderNodeMath")
            r.operation = "ROUND"
            nt.links.new(img.outputs["Alpha"], r.inputs[0])
            nt.links.new(r.outputs[0], bsdf.inputs["Alpha"])
        m["image"] = True
        return m
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
        t.inputs["Mortar Size"].default_value = 0.005 if tex == "brick" else 0.015   # ~1/3 of the area at 0.01: greyed bricks
        t.inputs["Color1"].default_value = base
        t.inputs["Color2"].default_value = dark
        t.inputs["Mortar"].default_value = [0.32, 0.3, 0.28, 1.0]
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
        if not m.use_nodes or "base_color" not in m or m.get("image"):
            continue
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        for link in list(bsdf.inputs["Base Color"].links):
            m.node_tree.links.remove(link)
        bsdf.inputs["Base Color"].default_value = list(m["base_color"])


# ----------------------------------------------------------------------------------------------- geometry
def make_object(name, V, F, mat, UV, materials, coll, tile=None, smooth=0.0):
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], F.tolist())
    used = sorted(set(mat.tolist()))
    for k in used:
        me.materials.append(materials[k])
    slot = {k: i for i, k in enumerate(used)}
    me.polygons.foreach_set("material_index", [slot[int(k)] for k in mat])
    uvl = me.uv_layers.new(name="UVMap")
    if tile is not None:                                                # image materials repeat every tile_m metres
        UV = UV / tile[mat][:, None, None]
    uvl.data.foreach_set("uv", UV.reshape(-1).astype(np.float32))      # from_pydata keeps face-corner order
    me.validate(clean_customdata=False)
    if smooth:                                                          # themes: rounded crowns, trunks, domes
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
        bm.to_mesh(me)
        bm.free()
        me.shade_smooth()
        me.set_sharp_from_angle(angle=np.radians(smooth))
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
    tile_m = np.array([float(p.get("tile_m", 1.0)) if p.get("image") else 1.0 for p in mats_p], np.float32)
    coll = bpy.data.collections.new("city")
    sc.collection.children.link(coll)
    groups = sorted({k.split("__")[0] for k in data.files if k.endswith("__F")})
    for g in groups:
        if len(data[f"{g}__F"]) and not g.startswith("proto_"):
            make_object(g, data[f"{g}__V"], data[f"{g}__F"], data[f"{g}__mat"], data[f"{g}__UV"], materials, coll, tile_m,
                        smooth=25.0 if g == "roofs" and any(p.get("image") for p in mats_p) else
                        float(cams.get("smooth_terrain", 0.0)) if g in ("ground", "backdrop") else 0.0)
    # instanced props: one mesh per prototype, linked duplicates per placement (glTF keeps the sharing)
    n_inst = 0
    props = bpy.data.collections.new("props")
    sc.collection.children.link(props)
    for g in groups:
        if not g.startswith("proto_") or f"inst_{g[6:]}__T" not in data.files:
            continue
        proto = make_object(g, data[f"{g}__V"], data[f"{g}__F"], data[f"{g}__mat"], data[f"{g}__UV"], materials, props, tile_m,
                            smooth=70.0 if any(p.get("image") for p in mats_p) else 0.0)
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
    E = float(cams.get("extent_m", 512.0))                        # tile (512 m) or multi-tile city
    R = 4000.0 * E / 512.0
    z = float(cams.get("surround_z", -0.02))    # a frame AROUND the tile, never under it; M12 terrain sets its own height
    sur.from_pydata([(-R, -R, z), (R, -R, z), (R, R, z), (-R, R, z), (0, 0, z), (E, 0, z), (E, E, z), (0, E, z)], [],
                    [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)])
    surround_mat = make_material({"name": "surround", "base_color": [0.32, 0.31, 0.29], "roughness": 0.95, "texture": None})
    sur.materials.append(surround_mat)
    sur_ob = bpy.data.objects.new("surround", sur)
    if cams.get("surround", True):                                # studio sheets: no surround (transparent film)
        sc.collection.objects.link(sur_ob)
    if "sea_z" in cams:                                           # themes: the sea continues to the horizon (render only)
        sea = bpy.data.meshes.new("sea_plane")
        zs = float(cams["sea_z"]) - 0.06
        sea.from_pydata([(-R, -R, zs), (R, -R, zs), (R, R, zs), (-R, R, zs)], [], [(0, 1, 2, 3)])
        wm = next((materials[k] for k, p in enumerate(mats_p) if p["name"] == "water"), surround_mat)
        sea.materials.append(wm)
        sc.collection.objects.link(bpy.data.objects.new("sea_plane", sea))
    t_scene = time.time() - t0
    # ---- light: physical sky + sun (same direction)
    world = bpy.data.worlds.new("sky")
    sc.world = world
    world.use_nodes = True
    nt = world.node_tree
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    el_deg, az_deg, sun_w, sun_col, sky_k, expo = LIGHTS[a.light]
    elev, rot = np.radians(el_deg), np.radians(az_deg if a.sun_az is None else a.sun_az)
    sky.sun_elevation, sky.sun_rotation = elev, rot
    sky.sun_disc = False
    nt.links.new(sky.outputs["Color"], nt.nodes["Background"].inputs["Color"])
    nt.nodes["Background"].inputs["Strength"].default_value = sky_k
    if a.light == "studio":                                   # landmark sheets: warm uniform ambient (watercolour-like
        for link in list(nt.nodes["Background"].inputs["Color"].links):          # shadows), the sky is not filmed
            nt.links.remove(link)
        nt.nodes["Background"].inputs["Color"].default_value = (1.0, 0.93, 0.84, 1.0)
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = sun_w
    sun.data.color = sun_col
    sun.data.angle = np.radians(0.6)
    sun.rotation_euler = (np.pi / 2 - elev, 0.0, rot + np.pi / 2)
    sc.collection.objects.link(sun)
    if a.light == "studio":                                   # landmark sheets: a warm fill from the opposite side
        fill = bpy.data.objects.new("fill", bpy.data.lights.new("fill", "SUN"))
        fill.data.energy = 0.35 * sun_w
        fill.data.color = (1.0, 0.9, 0.78)
        fill.data.angle = np.radians(30.0)
        fill.rotation_euler = (np.pi / 2 - np.radians(25.0), 0.0, rot + np.pi + np.pi / 2)
        sc.collection.objects.link(fill)
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
    a.view = a.view or ("Standard" if a.light == "studio" else "AgX")
    sc.view_settings.view_transform = a.view
    sc.view_settings.exposure = expo if a.exposure is None else a.exposure
    if a.light != "default" and a.view == "AgX":
        try:
            sc.view_settings.look = "AgX - Punchy"
        except TypeError:
            pass
    sc.render.image_settings.file_format = "PNG"
    if cams.get("transparent"):                                   # studio sheets: sky not filmed, composited on paper
        sc.render.film_transparent = True
        sc.render.image_settings.color_mode = "RGBA"
    out = tile / "renders"
    out.mkdir(exist_ok=True)
    timing = {"engine": a.engine, "samples": a.samples, "scene_build_s": round(t_scene, 2), "renders": {}}
    target = bpy.data.objects.new("target", None)
    sc.collection.objects.link(target)
    for view in a.views.split(","):
        cam = bpy.data.objects.new(f"cam_{view}", bpy.data.cameras.new(f"cam_{view}"))
        sc.collection.objects.link(cam)
        sc.camera = cam
        cam.data.clip_end = 6000 * E / 512.0
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
        if "res" in spec:                                        # per-view frame (portrait elevations of landmarks)
            sc.render.resolution_x, sc.render.resolution_y = int(spec["res"][0]), int(spec["res"][1])
        if "building" in spec:
            timing["closeup_building"] = spec["building"]
        sc.render.filepath = str(out / f"{view}{a.suffix}.png")
        t = time.time()
        bpy.ops.render.render(write_still=True)
        timing["renders"][view] = {"seconds": round(time.time() - t, 2), "res": [sc.render.resolution_x, sc.render.resolution_y],
                                   "camera": [round(x, 2) for x in cam.location]}
        print("rendered", view, timing["renders"][view], flush=True)
    sc.render.filepath = str(out / "last.png")
    if not a.no_export and not a.suffix:
        t = time.time()
        bpy.ops.wm.save_as_mainfile(filepath=str(tile / "scene.blend"), relative_remap=False)   # image paths survive a move
        timing["blend_save_s"] = round(time.time() - t, 2)
        plain_materials()
        t = time.time()
        for ob in list(sc.objects):
            ob.select_set(ob.type == "MESH" and ob.name not in ("surround", "sea_plane"))
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
    (tile / f"blender_timing{a.suffix}.json").write_text(json.dumps(timing, indent=1))
    print("TIMING", json.dumps(timing), flush=True)


main()
