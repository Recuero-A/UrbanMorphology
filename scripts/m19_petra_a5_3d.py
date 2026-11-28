"""Petra, M19 A5 (TNg_s0, seed 3): bare terrain + streets as a GLB for the Godot walk viewer.

    python scripts/m19_petra_a5_3d.py  ->  outputs/trialcities/petra/petra_A5_s3_streets/{scene.glb, viewer.json}

Terrain is the M19 site field (4 m), coloured by slope; streets are 6 m ribbons draped 0.25 m above it.
Dark grey = A0 streets, orange = connectors A5 added after growth. Godot axes: x = east, y = up, z = -north.
"""
import json
import struct
import warnings
import zlib
import pickle
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/trialcities/petra/petra_A5_s3_streets"
RUN = ROOT / "outputs/m19/arms/runs/petra__A0__TNg_s0__s3.pkl.gz"
WIDTH, LIFT, STEP = 6.0, 0.25, 4.0


def glb(meshes, path):
    """meshes: [(name, positions f32 (n,3), normals f32 (n,3), colors f32 (n,4) or None, indices u32, rgba)]."""
    blob, views, accs, gmeshes, mats, nodes = bytearray(), [], [], [], [], []

    def add(arr, target, comp, typ, minmax=False):
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        blob.extend(arr.tobytes())
        views.append({"buffer": 0, "byteOffset": off, "byteLength": arr.nbytes, "target": target})
        a = {"bufferView": len(views) - 1, "componentType": comp, "count": len(arr), "type": typ}
        if minmax:
            a["min"], a["max"] = arr.min(0).tolist(), arr.max(0).tolist()
        accs.append(a)
        return len(accs) - 1

    for name, P, N, C, I, rgba in meshes:
        attrs = {"POSITION": add(P, 34962, 5126, "VEC3", True), "NORMAL": add(N, 34962, 5126, "VEC3")}
        if C is not None:
            attrs["COLOR_0"] = add(C, 34962, 5126, "VEC4")
        mats.append({"name": name, "pbrMetallicRoughness": {"baseColorFactor": rgba, "metallicFactor": 0.0,
                                                            "roughnessFactor": 0.95}})
        gmeshes.append({"name": name, "primitives": [{"attributes": attrs, "indices": add(I, 34963, 5125, "SCALAR"),
                                                      "material": len(mats) - 1}]})
        nodes.append({"name": name, "mesh": len(gmeshes) - 1})
    while len(blob) % 4:
        blob.append(0)
    doc = {"asset": {"version": "2.0", "generator": "m19_petra_a5_3d"}, "scene": 0,
           "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": gmeshes, "materials": mats,
           "accessors": accs, "bufferViews": views, "buffers": [{"byteLength": len(blob)}]}
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(blob), 0x004E4942) + bytes(blob))


def normals(P, I):
    t = I.reshape(-1, 3)
    fn = np.cross(P[t[:, 1]] - P[t[:, 0]], P[t[:, 2]] - P[t[:, 0]])
    N = np.zeros_like(P)
    for k in range(3):
        np.add.at(N, t[:, k], fn)
    return (N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)).astype(np.float32)


def main():
    from m19_eval import fields
    from m19_maps import new_edges
    from urbanmorph.experiments.graph_io import graph_from_json
    S = fields("petra")[0][2].S
    E = S.elev.astype(float)
    h, w = E.shape
    xs = S.x0 + (np.arange(w) + 0.5) * S.res
    ys = S.y1 - (np.arange(h) + 0.5) * S.res
    X, Y = np.meshgrid(xs, ys)
    P = np.c_[X.ravel(), E.ravel(), -Y.ravel()].astype(np.float32)
    r = np.arange(h - 1)[:, None] * w + np.arange(w - 1)[None, :]
    I = np.c_[r.ravel(), (r + w).ravel(), (r + 1).ravel(), (r + 1).ravel(), (r + w).ravel(), (r + w + 1).ravel()]
    I = I.astype(np.uint32).ravel()
    gy, gx = np.gradient(E, S.res)
    sl = np.hypot(gx, gy).ravel() * 100                                       # slope %
    sand, rock = np.array([0.86, 0.72, 0.55]), np.array([0.62, 0.38, 0.30])
    t = np.clip(sl / 60, 0, 1)[:, None]
    C = np.c_[(1 - t) * sand + t * rock, np.ones(len(sl))].astype(np.float32)
    meshes = [("terrain", P, normals(P, I), None, I, [0.80, 0.58, 0.42, 1])]          # Petra sandstone

    run = pickle.loads(zlib.decompress(RUN.read_bytes()))
    k = run["ids"].index("petra")
    g0 = graph_from_json(run["graphs"][k])
    g5 = graph_from_json(run["A5"]["snapshots"][k][-1][1])
    s0 = g0.nodes[g0.edges]                                                    # A0 segments (m, 2, 2)

    def on_a0(seg):                                                            # a piece of an A0 street split by A5
        m = seg.mean(0)
        a, d = s0[:, 0], s0[:, 1] - s0[:, 0]
        t = np.clip(((m - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-9), 0, 1)
        return np.min(np.linalg.norm(a + d * t[:, None] - m, axis=1)) < 0.5

    added = {tuple(np.round(np.sort(s, axis=0).ravel(), 2)) for s in new_edges(g5, g0)}
    base, conn = [], []
    for a, b in g5.edges:
        seg = g5.nodes[[a, b]]
        new = tuple(np.round(np.sort(seg, axis=0).ravel(), 2)) in added and not on_a0(seg)
        (conn if new else base).append(seg)

    def ribbons(segs):
        Ps, Is = [], []
        for p, q in segs:
            L = float(np.hypot(*(q - p)))
            if L < 0.5:
                continue
            n = max(2, int(np.ceil(L / STEP)) + 1)
            pts = p + (q - p) * np.linspace(0, 1, n)[:, None]
            d = (q - p) / L
            off = np.array([-d[1], d[0]]) * WIDTH / 2
            side = np.r_[pts + off, pts - off]
            z = np.asarray(S.sample(side[:, 0], side[:, 1]), float) + LIFT
            if not np.isfinite(z).all():                                       # off the terrain grid
                continue
            o = sum(len(x) for x in Ps)
            Ps.append(np.c_[side[:, 0], z, -side[:, 1]])
            i = np.arange(n - 1)
            Is.append(np.c_[o + i, o + n + i, o + i + 1, o + i + 1, o + n + i, o + n + i + 1].ravel())
        P_ = np.concatenate(Ps).astype(np.float32)
        I_ = np.concatenate(Is).astype(np.uint32)
        return P_, normals(P_, I_), None, I_

    for name, segs, rgba in (("streets_A0", base, [0.18, 0.18, 0.2, 1]), ("connectors_A5", conn, [0.95, 0.45, 0.15, 1])):
        if segs:
            Pr, Nr, _, Ir = ribbons(segs)
            meshes.append((name, Pr, Nr, None, Ir, rgba))
    OUT.mkdir(parents=True, exist_ok=True)
    glb(meshes, OUT / "scene.glb")

    deg = np.bincount(g5.edges.ravel(), minlength=len(g5.nodes))
    c = g5.nodes[np.argmin(np.linalg.norm(g5.nodes - g5.nodes[deg >= 3].mean(0), axis=1))]
    z0 = float(np.asarray(S.sample(np.array([c[0]]), np.array([c[1]])))[0])
    look = c + np.array([60.0, 60.0])
    zl = float(np.asarray(S.sample(np.array([look[0]]), np.array([look[1]])))[0])
    view = {"spawn": [float(c[0]), z0 + 1.7, float(-c[1])], "look_at": [float(look[0]), zl + 1.7, float(-look[1])],
            "extent": float(S.res * w)}
    (OUT / "viewer.json").write_text(json.dumps(view))
    print(f"{len(base)} A0 street segments, {len(conn)} A5 connector segments -> {OUT / 'scene.glb'}")


if __name__ == "__main__":
    main()
