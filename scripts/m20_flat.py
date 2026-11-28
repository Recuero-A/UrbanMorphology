"""Milestone 20, D1: the flat TNg-R settlements (docs/M20_LOG.md D1-impl, "TNg-R selection rule", fixed before this ran).

    python scripts/m20_flat.py            # walk, gate, build and select; resumable (every OSM response is cached)

Pool: the train list only (data/processed/m20/pool/lists/train.csv.gz, R1-R4), in list order (rank) per region x size
stratum. Flat = built-up-weighted median slope < 6 % (m20_walk.terrain_one, FABDEM 32 m; values already in
train_terrain.csv are reused, new ones go to train_flat_terrain.csv). Gate = m20_walk.osm_gate unchanged (pinned
snapshot, osm_base freshness check; a failure is final). At most 2 settlements per 1 degree block in the train split,
counting the train-H settlements. Build = m20_towns osm / site / real unchanged (split label "train_flat"). Windows =
the build's kept windows with at least one vehicular edge.

Matching: each TNg-H stratum's target is its TNg-H added street-km (outputs/m20/tng/windows_H.csv). Strata are visited
round robin (sorted); each takes its next qualifying candidate until its km reaches the target; the last settlement
of a stratum contributes only the windows needed, in the order sha256("m20-D1-TNgR" + window id). A stratum that runs
out of candidates passes its remaining km to the other size of the same region; failing that it is reported.

Outputs: data/processed/m20/walk/train_flat_terrain.csv, train_flat_osm.csv (every candidate's outcome),
train_flat_accepted.csv (gate passes that were built), train_flat_selected.csv (+ train_flat_windows.json: the
windows each selected settlement contributes).
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import m20_towns as MT
import m20_walk as MW

WALK = MW.WALK
FLAT_MAX = 6.0
SALT = "m20-D1-TNgR"
LABEL = "train_flat"


def train_list():
    L = pd.read_csv(MW.POOL / "lists" / "train.csv.gz")
    L = L[L.region.isin(["R1", "R2", "R3", "R4"])].copy()
    L["stratum"] = [MW.stratum(r) for _, r in L.iterrows()]
    return L


def targets():
    W = pd.read_csv(ROOT / "outputs" / "m20" / "tng" / "windows_H.csv").fillna({"dropped": ""})
    A = pd.read_csv(WALK / "train_accepted.csv")
    W = W[W.dropped == ""].merge(A[["id", "stratum"]], left_on="town", right_on="id")
    return W.groupby("stratum").street_km.sum().to_dict()


def town_windows(sid):
    """(window id, street km) of the kept windows with a vehicular edge, in the order sha256(SALT + id)."""
    real = pickle.loads((MT.Paths(False).towns / sid / "real.pkl").read_bytes())
    ws = [(w["id"], MT.km(w["graph"])) for w in real["windows"] if w["keep"] and len(w["graph"].edges)]
    return sorted(ws, key=lambda x: hashlib.sha256((SALT + x[0]).encode()).hexdigest())


def main() -> int:
    import datetime as dt
    if dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat(MW.SNAPSHOT.replace("Z", "+00:00")):
        raise SystemExit("the pinned snapshot has not passed")
    L = train_list()
    T0 = pd.read_csv(WALK / "train_terrain.csv")
    tf_path, log_path = WALK / "train_flat_terrain.csv", WALK / "train_flat_osm.csv"
    TF = pd.read_csv(tf_path) if tf_path.exists() else pd.DataFrame(columns=["id"])
    terr = {r["id"]: r for r in T0.to_dict("records")} | {r["id"]: r for r in TF.to_dict("records")}
    tf_rows = TF.to_dict("records")
    tgt = targets()
    remaining = {s: tgt.get(s, 0.0) for s in sorted(L.stratum.unique()) if s != "village"}
    print("targets (TNg-H added km):", {k: round(v, 1) for k, v in remaining.items()}, flush=True)
    H = pd.read_csv(WALK / "train_accepted.csv")
    per_block = H.groupby("block").size().to_dict()
    hill = set(H.id)
    queues = {s: L[L.stratum == s].sort_values("rank").to_dict("records") for s in remaining}
    pos = {s: 0 for s in queues}
    log, accepted, selected, win_used = [], [], [], {}
    exhausted = set()
    P = MT.Paths(False)

    def terrain(r):
        if r["id"] in terr:
            return terr[r["id"]]
        poly = MW.polygons([r["id"]])
        MW.prefetch([r], poly)
        t = MW.terrain_one(r, poly[r["id"]])
        row = {k: r[k] for k in ("id", "rank", "stratum", "kind", "region", "iso3", "block", "lon", "lat")} | t
        tf_rows.append(row)
        terr[r["id"]] = row
        pd.DataFrame(tf_rows).to_csv(tf_path, index=False)
        return row

    def next_flat(s):
        """The next candidate of stratum s that is flat, under the block cap and passes the gate; built."""
        while pos[s] < len(queues[s]):
            r = queues[s][pos[s]]
            pos[s] += 1
            if r["id"] in hill:
                continue
            t = terrain(r)
            if t.get("status") != "ok" or not (float(t["wmedian_slope"]) < FLAT_MAX):
                continue                                        # not flat (or no DEM): not a TNg-R candidate
            rec = {k: r[k] for k in ("id", "rank", "stratum", "kind", "region", "iso3", "block", "lon", "lat")}
            rec |= {k: t.get(k) for k in ("status", "wmedian_slope", "median_slope", "built_m2", "cells", "utm", "dem_tiles")}
            if per_block.get(r["block"], 0) >= 2:
                log.append({**rec, "outcome": "block_cap"})
                continue
            g = MW.osm_gate(rec, MW.polygons([r["id"]])[r["id"]])
            rec |= g
            if not g["gate"]:
                log.append({**rec, "outcome": "gate_failed"})
                pd.DataFrame(log).to_csv(log_path, index=False)
                print(s, r["id"], r["iso3"], "gate_failed", g["vehicular_km"], g["coverage_any_way"], flush=True)
                continue
            row = {**rec, "eligible": False, "band": "", "outcome": "accepted", "split": LABEL}
            accepted.append(row)
            pd.DataFrame(accepted).drop(columns=["split"]).to_csv(WALK / f"{LABEL}_accepted.csv", index=False)
            MT.stage_osm([row], P, 8)
            if (P.raw / f"{r['id']}.json").exists():
                MT.stage_site([row], P)
            if not (P.towns / r["id"] / "real.pkl").exists():
                MT.stage_real([row], P) if (P.sites / f"{r['id']}.json").exists() else None
            if not (P.towns / r["id"] / "real.pkl").exists():
                log.append({**rec, "outcome": "build_failed_or_refused"})
                pd.DataFrame(log).to_csv(log_path, index=False)
                print(s, r["id"], "build failed or refused (oversize)", flush=True)
                continue
            per_block[r["block"]] = per_block.get(r["block"], 0) + 1
            log.append({**rec, "outcome": "accepted"})
            pd.DataFrame(log).to_csv(log_path, index=False)
            return row
        return None

    while True:
        active = [s for s in sorted(remaining) if remaining[s] > 1e-9 and s not in exhausted]
        if not active:
            break
        for s in active:
            row = next_flat(s)
            if row is None:
                exhausted.add(s)
                reg, size = s.split(":")
                other = f"{reg}:{'city' if size == 'town' else 'town'}"
                if other in remaining and other not in exhausted:
                    print(s, f"exhausted: {remaining[s]:.1f} km passed to {other}", flush=True)
                    remaining[other] += remaining[s]
                    remaining[s] = 0.0
                continue
            ws = town_windows(row["id"])
            take, km = [], 0.0
            for wid, k in ws:
                if km >= remaining[s]:
                    break
                take.append(wid)
                km += k
            remaining[s] -= km
            win_used[row["id"]] = take
            selected.append({**{k: row[k] for k in ("id", "rank", "stratum", "kind", "region", "iso3", "block", "lon",
                                                     "lat", "wmedian_slope", "vehicular_km", "coverage_any_way")},
                             "windows_available": len(ws), "windows_used": len(take), "street_km_used": round(km, 3),
                             "truncated": len(take) < len(ws)})
            pd.DataFrame(selected).to_csv(WALK / f"{LABEL}_selected.csv", index=False)
            (WALK / f"{LABEL}_windows.json").write_text(json.dumps(win_used, indent=1))
            print(s, row["id"], row["iso3"], f"slope {row['wmedian_slope']}", f"+{km:.1f} km ({len(take)}/{len(ws)} windows)",
                  f"remaining {max(remaining[s], 0):.1f}", flush=True)
    pd.DataFrame(log).to_csv(log_path, index=False)
    S = pd.DataFrame(selected)
    print("unfilled (km):", {k: round(v, 1) for k, v in remaining.items() if v > 1e-9} or "none")
    print(S.groupby("stratum").agg(n=("id", "size"), km=("street_km_used", "sum")).round(1).to_string() if len(S) else "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
