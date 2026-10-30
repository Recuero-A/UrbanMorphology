"""Milestone 2, step C: cached raw OSM -> quality-checked, split dataset (offline).

1. Process every planned tile whose raw data is cached (primaries + reserves).
2. Quality-check each tile; for each unusable or missing primary, pick a
   replacement reserve (same stratum first, then lowest rank in the city).
   Reserves that are needed but not yet downloaded are written to
   data/processed/m2/replacements_needed.csv (download them with
   ``scripts/m2_download.py --replacements``, then rerun this script).
3. Dataset-level checks (duplicate IDs / overlapping tiles / identical rasters).
4. Geographic splits with blocks + buffer.
5. Write samples.csv, per-sample JSON (with split) and per-city quality reports.
"""

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

from urbanmorph.config import M2_DIR, M2_OUTPUTS
from urbanmorph.data.m2_dataset import INDEX_PATH, index_row, process_sample, save_m2_sample, write_meta
from urbanmorph.data.osm import is_cached, load_raw
from urbanmorph.quality.checks import dataset_flags
from urbanmorph.sampling.plan_io import area_from_row, load_plan
from urbanmorph.sampling.study_area import load_config
from urbanmorph.splits.geographic import EXCLUDED, assign_splits, enforce_buffer, split_violations

REPLACEMENTS_PATH = M2_DIR / "replacements_needed.csv"


def choose_dataset(plan: pd.DataFrame, status: dict) -> tuple[list[str], list[dict]]:
    """Return (sample_ids in the dataset, replacement records)."""
    chosen, records, used = [], [], set()
    for code, grp in plan.groupby("city_code"):
        reserves = grp[grp["role"] == "reserve"].sort_values("rank")
        for _, p in grp[grp["role"] == "primary"].sort_values("rank").iterrows():
            st = status.get(p["sample_id"], "missing")
            if st == "usable":
                chosen.append(p["sample_id"])
                continue
            if st == "missing":  # not acquired yet: pending, not a failure
                records.append({"city_code": code, "primary_id": p["sample_id"], "reason": "pending_download",
                                "replacement_id": None, "replacement_status": None, "same_stratum": None})
                continue
            options = pd.concat([reserves[reserves["stratum"] == p["stratum"]], reserves[reserves["stratum"] != p["stratum"]]])
            options = options[~options["sample_id"].isin(used)]
            options = options[options["sample_id"].map(lambda s: status.get(s, "missing") != "unusable")]
            rec = {"city_code": code, "primary_id": p["sample_id"], "reason": st, "replacement_id": None,
                   "replacement_status": None, "same_stratum": None}
            if len(options):
                r = options.iloc[0]
                used.add(r["sample_id"])
                rec.update(replacement_id=r["sample_id"], replacement_status=status.get(r["sample_id"], "missing"),
                           same_stratum=bool(r["stratum"] == p["stratum"]))
                if rec["replacement_status"] == "usable":
                    chosen.append(r["sample_id"])
            records.append(rec)
    return chosen, records


def quality_report(code: str, table: pd.DataFrame, plan: pd.DataFrame, repl: list[dict], out: Path) -> None:
    t = table[table["city_code"] == code]
    lines = [f"# Data-quality report: {code}", "",
             f"Planned primaries: {int(((plan.city_code == code) & (plan.role == 'primary')).sum())}  ",
             f"Processed tiles (incl. reserves): {len(t)}  ",
             f"Usable in dataset: {int((t['in_dataset']).sum())}", "",
             "Severity: **error** = excluded; **warning** = kept, probably incomplete/unusual; "
             "**info** = not a defect (an empty green channel may be real or unmapped).", "",
             "| flag | severity | tiles |", "|---|---|---|"]
    for sev in ("error", "warning", "info"):
        codes = t[f"flags_{sev}"].fillna("").str.split(";").explode()
        for c, n in codes[codes != ""].value_counts().items():
            lines.append(f"| {c} | {sev} | {n} |")
    r = [x for x in repl if x["city_code"] == code]
    if r:
        lines += ["", "## Replacements", "", "| primary | reason | replacement | status | same stratum |", "|---|---|---|---|---|"]
        lines += [f"| {x['primary_id']} | {x['reason']} | {x['replacement_id']} | {x['replacement_status']} | {x['same_stratum']} |" for x in r]
    lines += ["", "## Tiles", "", "| sample | stratum | usable | split | OSM bldg cov | GHSL built | warnings | info |",
              "|---|---|---|---|---|---|---|---|"]
    for _, x in t.sort_values("sample_id").iterrows():
        lines.append(f"| {x.sample_id} | {x.stratum} | {x.usable} | {x.split} | {x.building_coverage_vector:.2f} | "
                     f"{x.ghsl_built_2020:.2f} | {x.flags_warning or ''} | {x.flags_info or ''} |")
    out.write_text("\n".join(lines) + "\n")


def main() -> int:
    cfg = load_config()
    plan = load_plan()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    metas, status = {}, {}
    for _, row in plan.iterrows():
        area = area_from_row(row, cfg)
        if not is_cached(area):
            continue
        try:
            layout, meta, sample = process_sample(area, load_raw(area), row.to_dict(), cfg, now)
        except Exception as exc:
            print(f"PROCESSING FAILED {row['sample_id']}: {exc!r}")
            status[row["sample_id"]] = "processing_failed"
            continue
        save_m2_sample(layout, meta, sample)
        metas[row["sample_id"]] = meta
        status[row["sample_id"]] = "usable" if meta["usable"] else "unusable"

    chosen, repl = choose_dataset(plan, status)
    pd.DataFrame(repl).to_csv(M2_DIR / "replacements.csv", index=False)
    needed = [r for r in repl if r["replacement_id"] and r["replacement_status"] == "missing"]
    pd.DataFrame(needed or [], columns=["city_code", "primary_id", "reason", "replacement_id",
                                        "replacement_status", "same_stratum"]).to_csv(REPLACEMENTS_PATH, index=False)

    table = pd.DataFrame([index_row(m) for m in metas.values()])
    table["in_dataset"] = table["sample_id"].isin(chosen)

    # dataset-level checks on the tiles that make up the dataset
    ds = table[table["in_dataset"]]
    for sid, flags in dataset_flags(ds, cfg["tile"]["size_m"]).items():
        if flags:
            metas[sid]["quality_flags"] += flags
            metas[sid]["usable"] = False
            table.loc[table.sample_id == sid, "usable"] = False
            table.loc[table.sample_id == sid, "flags_error"] += ";" + ";".join(f["code"] for f in flags)
            table.loc[table.sample_id == sid, "in_dataset"] = False

    sp = cfg["splits"]
    ds = assign_splits(table[table["in_dataset"]], sp["fractions"], sp["block_size_m"], cfg["seed"])
    ds = enforce_buffer(ds, sp["buffer_m"])
    viol = split_violations(ds, sp["buffer_m"])
    assert not viol, f"split buffer violated: {viol[:3]}"
    table = table.drop(columns=["split", "block_id"]).merge(ds[["sample_id", "split", "block_id"]], on="sample_id", how="left")
    table["split"] = table["split"].fillna("not_in_dataset")
    for _, r in table.iterrows():
        m = metas[r["sample_id"]]
        m["split"], m["block_id"], m["in_dataset"] = r["split"], r["block_id"] if pd.notna(r["block_id"]) else None, bool(r["in_dataset"])
        write_meta(m)

    table.to_csv(INDEX_PATH, index=False)
    qdir = M2_OUTPUTS / "quality"
    qdir.mkdir(parents=True, exist_ok=True)
    for code in sorted(plan["city_code"].unique()):
        quality_report(code, table, plan, repl, qdir / f"{code}.md")

    # City eligibility: median OSM/GHSL building-coverage ratio over the city's dataset tiles.
    ds = table[table["in_dataset"]].copy()
    ds["osm_ghsl_ratio"] = ds["building_coverage_vector"] / ds["ghsl_built_2020"]
    min_ratio = cfg["quality"].get("min_city_median_osm_to_ghsl_ratio")
    city_q = ds.groupby(["country", "city", "city_code"])["osm_ghsl_ratio"].median().rename("median_osm_ghsl_ratio").reset_index()
    city_q["eligible"] = city_q["median_osm_ghsl_ratio"] >= min_ratio if min_ratio is not None else True
    city_q.to_csv(M2_DIR / "city_completeness.csv", index=False)
    print(city_q.round(3).to_string(index=False))
    failed = city_q[~city_q["eligible"]]
    if len(failed):
        print(f"\nCITY ELIGIBILITY FAILED for {list(failed.city_code)} (median ratio < {min_ratio}); "
              "replace them with [[backup_cities]] in the config and rerun.")

    summary = (table[table["in_dataset"]].groupby(["country", "city"])["split"].value_counts().unstack(fill_value=0))
    print(summary.to_string())
    print(f"\n{int(table['in_dataset'].sum())} tiles in dataset; {len(needed)} replacements still to download; "
          f"{(table['split'] == EXCLUDED).sum()} excluded by buffer")
    print(f"Index: {INDEX_PATH}\nQuality reports: {qdir}")
    (M2_DIR / "build_summary.json").write_text(json.dumps(
        {"built_at_utc": now, "config_sha256_12": cfg["_hash"], "tiles_processed": len(table),
         "tiles_in_dataset": int(table["in_dataset"].sum()), "replacements_needed": len(needed)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
