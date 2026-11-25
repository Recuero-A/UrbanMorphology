"""Milestone 12: forward citation chasing (OpenAlex) from seed papers, to complement the keyword search.

    python scripts/m12_lit_citations.py   # -> outputs/m12/literature/citations.csv, raw/cites_*.json

Keyword search recalled only 5 of 14 known key papers (docs/M12_LITERATURE_REVIEW.md), so every work CITING a seed
is listed (OpenAlex filter cites:<id>, up to 200 per seed) and screened with the same mechanical rule as
m12_lit_search.py. Seeds are resolved by DOI, or by title when there is no DOI.
"""
import csv
import json
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m12_lit_search import A, B, C, D, OUT, fetch  # noqa: E402

SEEDS = {"fang2022": "doi:10.1080/13658816.2022.2072849", "gu2024": "doi:10.3390/ijgi13060203",
         "galin2010": "doi:10.1111/j.1467-8659.2009.01612.x", "galin2011": "doi:10.1111/j.1467-8659.2011.02055.x",
         "ntg2019": "title:Neural Turtle Graphics for Modeling City Road Layouts",
         "parish2001": "doi:10.1145/383259.383292", "chen2008": "doi:10.1145/1360612.1360702",
         "cav2024": "doi:10.1002/cav.2267", "birsak2022": "title:Large-Scale Auto-Regressive Modeling Of Street Networks",
         "emilien2012": "doi:10.1007/s00371-012-0699-7"}


def resolve(spec: str):
    kind, val = spec.split(":", 1)
    if kind == "doi":
        raw = fetch("https://api.openalex.org/works/doi:" + urllib.parse.quote(val))
        return json.loads(raw) if raw else None
    raw = fetch("https://api.openalex.org/works?" + urllib.parse.urlencode({"filter": f"title.search:{val}", "per-page": 1}))
    res = json.loads(raw or b"{}").get("results", [])
    return res[0] if res else None


def main() -> int:
    rows, seeds = {}, {}
    for name, spec in SEEDS.items():
        w = resolve(spec)
        if not w:
            seeds[name] = None
            continue
        wid = w["id"].rsplit("/", 1)[-1]
        seeds[name] = {"openalex": wid, "title": w.get("display_name"), "cited_by_count": w.get("cited_by_count")}
        raw = fetch("https://api.openalex.org/works?" + urllib.parse.urlencode({"filter": f"cites:{wid}", "per-page": 200}))
        (OUT / "raw" / f"cites_{name}.json").write_bytes(raw or b"")
        for c in json.loads(raw or b"{}").get("results", []):
            inv = c.get("abstract_inverted_index") or {}
            ab = " ".join(t for _, t in sorted((p, t) for t, ps in inv.items() for p in ps))
            text = f"{c.get('display_name') or ''} {ab}"
            r = rows.setdefault(c["id"], {"title": c.get("display_name"), "year": c.get("publication_year"),
                                          "doi": (c.get("doi") or "").replace("https://doi.org/", ""), "cites": set(),
                                          "roads": bool(A.search(text)), "generation": bool(B.search(text)),
                                          "terrain": bool(C.search(text)), "water_crossing": bool(D.search(text))})
            r["cites"].add(name)
        print(name, seeds[name], flush=True)
        time.sleep(1.0)
    out = sorted(rows.values(), key=lambda r: (-(r["roads"] and r["generation"] and (r["terrain"] or r["water_crossing"])), -len(r["cites"])))
    with open(OUT / "citations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["title", "year", "doi", "cites", "roads", "generation", "terrain", "water_crossing"])
        w.writeheader()
        for r in out:
            w.writerow({**r, "cites": ";".join(sorted(r["cites"]))})
    (OUT / "citations_seeds.json").write_text(json.dumps(seeds, indent=1))
    screen = [r for r in out if r["roads"] and r["generation"] and (r["terrain"] or r["water_crossing"])]
    print(json.dumps({"citing_works": len(out), "plus_terrain_or_water": len(screen)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
