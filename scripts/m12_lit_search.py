"""Milestone 12: reproducible literature search on terrain-aware street-network generation.

    python scripts/m12_lit_search.py          # -> outputs/m12/literature/{raw/*.json, candidates.csv, search_log.json}

Fixed queries (QUERIES) are sent to three open indexes, no API key, no personal data:
* OpenAlex  /works?search=  (journal and conference metadata with abstracts), top N_PER_QUERY by relevance;
* arXiv     /api/query       (preprints), top N_PER_QUERY by relevance;
* Semantic Scholar /graph/v1/paper/search (cross-venue), top N_PER_QUERY.
Records are de-duplicated by DOI, arXiv id, or normalised title. A first, MECHANICAL screen keeps a record when its
title + abstract mention (a) streets / roads and (b) generation / synthesis / modelling / planning, and flags whether
it also mentions (c) terrain (terrain, elevation, topograph*, slope, DEM, relief, hill*, mountain*) or (d) water /
crossings (river, water, lake, coast*, bridge*, tunnel*, viaduct*). Manual screening of the (a)+(b)+(c or d) set is
recorded in docs/M12_LITERATURE_REVIEW.md. Raw responses are saved so the search can be audited and re-run.
"""
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbanmorph.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "outputs" / "m12" / "literature"
N_PER_QUERY = 50
UA = {"User-Agent": "UrbanMorph literature search (research)"}
QUERIES = [
    "terrain-aware road network generation",
    "topography-aware street network generation",
    "terrain conditioned road generation deep learning",
    "sequential neural road network generation",
    "neural turtle graphics road layout",
    "autoregressive street network generation",
    "graph generation road network generative model",
    "learned road alignment terrain",
    "neural road planning terrain",
    "bridge aware road generation",
    "road network generation rivers bridges",
    "terrain conditioned diffusion road network",
    "procedural road generation terrain slope",
    "urban layout generation terrain elevation",
    "street network generation hilly areas",
    "road network synthesis digital elevation model",
]
A = re.compile(r"\b(street|streets|road|roads|highway|highways)\b", re.I)
B = re.compile(r"\b(generat\w*|synthes\w*|model(l)?ing|planning|layout|design)\b", re.I)
C = re.compile(r"\b(terrain|elevation|topograph\w*|slope\w*|dem|relief|hill\w*|mountain\w*)\b", re.I)
D = re.compile(r"\b(river\w*|water|lake\w*|coast\w*|bridge\w*|tunnel\w*|viaduct\w*)\b", re.I)


FAILED: list = []


def fetch(url: str, tries: int = 4) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception as e:
            print(f"  retry {k + 1} {type(e).__name__}: {url[:80]}", flush=True)
            time.sleep(10 * (k + 1))
    FAILED.append(url)
    return b""


def openalex(q):
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode({"search": q, "per-page": N_PER_QUERY})
    raw = fetch(url)
    recs = []
    for w in json.loads(raw or b"{}").get("results", []):
        inv = w.get("abstract_inverted_index") or {}
        words = sorted(((p, t) for t, ps in inv.items() for p in ps))
        recs.append({"title": w.get("display_name") or "", "year": w.get("publication_year"),
                     "doi": (w.get("doi") or "").replace("https://doi.org/", "").lower(),
                     "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
                     "abstract": " ".join(t for _, t in words), "arxiv": ""})
    return raw, recs


def arxiv(q):
    terms = " AND ".join(f"all:{t}" for t in q.split())
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": terms, "max_results": N_PER_QUERY})
    raw = fetch(url)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    recs = []
    if raw:
        for e in ET.fromstring(raw).findall("a:entry", ns):
            aid = (e.findtext("a:id", "", ns) or "").rsplit("/", 1)[-1]
            recs.append({"title": " ".join((e.findtext("a:title", "", ns) or "").split()),
                         "year": int((e.findtext("a:published", "0000", ns) or "0000")[:4]), "doi": "",
                         "venue": "arXiv", "abstract": " ".join((e.findtext("a:summary", "", ns) or "").split()),
                         "arxiv": re.sub(r"v\d+$", "", aid)})
    return raw, recs


def s2(q):
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(
        {"query": q, "limit": N_PER_QUERY, "fields": "title,year,venue,externalIds,abstract"})
    raw = fetch(url, tries=2)                                         # public endpoint is often rate-limited (429)
    recs = []
    for p in json.loads(raw or b"{}").get("data", []) or []:
        ids = p.get("externalIds") or {}
        recs.append({"title": p.get("title") or "", "year": p.get("year"), "doi": (ids.get("DOI") or "").lower(),
                     "venue": p.get("venue") or "", "abstract": p.get("abstract") or "", "arxiv": ids.get("ArXiv") or ""})
    return raw, recs


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def main() -> int:
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    log = {"date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_per_query": N_PER_QUERY, "runs": []}
    merged: dict = {}
    for qi, q in enumerate(QUERIES):
        for name, fn in (("openalex", openalex), ("arxiv", arxiv), ("semanticscholar", s2)):
            raw, recs = fn(q)
            (OUT / "raw" / f"{name}_{qi:02d}.json").write_bytes(raw or b"")
            log["runs"].append({"query": q, "source": name, "records": len(recs)})
            for r in recs:
                key = r["doi"] or (f"arxiv:{r['arxiv']}" if r["arxiv"] else "") or norm(r["title"])
                alt = norm(r["title"])
                hit = merged.get(key) or next((v for v in merged.values() if norm(v["title"]) == alt), None)
                if hit is None:
                    merged[key] = {**r, "sources": {name}, "queries": {qi}}
                else:
                    hit["sources"].add(name)
                    hit["queries"].add(qi)
                    for f in ("doi", "arxiv", "abstract", "venue"):
                        hit[f] = hit[f] or r[f]
            time.sleep(3.5 if name == "semanticscholar" else 1.0)     # stay inside the public rate limits
        print(q, flush=True)
    rows = []
    for r in merged.values():
        text = f"{r['title']} {r['abstract']}"
        rows.append({"title": r["title"], "year": r["year"], "venue": r["venue"], "doi": r["doi"], "arxiv": r["arxiv"],
                     "sources": ";".join(sorted(r["sources"])), "n_queries": len(r["queries"]),
                     "roads": bool(A.search(text)), "generation": bool(B.search(text)),
                     "terrain": bool(C.search(text)), "water_crossing": bool(D.search(text)),
                     "has_abstract": bool(r["abstract"])})
    rows.sort(key=lambda r: (-(r["roads"] and r["generation"] and (r["terrain"] or r["water_crossing"])), -r["n_queries"]))
    with open(OUT / "candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    screen = [r for r in rows if r["roads"] and r["generation"] and (r["terrain"] or r["water_crossing"])]
    log["unique_records"] = len(rows)
    log["failed_requests"] = FAILED
    log["mechanical_screen"] = {"roads_and_generation": sum(r["roads"] and r["generation"] for r in rows),
                                "plus_terrain_or_water": len(screen), "no_abstract": sum(not r["has_abstract"] for r in rows)}
    (OUT / "search_log.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log["mechanical_screen"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
