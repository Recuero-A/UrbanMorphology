# Milestone 12 progress log

Dated entries: decisions, experiments, results and limitations, in the order they happened. Summaries live in
`MILESTONE_12.md`; this log is the audit trail.

## 2026-11-04: Phase 3–5 start (after approval of Phase 2)

**Dataset expansion: Los Angeles and San Diego** (`scripts/m12_select_regions.py`, `m12_build_tiles.py`)
* **Rules** were fixed in the script before any terrain value was seen:
  * M2 urban eligibility (GHSL 2020 ≥ 0.15) inside the city boundary;
  * USGS 3DEP 32 m screening, landform p90 classes;
  * Los Angeles → train, steep first, at most 120 tiles; buffers to the LA val/test tiles (3 km blocks and
    1,536 m); ≥ 1,024 m spacing;
  * San Diego → a held-out city: 3 km blocks assigned at random to `val_t` / `test_t`, 1,536 m between them.
* **Eligible cells:** LA 3,206 (screening: 778 steep, 773 moderate, 1,655 flat); San Diego 1,618 (831 / 542 / 245).
* **Selected:**
  * LA: 120 steep train tiles;
  * San Diego `val_t`: 40 (23 steep, 7 moderate, 10 flat);
  * San Diego `test_t`: 34 (25 / 5 / 4), **not built, not fetched, not read**.
* **Built:** 160 tiles with the unchanged M2 processing; 0 skipped. Index: `data/processed/m12/samples_m12.csv`.
* **Downloads:**
  * GHSL tile R6_C8 (CC BY 4.0, the M2 product) was downloaded to apply the M2 eligibility rule in San Diego;
  * the San Diego city boundary: one Nominatim request, the M2 backup query;
  * OSM came from the cached SoCal extract.
* **San Francisco not added (yet).** San Diego already provides an independent hilly evaluation city, and LA
  supplies 120 steep training tiles. It is reconsidered if coverage stays insufficient after the audit.

**Spain MDT02.** Not downloaded. The CNIG download centre is an interactive web application, and no WCS for MDT02
was found. Its licence was quoted as CC BY 3.0 ES in one source and not verified. Its data type (float or integer)
is unverified, so it was not "demonstrably" better. Spanish street grades stay flagged (terracing, ±3%).

**FABDEM (Japan).** Not used. The licence (CC BY-NC-SA 4.0 per mirrors; one copy says otherwise) would permit
non-commercial research. However, the data come as 10° × 10° archives of several GB for 30 m bare-earth estimates,
and Japanese tiles already contribute water and streets without a DEM. TN trains with a `valid` channel, so
Japanese tiles have valid = 0. An exploratory FABDEM arm is deferred; this is a documented engineering decision.

**CAVW 2024 (doi 10.1002/cav.2267).**
* Authors: Zhou, Li, Zhang, Wang, Qin, Liu; vol. 35(3), 2024.
* OpenAlex lists a "bronze" open-access PDF, but the publisher returned 403 to a normal request. Circumventing that
  (browser impersonation) was refused and is not attempted.
* Its 21 references are mainly image-translation networks (pix2pix, CycleGAN, SPADE, Swin, LPIPS), NTG, Parish &
  Müller, Galin 2011, Nishida 2016 and Emilien 2012. This is consistent with the abstract: terrain probably enters
  through an image-translated density map, with a transformer producing the graph.
* **Graph representation, sequentiality and bridge handling: unknown.** The full text is needed, e.g. through KAIST
  library access.

**Implemented (tested):**
* `terrain/synthesis.py`: seamless, seeded noise; ridges; valleys; basins; 10 scenarios;
* `terrain/hydrology.py`: priority-flood, D8, drainage area, hydraulic-geometry rivers, lakes from deep
  depressions;
* `terrain/water.py`: OSM water and sea from coastline orientation;
* `terrain/road_feasibility.py`, `terrain/road_profiles.py`;
* `graphs/terrain_sequence.py` and `models/terrain_graph_turtle.py`: terrain crops, symmetry alignment, structure
  head;
* `scripts/m12_train.py`, with checkpointing.

**Pre-registration:** `docs/M12_EVALUATION_PROTOCOL.md`, written before any M12 model was trained.
