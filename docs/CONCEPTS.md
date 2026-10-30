# Concepts behind Milestone 1

Short explanations of the ideas the pipeline relies on, with examples from this repository.

## 1. What OpenStreetMap data represents

OSM is a crowd-sourced vector database of the world. It has three primitives:

- **node**: a point with lat/lon (a tree, an intersection, a corner of a building);
- **way**: an ordered list of nodes. Open ways are lines (a street centerline); closed ways
  can be areas (a building outline);
- **relation**: a group of elements (a building with a courtyard = outer + inner rings).

Meaning comes from free-form **tags** (`key=value`): `building=yes`, `highway=residential`,
`leisure=park`, `width=6`. Tags are voluntary and inconsistent. In our sample, Carrer de
València has `width=5` but `lanes=4`, which contradict each other. Many streets have no
`width` at all. The data describes what volunteers have mapped, which is not necessarily
everything that exists.

## 2. What a street network graph is

A graph `G = (V, E)`: **nodes** are intersections and dead ends, **edges** are the street
segments between them, each carrying attributes (length, highway class, name, geometry).
OSMnx builds a *directed multigraph*: a two-way street gives two edges (u→v and v→u), and
two distinct streets may join the same pair of nodes. "Simplification" merges the chain of
degree-2 nodes along a street into a single edge, which is why some attributes become lists
(e.g. `highway = ["living_street", "pedestrian"]`). The graph representation supports
topology questions (connectivity, routing, block shapes) that a raster cannot answer
directly. Milestone 1 only uses the edge geometries, but later milestones can use the topology.

## 3. What a coordinate reference system (CRS) is

A CRS defines what the numbers of a coordinate mean. It has two parts: a model of the Earth
(datum/ellipsoid, e.g. WGS84, ETRS89) and, optionally, a projection to a plane. EPSG codes
name them: `EPSG:4326` = WGS84 lat/lon in degrees; `EPSG:25831` = ETRS89 / UTM zone 31N in
metres, the official system for Catalonia.

## 4. Why projection is necessary

Lat/lon are *angles*. One degree of latitude ≈ 111 km everywhere, but one degree of longitude
≈ 111 km × cos(latitude). At Barcelona (41.4°N) that is ≈ 83.5 km. If you treated (lon, lat)
as (x, y), Barcelona would be stretched about 33% east–west, a "4 m buffer" would mean
4 *degrees* (≈ 400 km), and areas would be wrong. A projected CRS maps the region to a plane
where 1 unit = 1 m with negligible distortion. Only after projecting do "512 m × 512 m",
"buffer by 4 m" and "4 m per pixel" mean what they say. Note that the square is axis-aligned
in UTM, which is rotated about 0.5° relative to true north here (grid convergence).

## 5. What rasterization means

Converting vector shapes (exact polygons) into a grid of pixels. We use the *pixel-centre
rule*: pixel (r, c) is 1 if its centre point `(xmin + (c+½)·4, ymax − (r+½)·4)` lies inside
the polygon. Implementation: union all polygons of a layer, then run one vectorised
point-in-polygon test for all 16,384 centres (`shapely.intersects_xy`). Example: a 40 m × 40 m
building whose corner sits at (40 m, 40 m) from the north-west corner covers exactly
columns/rows 10–19, i.e. 100 pixels (see `tests/test_pipeline_synthetic.py`).

## 6. What an image channel represents

A channel is one 2-D grid of values describing one property of every location. RGB photos
have 3 channels (red, green, blue intensity). Ours has 3 *semantic* channels: "is this road?",
"is this building?", "is this green?". The array has shape `(C, H, W) = (3, 128, 128)`, which is
PyTorch's channel-first convention. Because channels are separate, one pixel can be
"building" *and* "green" (a roof garden), which a single colour-coded label image could not
express.

## 7. Why this representation suits neural networks

- **Fixed size**: every sample is the same tensor shape, so samples batch together trivially.
  Vector data has a variable number of polygons with variable numbers of vertices.
- **Spatial locality**: neighbouring pixels are neighbouring places, so convolutions and
  attention can learn local patterns (block corners, street widths) and reuse them everywhere
  (translation equivariance).
- **Known scale**: 4 m/px everywhere, so the model can learn real dimensions (Eixample
  blocks ≈ 113 m, chamfers ≈ 20 m).
- **Mature tooling**: VAEs, U-Nets and diffusion models are designed for image tensors, and
  binary channels map naturally to per-pixel Bernoulli/BCE losses.
- **Conditioning is easy**: targets like "green coverage = 15%" are just channel means.

## 8. What is lost when going from vector to raster

- **Sub-pixel detail**: anything narrower than ~4 m can vanish or become dashed (the 3 m
  Consell de Cent superilla lanes show up as broken diagonal lines), and edges become
  staircases.
- **Object identity**: 963 separate buildings become one mask, and party walls between
  adjacent buildings disappear.
- **Attributes**: names, heights, levels, road class, one-way, speed limits.
- **Topology**: which streets connect at an intersection, and the graph structure.
- **Exact geometry & area**: areas are quantised to multiples of 16 m².
- **Anything outside the square**: context beyond 512 m.

This is why the projected vectors are saved alongside the array
(`*_vectors.gpkg`). A future model may want graph or vector outputs instead.

---

# Concepts behind Milestone 2

## 9. Sampling bias and stratified sampling

If you pick neighbourhoods by hand ("the interesting ones near the centre"), the dataset
describes your taste, not the city. Two common biases:

* **Centre bias:** historic cores are over-represented. The suburbs and informal settlements that
  make up most of Lima's area would be missed.
* **Data-availability bias:** choosing tiles where OSM "looks good" selects well-mapped areas.

**Stratified random sampling** splits the population (all eligible grid cells) into *strata*
defined by measurable variables, then draws randomly *within* each stratum. That guarantees rare
but important types (e.g. post-2000 dense areas) appear at all. Because rare strata are
over-sampled, each tile carries an **inclusion probability** `p = n_h / N_h` and a **design weight**
`1/p`. A city-level average should be weighted, so that one tile from a stratum of 2 cells does
not count as much as one from a stratum of 300.

Example (Madrid): the `high|2000_2020` stratum has 1 eligible cell and is sampled with p = 1
(weight 1); the `low|by1975` stratum has hundreds of cells, and each sampled tile stands in for
many (weight ≫ 1).

## 10. Operational definitions

A category such as "traditional Spanish urbanism" cannot be measured, so it cannot be tested. An
*operational definition* is a rule anyone can apply and get the same answer, for example
"era = first GHSL epoch at which ≥ 15% of the tile is built-up". It may be imperfect (GHSL starts
in 1975), but it is explicit, reproducible and criticisable.

## 11. Spatial autocorrelation and data leakage

Tobler's first law: *near things are more related than distant things.* Two adjacent Eixample
tiles are almost copies of each other. If one lands in training and the other in test, the test
score measures memory, not generalisation. This is **leakage**. Remedies used here:

* **Spatial blocks:** 3 km squares are assigned *whole* to one split.
* **Buffer:** tiles in different splits must be ≥ 1 km apart (edge to edge).
* **City hold-out:** train on 8 cities, test on the 9th, which is the strongest test of
  "does it work in a new place?".

Note the difference between *geographic generalisation* (generating a realistic layout for an
unseen city) and *country classification* (guessing the country of a tile). A classifier can be
accurate by memorising city-specific cues, and that says little about generation.

## 12. Intersection density from a graph

An intersection is a graph node where ≥ 3 *streets* meet (`street_count ≥ 3`). Direction is
ignored, so a two-way street counts once. Dead ends (1) and bends (2) are not intersections.
Pixels can't give you this reliably: two roads crossing at a bridge look identical to a junction
in a raster.

Real junctions are often drawn as several nodes. A boulevard with two carriageways meets a cross
street at *two* nodes ~10 m apart. **Consolidation** buffers every node by 10 m and merges nodes
whose buffers overlap, so one crossroads counts once. In our pilot, Valencia's medieval core has
211 raw nodes but 105 consolidated intersections.

## 13. Street orientation: axial data, entropy and φ

A street bearing of 30° and one of 210° describe the same street, so orientations are **axial**
data: we fold them to [0°, 180°). Two consequences:

* **The histogram** has 18 bins of 10°, weighted by street length.
* **The average direction** can't be a plain mean: the average of 10° and 170° is 0° (both nearly
  N–S), not 90°. The fix is to *double* the angles (20° and 340°), take the vector mean, then halve
  it. For a grid, N–S and E–W cancel even after doubling, so we also *quadruple* the angles to get
  the grid orientation modulo 90°. Barcelona's Eixample comes out at about 45°.

**Entropy** `H = −Σ p ln p` measures disorder. A perfect grid fills 2 axial bins (`H = ln 2`); a
uniform spread fills all 18 (`H = ln 18`). **Boeing's order** `φ = 1 − ((H − ln 2)/(ln 18 − ln 2))²`
maps this to 1 for a perfect grid and 0 for no preferred direction.

## 14. OSM completeness, and why "missing" ≠ "absent"

OSM only contains what someone mapped. Our pilot compared OSM building coverage with GHSL, a
satellite-derived, OSM-independent estimate of built-up surface:

* Spain: OSM ≈ 0.8 × GHSL. Spanish OSM buildings largely come from the national cadastre.
* Lima: OSM ≈ 0.02 × GHSL. The streets are mapped, but most buildings are not.

A model trained on this would learn "Lima has almost no buildings", a *mapping artefact* presented
as urban form. That is why quality flags are severity-graded and why building comparisons across
countries are withheld until completeness is fixed. The reverse error exists too: GHSL counts
container yards as built-up, so a port looks "under-mapped" when OSM is actually right.

## 15. Reproducibility of live data

OSM changes every minute. "Download Barcelona" today and next month gives different data. To make
a dataset reproducible, we **pin a snapshot**: every tile records the replication timestamp of
the extract it came from (2026-10-28T21:21:02Z), the extract's MD5, the config hash, the software
versions and the git commit. Rebuilding from the cached raw files gives bit-identical arrays,
which is tested.

## 16. Choosing study areas by data quality: what it buys and what it costs

In v2 we kept only cities whose OSM building data is reasonably complete (median OSM/GHSL ratio
≥ 0.5). **What it buys:** the building channel now means the same thing in every city, so
comparisons and models are not driven by mapping effort. **What it costs:** the study population
becomes "well-mapped cities", and today that correlates with income and region (Herfort et al.
2023). A model trained on this set says nothing about Lima or Busan. This trade-off is a
*selection bias*. It is fine to make it, as long as it is stated as part of the population
definition and not hidden.

The same logic applies to every proxy we use. In v2, Japanese streets are mostly tagged
`unclassified`/`residential` and receive 7–8 m default widths, while many are about 4 m wide.
A metric is only comparable across places if the *measurement process* is comparable too.
