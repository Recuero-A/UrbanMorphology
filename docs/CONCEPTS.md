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
