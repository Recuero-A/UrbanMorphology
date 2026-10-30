# Development Log

## Milestone 1 — Real urban data → ML-ready layout (2026-10-30)

### What was implemented

- `src/urbanmorph/data/osm.py`: downloads the street graph, road-area polygons, buildings
  and green spaces for a fixed centre (41.3910 N, 2.1650 E) and radius, with retries and
  backoff over two Overpass endpoints. Raw responses are cached under
  `data/raw/barcelona_eixample_001/` (GraphML + GeoJSON + `download_info.json` with query,
  timestamp, versions, attribution). Later steps never touch the network.
- `preprocessing/`: projection to EPSG:25831, a 512 m square around the projected centre,
  road-width estimation and buffering, clipping, and a pixel-centre rasterizer.
- `data/dataset.py`: save / load / validate (shape, dtype, binary values, metadata keys,
  geometric consistency).
- `visualization/layout_plot.py`: per-channel maps, combined map, vector-vs-raster comparison.
- 29 tests (synthetic geometry tests + real-data tests that use only the local cache).

Result for `barcelona_eixample_001`: roads 19.7%, buildings 56.9%, green 1.25% of pixels;
overlaps road∩building 3 px, road∩green 5 px, building∩green 83 px; 3,721 px (22.7%)
unclassified (sidewalks, courtyards, plazas).

### Technical decisions

| Decision | Why |
|---|---|
| Centre point + fixed size, not a lat/lon bbox | A metric square is the actual sample unit; a lat/lon box is not square in metres. Download radius = half-diagonal + 100 m so buffers of roads just outside the square are included. |
| EPSG:25831 (ETRS89 / UTM 31N) | Official metric CRS for Catalonia; < 0.1% scale distortion here. |
| Own rasterizer (Shapely 2 `intersects_xy`) instead of `rasterio` | One dependency fewer, and the whole algorithm is ~10 readable lines. Same rule as rasterio's default (`all_touched=False`). |
| Pixel-centre rule, boundary counts as inside | Deterministic and unbiased: area is preserved on average. `all_touched` would systematically inflate thin features. |
| Road classes: motorway…residential, living_street, pedestrian | Barcelona maps sidewalks as separate `footway` ways; including them would double-count. Service roads and paths are excluded. |
| Width = max(`width` tag, `lanes` × 3.25 m) → class default | Tags conflict (`width=5, lanes=4` on Carrer de València); lanes give a physical lower bound. Final sources: 55 lanes, 33 width tag, 42 class default, 7 area polygons. |
| Road areas (`area=yes`) downloaded as polygons | In the graph their outline becomes edges, which buffered into hollow rings. |
| Tunnels excluded from roads | They occupy no surface. |
| Channels independent (multi-label), no priority | Lossless with respect to the vectors; overlaps are counted in metadata, and an exclusive label map can be derived later. Combined figure paint order: green < buildings < roads. |
| Green = vegetated land regardless of access | Not all green is `leisure=park`: interior-block gardens are `leisure=garden`, verges are `landuse=grass`. Points (e.g. single trees) are dropped, since only polygons have area. |
| Raw data kept in EPSG:4326, as received | Preserves the original source. Processed, clipped vectors are saved separately (`*_vectors.gpkg`). |
| Plot palette validated for colour-vision deficiency | The first attempt (terracotta vs. green) was indistinguishable under deuteranopia (ΔE 0.8). Now blue / amber / dark green. |

### Problems encountered

1. **Editable install invisible to Python.** The project lives on an iCloud-synced Desktop.
   iCloud sets the macOS `hidden` flag on files inside `.venv/`, and Python ≥ 3.12 ignores
   hidden `.pth` files, so `import urbanmorph` failed. `chflags nohidden` is undone within
   seconds. Fix: scripts insert `src/` into `sys.path`, and pytest uses `pythonpath = ["src"]`.
   Moving the project out of iCloud is the cleaner long-term fix (it also avoids syncing
   thousands of venv files).
2. **Hollow rings at pedestrian squares**: fixed with the road-area layer (see above).
3. **Thin, dashed streets**: real 3–3.6 m lanes (Consell de Cent superilla, Passeig de Gràcia
   side lanes) at 45° to the grid miss pixel centres. Contradictory width tags made more
   streets too thin than necessary, which was fixed with the lanes lower bound. The rest is
   inherent to 4 m pixels and is documented as a limitation.
4. **Slow Overpass responses**: the first download took ~3.8 min. Caching means this happens once.
5. **Numeric columns stringified in the vector export**: fixed so only object columns are converted.

### Known limitations

- Road widths are estimates of the carriageway, not measured street widths; sidewalks are
  not included. The road channel therefore under-represents public street space
  (Eixample streets are ~20 m façade to façade).
- Thin features (< 4 m) may be fragmented or lost; connectivity is not guaranteed in the raster.
- OSM completeness: green spaces in particular are under-mapped (private interior gardens).
  Nothing is inferred or fabricated.
- One sample only; no dataset split or statistics across neighbourhoods yet.
- The sample grid is aligned to UTM north, not to the street grid (Eixample is rotated ~45°),
  which is fine for learning but means orientation varies between cities.
- Building heights are not used (kept in raw data for the 3D milestone).
- Re-downloading later gives *current* OSM data, so the cache (or an OSM snapshot date) is
  what makes results reproducible.

### Next milestone (recommended)

**Milestone 2 — A small, documented multi-city dataset.**
1. Literature review (generative urban layouts, street-network generation, evaluation metrics).
2. Tile several neighbourhoods (Eixample, Gràcia, Ciutat Vella, plus contrasting cities) into
   hundreds of 512 m samples with a sliding window; store per-sample metadata.
3. Compute the conditioning variables per sample (building coverage, road coverage, green
   coverage, dominant street orientation, intersection density) and plot their distributions.
4. Define train/val/test splits **by area** (no spatial leakage between neighbouring tiles).
5. Only then: a first baseline model (e.g. a small conditional VAE) in Milestone 3.
