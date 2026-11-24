# Named terrain benchmark sites (EXPLORATORY)

User request (2026-11-24): "take the terrain of Machu Picchu and create different cities surrounded by jungle with all
the different algorithms and methods we have, with their respective realism score and other statistics ... just the map
of the streets ... tell me the score for adaptiveness to the terrain."

This is an exploratory comparison, not a milestone test. No sealed test split or declared held-out metro is used.
The M15 held-out metros for Peru are Lima and Trujillo; Machu Picchu is neither. Everything below the "Declared before
results" line was written and committed before any scores were computed. The generations had started; nothing had been
scored or looked at except a smoke test of four maps used to check the pipeline.

Code: `scripts/site_benchmark.py` (stages `data`, `generate`, `score`, `figures`). Site configs: `configs/sites/<site>.json`.
Data: `data/processed/sites/<site>/` (not tracked; regenerable). Results: `outputs/sites/<site>/`.

## Site 1: Machu Picchu (`configs/sites/machu_picchu.json`)

* **Domain.** A 2048 m × 2048 m square (4 × 4 windows of 512 m) in UTM 18S, with a 256 m terrain margin. It is centred
  75 m east and 150 m north of the citadel, so it holds:
  * the citadel saddle;
  * Huayna Picchu;
  * the Hiram Bingham switchbacks;
  * the Urubamba on the west and north-east.
* **Terrain.** Copernicus GLO-30 DSM (tile S14 W073), bilinear to 4 m. It is a surface model, so the forest canopy is
  included.
* **Water.** From OpenStreetMap (`data/raw/osm_extracts/cities/SITE_PER_MAPI.osm.pbf`, clipped from the Peru extract).
* **City limits.** Fixed and the same for every method. The limits are:
  * an ellipse along the ridge: centre (1060, 870) m, semi-axes 840 × 560 m, axis at 119.5°;
  * a noisy edge;
  * water removed.

  The limits are **not selected on slope**, so where each generator decides to build is part of the test. Outside the
  limits is jungle.
* **The terrain inside the limits** (`outputs/sites/machu_picchu/site.json`):
  * area 1.43 km²;
  * elevation 1967–2650 m;
  * median slope 73 %;
  * 87 % of the land is steeper than 45 %;
  * 22 % of the land is steeper than 100 %.

  This is far steeper than any training city.
* **Generators** (seeds 1, 2, 3), run on the whole domain with **no supplied roads**. At KAIST the real arterials were
  supplied; here none are, so the network comes from the generator alone.
  * The 15 methods of the KAIST comparison (`scripts/m16_kaist_all.py`):
    * M12-HR and P (M12);
    * A, M\*, M\* TNse and M13-SMC (M13);
    * NA-myopic, NA-fixed, NA-adaptive, NA-random and NA-SMC (M14);
    * M\* + grid, + organic, + radial and + cul-de-sac (M16).
  * The M17 F1 + F4 versions (junction reuse 15 m + degree prior) of A, M\*, M13-SMC, NA-fixed, NA-SMC, M\* + organic and
    M\* + radial.
  * The M15 Chen-inspired field tracer (grid and organic fields), run per 512 m window as in M15.

  Settings are as in the KAIST runs, except that the radial style is centred on the city centre rather than the domain
  centre.
* **Real references** (flagged; they are not a city):
  * the real OSM vehicular ways, which here are the Hiram Bingham road and service tracks;
  * the real paths, which are the Inca trails, the citadel footways and the steps.

## Declared before results

**Cropping.** Each network is cropped to the city limits as for the KAIST village (`kaist_village.crop`):
* edges with both ends inside are kept;
* dead-end stubs under 10 m are removed;
* isolated pieces under 120 m are removed.

There are no supplied main roads to keep.

**Realism.** Typicality (`scripts/m16_realism.py`, as in `outputs/m17/village`) measures how close a network is to real
neighbourhoods:
* it uses 13 street-morphology features (no terrain feature);
* each window's score is its mean distance to the 5 nearest of 156 real neighbourhoods, divided by the same quantity
  among the real neighbourhoods themselves;
* 1 = as typical as a real neighbourhood, and lower is more realistic.

It is averaged over the 8 windows of 512 m that lie at least 70 % inside the city limits (greedy, under 30 % overlap). The
same windows are used for every network, and a window without streets is skipped.

**Terrain adaptiveness score (TAS, 0–100, higher = better adapted).** It is computed on the cropped city network, using
ground edges only (bridges and tunnels are left out). Every edge is cut into pieces of at most 10 m, and the pieces are
read on a bare-earth proxy: the DSM low-passed with a Gaussian of σ = 16 m. The score has three components, each from 0
to 1:

* **G, gentle grades** = 1 − the share of street length steeper than 15 %.
* **C, contour following** = 1 − the contour index. For each piece the index is |cos θ| between the piece and the fall
  line, which equals the street grade divided by the terrain slope. It is length-weighted over pieces where the terrain
  is steeper than 5 %, as in the M12 definition, but on the 16 m proxy at the piece. Reference values:
  * along the contours, C = 1;
  * straight down the fall line, C = 0;
  * random orientation, C ≈ 0.36.
* **S, siting on gentler land** = the share of street length on the gentler half of the city's land, meaning below the
  city's median terrain slope (73 % here; set from the terrain only). Reference values:
  * a spread unrelated to slope gives S = 0.5;
  * all streets on the gentler half give S = 1.

**TAS = 100 × (G + C + S) / 3**, with equal weights fixed now.

**Statistics.** Inside the city limits:
* street km per km²;
* coverage (the share of city land within 60 m of a street);
* junctions per km²;
* the 3-way, 4-way and 5+-way shares;
* close15 (the share of junctions with another junction under 15 m away);
* dead ends per km, counted at least 32 m inside the edge;
* the share of street length in the largest connected piece;
* the median grade;
* the shares of street length steeper than 12, 15 and 20 %.

On the whole domain, the ground street length over water without a bridge. Also the generation time.

**Flag.** A city network with fewer than 5 km of street per km² is marked **sparse**: the generator did not build a
town there. Its TAS still describes the few streets it did build, but it is not a comparable city.

**Reporting.** Means over seeds 1–3, with the standard deviation across seeds. Maps are drawn for every method and seed;
the overview sheet uses seed 1. Rows are ordered by realism, as in the KAIST tables, and the TAS is shown next to it.
No winner is declared: the user chooses the city to build.
