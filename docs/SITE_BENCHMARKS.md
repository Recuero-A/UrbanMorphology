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

## Results: Machu Picchu (added after scoring; the definitions above are unchanged)

The full table is in `outputs/sites/machu_picchu/README.md`. Figures:
* `fig_maps_seed1.png` has every map;
* `fig_realism_vs_terrain.png` plots realism against TAS;
* `maps/` has one map per method and seed (local only, made by `figures`).

Values are means of seeds 1–3.

* **There is a strong trade-off between realism and terrain adaptation.** Across the 24 generator versions:
  * Spearman ρ between typicality and TAS is +0.85. A lower (more realistic) typicality goes with a lower TAS.
  * Street density and TAS: ρ = −0.77.

  No version is both realistic (≤ 1.5) and well adapted (TAS ≥ 60).
* **Dense, realistic, terrain-blind.** This group is:
  * M\* + organic and M\* + radial (realism 0.90–1.28, 31–39 km/km², 98–100 % connected);
  * the two tracers;
  * M\* + grid.

  Its TAS is 34–41, and its contour following, C = 0.36–0.43, is at the random-orientation level (0.36). 78–85 % of
  its street length is steeper than 15 %. The grid tracer's C = 0.36 and S = 0.51 match a terrain-blind layout, which is
  a sanity check of the score.
* **The middle: network-aware samplers (M14).** NA-SMC has realism 1.45, TAS 56, 19 km/km² and 90 % coverage.
  NA-fixed and NA-adaptive score TAS 64–65, but at only 8 km/km². Their streets follow the contours visibly (C 0.63–0.69).
* **Adapted but sparse.** This group is P, M12-HR, M\* TNse and M13-SMC:
  * TAS 71–83, with M12-HR at 10 % of its length steeper than 15 %;
  * only 0.8–6 km/km², fragmented (largest piece 7–29 %);
  * realism 1.98–4.27.

  Their high TAS comes from building little, and only where it is easy.
* **Real references.**
  * The Hiram Bingham switchbacks score TAS 73 with C 0.83. Of the generators, only M12-HR and M\* TNse follow the
    contours that well (C 0.83–0.84).
  * The Inca paths and steps score TAS 57: they climb straight up (63 % of their length steeper than 15 %).
* **Ground street over water.** A (learned, no terrain rules) puts about 1.3 km of ground street on mapped water, mostly
  without bridges: 0.4–0.9 km per seed on the Urubamba and the rest on streams. No other method puts more than 0.15 km there.
* **F1 + F4.** The M17 junction fixes change realism and TAS by less than the spread between seeds, except NA-SMC TAS
  56 → 59.

**Limits.**
* GLO-30 is a DSM with forest canopy, and the bare-earth proxy is a low-pass of it.
* The terrain is far outside the training range: the learned models clip slope at their training scale.
* The city limits are one ellipse.
* There are 3 seeds.
* This is exploratory: nothing here is a test of a pre-registered hypothesis.

## Site 2: Petra (`configs/sites/petra.json`)

User request (2026-11-25): "generate with the different algorithms we created some maps for Petra, Jordan ... tell me the
scores for terrain adaptiveness and realism." This section was written and committed **before any Petra generation or
score**. The only things looked at were the terrain, the OSM ways and the city limits (`outputs/sites/petra/site_terrain.png`).

* **Held-out check.** Jordan is not a training, validation, development or declared held-out region of any milestone
  (`configs/m17_holdout.json`, `configs/m15_benchmark/manifest.json`).
* **Domain.** A 2048 m × 2048 m square in UTM 36N (EPSG:32636) with a 256 m terrain margin. It is centred on the midpoint
  of the Treasury (Al-Khazneh) and the Monastery (Ad Deir), so it holds:
  * the end of the Siq;
  * the Royal Tombs;
  * the Colonnaded Street basin and Qasr al-Bint;
  * the Ad Deir plateau.
* **Terrain.** Copernicus GLO-30 DSM (tile N30 E035), bilinear to 4 m. It is bare sandstone and desert, so the surface
  model is close to the ground. The same σ = 16 m proxy is used as at Machu Picchu, so the two sites are scored the same
  way.
* **Water.** None. Petra has no permanent surface water. OSM maps its ephemeral wadis (Wadi Musa, Wadi Turkmaniyya) as
  `waterway=stream`. Rasterised as rivers, they would cut the city limits into pieces and force bridges over dry
  ground, so they are treated as dry ground (`dry_wadis: true`). The unbridged-water statistic is therefore 0 for every
  method.
* **City limits.** Fixed, the same for every method, and **not selected on slope**. The limits are:
  * an ellipse centred on the ancient city centre (30.3286 N, 35.4419 E, at (1037, 841) m);
  * its long axis along Wadi Musa (160°);
  * the same semi-axes (840 × 560 m) and edge noise as Machu Picchu.

  The limits hold the gentle Colonnaded Street basin and the sandstone massifs on both sides. Outside them is desert. The
  terrain inside the limits (`outputs/sites/petra/site.json`):
  * area 1.47 km²;
  * elevation 845–1161 m;
  * median slope 26 % (73 % at Machu Picchu);
  * 41 % of the land is steeper than 30 %;
  * 26 % of the land is steeper than 45 %;
  * 5 % of the land is steeper than 100 %, on the cliffs.
* **Generators, seeds, cropping, realism, TAS, statistics, flags and reporting.** Exactly as for Machu Picchu above,
  with no change to any code path of `generate` or `score`. S uses Petra's own median slope (26 %).
* **Real references** (flagged; they are not a city):
  * the real OSM vehicular ways, which are the tracks along Wadi Musa and the Colonnaded Street;
  * the real paths, which are the trails and the rock-cut steps.
* **Comparison across the two sites** (declared now). The report will give:
  * the Spearman ρ between realism (typicality) and TAS over the 24 generator versions at Petra, to compare with the
    Machu Picchu value;
  * the Spearman ρ of each score between the two sites, over the same 24 versions.

  No new threshold and no winner is declared.
