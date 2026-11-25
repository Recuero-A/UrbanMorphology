# Milestone 13 data protocol

## 1. Sources and licences

| Data | Source | Licence | Used for |
|---|---|---|---|
| Streets, water, structures | OpenStreetMap via Geofabrik extracts (cached: Catalonia, Madrid, Valencia, SoCal, Illinois, New York, Japan; NorCal **downloaded in M13**) | ODbL 1.0 | all tiles |
| DEM, Spain | IGN MDT05, PNOA LiDAR, WCS `Elevacion25830_5` | CC BY 4.0 | Spanish tiles, Girona, Manresa |
| DEM, US | USGS 3DEP 1 m bare earth (ImageServer) | public domain | US tiles, San Francisco |
| Built-up | GHSL GHS-BUILT-S R2023A (cached tiles) | CC BY 4.0 | tile eligibility only |
| DEM, Japan | — (GSI licence unresolved since M12) | — | **not used**: Japanese tiles have no terrain |

**Downloads in M13:**
* the NorCal OSM extract (655 MB, ODbL; md5 file stored next to it);
* IGN and 3DEP tiles for the generalisation cities.

No paid service, no private data.

## 2. Splits

| Split | Tiles | Content | Use in M13 |
|---|---|---|---|
| train | 685 | M2 / M3b train plus 120 steep LA tiles (M12) | training; potential calibration; feature normalisation |
| DEV-A | 25 (13 blocks) | half of the 52 development tiles with a DEM (`val` with a DEM ∪ San Diego `val_t`) | method selection; stop-bias calibration |
| DEV-B | 27 (14 blocks) | the other half | confirmatory comparisons |
| GEN | ≤ 100 | Girona (30), Manresa (30), San Francisco (≤ 40) | generalisation, evaluated once with frozen configurations |
| test / test_t | 19 / 34 | M2 held-out tiles; San Diego `test_t` (not built) | **untouched** (needs explicit authorisation) |

**DEV-A / DEV-B assignment** (`outputs/m13/splits.json`):
* blocks are a `val` city, or a 3 km cell of San Diego;
* blocks are shuffled with seed 20261106 and dealt alternately;
* no tile of a block appears in both halves.

**GEN rule** (pre-registered):
* every M2-grid 512 m cell inside the Nominatim city boundary with GHSL 2020 built-up ≥ 0.15;
* a seeded random sample of at most 30 / 30 / 40 tiles;
* **no terrain screening**;
* blocks of 3 km.

Plans and reports: `data/processed/m13/gen/{plan.csv, plan_report.json, build_report.json, dem_log.json}`.

## 3. Leakage controls on the terrain input

**A. Original DEM** (TN).

**B. Low-pass** (TN-lp, M12): Gaussian σ = 16 m on the input only.

**C. Street-erased** (TN-se, M13; `scripts/m13_street_erase.py`):
1. **Corridors:** every OSM street of the city clip, so the 256 m margin is erased too, buffered by f · w_c / 2.
   w_c is the class carriageway width: 9.75 / 10 / 8 / 6 m.
2. **Fill:** biharmonic inpainting (scikit-image `inpaint_biharmonic`).
3. **Micro-relief:** the original DEM's residual (z − G_8m z) is transplanted into the corridors from off-corridor
   pixels at a seeded random offset. Without it, the smooth fill would itself mark the corridors.
4. **Factor f:** chosen on **training tiles only**. The rule is the median on-street / off-street |Laplacian| ratio
   closest to 1 after erasure.

| f | Ratio after erasure | Mask share | Slope p90 shift |
|---|---|---|---|
| (original) | 0.93 | — | — |
| 1.5 | 1.56 | 0.17 | +0.0004 |
| 2.5 | 1.17 | 0.27 | +0.0002 |
| **4.0 (selected)** | **1.04** | **0.41** | −0.0010 |

   Source: `outputs/m13/audits/street_erasure.json`.

5. **Caveats.**
   * Erasure uses the real street geometry. It is a sensitivity analysis, not a reconstruction of pre-development
     terrain.
   * At f = 4 it rewrites about 40 % of each tile.
   * The transplanted roughness is spatially white at corridor scale.
   * Rules, potentials and every metric use the **original** DEM.

**D. Synthetic stress tests:** procedural terrain has no embedded streets (M12 regime C; M13 X5).

## 4. Coverage limits

* **Training structures** (tiles with a DEM): 107 / 40 / 17 / 4 bridge runs (major / mid / local / minor), 274
  tunnel edges, only **12** crossing angles measured over rivers, and 2 dry valley spans (M12). Structure potentials
  are weak priors; viaducts are exploratory only.
* **Development:** 52 tiles with a DEM; river tiles are few. Infrastructure results on DEV-B are descriptive.
* **Geography:** Spain and the US only have terrain. Japan contributes streets and water only.
