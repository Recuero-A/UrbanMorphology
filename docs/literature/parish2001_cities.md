# Parish & Müller 2001: Procedural Modeling of Cities

## 1. Header

- **Citation:** Yoav I. H. Parish, Pascal Müller. "Procedural Modeling of Cities." In *Proceedings of the 28th Annual Conference on Computer Graphics and Interactive Techniques* (SIGGRAPH 2001), ACM, pp. 301–308, August 2001.
- **DOI:** 10.1145/383259.383292
- **URLs opened:**
  - https://cgl.ethz.ch/Downloads/Publications/Papers/2001/p_Par01.pdf: author PDF from ETH CGL (8 pages; PDF page 1 = p. 301). Read in full.
  - https://api.crossref.org/works/10.1145/383259.383292: bibliographic metadata (venue, pages, date, event name "SIGGRAPH01").
- **Access level:** **FT** (full text read).
- **Code:** no official code (the system became the commercial CityEngine, excluded). Open re-implementations are listed in `docs/M18_BASELINES.md` (not re-checked here): `t-mw/citygen` (MIT), `josauder/procedural_city_generation` (MPL-2.0), `phiresky/procedural-cities` (AGPL-3.0).

## 2. BibTeX

```bibtex
@inproceedings{parish2001procedural,
  author    = {Parish, Yoav I. H. and M{\"u}ller, Pascal},
  title     = {Procedural Modeling of Cities},
  booktitle = {Proceedings of the 28th Annual Conference on Computer Graphics and Interactive Techniques},
  series    = {SIGGRAPH '01},
  publisher = {ACM},
  year      = {2001},
  month     = aug,
  pages     = {301--308},
  doi       = {10.1145/383259.383292}
}
```

## 3. What it does

CityEngine builds a whole virtual city from a few raster maps: water/land, elevation, population density, zoning, street-pattern control maps and a building-height map. Highways and streets are grown by an extended, "self-sensitive" L-system. The blocks between roads are split into lots, and a second L-system makes building geometry, with layered-grid procedural facade textures. The road part is the first widely cited rule-based street-growth model and the ancestor of most frontier-growth generators, NTG included.

## 4. How it generates

- **Procedural.** All rules and parameters are hand-written or set by the user. Nothing is fitted to data, although the authors say their pattern rules are based on data about New York, Paris, Tokyo and London.
- **Representation: vector.** Straight road segments in a street graph. A crossing creates a node and two edges, and high-curvature roads are smoothed by subdivision.
- **Algorithm, in plain words (§3):**
  1. **Axiom.** Start with one road segment placed in a legal area (not water, not a park).
  2. **Propose.** In each rewriting step the L-system proposes an "ideal successor" for every growing road end: a road module plus two branch modules, with parameters left unassigned.
  3. **`globalGoals` sets the parameters.**
     - *Highways* cast rays radially from the road end within a preset radius. They sum population density along each ray, weighted by inverse distance, and continue in the direction with the largest sum.
     - *Streets* follow the active pattern rule:
       - *Basic*: follow population;
       - *New York*: grid with a given angle and block size;
       - *Paris*: radial around a centre;
       - *San Francisco*: follow least elevation, with short connectors along the steepest elevation.
     - Several rules can be blended with weights read from grey-scale control maps. Branches start after a delay counter.
  4. **`localConstraints` adjusts or kills the proposal.**
     - If the segment end lies in an illegal area (water, park):
       - prune its length;
       - or rotate it within a maximum angle (this makes roads follow coastlines);
       - or, for highways only, let it cross the illegal area up to a specified length and flag it as a bridge or tunnel.
     - Then look for nearby roads:
       - an intersecting segment is cut back and becomes a crossing;
       - an end near an existing crossing is extended to it;
       - an end close to a road is extended to form an intersection.
     - After a crossing is made, the road stops growing. If no valid parameters exist, the module is FAILED and deleted.
  5. **Stop.** Streets stop growing when they reach an area with no population.
  6. **Downstream.** Blocks are split recursively into lots, and buildings are made with a stochastic parametric L-system.

## 5. Inputs, outputs and how terrain enters

- **Inputs (§1.3):**
  - *Geographical maps:* elevation maps; land/water/vegetation maps.
  - *Sociostatistical maps:* population density, zone maps, street-pattern maps, building-height maps.
  - Optional parameter files or statistics, such as block size or intersections per square mile.
- **Outputs:** road graph (highways and streets), allotments, building geometry and textures.
- **Terrain and water:**
  - *Water and parks* are illegal areas handled in `localConstraints` (prune, rotate, or bridge/tunnel flag for highways).
  - *Elevation* is listed as a local constraint (§3.2.1). In practice it acts through the **San Francisco rule**, a global goal: roads follow least elevation, and short connectors follow the steepest elevation.
  - The paper gives **no numeric slope or grade threshold**: a search of the text finds no "slope" or "grade". The bridge length limit is only "a specified length".

## 6. Evaluation

- **Qualitative only.**
  - Manhattan was regenerated from scanned maps. Fig. 9 shows the growth after 28 and 142 steps and the final roadmap, with "a real map of Manhattans streets for comparison".
  - The authors remark that the proposed bridges lie close to the real ones.
  - A second, imaginary city (20-mile diameter, from the Fig. 2 maps) is rendered.
- **Not reported:**
  - no metric;
  - no baseline;
  - no statistics over seeds;
  - no user study.
- **Runtime (§5):** street graph for Manhattan in under 10 s; lots and about 13,000 buildings in about 10 min. Hardware is not stated.

## 7. Quotes for citation

- p. 301, §1.2: "Our CityEngine creates urban environments from scratch, based on a hierarchical set of comprehensible rules that can be extended depending on the users needs."
- p. 303, §3.2.1: "local: land/water/park boundaries, elevation, crossing of streets"
- p. 304, §3.2.2: "San Francisco rule: This pattern lets streets and highways follow the route of the least elevation. Roads on different height levels are connected by smaller streets, that follow steepest elevation and are short. This pattern is usually observed in areas with large differences in ground elevation."
- p. 305, §3.3: "Highways are allowed to cross illegal area up to a specified length. The generated highway segment is flagged. At the geometry creation stage it can then be replaced by e.g. a bridge, or two tunnel entrances on both sides."
- p. 305, §3.3.1: "In traffic systems the dead end road is the exception. Most roads end when crossing other roads or circling back to themselves [9]."
- p. 305, §3.3.1: "Since we do not have an underlying model of transportation flow simulation on our roadmap, we can ignore the implications on street capacity."
- p. 305, Fig. 9 caption: "Lower: A real map of Manhattans streets for comparison."
- p. 305, §3.3.1: "Note how the proposed bridges cross the water very close to the locations of where the real bridges are built."
- p. 307, §5: "The creation of the street graph took less than ten seconds, the division into lots and the creation of buildings approximately 10 minutes."
- p. 307, §6: "Many more rules can be added to let the road creation mechanism behave more realistically."

The page numbers come from the PDF page plus 300. A script checked every quote against the PyMuPDF text of the PDF. The p. 304 quote differs only in PDF spacing before the colon.

## 8. Comparability with UrbanMorph

- **Matched inputs.** Mostly yes.
  - *Water mask:* becomes the illegal-area map.
  - *DEM:* drives the San Francisco rule.
  - *Entry points:* become axioms or seeds.
  - *Classes:* highway/street maps roughly onto our two coarse classes.
  - *Missing input:* the population-density map. Use a uniform map, or a density proxy shared by all methods, and document it.
  - *Our additions:* P&M has no max-grade rule. Applying our per-class grade limits inside `localConstraints` would be our addition and must be labelled as such. The San Francisco rule is under-specified (no formula), so any implementation is an interpretation.
- **Code and licence.**
  - No official code.
  - `t-mw/citygen` is MIT-licensed TypeScript with no terrain (per `docs/M18_BASELINES.md`). Its `localConstraints()` hook is where water and slope go.
- **Effort.**
  - *Per M18_BASELINES estimate:* 0.5–1 h as-is; 12–20 h adapted (headless runner, water/slope constraints, anchors, boundary, metric scale, GeoJSON export).
  - *My estimate:* a faithful San Francisco rule adds about 4–8 h.
- **Fair comparisons:**
  - **Full network growth** from entry points on our tile;
  - **completion given arterials**: seed streets from supplied major roads.
  - Not connector routing.
- **Main threat.** A weakened baseline. Tune it with the same budget and realism metric as our method, on held-out tiles.

## 9. Ideas we could adopt for evaluation

- **Bridge-siting agreement.** P&M only say their bridges land "very close" to the real ones; we can measure it:
  - for each real OSM bridge or water crossing on a tile, find the distance to the nearest generated crossing;
  - report precision and recall within a radius such as 50 m.
- **Dead-end share as a realism statistic.** P&M treat dead ends as "the exception", which matches our closed-dead-end potentials. Report the real-tile dead-end share as the target, not zero.
- **Contour-following test of the "least elevation" rule.** Report the share of street length whose bearing lies within X° of the local contour direction, for generated and real streets on the same DEM, split by road class. Steep connectors should instead align with the gradient. This directly tests the San Francisco intuition on real data.
- **Ablation baseline:** "P&M + our hard feasibility rules" isolates how much of our terrain score comes from the rules rather than from the learned model.
