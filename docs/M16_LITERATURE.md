# Milestone 16 literature: morphology grounding for style-aware proposals and topology rules

Compiled 2026-11-21, before any M16 implementation. Extends `docs/M15_LITERATURE.md` (its entries for Parish & Müller,
Chen, Galin 2010, CitySprout, CityStack, CaRoLS and RoadWeaver are reused and not contradicted) and builds on
`docs/M15_METHOD.md`. M15 reweights the K = 8 learned proposals with a heading term; it cannot create a junction or a
continuation that no proposal contains. That is why grid guidance could not raise the 4-way share (§2.1 below).

**Access levels:** **F** full text read · **F\*** full text read through a summarising fetch tool · **P** partial (table
of contents, preview pages or the author's own slides) · **A** abstract or metadata only. Items marked A support no
technical claim here. "Ours" marks arithmetic or interpretation added by us, not stated by the source.

## 1. Sources

| # | Citation | Access | URL / DOI |
|---|---|---|---|
| S1 | Marshall, S. (2005). *Streets and Patterns*. Spon Press, Abingdon. 318 pp., ISBN 0-415-31750-9 (the e-library edition is dated 2004 in Crossref) | **P**: table of contents and ch. 1 from the publisher preview; the author's slides "Characteristic structure", CUPUM/ECiD workshop, UCL, 2 July 2005; secondary summaries S1a and S1b. **Book text not read.** | doi:10.4324/9780203589397; casa.ucl.ac.uk/cupumecid_site/download/Marshallpres.pdf |
| S1a | Adam, R. & Jamieson, C. (2014). Identifying trends in masterplanning: a typological classification system. *URBAN DESIGN International* 19(4), 274–290 | F (accepted manuscript) | doi:10.1057/udi.2013.24 |
| S1b | Larkham, P. (2008). Review of Marshall, *Streets and Patterns*. *Urban Morphology* 12(1), 63–64 | F | doi:10.51347/jum.v12i1.4730 |
| S2 | Southworth, M. & Ben-Joseph, E. *Streets and the Shaping of Towns and Cities*. McGraw-Hill 1997 (ISBN 0-07-059808-8); 2nd ed., Island Press 2003 (ISBN 1-55963-916-4) | **A**: catalogue records and reviews only. **Book not read.** | (2003 ed.) review record doi:10.3368/lj.24.1.100 |
| S2a | Southworth, M. & Ben-Joseph, E. (2004). Reconsidering the cul-de-sac. *ACCESS* 24 (Spring 2004). Same authors; condenses the book's cul-de-sac material | F (the authors' manuscript, pp. 1–7, dated 9/7/03) | web.mit.edu/ebj/www/doc/culdesacs.pdf; escholarship.org/uc/item/1qn0g780 |
| S2b | Southworth, M. & Owens, P. (1993). The evolving metropolis… *J. Am. Planning Assoc.* 59(3), 271–287 | A (Crossref only; the source of the "gridiron … loops and lollipops" sequence; not used for claims) | doi:10.1080/01944369308975880 |
| S3 | Louf, R. & Barthelemy, M. (2014). A typology of street patterns. *J. R. Soc. Interface* 11(101), 20140924 | F (PMC HTML; equations are images, so the definitions are taken from the text) | doi:10.1098/rsif.2014.0924; pmc.ncbi.nlm.nih.gov/articles/PMC4223914 |
| S4 | Boeing, G. (2019). Urban spatial order: street network orientation, configuration, and entropy. *Applied Network Science* 4, 67 | F (arXiv:1808.00600, the August 2019 version; the publisher page redirected to a login). Table values come from the preprint. | doi:10.1007/s41109-019-0189-1 |
| S5 | Barthelemy, M. & Boeing, G. (2024). A review of the structure of street networks. *Findings* | F (arXiv:2409.08016) | doi:10.32866/001c.122117 |
| S6 | Barthelemy, M. & Boeing, G. (2025). Universal model of urban street networks. *Phys. Rev. Lett.* 135(13) | A (arXiv:2509.21931 abstract) | doi:10.1103/1vj4-n8vn |
| S7 | Kuncheria, A., Walker, J. L. & Macfarlane, J. (2025). Beyond centrality: urban street network typologies through intersection patterns | F\* (arXiv:2511.06747v1; preprint, no venue stated) | arxiv.org/abs/2511.06747 |
| G1 | Parish, Y. & Müller, P. (2001). Procedural modeling of cities. *SIGGRAPH 2001*, 301–308 | F (M15) | doi:10.1145/383259.383292 |
| G2 | Chen, G., Esch, G., Wonka, P., Müller, P. & Zhang, E. (2008). Interactive procedural street modeling. *ACM TOG* 27(3), 103 | F (M15) | doi:10.1145/1360612.1360702 |
| G3 | Galin, E., Peytavie, A., Maréchal, N. & Guérin, E. (2010). Procedural generation of roads. *CGF* 29(2), 429–438 | F (M15) | doi:10.1111/j.1467-8659.2009.01612.x |
| G4 | Galin, E., Peytavie, A., Guérin, E. & Beneš, B. (2011). Authoring hierarchical road networks. *CGF* 30(7), 2021–2030 (Pacific Graphics) | F (author PDF, cs.purdue.edu/homes/bbenes) | doi:10.1111/j.1467-8659.2011.02055.x |
| G5 | Chu, H., Li, D., Acuna, D., Kar, A., Shugrina, M., Wei, X., Liu, M.-Y., Torralba, A. & Fidler, S. (2019). Neural Turtle Graphics for modeling city road layouts. *ICCV 2019*, 4521–4529 | F (arXiv:1910.02055) | doi:10.1109/ICCV.2019.00462 |
| G6 | Dobbie & Johansson (2026). Comparing urban network morphologies… (tool: CitySprout). *EPJ Data Science* 15, 76 | F\* / F supplement (M15) | doi:10.1140/epjds/s13688-026-00694-7 |
| G7 | Post, Ledoux & Patil (2026). A proposed framework for generating plausible cities (CityStack). *ISPRS Annals* XII-4/W1-2026, 277–284 | F (M15) | doi:10.5194/isprs-annals-XII-4-W1-2026-277-2026 |
| G8 | Feng, Li, Li, Li & Shen (2025). CaRoLS. *Computers & Graphics* 133, 104451 | A (M15; still closed access) | doi:10.1016/j.cag.2025.104451 |
| G9 | Li, Y., Chen, Z., Xi, W., Jiang, M., Zhang, S., Zhuang, H. & Yang, M. (2026). RoadWeaver: large-scale lane-level HD map generation from scratch… arXiv:2608.11580v1 | F\* (HTML) | arxiv.org/abs/2608.11580 |
| G10 | Li, T. et al. (2024). RoBus: a multimodal dataset for controllable road networks and building layouts generation. arXiv:2407.07835 | A | arxiv.org/abs/2407.07835 |
| G11 | Gao, J. & Patel, S. V. (2026). From sketch to design: a cross-scale workflow for procedural generative urban design. arXiv:2611.11561 | A | arxiv.org/abs/2611.11561 |

The author lists of G6 and G7 are Crossref family names only.
Other works carried over from M15 (Gu 2024, Fang 2022, Zhou 2024, Kasiński & Szufel 2026) keep their M15 access levels.

## 2. Morphology grounding per pattern

### 2.1 Grid

* **Typology.** Marshall's four broad types are "Altstadt", "Bilateral", "Characteristic/Conjoint" and "Distributory"
  (S1, p. 85, as reported by S1a). Bilateral is the grid type. S1a notes that Marshall classifies patterns by "the
  configuration of streets…the shape of interstices…the alignment of the routes" (S1, p. 77). The book's own junction
  mix for each ABCD type is **not verified**.
* **Orientation order** (S4, Methods, eqs. 1–3):
  * Bearings and their reciprocals go into 36 bins of 10°, shifted by −5°. H_o is the Shannon entropy in nats.
    H_max = 3.584 and a perfect 4-way grid has H_g = 1.386.
  * φ = 1 − ((H_o − H_g)/(H_max − H_g))². The text says "the value is squared"; the equation was garbled in our
    extraction, but this form matches the stated end points.
  * Chicago has φ = 0.899, the highest of the 100 cities. Miami has 0.811, Minneapolis 0.749, Manhattan 0.669 (Table 1).
    The US/Canada mean is 0.427 against a European mean of 0.033 (Table 2).
* **Junctions and connectedness** (S4, Table 1):

  | City | P4w | Pde | k̄ | ς | Median segment l̃ (m) |
  |---|---|---|---|---|---|
  | Chicago | 0.507 | 0.074 | 3.343 | 1.016 | 105.3 |
  | Manhattan | 0.572 | 0.027 | 3.508 | 1.017 | 82.2 |
  | Buenos Aires | 0.576 | 0.027 | 3.548 | 1.011 | 104.8 |
  | Minneapolis | 0.521 | 0.053 | — | — | — |
  | US/Canada mean (Table 2) | 0.334 | 0.116 | 3.090 | 1.043 | 98.8 |

  Buenos Aires, Detroit and Chicago are the least circuitous, "only 1.1%–1.6% more circuitous than straight-line
  distances" (Results, p. 9).
* **Correlations** (S4, Results, p. 15): r(φ, P4w) = 0.634, r(φ, k̄) = 0.518, r(φ, Pde) = −0.376 and
  r(φ, ς) = −0.432. A grid is therefore a *topological* pattern (crossings, few dead ends) as well as an orientation
  pattern. **Ours:** this is why the heading-only M15 term failed. The 4-way junctions have to be created by straight
  continuation through existing junctions and by crossing, and the K learned proposals rarely contain those moves.
* **Spacing and blocks.**
  * Manhattan's 1811 grid has "800-foot × 200-foot blocks", rotated 29° (S4, p. 11), about 244 × 61 m (ours).
  * Block shape factor Φ is "the ratio between the area of the block and the area of the circumscribed circle", and
    Φ < 1 (S3, §3, eq. 3.2). Manhattan shows "two sharp peaks at Φ ≈ 0.3 and Φ ≈ 0.5, which are the signature of a
    grid-like pattern" (S3, §5, Fig. 6).
  * **Ours:** an a × b rectangle has Φ = 4ab / (π(a² + b²)). That gives 0.64 for a square, 0.51 for 2:1 and 0.30 for
    4:1. Manhattan's 800 × 200 ft block is 4:1, which matches the 0.3 peak.
  * Buenos Aires (S3 group 1, the only member) has blocks "dominated by the square shape and regular rectangles"
    (§4, Fig. 5).

### 2.2 Organic (Marshall's "Altstadt", and "characteristic structure")

* **Orientation.** Near-uniform: Rome φ = 0.005, Paris 0.016, London 0.015, São Paulo 0.002 (S4, Table 1). Charlotte,
  Rome and São Paulo "have nearly uniform distributions of street orientations" (p. 13).
* **Junctions and grain** (S4, Table 1):

  | City | P4w | Pde | k̄ | ς | l̃ (m) |
  |---|---|---|---|---|---|
  | Rome | 0.145 | 0.161 | 2.820 | 1.070 | 73.7 |
  | Venice | 0.073 | 0.300 | 2.474 | 1.090 | 23.2 |
  | Europe mean (Table 2) | 0.172 | 0.172 | 2.814 | 1.061 | 78.7 |

  European streets are "42% more circuitous" than US/Canadian ones (Discussion). **Ours:** this refers to the excess
  circuity, 0.061 against 0.043.
* **Characteristic structure** (S1, §6, p. 151 per the table of contents; the wording below is from the author's
  2005 slides):
  * "three-way junctions are typically in the majority, but likelihood of at least some crossroads and culs-de-sac";
  * "a mixture of short and long routes";
  * relative connectivity "around 0.35–0.45";
  * complexity Ω "typically in the range 0.35 to 0.6".

  The definitions of these indices are in the book and were **not verified**.
* **Junction angles** (S7, §4.3):
  * Across the San Francisco Bay Area, 74% of 3-way junctions are right-angled T's (Type 1), and 69% of 4-way
    junctions have four right angles.
  * The "organic" cluster has a high share of non-right-angle 3-way junctions. Per-cluster numbers are only in
    figures (§5, Fig. 6).
* **Blocks.**
  * The area distribution has a power-law tail with "τ ≈ 2". The authors call this "seemingly universal", so it does
    not distinguish cities (S3, §3, eq. 3.1).
  * All European cities except Athens fall in one subgroup of group 3, which has "various shapes" (S3, §4, Fig. 4).
  * Athens (group 2) has "a dominant fraction of small blocks with shapes broadly distributed".

### 2.3 Radial

* **No source read gives a quantitative radial signature.** Global orientation entropy cannot detect one: radial and
  ring cities such as Paris (φ = 0.016) and Moscow (φ = 0.007) score like organic cities (S4, Table 1).
* Radial patterns appear only as generator rules (G1 "Paris"; G2 radial basis field; G6 "Hex" arms; G11 radial
  tensor fields).
* M15 already flagged that "no paper read defines radialness numerically". Any centre-referenced metric is **ours**
  and must be labelled as such. Marshall's composition and configuration layers (S1, p. 85) may treat radiality;
  **not verified**.

### 2.4 Cul-de-sac / loops-and-lollipops (Marshall's "Distributory")

* **Distributory** is the tree-like, hierarchical type (S1 via S1a; also Wan Mohamad et al. 2020, *Planning Malaysia*
  18(2), 193–205, a weak secondary source).
* **The suburban cul-de-sac** (S2a, p. 1):
  * "usually less than 1,000 feet, serving up to 20 dwellings";
  * a turnaround of "typical radius of 35 to 40 feet";
  * a single entrance that "joins onto a collector street that services other similar cul-de-sacs".
  * A loop street "has two access points" and is usually longer. Both are "often found in the same development".
* **Cost.** The grid "requires up to 50 percent more road construction" (S2a, p. 3).
* **Origins and variants** (S2a, pp. 4–5):
  * Hampstead Garden Suburb (1904) used short, narrow cul-de-sacs with no turnaround, joined by mid-block footpaths.
  * Radburn (1928) clustered houses on cul-de-sacs.
  * Berkeley's barriers turned a grid into cul-de-sacs and loops for cars only.
  * The book's account of FHA/ITE subdivision standards is **not verified** (S2 not read).
* **Dead-end shares** (S4, Table 1):

  | City | Pde | Note |
  |---|---|---|
  | Helsinki | 0.395 | k̄ = 2.348 |
  | Charlotte | 0.288 | φ = 0.002 |
  | Las Vegas | 0.230 | φ = 0.542 |
  | Phoenix | 0.186 | φ = 0.586 |
  | Manhattan, Buenos Aires | 0.027 | for contrast |

  * Phoenix and Las Vegas sit in the "sprawling but relatively low-entropy" cluster (p. 16). **Ours:** a high φ
    together with a high Pde is consistent with an ordered arterial grid filled with dead-end local streets, i.e. a
    mixed pattern.
  * Blocks do not register dead ends: "the block representation does not take into account dead-ends" (S3, Fig. 1).
    Block metrics alone cannot evaluate this pattern.
* **Mechanism precedent** (S6, abstract only): a "spanning tree as a backbone", to which edges are then added
  iteratively to match empirical degree distributions. This supports tree-first growth with controlled loop closure;
  it is not used as evidence for numbers.

## 3. Key table: rules to test, evidence, independent measures, exceptions

The *independent measures* are not directly optimised by the rule. Measures that the rule does target, such as φ for
a heading-constrained grid, are reported only as manipulation checks.

| Pattern | Rule / mechanism to test | Observed morphology supporting it | Independent evaluation measures | Known exceptions |
|---|---|---|---|---|
| **Grid** | **Proposals:** add rule candidates (straight continuation; ±90° branches at configurable spacing d, aligned to a local grid axis) to the K learned ones. **Topology rule:** close blocks by crossing or continuing through junctions where feasible (a T becomes an X) within a snap tolerance. | High P4w (Chicago 0.507, Manhattan 0.572), low Pde (0.027–0.074), k̄ > 3.3, ς ≈ 1.01–1.02 (S4, Table 1); φ correlates with P4w at r = 0.634 (S4, p. 15); rectangular Φ peaks of 0.3 and 0.5 (S3, §5) | P4w, k̄, Pde, ς; block Φ histogram peakedness; block-area spread in the [10³, 10⁵] m² bins (S3 §4); median segment length against d. φ is a manipulation check only. | Multiple non-aligned grids: Buenos Aires φ = 0.151 yet P4w = 0.576 (S4, Discussion); Detroit has two rotated grids; Seattle's downtown is rotated 32°/49° (S4, p. 12). Hilly grid: San Francisco φ = 0.278, P4w = 0.454 (Table 1); S4 does not quantify slope effects. Barcelona city-wide φ = 0.108; the Eixample is a distinct district (S3, §5); chamfer geometry is not in the sources read. Boston has local grids that do not align (S4, p. 11). |
| **Organic** | **Proposals:** persistent but slowly drifting axes, i.e. continuation of the current stroke with curvature or noise (G2 rotation noise as precedent). Varied junction angles: branch angles sampled away from 90°. Asymmetric branching: one side per junction. **Infill rule:** connect into existing streets so that blocks are uneven, without requiring X closure. | Near-uniform orientation (Rome φ = 0.005); P4w ≈ 0.15–0.17; Pde ≈ 0.16–0.17; ς ≈ 1.06–1.07 (S4, Tables 1–2); 3-way junctions in the majority, connectivity 0.35–0.45 (S1 slides); many non-right-angle T's (S7); broad Φ, small blocks (S3, group 2) | Block Φ spread and area distribution (τ ≈ 2 tail, S3); P3w/P4w ratio; ς; median segment length; route continuity (CityStack-style stroke length, M15 §6). Junction-angle share is a manipulation check if angles are sampled directly. | Terrain-driven circuity, not organic growth: Caracas, Hong Kong and Sarajevo, ς 1.133–1.148, "due largely to topography" (S4, p. 9). Planned curvilinear suburbs (C/D types). Paris is near-uniform in orientation but low in dead ends (Pde = 0.050, S4). S3 §6 (citing their ref. 7) attributes Paris's varied block shapes to "urban modification that does not respect the existing geometry"; any link to the low Pde is ours. |
| **Radial** | **Explicit centre c.** **Proposals:** spoke continuation (bearing to/from c) and ring-arc continuation (tangent at constant radius), as separate candidate families. **Topology rules:** spokes must stay continuous (no termination before the ring or the edge without cause); ring arcs close onto spokes (T or X). Ring radii are configurable; partial rings are allowed. | No quantitative source (§2.3). Generator precedent: G1, G2, G6, G11. | Ours, labelled as such: length-weighted radial/tangential deviation by distance band; share of spokes reaching within r₀ of c; ring closure (cycles enclosing c); network/Euclidean distance to c by band (ς to centre); betweenness concentration near c (S5 reviews betweenness as a structural measure). | Partial rings and missing sectors (water or relief). Radial-concentric cities look organic in φ (Paris, Moscow). Radial arterials over local grids or cul-de-sacs: mixed pattern. |
| **Cul-de-sac** | **Backbone:** a collector (learned or grid/organic). **Proposals:** short terminal branches off the collector, length capped at L_max (S2a guideline: < 1,000 ft ≈ 305 m); loop candidates with two access points. **Topology rules:** forbid through-connections between branches (no merge with another component's local branch); allow pedestrian links as a separate layer (not modelled). | Pde up to 0.29–0.40 (Charlotte, Helsinki); k̄ ≈ 2.35–2.55 (S4, Table 1); single entrance onto a collector (S2a, p. 1); 50% less road length than a grid (S2a, p. 3) | Pde (window-closed, M14), cycle ratio, k̄, ς between random origin–destination pairs (detour cost), branch depth from the collector, cul-de-sac length distribution against L_max, road length per km² (grid comparison, S2a) | Loops-and-lollipops mixing loops with cul-de-sacs (S2a, p. 1); grid geometry with cul-de-sac topology for cars (Berkeley barriers, S2a, p. 5); Radburn and Hampstead (pedestrian-connected); ordered arterial grid with dead-end infill (Phoenix, Las Vegas; ours). |
| **Mixed** | The style chosen per road class or per region: e.g. a radial or grid backbone (major/mid) with cul-de-sac or organic locals. This reuses the M15 class gain g(c) and G2's per-region minor fields. | Real cities mix patterns by district (S3 §5–6; S4 Discussion "centrally planned and self-organized spatial patterns coexist") | All of the above, computed per class and per district | Region boundaries are arbitrary (S3 §6 on administrative boroughs). |

## 4. How generators handle style and terrain

| Generator | Learned? | Style mechanism | Where style acts | Explicit topology rules | Terrain |
|---|---|---|---|---|---|
| Parish & Müller 2001 (G1) | No | Rules: grid (angle, maximum block size), Paris radial, SF elevation, Basic; weighted blend by maps | **Proposal** (the rule emits the next segment's parameters) | Local constraints: cut at crossings, extend to join nearby roads, snap | Water masks with prune/rotate/bridge; elevation only via the SF rule; no grade limit |
| Chen et al. 2008 (G2) | No | Tensor basis fields (grid, radial, boundary, height), RBF sum, rotation noise | **Proposal** (the streamline direction is the style) | Interleaved major/minor tracing, d_sep, look-ahead to join | Height field steers headings; water boundary; bridges |
| Galin et al. 2010 (G3) | No | None (single road) | — | — | Anisotropic cost; hard slope maximum; bridges and tunnels |
| Galin et al. 2011 (G4) | No | "Density and pattern" via a γ-proximity graph under a terrain metric (γ = 2 gives a Gabriel graph, γ → ∞ a relative-neighbour graph; Table 1: γ = 1.2 → 19 major / 70 minor roads, γ = 8 → 9 / 29); control cost h(p); waypoints | Graph selection between given cities (inter-city scale, not urban street style) | Path merging into Steiner junctions (Fréchet distance) | Terrain-induced non-Euclidean metric; water; vegetation; bridges and tunnels |
| NTG 2019 (G5) | Yes (encoder–decoder GRU) | One-hot city identity embedding; local style selection while sketching (Fig. 7) | Learned conditioning of the proposal distribution | None stated beyond the generation loop | None |
| CitySprout 2026 (G6) | No | Four fixed L-system grammars (Organic, Grid, Hex, Line) | Proposal (production rules) | Cut back at crossings; merge close nodes | None |
| CityStack 2026 (G7) | Partly (GMM typologies) | 13 data-derived typologies; segment length, angle and branching sampled per cell | Proposal (statistics per cell) | Not implemented | Not implemented (future work) |
| CaRoLS 2025 (G8) | Yes (latent diffusion) | Condition image ("natural and socio-economic") | Not verified | Not verified | Not verified |
| RoadWeaver 2026 (G9) | Hybrid | 11-D condition (6-D style code, density, gridness, radialness, organicness, bearing entropy) with no formulas given; style-dependent angular perturbation Δφ in rule-based growth (§III-C) | Learned skeleton (VQ-VAE + masked Transformer), then **rule-based** structure-tensor growth | Snapping, A* reconnection over a learned cost, largest-component filter (§III-C). Reports dead-end ratio 10.7 ± 3.6% and cycle ratio 85.2 ± 6.0% (Table II). Style and mixing **not tested** (§IV-B). | None |
| RoBus 2024 (G10) | Yes (baselines) | Road orientation and building density as controls | Not verified | Not verified | Not verified |
| Gao & Patel 2026 (G11) | No | Sketch symbols mapped to grid/radial tensor-field attractors | Proposal (field) | Not verified | Not verified |
| Barthelemy & Boeing 2025 (S6) | No (statistical model) | One density parameter | Mechanism: spanning tree, then edge additions | Implicit (tree, then loops) | None |
| **UrbanMorph M15** | Yes (terrain-conditioned turtle) | Heading reweighting κ·(cos 4Δθ − 1)/2; loop penalty; stop bias | **Rerank only** (K = 8 learned proposals) | M14 snap/merge/split/clip; network potentials | Elevation, slope and water inputs; hard water; optional grade cap |

## 5. Novelty assessment

**Already done:**
1. **Style-aware proposals are the core of procedural modelling.** G1 emits grid and radial continuations with
   block-size limits. G2's streamlines *are* style proposals, already composed with terrain-aligned fields. G6 and G7
   sample per-style or per-typology lengths, angles and branching. Grid spacing, radial centres, organic noise and
   per-class or per-region styles are all prior art (M15 §5 already conceded this for the field terms).
2. **Explicit topology rules on generated streets are standard:** cut and snap at crossings, extend to join (G1),
   look-ahead joining (G2), merging into Steiner junctions (G4), node merging (G6).
3. **Learned generation plus rule-based growth and repair with style conditioning exists.** RoadWeaver (G9) runs
   learned skeleton → rule-based tensor growth with a style-dependent perturbation → snapping, A* reconnection and
   component filtering. It has no terrain, no attribute formulas, no style test and no code yet.
4. **Terrain plus style plus topology rules exists procedurally** in G2. Hierarchical terrain-aware networks exist in
   G4, but at inter-city scale.

**What remains (narrow, conditional):** among the works read, none
1. mixes *rule-generated, style-specific candidates* (grid continuation and closure, radial spoke and ring, organic
   drift, cul-de-sac branch) with candidates drawn from a *learned, terrain-conditioned* vector prior trained without
   style input, in one candidate set;
2. selects among them with that prior's terrain and network potentials under hard water and grade feasibility;
3. evaluates the patterns on held-out real terrain with morphology measures the rules do not target (P4w, Pde, k̄, ς,
   block Φ and area, §3), stratified by slope.

This is an engineering combination plus an evaluation protocol, not a new mechanism.

**Risks to the claim:**
* With rule candidates in the set, the learned prior may only act as a terrain filter. An ablation is needed:
  rule-only proposals with the same potentials (close to a G2-style baseline) against a mixed candidate set.
* The claim stays conditional on reading CaRoLS, DRoLaS and Zhou 2024 in full, on a Scopus or Web of Science search,
  and on Marshall's and Southworth & Ben-Joseph's books (both unread). M15 §8 already lists these open items.

**Lessons for M16 design:**
* Grids need X-junction closure rules, not heading terms (S4 correlations).
* Radial needs our own centre-referenced metrics (no literature metric).
* Cul-de-sac evaluation must use node-level Pde, because block metrics ignore dead ends (S3).
* Every grid result needs a hilly or multi-grid exception analysis (San Francisco, Detroit, Buenos Aires in S4).
