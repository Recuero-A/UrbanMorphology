# Berwaldt, Bettker, Pozzer 2020: Procedural Generation of Favela Layouts on Arbitrary Terrains

**Added item: this paper is not in `docs/M18_LITERATURE.md`.** It appears only as a one-line row in `docs/M12_LITERATURE_REVIEW.md` (abstract-level). It is included because it targets **steep informal settlements**, with zigzag roads on steep ground and many dead ends. That is the regime of our hill-town benchmarks.

## 1. Header

- **Citation:** Natan Luiz Paetzhold Berwaldt, Rafael Vales Bettker, Cesar Tadeu Pozzer. "Procedural Generation of Favela Layouts on Arbitrary Terrains." In *2020 19th Brazilian Symposium on Computer Games and Digital Entertainment (SBGames)*, Recife, Brazil, 7–10 Nov 2020, IEEE, pp. 136–144.
- **DOI:** 10.1109/sbgames51465.2020.00027
- **URLs opened:**
  - https://api.crossref.org/works/10.1109/sbgames51465.2020.00027: title, authors, venue, pages, event (Crossref holds no abstract).
  - https://api.openalex.org/works/doi:10.1109/sbgames51465.2020.00027: abstract (reconstructed from OpenAlex's inverted index). OA status "closed"; no repository copy listed.
- **Access level:** **AO** (abstract only). Everything beyond the abstract is **not verified**.

## 2. BibTeX

```bibtex
@inproceedings{berwaldt2020favela,
  author    = {Berwaldt, Natan Luiz Paetzhold and Bettker, Rafael Vales and Pozzer, Cesar Tadeu},
  title     = {Procedural Generation of Favela Layouts on Arbitrary Terrains},
  booktitle = {2020 19th Brazilian Symposium on Computer Games and Digital Entertainment (SBGames)},
  publisher = {IEEE},
  pages     = {136--144},
  year      = {2020},
  doi       = {10.1109/sbgames51465.2020.00027}
}
```

## 3. What it does

Per the abstract, it generates favela layouts on arbitrary terrain:

- a road system from A\* on the heightmap, with zigzag roads on steep elevations, more dead ends and large blocks;
- quadtree subdivision of blocks into building lots;
- an alley system connecting every building in a block to the road.

## 4. How it generates

- **Procedural** (A\* and quadtree), per the abstract.
- **Representation:** not verified. Roads come from A\* on the heightmap, presumably grid paths.
- **Not verified:** the step-by-step algorithm (cost terms, how zigzags are produced, how road targets are chosen).

## 5. Inputs, outputs and how terrain enters

- **Inputs:** a heightmap (per the abstract); other inputs not verified.
- **Outputs:** roads, lots, alleys, building placement.
- **Terrain:** the road A\* "is based on the heightmap", and zigzag roads adapt to steep elevations. Cost form and thresholds are **not verified**.
- **Water:** not mentioned in the abstract.

## 6. Evaluation

**Not verified.** The abstract mentions no metrics, real-site comparison, baselines or user study.

## 7. Quotes for citation

From the abstract as reconstructed from OpenAlex. No page numbers.

- Abstract: "The road system uses the A* algorithm and is based on the heightmap, having the characteristics of favelas on uneven terrains."
- Abstract: "The method allows zigzag roads to adapt to the steep elevations, a higher number of dead ends, and large blocks, fitting several buildings, and an alley system."
- Abstract: "Our purpose is to generate the layout of a favela on arbitrary terrain procedurally."

A script checked every quote against the saved OpenAlex abstract.

## 8. Comparability with UrbanMorph

- **Inputs:** probably runnable on our DEM tile (heightmap). Water, entry points and classes are not verified.
- **Code:** none found (not searched beyond the metadata APIs).
- **Effort:** not estimable without the full text. An A\*-on-heightmap road system with switchbacks is in the 15–30 h range (my estimate, labelled, assuming the method is as simple as the abstract suggests).
- **Fair comparison:** probably full network growth on steep informal-settlement tiles. Not verified.

## 9. Ideas we could adopt for evaluation

- The abstract treats **"a higher number of dead ends"** as a target property of steep informal settlements. Our dead-end and closed-island potentials must therefore be judged against the **real** dead-end share of each tile's terrain and settlement type, not pushed toward zero.
- **Zigzag / switchback presence** on steep tiles is a qualitative signature worth quantifying: the count of turns above 120° per km on segments above X % grade, real vs generated.
