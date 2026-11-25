# Mizdal & Pozzer 2018: Procedural Content Generation of Villages and Road System on Arbitrary Terrains

**Added item: this paper is not in `docs/M18_LITERATURE.md`.** It appears only as a one-line row in `docs/M12_LITERATURE_REVIEW.md` (abstract-level). It is included because its road system is A\* with slope-aware costs connecting terrain-sited villages, with T-junctions.

## 1. Header

- **Citation:** Tiago Boelter Mizdal, Cesar Tadeu Pozzer. "Procedural Content Generation of Villages and Road System on Arbitrary Terrains." In *2018 17th Brazilian Symposium on Computer Games and Digital Entertainment (SBGames)*, Foz do Iguaçu, Brazil, 29 Oct – 1 Nov 2018, IEEE.
  - Crossref gives the pages as "205-2056", which looks like a data-entry error; the true page range is **not verified**.
- **DOI:** 10.1109/sbgames.2018.00032
- **URLs opened:**
  - https://api.crossref.org/works/10.1109/sbgames.2018.00032: title, authors, venue, event.
  - https://api.openalex.org/works/doi:10.1109/sbgames.2018.00032: abstract (reconstructed from OpenAlex's inverted index). OA status "closed".
  - https://www.sbgames.org/sbgames2018/files/papers/ComputacaoFull/188241.pdf (found via web search): connection refused on 2026-11-25. It was not read, and it is **not verified** to be this paper.
- **Access level:** **AO**.

## 2. BibTeX

```bibtex
@inproceedings{mizdal2018villages,
  author    = {Mizdal, Tiago Boelter and Pozzer, Cesar Tadeu},
  title     = {Procedural Content Generation of Villages and Road System on Arbitrary Terrains},
  booktitle = {2018 17th Brazilian Symposium on Computer Games and Digital Entertainment (SBGames)},
  publisher = {IEEE},
  year      = {2018},
  doi       = {10.1109/sbgames.2018.00032}
}
```

## 3. What it does

Per the abstract, it analyses the terrain to delimit suitable village areas, places buildings in acceptable areas, and generates the roads connecting the villages. The roads use A\* with custom slope-aware costs, can create T-junctions, and run "from a starting point to any other road".

## 4. How it generates

- **Procedural** (A\*). The rest is not verified.
- **Representation:** not verified (A\* paths on a grid, presumably).
- **Algorithm details:** not verified.

## 5. Inputs, outputs and how terrain enters

- **Inputs:** terrain (heightmap); seed.
- **Outputs:** village areas, buildings, a road network with T-junctions.
- **Terrain:** a slope-aware A\* cost and terrain-defined village areas. Cost form and thresholds are **not verified**.

## 6. Evaluation

**Not verified** (the abstract mentions none).

## 7. Quotes for citation

From the abstract as reconstructed from OpenAlex. No page numbers.

- Abstract: "In this paper we propose a solution that analyzes the terrain, defines proper areas for the villages and generates the roads connecting them."
- Abstract: "The road system is generated using the A* algorithm with our own cost functions that consider the slope of the terrain."
- Abstract: "Our method is also able to create T-junctions and works from a starting point to any other road."

A script checked every quote against the saved OpenAlex abstract.

## 8. Comparability with UrbanMorph

- **Inputs:** probably DEM plus anchors, as in Emilien 2012 or Galin 2010. Not verified.
- **Code:** none found.
- **Effort:** not estimable without the full text.
- **Fair comparison** (if it is as described): connector routing between supplied points, with T-junction attachment to existing roads.

## 9. Ideas we could adopt for evaluation

- **"Route to the nearest existing road".** The abstract's "from a starting point to any other road" is the same attachment rule as Emilien's road reuse. Count T-junctions vs X-junctions (degree-3 vs degree-4 share) for real vs generated on steep tiles, where T-junctions should dominate.
