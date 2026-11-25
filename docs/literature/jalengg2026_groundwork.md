# "Groundwork" (GitHub jalengg) 2026: CaRoLS-style terrain-conditioned latent diffusion for US suburban road networks

**NEW: not in the M18 list.** This is an unpublished research prototype on GitHub, not a peer-reviewed paper. It is included because it is a 2026 learned road-layout generator with released code that takes elevation and water as inputs.

## 1. Header

- **Citation:** jalengg (GitHub user; commit author name "Jalen"; full name not verified). *Groundwork*. GitHub repository, created 2026-03-08, last push 2026-06-03, MIT licence.
- **DOI:** none.
- **URLs opened:**
  - https://api.github.com/repos/jalengg/groundwork (licence, dates, language).
  - https://raw.githubusercontent.com/jalengg/groundwork/main/README.md
  - https://raw.githubusercontent.com/jalengg/groundwork/main/docs/carols_spec_audit.md
  - https://raw.githubusercontent.com/jalengg/groundwork/main/docs/findings.md
  - https://raw.githubusercontent.com/jalengg/groundwork/main/data_pipeline/cities.yaml
  - https://raw.githubusercontent.com/jalengg/groundwork/main/requirements.txt
  - https://raw.githubusercontent.com/jalengg/groundwork/main/model/eval_metrics.py
  - https://raw.githubusercontent.com/jalengg/groundwork/main/data_pipeline/elevation_layer.py
  - https://raw.githubusercontent.com/jalengg/groundwork/main/.gitignore
  - https://api.github.com/repos/jalengg/groundwork/git/trees/main?recursive=1
  - https://api.github.com/repos/jalengg/groundwork/releases (empty)
  - https://api.github.com/repos/jalengg/groundwork/commits?per_page=5
- **Access level:** README, design documents and selected source files read. No code was run.

## 2. BibTeX

```bibtex
@misc{jalengg2026groundwork,
  author       = {{jalengg}},
  title        = {Groundwork},
  year         = {2026},
  howpublished = {\url{https://github.com/jalengg/groundwork}},
  note         = {GitHub repository, MIT licence, accessed 2026-11-25}
}
```

## 3. What it does

A loose reimplementation of CaRoLS (Feng et al. 2025) generates 5-class road rasters for US suburbs, conditioned on terrain and land use. The intended output is a road map for a Cities: Skylines mod; the mod is "not yet built". The author's own findings report that outputs lack realistic suburban structure.

## 4. Learned or procedural, and HOW

- **Learned:** two-stage **latent diffusion**.
  1. A VAE maps a 5 × 512 × 512 one-hot road raster (background, residential, tertiary, primary/secondary, motorway/trunk) to a 4 × 64 × 64 latent.
  2. A conditional U-Net with LDE/GCI "condition-aware decoder blocks" runs DDPM training and DDIM 50-step sampling with classifier-free guidance.
- **Data:**
  - 17 US suburban cities, 150 tiles each (`cities.yaml`). One city, Irving TX, is held out for validation.
  - Tiles are 512 × 512 at 5 m (2.56 km).
  - OSM roads, OSM land use, and SRTM1 elevation downloaded from the AWS terrain tiles.
- **Raster to graph:** per the README, "dilate → skeletonize → prune". `requirements.txt` lists `sknw`, `networkx` and `rdp`, which suggests skeleton-to-graph conversion with simplification; the exact chain was not verified in `postprocess.py`.

## 5. Inputs/outputs and how terrain enters

- **Conditioning** is a 7 × 512 × 512 stack: [elevation, water, residential, commercial, industrial, parkland, agricultural].
- **Elevation** is SRTM 30 m, bilinearly reprojected and "normalized 0-1 within the tile" (`elevation_layer.py` docstring). Absolute elevation is discarded; relief is relative per tile.
- **Water** comes from OSM. There is no slope channel.
- An earlier "existing-roads" conditioning channel was removed because "model was doing semantic segmentation, not generation" (`findings.md`).
- **Output:** a 5-class road raster, then (post-processed) a skeleton/graph.

## 6. Evaluation

- `eval_metrics.py` has:
  - a connectivity index (mean degree);
  - transport convenience (mean Euclidean/shortest-path ratio over 200 random node pairs, with unreachable pairs **skipped**);
  - FID and KID via torchmetrics (2048-d features).
  - `vlm_eval.py` (not read) suggests a vision-language-model judge.
- **No published results.** `findings.md` reports qualitative failure modes:
  - predictions that look like a "honeycomb texture" instead of parallel/grid suburban streets;
  - at one stage, "dense salt-and-pepper noise with no road topology";
  - a CDB variant faithful to the paper "stuck at 0.808" loss.

## 7. Verbatim quotes (repository documents, not a paper)

- `README.md`: "Status: research prototype. The ML pipeline trains and produces road-shaped outputs. Known limitations: residential street channel is noisy/scattered, and the in-game mod integration is not yet built."
- `docs/findings.md`, "What we ruled out empirically": "Was a leak — model was doing semantic segmentation, not generation. Removed." This refers to the existing-roads conditioning channel.
- `docs/findings.md`, "The honeycomb-vs-grid problem (open)": "Even with a model that trains well, predictions look like a **honeycomb texture** — irregular cellular shapes — instead of real US suburban road networks".
- `docs/findings.md`: "Validation split: held out city `irving_tx`. Caveat: stylistically similar to training cities, so this is a weak generalization test".

## 8. Comparability with UrbanMorph

- **Code and licence:** https://github.com/jalengg/groundwork, **MIT**.
- **Weights:** **none**. `*.pt`/`*.pth` and `checkpoints/` are git-ignored, and there are no GitHub releases.
- **Dependencies:** `torch>=2.0.0` and `torchvision>=0.15.0`, so recent PyTorch with MPS support. Training scripts target A100/H100 SLURM jobs.
- **macOS without CUDA:** plausible for inference and small training runs on MPS/CPU; **not tested**.
- **Inputs we could supply:**
  - DEM: yes, though they normalise per tile, so we would keep or alter that.
  - Water mask: yes.
  - Land-use classes: from OSM.
  - Boundary: implied by the tile.
  - Surrounding streets: not supported (the channel was removed as a leak).
  - Resolution: 512 px × 5 m = 2.56 km tiles, vs our 512 m at 4 m. We would retrain at 128 px × 4 m or crop.
- **Fair comparison:** terrain-only growth, the same setting as UrbanMorph.
- **Retraining:** required.
- **Credibility:** low. It is unpublished, not peer-reviewed, and self-reports unrealistic outputs. As an external baseline it would invite "strawman" criticism.
- **Effort (estimate):** 15–30 h human time to adapt the data pipeline to our tiles, train and post-process, plus 1–3 days of compute on Apple Silicon. **Use at most as an implementation reference for a CaRoLS-style baseline, not as a published baseline.**

## 9. Ideas we could adopt

1. **Leakage audit by channel ablation:** check that removing a suspect channel (for us, a high-resolution DEM that may carry street cuts) does not collapse performance from "generation" to "segmentation".
2. **Spectral/orientation statistics for realism:** their honeycomb diagnosis, an isotropic ring in |FFT| vs axis-aligned peaks for grids, suggests an orientation-spectrum or bearing-entropy distance as a realism metric that catches "texture without structure".
3. **Score unreachable OD pairs explicitly.** Their transport-convenience code silently skips unreachable pairs, which is the pitfall flagged in `docs/M18_LITERATURE.md`. Our metric should count them, e.g. as 0, or report them separately.
