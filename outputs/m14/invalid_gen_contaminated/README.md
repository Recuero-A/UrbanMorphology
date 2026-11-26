# Invalid GEN results (set aside, not deleted)

These 28 runs and `real/gen.json` were labelled `gen` (M13 GEN: Girona, Manresa, San Francisco) but contain the GEN2 tiles.
`scripts/m14_build_gen2.py` re-pointed `m13_build_gen_cities.GEN` to GEN2 at import time, so any worker process that had
already loaded a GEN2 run read GEN2 when it later loaded `gen`. Found on 2026-11-18 when the GEN and GEN2 panels of
`fig_tradeoff.png` came out identical. The patch now happens only in the build script's `main()`; the GEN runs were re-run.
No other tile set goes through that code path.
