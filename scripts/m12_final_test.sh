#!/bin/bash
# Milestone 12: the ONE-TIME final test evaluation. Requires explicit user authorisation:
#     M12_TEST_AUTHORISED=yes bash scripts/m12_final_test.sh
# Everything it uses is frozen BEFORE it runs: checkpoints (outputs/m12/training/*/summary.json), thresholds
# (configs/m12_feasibility.json), normalisation (data/processed/m12/terrain/norm.json), T = 0.5, the metrics and
# criteria of docs/M12_EVALUATION_PROTOCOL.md. Nothing is tuned on its output.
# Steps:
#   1. build the San Diego test_t tiles (scripts/m12_build_tiles.py --authorised-test);
#   2. fetch DEMs for test_t and for the M2 test tiles of Spain and the US (Japan: no DEM);
#   3. 3D-aware graphs + terrain arrays for the test tiles;
#   4. regimes A (conditional) and B (S10 vs the 19 M2 test tiles) on the test splits; tests H1-H5 and H1lp-H3lp.
set -e
if [ "$M12_TEST_AUTHORISED" != "yes" ]; then
  echo "refusing: the held-out test splits may only be used after explicit authorisation (set M12_TEST_AUTHORISED=yes)"
  exit 1
fi
cd "$(dirname "$0")/.."
echo "final test evaluation started $(date -u +%FT%TZ)" | tee -a outputs/m12/final_test.log
.venv/bin/python scripts/m12_build_tiles.py --authorised-test
M12_INCLUDE_TEST=yes .venv/bin/python scripts/m12_fetch_dem.py --country spain
M12_INCLUDE_TEST=yes .venv/bin/python scripts/m12_fetch_dem.py --country united_states
M12_INCLUDE_TEST=yes .venv/bin/python scripts/m12_graph_levels_effect.py
M12_INCLUDE_TEST=yes .venv/bin/python scripts/m12_prepare_data.py
M12_INCLUDE_TEST=yes .venv/bin/python scripts/m12_evaluate.py --regime A --split test
M12_INCLUDE_TEST=yes .venv/bin/python scripts/m12_evaluate.py --regime B --split test
M12_INCLUDE_TEST=yes .venv/bin/python scripts/m12_evaluate.py --regime stats --split test
echo "final test evaluation finished $(date -u +%FT%TZ)" | tee -a outputs/m12/final_test.log
