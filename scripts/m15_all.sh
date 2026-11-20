#!/bin/bash
# Milestone 15, end to end (docs/M15_PROTOCOL.md): audit replay, benchmark build, development sweep, evaluation and
# real-city pilot runs, report. Resumable: finished runs are skipped. Development data only; no held-out metro is built.
#   bash scripts/m15_all.sh [workers]        (default 3 CPU workers, 2 threads each)
set -e
cd "$(dirname "$0")/.."
P=.venv/bin/python
W=${1:-3}
$P scripts/m15_replay_baseline.py --out replay_rerun.json                 # M14 baselines replay bit for bit (the
                                                                          # archived records replay*.json are kept)
$P scripts/m15_build_benchmark.py --stage all                             # manifest, GLO-30, graphs, strata
run_plan () {                                                             # one plan on W parallel workers
  for k in $(seq 0 $((W - 1))); do $P scripts/m15_run.py --plan "configs/m15_plans/$1.json" --worker "$k/$W" & done
  wait
}
run_plan dev_tune                                                         # development sweep (strengths are frozen in
$P scripts/m15_dev_select.py                                              # configs/m15_style.json; prints the rule's choice)
$P scripts/m15_plans.py eval && run_plan eval                             # synthetic interventions
$P scripts/m15_plans.py real && run_plan real                             # real-city pilot (development metros)
$P scripts/m15_report.py                                                  # outputs/m15/REPORT.md and figures
