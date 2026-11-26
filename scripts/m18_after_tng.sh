#!/bin/bash
# M18 D7 (c): once TNg seed 2 has finished training, evaluate TNg exactly as TN was evaluated.
cd "$(dirname "$0")/.."
while pgrep -f "m18_train_tng.py --seed 2" > /dev/null; do sleep 60; done
test -f checkpoints/m12/TNg_s2/best.pt || { echo "TNg_s2 missing"; exit 1; }
.venv/bin/python scripts/m14_experiments.py --plan configs/m18_plans/gen2_tng.json --device cpu --worker 0/2 > outputs/m18/glo30/gen2_w0.log 2>&1 &
.venv/bin/python scripts/m14_experiments.py --plan configs/m18_plans/gen2_tng.json --device cpu --worker 1/2 > outputs/m18/glo30/gen2_w1.log 2>&1 &
.venv/bin/python scripts/m18_natural_terrain.py generate > outputs/m18/natural/generate_tng.log 2>&1 && .venv/bin/python scripts/m18_natural_terrain.py score > outputs/m18/natural/score_tng.log 2>&1
wait
echo "TNg evaluation done $(date +%H:%M)"
